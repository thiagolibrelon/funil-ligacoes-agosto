"""Teste GPT via llm-gate: modelos disponiveis, 5 ligacoes de agosto com gabarito
Gemini, e a configuracao que gasta menos tokens chegando no mesmo resultado.

RODA NA MAQUINA/REDE DA LOCALIZA. Precisa de: Python 3 + requests, amostra_5.json
nesta pasta, e a chave do llm-gate (env var API_KEY) — ou o pacote src.settings
do time (get_llm_url/get_llm_headers), que e usado automaticamente se importavel.

Uso:
  python teste_gpt.py modelos            # etapa 1: quais modelos o llm-gate aceita
  python teste_gpt.py rodar              # etapas 2-3: matriz de configuracoes x 5 ligacoes
  python teste_gpt.py comparar           # relatorio: tokens, custo e concordancia com o Gemini

Opcoes do rodar:
  --configs C0,C3        roda so essas configuracoes (padrao: todas)
  --simular              nao chama a API; gera respostas a partir do gabarito (teste do script)
  --refazer              refaz as configs pedidas mesmo que ja tenham rodado
  --amostra 100          usa amostra_100.json (tambem 300, 500; padrao 5) — vale para rodar e comparar
  --workers 4            pedidos em paralelo (padrao 4)

Saidas (nesta pasta): modelos_disponiveis.json, resultados.jsonl, relatorio.csv,
relatorio_detalhe.csv. Nenhuma delas contem transcricao — sao essas que voltam.
"""
import argparse
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import requests

AQUI = Path(__file__).resolve().parent
AMOSTRA = AQUI / "amostra_5.json"  # trocado por --amostra
MODELOS_JSON = AQUI / "modelos_disponiveis.json"
RESULTADOS = AQUI / "resultados.jsonl"
RELATORIO = AQUI / "relatorio.csv"
RELATORIO_DETALHE = AQUI / "relatorio_detalhe.csv"

URL_PADRAO = "https://llm-gate-np.localiza.dev/llm-gate/v2/chat/completions"


def _carregar_classificador():
    """C12 usa o prompt e a normalizacao do classificar_ligacoes_diario.py (mesma pasta ou ../scripts)."""
    for pasta in (AQUI, AQUI.parent / "scripts"):
        if (pasta / "classificar_ligacoes_diario.py").exists():
            sys.path.insert(0, str(pasta))
            import classificar_ligacoes_diario as cld
            return cld
    return None


CLD = _carregar_classificador()
TIMEOUT_S = 120
MAX_TENTATIVAS = 3

CANDIDATOS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
              "gpt-5", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3-mini"]
# modelos testados na matriz alem de gpt-4o / gpt-4o-mini, se a etapa 1 achar disponiveis
EXTRAS_MATRIZ = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini"]


# ---------------------------------------------------------------------------
# Conexao — mesma origem de URL/headers que a funcao do time (src.settings),
# com fallback para URL conhecida + API_KEY
# ---------------------------------------------------------------------------
def conexao():
    try:
        from src.settings import get_llm_headers, get_llm_url
        return get_llm_url(), dict(get_llm_headers()), "src.settings"
    except Exception:
        chave = os.getenv("API_KEY")
        if not chave:
            sys.exit("Defina a variavel de ambiente API_KEY (chave do llm-gate) ou rode de dentro do projeto que tem src.settings.")
        return URL_PADRAO, {"Content-Type": "application/json", "api_key": chave}, "API_KEY"


def eh_raciocinio(modelo):
    return modelo.startswith(("gpt-5", "o1", "o3", "o4"))


def montar_payload(modelo, mensagens, max_tokens, json_mode=True):
    p = {"model": modelo, "messages": mensagens}
    if eh_raciocinio(modelo):
        # modelos de raciocinio: sem temperature; o teto inclui tokens de raciocinio
        p["max_completion_tokens"] = max_tokens * 4
        p["reasoning_effort"] = "minimal" if modelo.startswith("gpt-5") else "low"
    else:
        p["temperature"] = 0
        p["max_tokens"] = max_tokens
    if json_mode:
        p["response_format"] = {"type": "json_object"}
    return p


def extrair_uso(j):
    u = j.get("usage") or {}
    det_in = u.get("prompt_tokens_details") or {}
    det_out = u.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens") or u.get("promptTokens") or 0,
        "completion_tokens": u.get("completion_tokens") or u.get("completionTokens") or 0,
        "cached_tokens": det_in.get("cached_tokens") or 0,
        "reasoning_tokens": det_out.get("reasoning_tokens") or 0,
        "custo": ((j.get("cost") or {}).get("token") or {}).get("total"),
        "modelo_resposta": j.get("model"),
    }


def post(url, headers, payload):
    h = dict(headers)
    h["X-Correlation-ID"] = str(uuid.uuid4())
    t0 = time.time()
    r = requests.post(url, headers=h, json=payload, timeout=TIMEOUT_S)
    return r, round(time.time() - t0, 2)


# ---------------------------------------------------------------------------
# Etapa 1 — modelos
# ---------------------------------------------------------------------------
def cmd_modelos(_args):
    url, headers, origem = conexao()
    print(f"conexao via {origem}: {url}")
    saida = {"url": url, "listagem": None, "sondagem": []}

    url_models = re.sub(r"/chat/completions/?$", "/models", url)
    try:
        h = dict(headers); h["X-Correlation-ID"] = str(uuid.uuid4())
        r = requests.get(url_models, headers=h, timeout=30)
        saida["listagem"] = {"url": url_models, "status": r.status_code, "corpo": r.text[:5000]}
        print(f"GET {url_models} -> {r.status_code}")
        if r.ok:
            print(r.text[:2000])
    except Exception as e:
        saida["listagem"] = {"url": url_models, "erro": str(e)}
        print(f"GET {url_models} falhou: {e}")

    print("\nsondando modelos (1 pedido minimo cada):")
    for m in CANDIDATOS:
        payload = montar_payload(m, [{"role": "user", "content": "Responda apenas: ok"}], 5, json_mode=False)
        try:
            r, lat = post(url, headers, payload)
            item = {"modelo": m, "status": r.status_code, "latencia_s": lat, "disponivel": r.ok}
            if r.ok:
                item.update(extrair_uso(r.json()))
            else:
                item["erro"] = r.text[:300]
        except Exception as e:
            item = {"modelo": m, "disponivel": False, "erro": str(e)[:300]}
        saida["sondagem"].append(item)
        print(f"  {m:<14} {'OK ' if item.get('disponivel') else 'NAO'}  {item.get('status', '')}  "
              f"{item.get('modelo_resposta') or item.get('erro', '')[:90]}")

    MODELOS_JSON.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {MODELOS_JSON.name}")


# ---------------------------------------------------------------------------
# Prompts — taxonomia identica a scripts/analisar_gemini_julho.py (a mesma usada
# pelo Gemini pago de agosto), trocando CSV por JSON
# ---------------------------------------------------------------------------
CONTEXTO = (
    "Empresa: Localiza (locacao de veiculos mensais para PJ)\n"
    'Vendedor = "A" | Cliente = "C"\n'
    "STT fragmentado - junte turns do mesmo speaker. <unk> = inaudivel, ignore."
)

PAPEL = {
    "P1_P2": "Voce e analista de comportamento comercial em locacao de veiculos corporativos.",
    "P3": "Voce e analista de Customer Success e de expansao de receita em contas B2B de locacao de frotas corporativas.",
    "P5_P6": "Voce e especialista em Challenger Sale e coaching de vendas B2B.",
    "P9": "Voce e especialista em comportamento comercial e Customer Success B2B.",
}

TAXO = {
    "P1_P2": """## P1 (Conversao)
desfecho: fechou_novo|fechou_renovacao|fechou_upsell|interessou_nao_fechou|nao_era_venda
intencao_entrada: cliente_queria|vendedor_puxou|misto|indefinido
tentativa_comercial: SIM|NAO
B1-B10 (comportamentos do vendedor presentes na ligacao):
B1=diagnostico antes do produto | B2=ancora de preco alto | B3=usou historico do cliente
B4=avisou risco ao cliente | B5=upsell conectado a dor | B6=urgencia com fato verificavel
B7=prometeu batalhar internamente | B8=ligou proativamente | B9=conhecimento pessoal
B10=perguntou sobre concorrente/frota propria

## P2 (Tipo de ligacao)
tipo: nova_venda|renovacao|upsell|retencao|suporte_operacional|pos_venda_sinistro|pos_venda_manutencao|cobranca|onboarding|duvida_contrato|misto|sem_conteudo
quem_iniciou: vendedor|cliente|indefinido""",

    "P3": """## P3 (Problemas)
problemas_identificados: codigos, ou ["nenhum"] se a ligacao nao teve problema.
So marque um codigo se o problema for tema central ou relevante da ligacao — NAO marque por mencao
tangencial de 1 palavra-chave sem um problema real por tras dela.
P1=Acesso/sistema(portal fora do ar,senha,biometria) | P2=Faturamento/cobranca(boleto errado,fatura errada)
P3=Multa/infracao | P4=Manutencao/substituto(carro em oficina,sem substituto)
P5=Condutor/cadastro(troca de condutor,habilitacao) | P6=Km excedente(cobranca incorreta,plano errado)
P7=Disponibilidade(carro reservado indisponivel) | P8=Franquia/agencia(processo diferente em franqueada)
P9=Sinistro(acidente,franquia de dano)
P10=Reserva/sistema(reserva nao aparece no sistema, status incorreto, erro ao criar/editar uma reserva,
bug reproduzivel do fluxo de reserva). NAO marcar P10 quando for so duvida de como navegar/usar o
portal sem nenhum erro real (isso e "nenhum", nao P10), nem quando o assunto real da ligacao for outro
(fatura, cadastro, troca de contato comercial) e "reserva" so aparecer de passagem.
P11=Outro
P12=Manipulacao de pesquisa/NPS (vendedor pede/dita a nota — quem_relatou=vendedor_detectou, foi_resolvido_na_ligacao=NAO)
quem_relatou: cliente|vendedor_detectou|nenhum
foi_resolvido_na_ligacao: SIM|NAO|PARCIAL|nenhum
escalou_para: 0800|supervisora|IT|nao_escalou|nenhum""",

    "P5_P6": """## P5 (Challenger)
CH1=Custo oculto revelado | CH2=ROI calculado | CH3=Urgencia por regra real
CH4=Concorrente superado com dado | CH5=Dor->produto conectado
CH6=Antecipacao de problema | CH7=Descoberta de necessidade nao declarada
resultado_imediato: cliente_aceitou|cliente_refletiu|sem_reacao_visivel
qualidade: alta|media|baixa
Se nao teve challenger: teve_challenger=NAO, codigos_challenger=[], resultado_imediato e qualidade = "nenhum"

## P6 (Oportunidades perdidas)
OP1=Suporte sem pergunta comercial | OP2=Dor sem produto conectado | OP3=Fechamento sem urgencia
OP4=Referral nao solicitado | OP5=Dado estrategico nao explorado
OP6=Upsell obvio nao tentado | OP7=Churn nao detectado | OP8=Escalada desnecessaria
valor_potencial_R$: estimativa mensal (telemetria=50/v, protecao=80/v, anual=200, eletrico=3600/v)
Se nao ha oportunidade perdida: oportunidades_perdidas=["nenhuma"]""",

    "P9": """## P9 (Abertura e fechamento)
AB1=Contextualizada | AB2=Relacional | AB3=Generica | AB4=Reativa | AB5=De protecao
FC1=Compromisso duplo(vendedor+cliente+prazo) | FC2=Proximo passo so do vendedor
FC3=Convite generico("se precisar me chama") | FC4=Aberto | FC5=Diretivo
usou_nome: SIM|NAO | proximo_passo_claro: SIM|NAO | prazo_definido: SIM|NAO""",

    "P7": """## P7 (Cross-sell e upsell)
PR1=Aluguel mensal leve(RAC,core) | PR2=Aluguel pesado(caminhao,carreta — NAO inclui PR1) | PR3=Telemetria
PR4=Protecao total/cobertura de avarias | PR5=Upgrade de categoria | PR6=Km adicional | PR7=Contrato anual
PR8=Veiculos eletricos | PR9=ZARP | PR10=Meoo | PR11=Venda de Seminovos
PR12=Gestao de frotas (contrato longo, cliente adquire veiculo personalizavel, administrado pela Localiza — diferente de PR7)
janela_perdida: SIM se o cliente disse algo que se conecta a um produto e o vendedor NAO ofereceu | NAO caso contrario
produto_janela: codigo PR da janela perdida (ou "nenhum")
Exemplos de janela: multa em condutor errado->PR3 | frota propria/personalizacao->PR12 | sinistro com franquia alta->PR4 | renova ha 3+ meses sem personalizacao->PR7 | perfil sustentavel->PR8""",

    "P8": """## P8 (Objecoes) — lista com 0 a N objecoes; lista vazia [] se nao houve objecao
tipo_objecao: OB1=preco/valor | OB2=prazo/adiamento | OB3=demanda baixa | OB4=produto inadequado | OB5=problema nao resolvido | OB6=burocracia/aprovacao | OB7=indisponibilidade
resposta_vendedor_codigo: R1=validou e explorou | R2=usou dado do cliente | R3=criou urgencia | R4=posicionou como aliado | R5=capitulou sem explorar | R6=argumentou sem ouvir
eficacia: alta|media|baixa | desfecho_da_objecao: superada|adiada|perdida""",

    "P10": """## P10 (Promessas do vendedor) — lista com 0 a N promessas; lista vazia [] se nao houve promessa
PM1=Resolve na ligacao | PM2=Envia documento | PM3=Liga de volta | PM4=Encaminha terceiro | PM5=Negocia internamente | PM6=Promessa implicita
prazo_prometido: texto curto (ex: hoje, amanha, sem prazo) | risco_nao_cumprimento: baixo|medio|alto|muito_alto""",

    "P11": """## P11 (Inteligencia competitiva) — lista com 0 a N sinais; lista vazia [] se nao houve sinal
tipo_sinal: IC1=mencao direta a concorrente | IC2=confirmacao de exclusividade | IC3=historico de migracao | IC4=frota propria revelada | IC5=comparacao implicita(sem citar empresa) | IC6=vendedor sondou ativamente | IC7=risco competitivo detectado(cliente aberto a avaliar alternativas)
concorrente_mencionado: unidas|movida|localfrio|ouro_verde|outro|nenhum
vendedor_explorou: SIM|NAO|nao_aplicavel""",

    "P15": """## P15 (Eventos raros de alto impacto) — lista com 0 a N eventos; lista vazia [] na grande maioria das ligacoes (nao invente)
categoria: R1=expansao(contratacao,nova filial,crescimento de frota,novo contrato) | R2=churn(ameaca explicita de saida,insatisfacao recorrente) | R3=inteligencia competitiva(migracao de fornecedor,teste com concorrente) | R4=influencia(indicacao oferecida,contato apresentado) | R5=mudanca estrutural(fusao,aquisicao,troca de gestor do cliente)
impacto_potencial: alto|medio|baixo""",
}

FORMATO_COMPLETO = {
    "P1_P2": '{"P1": {"desfecho": "...", "intencao_entrada": "...", "tentativa_comercial": "SIM|NAO", "B": ["B1", ...] (lista vazia [] se nenhum)}, '
             '"P2": {"tipo": "...", "subtipo": "<curto ou vazio>", "quem_iniciou": "...", "tentativa_comercial": "SIM|NAO"}}',
    "P3": '{"P3": {"problemas_identificados": ["P1", ...] ou ["nenhum"], "quem_relatou": "...", "foi_resolvido_na_ligacao": "...", "escalou_para": "..."}}',
    "P5_P6": '{"P5": {"teve_challenger": "SIM|NAO", "codigos_challenger": [...], "frase_resumo": "...", "resultado_imediato": "...", "qualidade": "..."}, '
             '"P6": {"oportunidades_perdidas": [...] ou ["nenhuma"], "descricao_curta": "...", "valor_potencial_R$": "<numero>"}}',
    "P9": '{"P9": {"tipo_abertura": "AB1..AB5", "usou_nome": "SIM|NAO", "tipo_fechamento": "FC1..FC5", "proximo_passo_claro": "SIM|NAO", "prazo_definido": "SIM|NAO", "desfecho": "<curto>"}}',
}

# saida enxuta: so codigos (sem frase_resumo, descricao_curta, subtipo, desfecho livre do P9)
FORMATO_ENXUTO = (
    '{"P1": {"desfecho": "...", "intencao_entrada": "...", "tentativa_comercial": "SIM|NAO", "B": [...]}, '
    '"P2": {"tipo": "...", "quem_iniciou": "..."}, '
    '"P3": {"problemas_identificados": [...] ou ["nenhum"], "quem_relatou": "...", "foi_resolvido_na_ligacao": "...", "escalou_para": "..."}, '
    '"P5": {"teve_challenger": "SIM|NAO", "codigos_challenger": [...], "resultado_imediato": "...", "qualidade": "..."}, '
    '"P6": {"oportunidades_perdidas": [...] ou ["nenhuma"], "valor_potencial_R$": <numero>}, '
    '"P9": {"tipo_abertura": "AB1|AB2|AB3|AB4|AB5", "usou_nome": "SIM|NAO", "tipo_fechamento": "FC1|FC2|FC3|FC4|FC5", "proximo_passo_claro": "SIM|NAO", "prazo_definido": "SIM|NAO"}}\n'
    "Use SEMPRE os codigos (B1..B10, P1..P12, CH1..CH7, OP1..OP8, AB1..AB5, FC1..FC5), nunca o nome por extenso."
)

GRUPOS = ["P1_P2", "P3", "P5_P6", "P9"]
GRUPOS_TUDO = ["P1_P2", "P3", "P5_P6", "P7", "P8", "P9", "P10", "P11", "P15"]
EXEMPLOS = AQUI / "exemplos_fewshot.json"

FORMATO_TUDO = (
    FORMATO_ENXUTO.split("\n")[0][:-1] + ", "
    '"P7": {"produtos_mencionados": [...], "produtos_ofertados": [...], "janela_perdida": "SIM|NAO", "produto_janela": "PR..|nenhum"}, '
    '"P8": {"objecoes": [{"tipo_objecao": "OB..", "resposta_vendedor_codigo": "R..", "eficacia": "...", "desfecho_da_objecao": "..."}]}, '
    '"P10": {"promessas": [{"tipo_promessa": "PM..", "prazo_prometido": "...", "risco_nao_cumprimento": "..."}]}, '
    '"P11": {"sinais": [{"tipo_sinal": "IC..", "concorrente_mencionado": "...", "vendedor_explorou": "SIM|NAO|nao_aplicavel"}]}, '
    '"P15": {"eventos": [{"categoria": "R..", "impacto_potencial": "..."}]}}\n'
    "Use SEMPRE os codigos (B1..B10, P1..P12, CH1..CH7, OP1..OP8, PR1..PR12, OB1..OB7, R1..R6, PM1..PM6, "
    "IC1..IC7, AB1..AB5, FC1..FC5), nunca o nome por extenso."
)


def system_grupo(g):
    return (f"{PAPEL[g]}\n{CONTEXTO}\n\nAnalise a ligacao enviada pelo usuario e classifique. "
            f"Nao invente informacao que nao esta na transcricao.\n\n{TAXO[g]}\n\n"
            f"Responda SOMENTE um objeto JSON valido, sem texto fora dele, no formato:\n{FORMATO_COMPLETO[g]}")


def system_unico():
    return (f"Voce e analista de ligacoes comerciais B2B de locacao de frotas corporativas "
            f"(comportamento comercial, Customer Success, Challenger Sale).\n{CONTEXTO}\n\n"
            f"Analise a ligacao enviada pelo usuario e classifique TODAS as dimensoes abaixo. "
            f"Nao invente informacao que nao esta na transcricao.\n\n"
            + "\n\n".join(TAXO[g] for g in GRUPOS)
            + f"\n\nResponda SOMENTE um objeto JSON valido, sem texto fora dele, no formato:\n{FORMATO_ENXUTO}")


def bloco_exemplos():
    """Exemplos few-shot (exemplos_fewshot.json): trecho + classificacao + motivo. Ficam no system (prefixo fixo = cache)."""
    ex = json.loads(EXEMPLOS.read_text(encoding="utf-8"))["exemplos"]
    partes = []
    for i, e in enumerate(ex, 1):
        classif = "; ".join(f"{p}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                            for p, v in e["classificacao"].items())
        partes.append(f"### Exemplo {i} ({', '.join(e['prompts'])})\nTrecho:\n{e['trecho']}\n"
                      f"Classificacao: {classif}\nPor que: {e['motivo']}")
    return ("# EXEMPLOS RESOLVIDOS (trechos de outras ligacoes; mostram so as dimensoes indicadas — use como "
            "referencia de criterio, nao copie)\n\n" + "\n\n".join(partes))


def system_tudo(com_exemplos):
    return (f"Voce e analista de ligacoes comerciais B2B de locacao de frotas corporativas "
            f"(comportamento comercial, Customer Success, Challenger Sale, inteligencia competitiva).\n{CONTEXTO}\n\n"
            f"Analise a ligacao enviada pelo usuario e classifique TODAS as dimensoes abaixo. "
            f"Nao invente informacao que nao esta na transcricao.\n\n"
            + "\n\n".join(TAXO[g] for g in GRUPOS_TUDO)
            + (f"\n\n{bloco_exemplos()}" if com_exemplos else "")
            + f"\n\nResponda SOMENTE um objeto JSON valido, sem texto fora dele, no formato:\n{FORMATO_TUDO}")


# ---------------------------------------------------------------------------
# Transcricao
# ---------------------------------------------------------------------------
URA = re.compile(
    r"(se voc[eê] disser seu nome e o motivo da liga[cç][aã]o.*?dispon[ií]vel\.?"
    r"|permane[cç]a na linha!?\.?"
    r"|esta pessoa n[aã]o est[aá] dispon[ií]vel.*$"
    r"|vamos entregar o seu recado.*$"
    r"|grave (a )?sua mensagem.*$)", re.I)


def compactar(t):
    """A:/C:, junta turnos seguidos do mesmo falante, tira URA/caixa postal e ruido de 1-2 chars."""
    t = re.sub(r"\bAGENTE\s*:", "\nA:", t or "")
    t = re.sub(r"\bCLIENTE\s*:", "\nC:", t)
    t = t.replace(" | ", "\n").replace("|", "\n")
    turnos = []
    for ln in t.splitlines():
        ln = ln.strip()
        if ln[:2] in ("A:", "C:"):
            quem, fala = ln[0], ln[2:].strip()
        elif turnos:
            quem, fala = turnos[-1][0], ln
        else:
            continue
        fala = URA.sub("", fala).strip()
        if len(fala) <= 2:
            continue
        if turnos and turnos[-1][0] == quem:
            turnos[-1][1] += " " + fala
        else:
            turnos.append([quem, fala])
    return "\n".join(f"{q}: {f}" for q, f in turnos)


def mensagem_usuario(lig, modo_transcricao):
    t = lig["transcricao_limpa"] if modo_transcricao == "bruta" else compactar(lig["transcricao_limpa"])
    # mesmo formato que a analisa_transcricoes_llm do time monta (data + tipo + transcricao)
    return f"Data: {lig.get('data', '')} | Tipo de ligacao: {lig.get('direcao', '')}\nTranscricao:\n{t}"


# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
def configs_base():
    return [
        {"id": "C0", "descricao": "como o time faz hoje", "modelo": "gpt-4o", "modo": "grupos", "transcricao": "bruta", "max_tokens": 800},
        {"id": "C1", "descricao": "mini", "modelo": "gpt-4o-mini", "modo": "grupos", "transcricao": "bruta", "max_tokens": 800},
        {"id": "C2", "descricao": "mini + transcricao compacta", "modelo": "gpt-4o-mini", "modo": "grupos", "transcricao": "compacta", "max_tokens": 800},
        {"id": "C3", "descricao": "mini + 1 pedido + saida enxuta", "modelo": "gpt-4o-mini", "modo": "unico", "transcricao": "compacta", "max_tokens": 600},
        {"id": "C4", "descricao": "4o + 1 pedido + saida enxuta", "modelo": "gpt-4o", "modo": "unico", "transcricao": "compacta", "max_tokens": 600},
        {"id": "C7", "descricao": "mini + 11 prompts em 1 pedido, sem exemplos", "modelo": "gpt-4o-mini", "modo": "tudo",
         "transcricao": "compacta", "max_tokens": 1500},
        {"id": "C8", "descricao": "mini + 11 prompts em 1 pedido + exemplos", "modelo": "gpt-4o-mini", "modo": "tudo_exemplos",
         "transcricao": "compacta", "max_tokens": 1500},
        {"id": "C12", "descricao": "mini + foco: venda, Challenger, problemas", "modelo": "gpt-4o-mini", "modo": "foco",
         "transcricao": "compacta", "max_tokens": 1500},
    ]


def todas_configs():
    cfgs = configs_base()
    disponiveis = set()
    if MODELOS_JSON.exists():
        dados = json.loads(MODELOS_JSON.read_text(encoding="utf-8"))
        disponiveis = {s["modelo"] for s in dados.get("sondagem", []) if s.get("disponivel")}
    if not EXEMPLOS.exists():
        cfgs = [c for c in cfgs if c["modo"] != "tudo_exemplos"]
    if CLD is None:
        cfgs = [c for c in cfgs if c["modo"] != "foco"]
    for i, m in enumerate([m for m in EXTRAS_MATRIZ if m in disponiveis], start=9):
        cfgs.append({"id": f"C{i}", "descricao": f"{m} + 1 pedido + saida enxuta", "modelo": m,
                     "modo": "unico", "transcricao": "compacta", "max_tokens": 600})
    return cfgs


def pedidos(cfg, lig):
    """Lista de (grupo, mensagens) para uma ligacao nesta config. System sempre primeiro (cache de prefixo)."""
    usuario = mensagem_usuario(lig, cfg["transcricao"])
    if cfg["modo"] == "unico":
        return [("TODOS", [{"role": "system", "content": system_unico()}, {"role": "user", "content": usuario}])]
    if cfg["modo"] == "foco":
        # mesma mensagem do classificador diario
        return [("FOCO", [{"role": "system", "content": CLD.SYSTEM_FOCO},
                          {"role": "user", "content": f"Transcricao:\n{compactar(lig['transcricao_limpa'])}"}])]
    if cfg["modo"] in ("tudo", "tudo_exemplos"):
        system = system_tudo(cfg["modo"] == "tudo_exemplos")
        return [("TUDO", [{"role": "system", "content": system}, {"role": "user", "content": usuario}])]
    return [(g, [{"role": "system", "content": system_grupo(g)}, {"role": "user", "content": usuario}]) for g in GRUPOS]


# ---------------------------------------------------------------------------
# Etapas 2-3 — rodar
# ---------------------------------------------------------------------------
def resposta_simulada(lig, grupo, mensagens):
    """Resposta falsa montada do gabarito — so para testar o script sem rede."""
    g = lig["gabarito_gemini"]
    if grupo == "FOCO":
        teve = g["P5"]["teve_challenger"] == "SIM"
        chs = set(_conj(g["P5"]["codigos_challenger"])) if teve else set()
        pts = {"alta": 3, "media": 2, "baixa": 1}.get(g["P5"]["qualidade"], 1) if teve else 0
        regua = ["trouxe_dado_concreto", "conectou_a_situacao_do_cliente", "cliente_reagiu"]
        resp = {"venda": {"evidencia": "x", "era_venda": "NAO" if g["P1"]["desfecho"] == "nao_era_venda" else "SIM",
                          "tentativa_comercial": g["P1"]["tentativa_comercial"], "desfecho": g["P1"]["desfecho"],
                          "tipo": g["P2"]["tipo"]},
                "challenger": {"frase_vendedor": "x" if teve else "", "reacao_cliente": "",
                               **{f"CH{i}": "SIM" if f"CH{i}" in chs else "NAO" for i in range(1, 8)},
                               **{k: "SIM" if n < pts else "NAO" for n, k in enumerate(regua)}},
                "problemas": {"lista": [{"evidencia": "x", "codigo": c} for c in sorted(_conj(g["P3"]["problemas_identificados"]))],
                              "tem_problema": "SIM" if _conj(g["P3"]["problemas_identificados"]) else "NAO",
                              "principal": "", "foi_resolvido_na_ligacao": g["P3"]["foi_resolvido_na_ligacao"]}}
        n_in = sum(len(m["content"]) for m in mensagens) // 4
        return resp, {"prompt_tokens": n_in, "completion_tokens": len(json.dumps(resp)) // 4, "cached_tokens": 0,
                      "reasoning_tokens": 0, "custo": None, "modelo_resposta": "simulado"}

    def lista(s, vazio):
        itens = [x for x in re.split(r"[+;,]", s) if x and x.lower() not in ("nenhum", "nenhuma")]
        return itens or ([vazio] if vazio else [])

    blocos = {
        "P1": {"desfecho": g["P1"]["desfecho"], "intencao_entrada": g["P1"]["intencao_entrada"],
               "tentativa_comercial": g["P1"]["tentativa_comercial"],
               "B": [f"B{i}" for i in range(1, 11) if g["P1"][f"B{i}"] == "SIM"]},
        "P2": {"tipo": g["P2"]["tipo"], "quem_iniciou": g["P2"]["quem_iniciou"]},
        "P3": {"problemas_identificados": lista(g["P3"]["problemas_identificados"], "nenhum"),
               "quem_relatou": g["P3"]["quem_relatou"], "foi_resolvido_na_ligacao": g["P3"]["foi_resolvido_na_ligacao"],
               "escalou_para": g["P3"]["escalou_para"]},
        "P5": {"teve_challenger": g["P5"]["teve_challenger"], "codigos_challenger": lista(g["P5"]["codigos_challenger"], None),
               "resultado_imediato": g["P5"]["resultado_imediato"], "qualidade": g["P5"]["qualidade"]},
        "P6": {"oportunidades_perdidas": lista(g["P6"]["oportunidades_perdidas"], "nenhuma"), "valor_potencial_R$": 0},
        "P9": {k: g["P9"][k] for k in ("tipo_abertura", "usou_nome", "tipo_fechamento", "proximo_passo_claro", "prazo_definido")},
        "P7": {"produtos_mencionados": lista(g["P7"]["produtos_mencionados"], None),
               "produtos_ofertados": lista(g["P7"]["produtos_ofertados"], None),
               "janela_perdida": g["P7"]["janela_perdida"], "produto_janela": g["P7"]["produto_janela"]},
        "P8": {"objecoes": []}, "P10": {"promessas": []}, "P11": {"sinais": g["P11"]["sinais"]}, "P15": {"eventos": []},
    }
    chaves = {"TUDO": list(blocos), "TODOS": ["P1", "P2", "P3", "P5", "P6", "P9"], "P1_P2": ["P1", "P2"], "P3": ["P3"], "P5_P6": ["P5", "P6"], "P9": ["P9"]}[grupo]
    resp = {k: blocos[k] for k in chaves}
    n_in = sum(len(m["content"]) for m in mensagens) // 4
    return resp, {"prompt_tokens": n_in, "completion_tokens": len(json.dumps(resp)) // 4, "cached_tokens": 0,
                  "reasoning_tokens": 0, "custo": None, "modelo_resposta": "simulado"}


def chamar(url, headers, cfg, mensagens):
    payload = montar_payload(cfg["modelo"], mensagens, cfg["max_tokens"])
    ultimo_erro = ""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            r, lat = post(url, headers, payload)
            if r.ok:
                j = r.json()
                conteudo = j["choices"][0]["message"]["content"]
                uso = extrair_uso(j)
                uso["latencia_s"] = lat
                uso["finish_reason"] = j["choices"][0].get("finish_reason")
                try:
                    return json.loads(conteudo), uso, None
                except json.JSONDecodeError:
                    return None, uso, f"JSON invalido (finish_reason={uso['finish_reason']}): {conteudo[:200]}"
            ultimo_erro = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 400 and "reasoning_effort" in r.text and "reasoning_effort" in payload:
                payload.pop("reasoning_effort")
                continue
            if r.status_code in (400, 401, 403, 404):
                break
        except Exception as e:
            ultimo_erro = str(e)[:300]
        time.sleep(5 * tentativa)
    return None, None, ultimo_erro


def usar_amostra(args):
    """--amostra 100 / 300 / 500 / 5 (padrao) ou nome de arquivo; relatorios ganham o sufixo da amostra."""
    global AMOSTRA, RELATORIO, RELATORIO_DETALHE
    a = str(getattr(args, "amostra", "") or "5")
    AMOSTRA = AQUI / (a if a.endswith(".json") else f"amostra_{a}.json")
    if not AMOSTRA.exists():
        sys.exit(f"Nao achei {AMOSTRA.name} nesta pasta.")
    sufixo = "" if AMOSTRA.stem == "amostra_5" else "_" + AMOSTRA.stem.replace("amostra_", "")
    RELATORIO = AQUI / f"relatorio{sufixo}.csv"
    RELATORIO_DETALHE = AQUI / f"relatorio_detalhe{sufixo}.csv"


def cmd_rodar(args):
    usar_amostra(args)
    amostra = json.loads(AMOSTRA.read_text(encoding="utf-8"))["ligacoes"]
    cfgs = todas_configs()
    if args.configs:
        pedidas = set(args.configs.split(","))
        cfgs = [c for c in cfgs if c["id"] in pedidas]
    if not args.simular:
        url, headers, origem = conexao()
        print(f"conexao via {origem}: {url}")
        if not MODELOS_JSON.exists():
            print("aviso: rode 'python teste_gpt.py modelos' antes para incluir os modelos extras (C5+) na matriz.")

    feitos = set()
    if RESULTADOS.exists():
        for ln in RESULTADOS.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln)
            if args.refazer and rec["config"] in {c["id"] for c in cfgs}:
                continue
            if rec.get("ok") and bool(rec.get("simulado")) == bool(args.simular):
                feitos.add((rec["config"], rec["cd_segmento"], rec["grupo"]))

    tarefas = [(cfg, lig, grupo, mensagens) for cfg in cfgs for lig in amostra
               for grupo, mensagens in pedidos(cfg, lig) if (cfg["id"], lig["cd_segmento"], grupo) not in feitos]
    total = sum(len(pedidos(c, amostra[0])) for c in cfgs) * len(amostra)
    print(f"{AMOSTRA.name}: {len(cfgs)} configs x {len(amostra)} ligacoes = {total} pedidos "
          f"({total - len(tarefas)} ja feitos serao pulados), {args.workers} em paralelo")
    trava = threading.Lock()
    feitas = [0]

    def executar(tarefa):
        cfg, lig, grupo, mensagens = tarefa
        if args.simular:
            resp, uso, erro = (*resposta_simulada(lig, grupo, mensagens), None)
        else:
            resp, uso, erro = chamar(url, headers, cfg, mensagens)
        rec = {"config": cfg["id"], "modelo": cfg["modelo"], "modo": cfg["modo"],
               "transcricao": cfg["transcricao"], "cd_segmento": lig["cd_segmento"], "grupo": grupo,
               "ok": resp is not None, "resposta": resp, "uso": uso, "erro": erro,
               "simulado": bool(args.simular), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        uso = uso or {}
        with trava:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            feitas[0] += 1
            print(f"  [{feitas[0]}/{len(tarefas)}] {cfg['id']:<3} {cfg['modelo']:<13} {lig['cd_segmento'][:8]} {grupo:<6} "
                  f"{'ok ' if resp is not None else 'ERRO'} in={uso.get('prompt_tokens', '-')} "
                  f"out={uso.get('completion_tokens', '-')} cache={uso.get('cached_tokens', '-')}"
                  + (f"  {erro[:80]}" if erro else ""))

    with open(RESULTADOS, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            list(ex.map(executar, tarefas))
    sufixo = "" if AMOSTRA.stem == "amostra_5" else f" --amostra {AMOSTRA.stem.replace('amostra_', '')}"
    print(f"\n-> {RESULTADOS.name}. Agora rode: python teste_gpt.py comparar{sufixo}")


# ---------------------------------------------------------------------------
# Comparacao com o gabarito Gemini
# ---------------------------------------------------------------------------
def _conj(v, vazios=("nenhum", "nenhuma", "na", "")):
    if isinstance(v, list):
        itens = v
    else:
        itens = re.split(r"[+;,/ ]", str(v or ""))
    return frozenset(str(x).strip().upper() for x in itens if str(x).strip().lower() not in vazios)


# nome por extenso -> codigo (P9), para nao contar como erro um acerto escrito por extenso
NOMES_P9 = {"contextualizada": "ab1", "relacional": "ab2", "generica": "ab3", "genérica": "ab3", "reativa": "ab4",
            "de protecao": "ab5", "de proteção": "ab5", "compromisso duplo": "fc1",
            "proximo passo so do vendedor": "fc2", "convite generico": "fc3", "convite genérico": "fc3",
            "aberto": "fc4", "diretivo": "fc5"}


def _conj_b(v, prefixo="B"):
    """Conjunto de codigos aceitando o numero puro (3 -> B3, 10 -> P10)."""
    return frozenset(f"{prefixo}{x}" if x.isdigit() else x for x in _conj(v))


def _norm(v):
    s = str(v or "").strip().lower()
    return {"vendedor_puxou": "vendedor", "cliente_queria": "cliente", **NOMES_P9}.get(s, s) if s else ""


def campos_comparados(gab, gpt):
    """Lista de (campo, valor_gabarito, valor_gpt, conta_no_funil). Campo so entra se o gabarito for valido."""
    p = lambda k: gpt.get(k) or {}
    g1, g2, g3, g5, g6, g9 = (gab[k] for k in ("P1", "P2", "P3", "P5", "P6", "P9"))
    out = [
        ("P1.desfecho", _norm(g1["desfecho"]), _norm(p("P1").get("desfecho")), True),
        ("P1.intencao_entrada", _norm(g1["intencao_entrada"]), _norm(p("P1").get("intencao_entrada")), False),
        ("P1.B (conjunto)", _conj([f"B{i}" for i in range(1, 11) if g1[f"B{i}"] == "SIM"]), _conj_b(p("P1").get("B", [])), False),
        ("P2.tipo", _norm(g2["tipo"]), _norm(p("P2").get("tipo")), True),
        ("P3.problemas (conjunto)", _conj(g3["problemas_identificados"]), _conj_b(p("P3").get("problemas_identificados"), "P"), True),
        ("P3.foi_resolvido", _norm(g3["foi_resolvido_na_ligacao"]), _norm(p("P3").get("foi_resolvido_na_ligacao")), False),
        ("P5.teve_challenger", _norm(g5["teve_challenger"]), _norm(p("P5").get("teve_challenger")), False),
        ("P6.oportunidades (conjunto)", _conj(g6["oportunidades_perdidas"]), _conj(p("P6").get("oportunidades_perdidas")), False),
        ("P9.tipo_abertura", _norm(g9["tipo_abertura"]), _norm(p("P9").get("tipo_abertura")), False),
        ("P9.tipo_fechamento", _norm(g9["tipo_fechamento"]), _norm(p("P9").get("tipo_fechamento")), False),
        ("P9.proximo_passo_claro", _norm(g9["proximo_passo_claro"]), _norm(p("P9").get("proximo_passo_claro")), False),
        ("P9.prazo_definido", _norm(g9["prazo_definido"]), _norm(p("P9").get("prazo_definido")), False),
    ]
    if _norm(g1["tentativa_comercial"]) in ("sim", "nao"):
        out.append(("P1.tentativa_comercial", _norm(g1["tentativa_comercial"]), _norm(p("P1").get("tentativa_comercial")), False))
    if _norm(g5["teve_challenger"]) == "sim":  # qualidade/codigos so fazem sentido quando houve Challenger
        out.append(("P5.qualidade", _norm(g5["qualidade"]), _norm(p("P5").get("qualidade")), False))
        out.append(("P5.codigos (conjunto)", _conj(g5["codigos_challenger"]), _conj(p("P5").get("codigos_challenger")), False))
    return out


def _p11(cods):
    """Codigos novos (P13+) nao existem no gabarito de agosto: contam como P11 (Outro) na comparacao."""
    return frozenset(c if c in {f"P{i}" for i in range(1, 13)} else "P11" for c in cods)


def campos_foco(gab, resp):
    """C12: mesmas metricas do funil + as perguntas de negocio (era venda? tem problema?)."""
    c = CLD.normalizar_foco(resp)
    g1, g2, g3, g5 = gab["P1"], gab["P2"], gab["P3"], gab["P5"]
    gprob = _conj(g3["problemas_identificados"])
    out = [
        ("era_venda", "sim" if g1["desfecho"] != "nao_era_venda" else "nao", c["P1"]["era_venda"].lower(), False),
        ("P1.desfecho", _norm(g1["desfecho"]), _norm(c["P1"]["desfecho"]), True),
        ("P2.tipo", _norm(g2["tipo"]), _norm(c["P2"]["tipo"]), True),
        ("tem_problema", "sim" if gprob else "nao", c["P3"]["tem_problema"].lower(), False),
        ("P3.problemas (conjunto)", gprob, _p11(c["P3"]["problemas_identificados"]), True),
        ("P5.teve_challenger", _norm(g5["teve_challenger"]), c["P5"]["teve_challenger"].lower(), False),
    ]
    if gprob:
        out.append(("P3.foi_resolvido", _norm(g3["foi_resolvido_na_ligacao"]), _norm(c["P3"]["foi_resolvido_na_ligacao"]), False))
        out.append(("P3.principal no gabarito", "sim", "sim" if (_p11([c["P3"]["problema_principal"]]) & gprob) else "nao", False))
    if _norm(g5["teve_challenger"]) == "sim":
        out.append(("P5.qualidade", _norm(g5["qualidade"]), _norm(c["P5"]["qualidade"]), False))
        out.append(("P5.codigos (conjunto)", _conj(g5["codigos_challenger"]), frozenset(c["P5"]["codigos_challenger"]), False))
    return out


def campos_extras(gab, gpt):
    """P7 e P11 (so existem nas configs de 11 prompts). Ficam fora da concordancia geral para manter C1-C4 comparaveis."""
    if "P7" not in gpt and "P11" not in gpt:
        return []
    g7, p7 = gab["P7"], gpt.get("P7") or {}
    sinais = (gpt.get("P11") or {}).get("sinais") or []
    return [
        ("P7.janela_perdida", _norm(g7["janela_perdida"]), _norm(p7.get("janela_perdida"))),
        ("P7.produto_janela", _norm(g7["produto_janela"]).upper() if _norm(g7["produto_janela"]) != "nenhum" else "nenhum",
         (_norm(p7.get("produto_janela")).upper() or "nenhum") if _norm(p7.get("produto_janela")) not in ("", "nenhum") else "nenhum"),
        ("P7.produtos_ofertados (conjunto)", _conj(g7["produtos_ofertados"]), _conj(p7.get("produtos_ofertados"))),
        ("P11.sinais (conjunto)", frozenset(x["tipo_sinal"].upper() for x in gab["P11"]["sinais"]),
         frozenset(str(x.get("tipo_sinal", "")).upper() for x in sinais if isinstance(x, dict))),
    ]


def _fmt(v):
    return "+".join(sorted(v)) or "nenhum" if isinstance(v, frozenset) else v


def cmd_comparar(args):
    usar_amostra(args)
    amostra = {l["cd_segmento"]: l for l in json.loads(AMOSTRA.read_text(encoding="utf-8"))["ligacoes"]}
    recs = [r for r in (json.loads(ln) for ln in RESULTADOS.read_text(encoding="utf-8").splitlines() if ln.strip())
            if r["cd_segmento"] in amostra]
    cfg_info = {c["id"]: c for c in todas_configs()}

    # ultima resposta ok por (config, ligacao, grupo)
    ultimo = {}
    for r in recs:
        ultimo[(r["config"], r["cd_segmento"], r["grupo"])] = r

    por_cfg = {}
    for (cfg, cd, grupo), r in ultimo.items():
        d = por_cfg.setdefault(cfg, {"lig": {}, "uso": {}, "erros": 0, "pedidos": 0, "simulado": r.get("simulado")})
        d["pedidos"] += 1
        if not r["ok"]:
            d["erros"] += 1
            continue
        d["lig"].setdefault(cd, {}).update(r["resposta"] or {})
        u = d["uso"].setdefault(cd, {"in": 0, "out": 0, "cache": 0, "raciocinio": 0, "custo": 0.0, "custo_ok": True, "lat": 0.0})
        uso = r["uso"] or {}
        u["in"] += uso.get("prompt_tokens") or 0
        u["out"] += uso.get("completion_tokens") or 0
        u["cache"] += uso.get("cached_tokens") or 0
        u["raciocinio"] += uso.get("reasoning_tokens") or 0
        u["lat"] += uso.get("latencia_s") or 0
        if uso.get("custo") is None:
            u["custo_ok"] = False
        else:
            u["custo"] += uso["custo"]

    linhas, detalhe = [], []
    for cfg_id in sorted(por_cfg, key=lambda c: int(c[1:])):
        d = por_cfg[cfg_id]
        info = cfg_info.get(cfg_id, {})
        acertos = total = acertos_funil = total_funil = acertos_extra = total_extra = 0
        preenchidos = {"P8": 0, "P10": 0, "P15": 0}
        por_campo = {}
        tp = fp = fn = 0
        for cd, gpt in d["lig"].items():
            foco = "venda" in gpt
            comparados = (campos_foco if foco else campos_comparados)(amostra[cd]["gabarito_gemini"], gpt)
            for campo, vg, vp, funil in comparados:
                if campo == "P3.problemas (conjunto)":
                    tp += len(vg & vp); fp += len(vp - vg); fn += len(vg - vp)
                bate = vg == vp
                acertos += bate; total += 1
                if funil:
                    acertos_funil += bate; total_funil += 1
                c = por_campo.setdefault(campo, [0, 0]); c[0] += bate; c[1] += 1
                detalhe.append({"config": cfg_id, "cd_segmento": cd[:8], "campo": campo,
                                "gemini": _fmt(vg), "gpt": _fmt(vp), "bate": "SIM" if bate else "NAO"})
            for campo, vg, vp in campos_extras(amostra[cd]["gabarito_gemini"], gpt):
                bate = vg == vp
                acertos_extra += bate; total_extra += 1
                c = por_campo.setdefault(campo, [0, 0]); c[0] += bate; c[1] += 1
                detalhe.append({"config": cfg_id, "cd_segmento": cd[:8], "campo": campo,
                                "gemini": _fmt(vg), "gpt": _fmt(vp), "bate": "SIM" if bate else "NAO"})
            for pr, chave in (("P8", "objecoes"), ("P10", "promessas"), ("P15", "eventos")):
                if ((gpt.get(pr) or {}).get(chave) or []):
                    preenchidos[pr] += 1
                    detalhe.append({"config": cfg_id, "cd_segmento": cd[:8], "campo": f"{pr} (sem gabarito)",
                                    "gemini": "", "gpt": json.dumps(gpt[pr][chave], ensure_ascii=False)[:300], "bate": ""})
        n = len(d["uso"]) or 1
        soma = lambda k: sum(u[k] for u in d["uso"].values())
        custo_ok = d["uso"] and all(u["custo_ok"] for u in d["uso"].values())
        linha = {
            "config": cfg_id, "descricao": info.get("descricao", ""), "modelo": info.get("modelo", ""),
            "modo": info.get("modo", ""), "transcricao": info.get("transcricao", ""),
            "ligacoes_ok": len(d["lig"]), "pedidos_com_erro": d["erros"],
            "tokens_entrada_por_lig": round(soma("in") / n), "tokens_cache_por_lig": round(soma("cache") / n),
            "tokens_saida_por_lig": round(soma("out") / n), "tokens_raciocinio_por_lig": round(soma("raciocinio") / n),
            "tokens_total_por_lig": round((soma("in") + soma("out")) / n),
            "custo_por_lig": round(soma("custo") / n, 6) if custo_ok else "",
            "latencia_por_lig_s": round(soma("lat") / n, 1),
            "concordancia_geral_pct": round(100 * acertos / total, 1) if total else "",
            "concordancia_funil_pct": round(100 * acertos_funil / total_funil, 1) if total_funil else "",
            "concordancia_P7_P11_pct": round(100 * acertos_extra / total_extra, 1) if total_extra else "",
            "problemas_precisao_pct": round(100 * tp / (tp + fp), 1) if tp + fp else "",
            "problemas_cobertura_pct": round(100 * tp / (tp + fn), 1) if tp + fn else "",
            "ligacoes_com_P8_objecao": preenchidos["P8"] if total_extra else "",
            "ligacoes_com_P10_promessa": preenchidos["P10"] if total_extra else "",
            "ligacoes_com_P15_evento": preenchidos["P15"] if total_extra else "",
        }
        for campo, (a, t) in sorted(por_campo.items()):
            linha[f"conc_{campo}"] = f"{a}/{t}"
        linhas.append(linha)

    cols = list(dict.fromkeys(k for l in linhas for k in l))
    for caminho, dados, campos in ((RELATORIO, linhas, cols),
                                   (RELATORIO_DETALHE, detalhe, ["config", "cd_segmento", "campo", "gemini", "gpt", "bate"])):
        with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos, delimiter=";")
            w.writeheader(); w.writerows(dados)

    print(f"{'cfg':<4}{'modelo':<14}{'modo':<15}{'transc':<9}{'lig':>4}{'entrada':>9}{'cache':>7}{'saida':>7}"
          f"{'custo/lig':>11}{'geral%':>8}{'funil%':>8}{'P7P11%':>8}{'prob.prec%':>11}{'prob.cob%':>10}")
    for l in linhas:
        print(f"{l['config']:<4}{l['modelo']:<14}{l['modo']:<15}{l['transcricao']:<9}{l['ligacoes_ok']:>4}"
              f"{l['tokens_entrada_por_lig']:>9}{l['tokens_cache_por_lig']:>7}{l['tokens_saida_por_lig']:>7}"
              f"{str(l['custo_por_lig']):>11}{str(l['concordancia_geral_pct']):>8}{str(l['concordancia_funil_pct']):>8}"
              f"{str(l['concordancia_P7_P11_pct']):>8}{str(l['problemas_precisao_pct']):>11}{str(l['problemas_cobertura_pct']):>10}")
    foco = [l for l in linhas if l["modo"] == "foco"]
    for l in foco:
        conc = lambda k: l.get(f"conc_{k}", "-")
        print(f"  {l['config']} (foco): era venda {conc('era_venda')} | tem problema {conc('tem_problema')} | "
              f"principal no gabarito {conc('P3.principal no gabarito')} | Challenger teve {conc('P5.teve_challenger')}, "
              f"qualidade {conc('P5.qualidade')} (o Gemini superestima Challenger: revise as divergencias)")
    extras = [l for l in linhas if l["ligacoes_com_P8_objecao"] != ""]
    for l in extras:
        print(f"  {l['config']}: sem gabarito, so contagem — P8 objecao em {l['ligacoes_com_P8_objecao']}, "
              f"P10 promessa em {l['ligacoes_com_P10_promessa']}, P15 evento em {l['ligacoes_com_P15_evento']} ligacoes "
              f"(conteudo em {RELATORIO_DETALHE.name})")

    base = next((l for l in linhas if l["config"] == "C0"), None) or next((l for l in linhas if l["config"] == "C1"), None)
    if base and base["concordancia_geral_pct"] != "":
        custo = lambda l: l["custo_por_lig"] if l["custo_por_lig"] != "" else l["tokens_entrada_por_lig"] + 4 * l["tokens_saida_por_lig"]
        aptas = [l for l in linhas if l["ligacoes_ok"] == len(amostra) and l["pedidos_com_erro"] == 0
                 and l["concordancia_geral_pct"] >= base["concordancia_geral_pct"]
                 and l["concordancia_funil_pct"] >= base["concordancia_funil_pct"]]
        if aptas:
            v = min(aptas, key=custo)
            print(f"\nMais barata com concordancia >= {base['config']} ({base['descricao']}): {v['config']} — {v['descricao']}")
            print(f"  tokens/ligacao: {v['tokens_total_por_lig']} ({base['config']}: {base['tokens_total_por_lig']})")
        else:
            print(f"\nNenhuma configuracao empatou ou superou o {base['config']} em concordancia sem erro — ver relatorio_detalhe.csv.")
    print(f"\n-> {RELATORIO.name}, {RELATORIO_DETALHE.name} (sem transcricao — pode trazer de volta)")
    if any(d.get("simulado") for d in por_cfg.values()):
        print("ATENCAO: resultados SIMULADOS (sem API). Apague resultados.jsonl antes do teste real.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("modelos")
    r = sub.add_parser("rodar")
    r.add_argument("--configs", default="")
    r.add_argument("--simular", action="store_true")
    r.add_argument("--refazer", action="store_true", help="refaz as configs pedidas mesmo se ja rodaram")
    r.add_argument("--amostra", default="5", help="5 (padrao), 100, 300 ou 500")
    r.add_argument("--workers", type=int, default=4, help="pedidos em paralelo (padrao 4)")
    c = sub.add_parser("comparar")
    c.add_argument("--amostra", default="5", help="5 (padrao), 100, 300 ou 500")
    args = ap.parse_args()
    {"modelos": cmd_modelos, "rodar": cmd_rodar, "comparar": cmd_comparar}[args.cmd](args)


if __name__ == "__main__":
    main()
