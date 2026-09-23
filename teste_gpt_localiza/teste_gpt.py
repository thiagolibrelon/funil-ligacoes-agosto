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

Saidas (nesta pasta): modelos_disponiveis.json, resultados.jsonl, relatorio.csv,
relatorio_detalhe.csv. Nenhuma delas contem transcricao — sao essas que voltam.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import requests

AQUI = Path(__file__).resolve().parent
AMOSTRA = AQUI / "amostra_5.json"
MODELOS_JSON = AQUI / "modelos_disponiveis.json"
RESULTADOS = AQUI / "resultados.jsonl"
RELATORIO = AQUI / "relatorio.csv"
RELATORIO_DETALHE = AQUI / "relatorio_detalhe.csv"

URL_PADRAO = "https://llm-gate-np.localiza.dev/llm-gate/v2/chat/completions"
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
    '"P9": {"tipo_abertura": "...", "usou_nome": "SIM|NAO", "tipo_fechamento": "...", "proximo_passo_claro": "SIM|NAO", "prazo_definido": "SIM|NAO"}}'
)

GRUPOS = ["P1_P2", "P3", "P5_P6", "P9"]


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
    ]


def todas_configs():
    cfgs = configs_base()
    disponiveis = set()
    if MODELOS_JSON.exists():
        dados = json.loads(MODELOS_JSON.read_text(encoding="utf-8"))
        disponiveis = {s["modelo"] for s in dados.get("sondagem", []) if s.get("disponivel")}
    for i, m in enumerate([m for m in EXTRAS_MATRIZ if m in disponiveis], start=5):
        cfgs.append({"id": f"C{i}", "descricao": f"{m} + 1 pedido + saida enxuta", "modelo": m,
                     "modo": "unico", "transcricao": "compacta", "max_tokens": 600})
    return cfgs


def pedidos(cfg, lig):
    """Lista de (grupo, mensagens) para uma ligacao nesta config. System sempre primeiro (cache de prefixo)."""
    usuario = mensagem_usuario(lig, cfg["transcricao"])
    if cfg["modo"] == "unico":
        return [("TODOS", [{"role": "system", "content": system_unico()}, {"role": "user", "content": usuario}])]
    return [(g, [{"role": "system", "content": system_grupo(g)}, {"role": "user", "content": usuario}]) for g in GRUPOS]


# ---------------------------------------------------------------------------
# Etapas 2-3 — rodar
# ---------------------------------------------------------------------------
def resposta_simulada(lig, grupo, mensagens):
    """Resposta falsa montada do gabarito — so para testar o script sem rede."""
    g = lig["gabarito_gemini"]

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
    }
    chaves = {"TODOS": list(blocos), "P1_P2": ["P1", "P2"], "P3": ["P3"], "P5_P6": ["P5", "P6"], "P9": ["P9"]}[grupo]
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


def cmd_rodar(args):
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
            if rec.get("ok") and bool(rec.get("simulado")) == bool(args.simular):
                feitos.add((rec["config"], rec["cd_segmento"], rec["grupo"]))

    total = sum(len(pedidos(c, amostra[0])) for c in cfgs) * len(amostra)
    print(f"{len(cfgs)} configs x {len(amostra)} ligacoes = {total} pedidos ({len(feitos)} ja feitos serao pulados)")
    with open(RESULTADOS, "a", encoding="utf-8") as out:
        for cfg in cfgs:
            for lig in amostra:
                for grupo, mensagens in pedidos(cfg, lig):
                    if (cfg["id"], lig["cd_segmento"], grupo) in feitos:
                        continue
                    if args.simular:
                        resp, uso, erro = (*resposta_simulada(lig, grupo, mensagens), None)
                    else:
                        resp, uso, erro = chamar(url, headers, cfg, mensagens)
                    rec = {"config": cfg["id"], "modelo": cfg["modelo"], "modo": cfg["modo"],
                           "transcricao": cfg["transcricao"], "cd_segmento": lig["cd_segmento"], "grupo": grupo,
                           "ok": resp is not None, "resposta": resp, "uso": uso, "erro": erro,
                           "simulado": bool(args.simular), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    uso = uso or {}
                    print(f"  {cfg['id']:<3} {cfg['modelo']:<13} {lig['cd_segmento'][:8]} {grupo:<6} "
                          f"{'ok ' if resp is not None else 'ERRO'} in={uso.get('prompt_tokens', '-')} "
                          f"out={uso.get('completion_tokens', '-')} cache={uso.get('cached_tokens', '-')}"
                          + (f"  {erro[:80]}" if erro else ""))
    print(f"\n-> {RESULTADOS.name}. Agora rode: python teste_gpt.py comparar")


# ---------------------------------------------------------------------------
# Comparacao com o gabarito Gemini
# ---------------------------------------------------------------------------
def _conj(v, vazios=("nenhum", "nenhuma", "na", "")):
    if isinstance(v, list):
        itens = v
    else:
        itens = re.split(r"[+;,/ ]", str(v or ""))
    return frozenset(str(x).strip().upper() for x in itens if str(x).strip().lower() not in vazios)


def _norm(v):
    s = str(v or "").strip().lower()
    return {"vendedor_puxou": "vendedor", "cliente_queria": "cliente"}.get(s, s) if s else ""


def campos_comparados(gab, gpt):
    """Lista de (campo, valor_gabarito, valor_gpt, conta_no_funil). Campo so entra se o gabarito for valido."""
    p = lambda k: gpt.get(k) or {}
    g1, g2, g3, g5, g6, g9 = (gab[k] for k in ("P1", "P2", "P3", "P5", "P6", "P9"))
    out = [
        ("P1.desfecho", _norm(g1["desfecho"]), _norm(p("P1").get("desfecho")), True),
        ("P1.intencao_entrada", _norm(g1["intencao_entrada"]), _norm(p("P1").get("intencao_entrada")), False),
        ("P1.B (conjunto)", _conj([f"B{i}" for i in range(1, 11) if g1[f"B{i}"] == "SIM"]), _conj(p("P1").get("B", [])), False),
        ("P2.tipo", _norm(g2["tipo"]), _norm(p("P2").get("tipo")), True),
        ("P3.problemas (conjunto)", _conj(g3["problemas_identificados"]), _conj(p("P3").get("problemas_identificados")), True),
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


def _fmt(v):
    return "+".join(sorted(v)) or "nenhum" if isinstance(v, frozenset) else v


def cmd_comparar(_args):
    amostra = {l["cd_segmento"]: l for l in json.loads(AMOSTRA.read_text(encoding="utf-8"))["ligacoes"]}
    recs = [json.loads(ln) for ln in RESULTADOS.read_text(encoding="utf-8").splitlines() if ln.strip()]
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
        acertos = total = acertos_funil = total_funil = 0
        por_campo = {}
        for cd, gpt in d["lig"].items():
            for campo, vg, vp, funil in campos_comparados(amostra[cd]["gabarito_gemini"], gpt):
                bate = vg == vp
                acertos += bate; total += 1
                if funil:
                    acertos_funil += bate; total_funil += 1
                c = por_campo.setdefault(campo, [0, 0]); c[0] += bate; c[1] += 1
                detalhe.append({"config": cfg_id, "cd_segmento": cd[:8], "campo": campo,
                                "gemini": _fmt(vg), "gpt": _fmt(vp), "bate": "SIM" if bate else "NAO"})
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

    print(f"{'cfg':<4}{'modelo':<14}{'modo':<7}{'transc':<9}{'lig':>4}{'entrada':>9}{'cache':>7}{'saida':>7}"
          f"{'custo/lig':>11}{'geral%':>8}{'funil%':>8}")
    for l in linhas:
        print(f"{l['config']:<4}{l['modelo']:<14}{l['modo']:<7}{l['transcricao']:<9}{l['ligacoes_ok']:>4}"
              f"{l['tokens_entrada_por_lig']:>9}{l['tokens_cache_por_lig']:>7}{l['tokens_saida_por_lig']:>7}"
              f"{str(l['custo_por_lig']):>11}{str(l['concordancia_geral_pct']):>8}{str(l['concordancia_funil_pct']):>8}")

    base = next((l for l in linhas if l["config"] == "C0"), None)
    if base and base["concordancia_geral_pct"] != "":
        custo = lambda l: l["custo_por_lig"] if l["custo_por_lig"] != "" else l["tokens_entrada_por_lig"] + 4 * l["tokens_saida_por_lig"]
        aptas = [l for l in linhas if l["ligacoes_ok"] == len(amostra) and l["pedidos_com_erro"] == 0
                 and l["concordancia_geral_pct"] >= base["concordancia_geral_pct"]
                 and l["concordancia_funil_pct"] >= base["concordancia_funil_pct"]]
        if aptas:
            v = min(aptas, key=custo)
            print(f"\nMais barata com concordancia >= C0 (gpt-4o como hoje): {v['config']} — {v['descricao']}")
            print(f"  tokens/ligacao: {v['tokens_total_por_lig']} (C0: {base['tokens_total_por_lig']})")
        else:
            print("\nNenhuma configuracao empatou ou superou o C0 em concordancia sem erro — ver relatorio_detalhe.csv.")
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
    sub.add_parser("comparar")
    args = ap.parse_args()
    {"modelos": cmd_modelos, "rodar": cmd_rodar, "comparar": cmd_comparar}[args.cmd](args)


if __name__ == "__main__":
    main()
