"""Classifica as ligacoes de um dia com GPT (llm-gate) e grava o resultado em JSON e CSV.

COMO USAR (maquina/rede da Localiza):
  python classificar_ligacoes_diario.py
    1) abre uma janela para escolher o arquivo do dia (o chamadas/<mes>/<DD> gerado pelo
       compilar_ligacoes_diarias.py, ou o proprio CSV exportado do sistema);
    2) abre outra janela para escolher a pasta onde salvar o resultado;
    3) se a variavel API_KEY nao existir, pede a chave do llm-gate numa janela (nao fica salva);
    4) classifica e mostra um resumo no fim.
  Sem janelas (para agendar):  python classificar_ligacoes_diario.py --entrada <arquivo> --saida <pasta>

SAIDA (na pasta escolhida), com o nome do dia (ex: classificacao_2026-09-22):
  .json  — uma entrada por ligacao, com todos os prompts detalhados (objecoes, promessas, sinais...)
  .csv   — uma linha por ligacao, colunas no padrao do CONSOLIDADO de agosto (p1_desfecho, p2_tipo...),
           separador ";" (abre direto no Excel)
  _andamento.jsonl — respostas salvas uma a uma; se cair no meio, rode de novo com o mesmo arquivo e
           a mesma pasta que ele continua de onde parou.
  consumo_diario.csv — 1 linha por dia (ligacoes, tokens, custo estimado), para acompanhar o consumo do mes.
Na tela aparece cada ligacao (desfecho, tipo, tokens) e o acumulado de tokens e custo; --silencioso mostra
so 1 linha a cada 25.

FORMATO (vencedor do teste teste_gpt_localiza/, config C7): 11 prompts por ligacao em 1 pedido so
(P1, P2, P3, P5, P6, P7, P8, P9, P10, P11, P15), transcricao compactada, resposta so com codigos, cache
do llm-gate. Taxonomia identica a scripts/analisar_gemini_julho.py (a do Gemini de agosto).
Regras fora do modelo:
  - transcricao com menos de TAMANHO_MINIMO caracteres = "sem_conteudo" automatico, sem chamar a API
    (mesmo corte de agosto — caixa postal, recado, transferencia);
  - B8 (ligou proativamente) = SIM quando a ligacao e Outbound (dado do sistema, nao do modelo);
  - score_comercial recalculado pelos pesos de B1-B10 (nao confia na conta do modelo).
Para trocar de modelo (ex: quando o gpt-4o for liberado), mude so a constante MODELO.
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

csv.field_size_limit(10_000_000)

MODELO = "gpt-4o-mini"
URL_PADRAO = "https://llm-gate-np.localiza.dev/llm-gate/v2/chat/completions"
MAX_TOKENS_SAIDA = 1500
WORKERS = 4
MAX_TENTATIVAS = 3
TIMEOUT_S = 120
TAMANHO_MINIMO = 600          # caracteres da transcricao compactada — abaixo disso, sem_conteudo automatico
MIN_CHARS_TRANSCRICAO = 40    # abaixo disso a linha e ignorada (transcricao vazia/quebrada)
TIME_ALVO = "LL_GRVIN"        # usado so se o arquivo tiver a coluna time_agente_1; "" = todos os times
FUSO_H = -3
# precos publicos OpenAI por milhao de tokens (entrada, entrada em cache, saida) — so para a estimativa
PRECOS = {"gpt-4o-mini": (0.15, 0.075, 0.60), "gpt-4o": (2.50, 1.25, 10.00)}

PESOS_B = {"B1": 2, "B2": 1, "B3": 2, "B4": 2, "B5": 2, "B6": 3, "B7": 1, "B8": 1, "B9": 1, "B10": 3}

# ---------------------------------------------------------------------------
# Prompt (config C7 do teste)
# ---------------------------------------------------------------------------
CONTEXTO = (
    "Empresa: Localiza (locacao de veiculos mensais para PJ)\n"
    'Vendedor = "A" | Cliente = "C"\n'
    "STT fragmentado - junte turns do mesmo speaker. <unk> = inaudivel, ignore."
)

TAXONOMIA = """## P1 (Conversao)
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
quem_iniciou: vendedor|cliente|indefinido

## P3 (Problemas)
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
escalou_para: 0800|supervisora|IT|nao_escalou|nenhum

## P5 (Challenger)
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
Se nao ha oportunidade perdida: oportunidades_perdidas=["nenhuma"]

## P7 (Cross-sell e upsell)
PR1=Aluguel mensal leve(RAC,core) | PR2=Aluguel pesado(caminhao,carreta — NAO inclui PR1) | PR3=Telemetria
PR4=Protecao total/cobertura de avarias | PR5=Upgrade de categoria | PR6=Km adicional | PR7=Contrato anual
PR8=Veiculos eletricos | PR9=ZARP | PR10=Meoo | PR11=Venda de Seminovos
PR12=Gestao de frotas (contrato longo, cliente adquire veiculo personalizavel, administrado pela Localiza — diferente de PR7)
janela_perdida: SIM se o cliente disse algo que se conecta a um produto e o vendedor NAO ofereceu | NAO caso contrario
produto_janela: codigo PR da janela perdida (ou "nenhum")
Exemplos de janela: multa em condutor errado->PR3 | frota propria/personalizacao->PR12 | sinistro com franquia alta->PR4 | renova ha 3+ meses sem personalizacao->PR7 | perfil sustentavel->PR8

## P8 (Objecoes) — lista com 0 a N objecoes; lista vazia [] se nao houve objecao
tipo_objecao: OB1=preco/valor | OB2=prazo/adiamento | OB3=demanda baixa | OB4=produto inadequado | OB5=problema nao resolvido | OB6=burocracia/aprovacao | OB7=indisponibilidade
resposta_vendedor_codigo: R1=validou e explorou | R2=usou dado do cliente | R3=criou urgencia | R4=posicionou como aliado | R5=capitulou sem explorar | R6=argumentou sem ouvir
eficacia: alta|media|baixa | desfecho_da_objecao: superada|adiada|perdida

## P9 (Abertura e fechamento)
AB1=Contextualizada | AB2=Relacional | AB3=Generica | AB4=Reativa | AB5=De protecao
FC1=Compromisso duplo(vendedor+cliente+prazo) | FC2=Proximo passo so do vendedor
FC3=Convite generico("se precisar me chama") | FC4=Aberto | FC5=Diretivo
usou_nome: SIM|NAO | proximo_passo_claro: SIM|NAO | prazo_definido: SIM|NAO

## P10 (Promessas do vendedor) — lista com 0 a N promessas; lista vazia [] se nao houve promessa
PM1=Resolve na ligacao | PM2=Envia documento | PM3=Liga de volta | PM4=Encaminha terceiro | PM5=Negocia internamente | PM6=Promessa implicita
prazo_prometido: texto curto (ex: hoje, amanha, sem prazo) | risco_nao_cumprimento: baixo|medio|alto|muito_alto

## P11 (Inteligencia competitiva) — lista com 0 a N sinais; lista vazia [] se nao houve sinal
tipo_sinal: IC1=mencao direta a concorrente | IC2=confirmacao de exclusividade | IC3=historico de migracao | IC4=frota propria revelada | IC5=comparacao implicita(sem citar empresa) | IC6=vendedor sondou ativamente | IC7=risco competitivo detectado(cliente aberto a avaliar alternativas)
concorrente_mencionado: unidas|movida|localfrio|ouro_verde|outro|nenhum
vendedor_explorou: SIM|NAO|nao_aplicavel

## P15 (Eventos raros de alto impacto) — lista com 0 a N eventos; lista vazia [] na grande maioria das ligacoes (nao invente)
categoria: R1=expansao(contratacao,nova filial,crescimento de frota,novo contrato) | R2=churn(ameaca explicita de saida,insatisfacao recorrente) | R3=inteligencia competitiva(migracao de fornecedor,teste com concorrente) | R4=influencia(indicacao oferecida,contato apresentado) | R5=mudanca estrutural(fusao,aquisicao,troca de gestor do cliente)
impacto_potencial: alto|medio|baixo"""

FORMATO = (
    '{"P1": {"desfecho": "...", "intencao_entrada": "...", "tentativa_comercial": "SIM|NAO", "B": [...]}, '
    '"P2": {"tipo": "...", "quem_iniciou": "..."}, '
    '"P3": {"problemas_identificados": [...] ou ["nenhum"], "quem_relatou": "...", "foi_resolvido_na_ligacao": "...", "escalou_para": "..."}, '
    '"P5": {"teve_challenger": "SIM|NAO", "codigos_challenger": [...], "resultado_imediato": "...", "qualidade": "..."}, '
    '"P6": {"oportunidades_perdidas": [...] ou ["nenhuma"], "valor_potencial_R$": <numero>}, '
    '"P9": {"tipo_abertura": "AB1|AB2|AB3|AB4|AB5", "usou_nome": "SIM|NAO", "tipo_fechamento": "FC1|FC2|FC3|FC4|FC5", "proximo_passo_claro": "SIM|NAO", "prazo_definido": "SIM|NAO"}, '
    '"P7": {"produtos_mencionados": [...], "produtos_ofertados": [...], "janela_perdida": "SIM|NAO", "produto_janela": "PR..|nenhum"}, '
    '"P8": {"objecoes": [{"tipo_objecao": "OB..", "resposta_vendedor_codigo": "R..", "eficacia": "...", "desfecho_da_objecao": "..."}]}, '
    '"P10": {"promessas": [{"tipo_promessa": "PM..", "prazo_prometido": "...", "risco_nao_cumprimento": "..."}]}, '
    '"P11": {"sinais": [{"tipo_sinal": "IC..", "concorrente_mencionado": "...", "vendedor_explorou": "SIM|NAO|nao_aplicavel"}]}, '
    '"P15": {"eventos": [{"categoria": "R..", "impacto_potencial": "..."}]}}\n'
    "Use SEMPRE os codigos (B1..B10, P1..P12, CH1..CH7, OP1..OP8, PR1..PR12, OB1..OB7, R1..R6, PM1..PM6, "
    "IC1..IC7, AB1..AB5, FC1..FC5), nunca o nome por extenso."
)

SYSTEM = (
    "Voce e analista de ligacoes comerciais B2B de locacao de frotas corporativas "
    f"(comportamento comercial, Customer Success, Challenger Sale, inteligencia competitiva).\n{CONTEXTO}\n\n"
    "Analise a ligacao enviada pelo usuario e classifique TODAS as dimensoes abaixo. "
    f"Nao invente informacao que nao esta na transcricao.\n\n{TAXONOMIA}\n\n"
    f"Responda SOMENTE um objeto JSON valido, sem texto fora dele, no formato:\n{FORMATO}"
)

URA = re.compile(
    r"(se voc[eê] disser seu nome e o motivo da liga[cç][aã]o.*?dispon[ií]vel\.?"
    r"|permane[cç]a na linha!?\.?"
    r"|esta pessoa n[aã]o est[aá] dispon[ií]vel.*$"
    r"|vamos entregar o seu recado.*$"
    r"|grave (a )?sua mensagem.*$)", re.I)


def compactar(t):
    """A:/C:, junta turnos seguidos do mesmo falante, tira URA/caixa postal e ruido de 1-2 caracteres."""
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


# ---------------------------------------------------------------------------
# Janelas (tkinter vem com o Python do Windows)
# ---------------------------------------------------------------------------
def _tk():
    import tkinter as tk
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    return raiz


def escolher_arquivo():
    from tkinter import filedialog
    raiz = _tk()
    caminho = filedialog.askopenfilename(
        parent=raiz, title="Escolha o arquivo de ligacoes do dia",
        filetypes=[("Todos os arquivos", "*.*"), ("CSV", "*.csv")])
    raiz.destroy()
    return caminho


def escolher_pasta():
    from tkinter import filedialog
    raiz = _tk()
    caminho = filedialog.askdirectory(parent=raiz, title="Escolha a pasta onde salvar o resultado")
    raiz.destroy()
    return caminho


def pedir_chave():
    from tkinter import simpledialog
    raiz = _tk()
    chave = simpledialog.askstring("Chave do llm-gate", "Cole a chave do llm-gate (API_KEY):", show="*", parent=raiz)
    raiz.destroy()
    return chave


def avisar(titulo, texto, erro=False):
    print(texto)
    try:
        from tkinter import messagebox
        raiz = _tk()
        (messagebox.showerror if erro else messagebox.showinfo)(titulo, texto, parent=raiz)
        raiz.destroy()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Leitura do arquivo do dia
# ---------------------------------------------------------------------------
def dia_local(valor):
    v = (valor or "").strip()
    em_utc = v.upper().endswith("UTC")
    v = v.replace("UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(v[:26], fmt)
            break
        except ValueError:
            continue
    else:
        return ""
    if em_utc:
        dt = dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=FUSO_H)))
    return dt.date().isoformat()


def ler_ligacoes(caminho):
    """Aceita o arquivo do compilador (chamadas/<mes>/<DD>, ';') ou o CSV exportado do sistema (',' ou ';')."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(caminho, encoding=enc, newline="") as f:
                cab = f.readline()
                sep = ";" if cab.count(";") > cab.count(",") else ","
                f.seek(0)
                leitor = csv.DictReader(f, delimiter=sep)
                linhas, colunas = list(leitor), leitor.fieldnames or []
            break
        except UnicodeDecodeError:
            continue
    faltando = [c for c in ("cd_segmento", "transcricao_limpa") if c not in colunas]
    if faltando:
        raise ValueError(f"o arquivo nao tem as colunas {faltando}. Colunas encontradas: {colunas[:12]}")
    ligacoes, vistos, ignoradas = [], set(), {"outro_time": 0, "transcricao_nao_success": 0, "vazia": 0, "duplicada": 0}
    for r in linhas:
        if TIME_ALVO and "time_agente_1" in colunas and (r.get("time_agente_1") or "").strip() != TIME_ALVO:
            ignoradas["outro_time"] += 1
            continue
        if "status_transcricao" in colunas and (r.get("status_transcricao") or "").strip().lower() != "success":
            ignoradas["transcricao_nao_success"] += 1
            continue
        if len((r.get("transcricao_limpa") or "").strip()) < MIN_CHARS_TRANSCRICAO:
            ignoradas["vazia"] += 1
            continue
        cd = (r.get("cd_segmento") or "").strip()
        if cd in vistos:
            ignoradas["duplicada"] += 1
            continue
        vistos.add(cd)
        ligacoes.append({
            "cd_segmento": cd,
            "data": (r.get("Data") or "").strip() or dia_local(r.get("data_hora_inicio")),
            "data_hora_inicio": r.get("data_hora_inicio", ""),
            "direcao": r.get("direcao", ""),
            "nome_agente_1": r.get("nome_agente_1", ""),
            "time_agente_1": r.get("time_agente_1", ""),
            "transcricao_limpa": r["transcricao_limpa"],
        })
    return ligacoes, ignoradas


# ---------------------------------------------------------------------------
# Chamada ao llm-gate
# ---------------------------------------------------------------------------
def conexao(chave_janela=None):
    try:
        from src.settings import get_llm_headers, get_llm_url
        return get_llm_url(), dict(get_llm_headers())
    except Exception:
        chave = os.getenv("API_KEY") or chave_janela
        return URL_PADRAO, {"Content-Type": "application/json", "api_key": chave}


def montar_payload(transcricao):
    usuario = f"Transcricao:\n{transcricao}"
    p = {"model": MODELO, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": usuario}],
         "response_format": {"type": "json_object"}}
    if MODELO.startswith(("gpt-5", "o1", "o3", "o4")):
        p["max_completion_tokens"] = MAX_TOKENS_SAIDA * 4
    else:
        p["temperature"] = 0
        p["max_tokens"] = MAX_TOKENS_SAIDA
    return p


def chamar(url, headers, transcricao):
    payload = montar_payload(transcricao)
    erro = ""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            h = dict(headers)
            h["X-Correlation-ID"] = str(uuid.uuid4())
            r = requests.post(url, headers=h, json=payload, timeout=TIMEOUT_S)
            if r.ok:
                j = r.json()
                u = j.get("usage") or {}
                uso = {"prompt_tokens": u.get("prompt_tokens") or 0,
                       "completion_tokens": u.get("completion_tokens") or 0,
                       "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}
                try:
                    return json.loads(j["choices"][0]["message"]["content"]), uso, ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    erro = f"resposta nao e JSON valido (finish_reason={j['choices'][0].get('finish_reason')})"
            else:
                erro = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 401, 403, 404):
                    break
        except Exception as e:
            erro = str(e)[:200]
        time.sleep(5 * tentativa)
    return None, None, erro


def resposta_simulada(transcricao):
    return ({"P1": {"desfecho": "nao_era_venda", "intencao_entrada": "indefinido", "tentativa_comercial": "NAO", "B": []},
             "P2": {"tipo": "suporte_operacional", "quem_iniciou": "cliente"},
             "P3": {"problemas_identificados": ["nenhum"], "quem_relatou": "nenhum", "foi_resolvido_na_ligacao": "nenhum", "escalou_para": "nenhum"},
             "P5": {"teve_challenger": "NAO", "codigos_challenger": [], "resultado_imediato": "nenhum", "qualidade": "nenhum"},
             "P6": {"oportunidades_perdidas": ["OP1"], "valor_potencial_R$": 0},
             "P7": {"produtos_mencionados": ["PR1"], "produtos_ofertados": [], "janela_perdida": "NAO", "produto_janela": "nenhum"},
             "P8": {"objecoes": []}, "P9": {"tipo_abertura": "AB4", "usou_nome": "SIM", "tipo_fechamento": "FC3",
                                           "proximo_passo_claro": "NAO", "prazo_definido": "NAO"},
             "P10": {"promessas": [{"tipo_promessa": "PM3", "prazo_prometido": "hoje", "risco_nao_cumprimento": "baixo"}]},
             "P11": {"sinais": []}, "P15": {"eventos": []}},
            {"prompt_tokens": len(SYSTEM + transcricao) // 4, "completion_tokens": 350, "cached_tokens": len(SYSTEM) // 4}, "")


# ---------------------------------------------------------------------------
# Normalizacao da resposta
# ---------------------------------------------------------------------------
def _codigos(v, prefixo):
    itens = v if isinstance(v, list) else re.split(r"[+;,/ ]", str(v or ""))
    out = []
    for x in itens:
        x = str(x).strip().upper()
        if not x or x.lower() in ("nenhum", "nenhuma", "na", "none"):
            continue
        if x.isdigit():
            x = f"{prefixo}{x}"
        if re.fullmatch(fr"{prefixo}\d+", x) and x not in out:
            out.append(x)
    return out


def _txt(v):
    return str(v).strip() if v is not None else ""


def _lista_dicts(v):
    return [x for x in (v or []) if isinstance(x, dict)] if isinstance(v, list) else []


def resultado_sem_conteudo():
    return {"P1": {"desfecho": "nao_era_venda", "intencao_entrada": "indefinido", "tentativa_comercial": "NAO", "B": [],
                   "score_comercial": 0},
            "P2": {"tipo": "sem_conteudo", "quem_iniciou": "indefinido"},
            "P3": {"problemas_identificados": [], "quem_relatou": "nenhum", "foi_resolvido_na_ligacao": "nenhum", "escalou_para": "nenhum"},
            "P5": {"teve_challenger": "NAO", "codigos_challenger": [], "resultado_imediato": "nenhum", "qualidade": "nenhum"},
            "P6": {"oportunidades_perdidas": [], "valor_potencial_R$": 0},
            "P7": {"produtos_mencionados": [], "produtos_ofertados": [], "janela_perdida": "NAO", "produto_janela": "nenhum"},
            "P8": {"objecoes": []}, "P9": {}, "P10": {"promessas": []}, "P11": {"sinais": []}, "P15": {"eventos": []}}


def normalizar(resp, direcao):
    g = lambda k: resp.get(k) if isinstance(resp.get(k), dict) else {}
    p1, p2, p3, p5, p6, p7, p9 = (g(k) for k in ("P1", "P2", "P3", "P5", "P6", "P7", "P9"))
    b = _codigos(p1.get("B"), "B")
    if (direcao or "").strip().lower() == "outbound" and "B8" not in b:
        b.append("B8")  # regra de sistema: ligacao ativa do vendedor
    teve = _txt(p5.get("teve_challenger")).upper() or "NAO"
    return {
        "P1": {"desfecho": _txt(p1.get("desfecho")), "intencao_entrada": _txt(p1.get("intencao_entrada")),
               "tentativa_comercial": _txt(p1.get("tentativa_comercial")).upper(), "B": sorted(b, key=lambda x: int(x[1:])),
               "score_comercial": sum(PESOS_B.get(x, 0) for x in b)},
        "P2": {"tipo": _txt(p2.get("tipo")), "quem_iniciou": _txt(p2.get("quem_iniciou"))},
        "P3": {"problemas_identificados": _codigos(p3.get("problemas_identificados"), "P"),
               "quem_relatou": _txt(p3.get("quem_relatou")), "foi_resolvido_na_ligacao": _txt(p3.get("foi_resolvido_na_ligacao")).upper(),
               "escalou_para": _txt(p3.get("escalou_para"))},
        "P5": {"teve_challenger": teve, "codigos_challenger": _codigos(p5.get("codigos_challenger"), "CH") if teve == "SIM" else [],
               "resultado_imediato": _txt(p5.get("resultado_imediato")) if teve == "SIM" else "nenhum",
               "qualidade": _txt(p5.get("qualidade")) if teve == "SIM" else "nenhum"},
        "P6": {"oportunidades_perdidas": _codigos(p6.get("oportunidades_perdidas"), "OP"),
               "valor_potencial_R$": p6.get("valor_potencial_R$", 0)},
        "P7": {"produtos_mencionados": _codigos(p7.get("produtos_mencionados"), "PR"),
               "produtos_ofertados": _codigos(p7.get("produtos_ofertados"), "PR"),
               "janela_perdida": _txt(p7.get("janela_perdida")).upper() or "NAO",
               "produto_janela": (_codigos(p7.get("produto_janela"), "PR") or ["nenhum"])[0]},
        "P8": {"objecoes": _lista_dicts((resp.get("P8") or {}).get("objecoes") if isinstance(resp.get("P8"), dict) else [])},
        "P9": {k: _txt(p9.get(k)).upper() for k in ("tipo_abertura", "usou_nome", "tipo_fechamento", "proximo_passo_claro", "prazo_definido")},
        "P10": {"promessas": _lista_dicts((resp.get("P10") or {}).get("promessas") if isinstance(resp.get("P10"), dict) else [])},
        "P11": {"sinais": _lista_dicts((resp.get("P11") or {}).get("sinais") if isinstance(resp.get("P11"), dict) else [])},
        "P15": {"eventos": _lista_dicts((resp.get("P15") or {}).get("eventos") if isinstance(resp.get("P15"), dict) else [])},
    }


def linha_csv(reg):
    c = reg["classificacao"]
    j = lambda xs: "+".join(xs) if xs else "nenhum"
    p1, p3, p5, p6, p7, p9 = c["P1"], c["P3"], c["P5"], c["P6"], c["P7"], c["P9"]
    return {
        "cd_segmento": reg["cd_segmento"], "data": reg["data"], "data_hora_inicio": reg["data_hora_inicio"],
        "direcao": reg["direcao"], "nome_agente_1": reg["nome_agente_1"], "time_agente_1": reg["time_agente_1"],
        "fonte_classificacao": reg["fonte_classificacao"], "modelo": reg["modelo"],
        "p1_desfecho": p1["desfecho"], "p1_intencao_entrada": p1["intencao_entrada"],
        "p1_tentativa_comercial": p1["tentativa_comercial"],
        **{f"p1_B{i}": "SIM" if f"B{i}" in p1["B"] else "NAO" for i in range(1, 11)},
        "p1_score_comercial": p1.get("score_comercial", 0),
        "p2_tipo": c["P2"]["tipo"], "p2_quem_iniciou": c["P2"]["quem_iniciou"],
        "p3_problemas_identificados": j(p3["problemas_identificados"]), "p3_quem_relatou": p3["quem_relatou"],
        "p3_foi_resolvido_na_ligacao": p3["foi_resolvido_na_ligacao"], "p3_escalou_para": p3["escalou_para"],
        "p5_teve_challenger": p5["teve_challenger"], "p5_codigos_challenger": j(p5["codigos_challenger"]),
        "p5_resultado_imediato": p5["resultado_imediato"], "p5_qualidade": p5["qualidade"],
        "p6_oportunidades_perdidas": j(p6["oportunidades_perdidas"]), "p6_valor_potencial_R$": p6["valor_potencial_R$"],
        "p7_produtos_mencionados": j(p7["produtos_mencionados"]), "p7_produtos_ofertados": j(p7["produtos_ofertados"]),
        "p7_janela_perdida": p7["janela_perdida"], "p7_produto_janela": p7["produto_janela"],
        "p8_qtd_objecoes": len(c["P8"]["objecoes"]),
        "p8_objecoes": j([f"{_txt(o.get('tipo_objecao'))}:{_txt(o.get('resposta_vendedor_codigo'))}:{_txt(o.get('desfecho_da_objecao'))}"
                          for o in c["P8"]["objecoes"]]),
        "p9_tipo_abertura": p9.get("tipo_abertura", ""), "p9_usou_nome": p9.get("usou_nome", ""),
        "p9_tipo_fechamento": p9.get("tipo_fechamento", ""), "p9_proximo_passo_claro": p9.get("proximo_passo_claro", ""),
        "p9_prazo_definido": p9.get("prazo_definido", ""),
        "p10_qtd_promessas": len(c["P10"]["promessas"]),
        "p10_promessas": j([f"{_txt(p.get('tipo_promessa'))}:{_txt(p.get('risco_nao_cumprimento'))}" for p in c["P10"]["promessas"]]),
        "p11_sinais": j([_txt(s.get("tipo_sinal")) for s in c["P11"]["sinais"]]),
        "p11_concorrentes": j(sorted({_txt(s.get("concorrente_mencionado")) for s in c["P11"]["sinais"]} - {"", "nenhum"})),
        "p15_eventos": j([f"{_txt(e.get('categoria'))}:{_txt(e.get('impacto_potencial'))}" for e in c["P15"]["eventos"]]),
        "tokens_entrada": reg["uso"].get("prompt_tokens", 0), "tokens_cache": reg["uso"].get("cached_tokens", 0),
        "tokens_saida": reg["uso"].get("completion_tokens", 0), "erro": reg.get("erro", ""),
    }


def registrar_consumo(saida, nome, arquivo_entrada, registros, fontes, tok, custo):
    """consumo_diario.csv na pasta de saida: 1 linha por dia/arquivo (rodar de novo o mesmo dia atualiza a linha)."""
    caminho = saida / "consumo_diario.csv"
    campos = ["resultado", "arquivo_entrada", "atualizado_em", "modelo", "ligacoes", "pelo_gpt", "curtas_sem_conteudo",
              "com_erro", "tokens_entrada", "tokens_cache", "tokens_saida", "tokens_total", "custo_estimado_usd"]
    linhas = []
    if caminho.exists():
        with open(caminho, encoding="utf-8-sig", newline="") as f:
            linhas = [r for r in csv.DictReader(f, delimiter=";") if r.get("resultado") != nome]
    linhas.append({
        "resultado": nome, "arquivo_entrada": arquivo_entrada, "atualizado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo": MODELO, "ligacoes": len(registros), "pelo_gpt": fontes["gpt"] + fontes["simulado"],
        "curtas_sem_conteudo": fontes["auto_curta"], "com_erro": fontes["erro"],
        "tokens_entrada": tok["prompt_tokens"], "tokens_cache": tok["cached_tokens"], "tokens_saida": tok["completion_tokens"],
        "tokens_total": tok["prompt_tokens"] + tok["completion_tokens"], "custo_estimado_usd": f"{custo:.4f}".replace(".", ","),
    })
    linhas.sort(key=lambda r: r["resultado"])
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        w.writeheader()
        w.writerows(linhas)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", help="arquivo do dia (sem isso, abre uma janela)")
    ap.add_argument("--saida", help="pasta de saida (sem isso, abre uma janela)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--simular", action="store_true", help="nao chama a API (teste do script)")
    ap.add_argument("--silencioso", action="store_true", help="mostra so 1 linha a cada 25 ligacoes")
    args = ap.parse_args()

    entrada = args.entrada or escolher_arquivo()
    if not entrada:
        sys.exit("Nenhum arquivo escolhido.")
    saida = args.saida or escolher_pasta()
    if not saida:
        sys.exit("Nenhuma pasta escolhida.")
    entrada, saida = Path(entrada), Path(saida)
    saida.mkdir(parents=True, exist_ok=True)

    try:
        ligacoes, ignoradas = ler_ligacoes(entrada)
    except Exception as e:
        avisar("Erro no arquivo", f"Nao consegui ler {entrada.name}: {e}", erro=True)
        sys.exit(1)
    if not ligacoes:
        avisar("Nada para classificar", f"{entrada.name} nao tem ligacoes validas. Ignoradas: {ignoradas}", erro=True)
        sys.exit(1)

    url = headers = None
    if not args.simular:
        chave = None if os.getenv("API_KEY") else pedir_chave()
        url, headers = conexao(chave)
        if not headers.get("api_key") and "src.settings" not in sys.modules:
            avisar("Sem chave", "Sem a chave do llm-gate nao da para classificar.", erro=True)
            sys.exit(1)

    datas = sorted({l["data"] for l in ligacoes if l["data"]})
    nome = f"classificacao_{datas[0]}" if len(datas) == 1 else f"classificacao_{entrada.stem}"
    andamento = saida / f"{nome}_andamento.jsonl"
    feitos = {}
    if andamento.exists():
        for ln in andamento.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                reg = json.loads(ln)
                if reg["fonte_classificacao"] != "erro":
                    feitos[reg["cd_segmento"]] = reg

    pendentes = [l for l in ligacoes if l["cd_segmento"] not in feitos]
    print(f"{entrada.name}: {len(ligacoes)} ligacoes validas ({len(feitos)} ja classificadas, {len(pendentes)} a fazer) "
          f"| ignoradas: {ignoradas} | modelo {MODELO}")
    trava = threading.Lock()
    contador = [0]
    acum = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
    pin, pcache, pout = PRECOS.get(MODELO, (0, 0, 0))
    fmt = lambda n: f"{n:,}".replace(",", ".")
    if not args.silencioso:
        print(f"{'':>13} {'ligacao':<9}{'fonte':<11}{'desfecho':<22}{'tipo':<21}{'entrada':>8}{'cache':>7}{'saida':>6}"
              f"   acumulado")

    def processar(lig):
        t = compactar(lig["transcricao_limpa"])
        base = {k: lig[k] for k in ("cd_segmento", "data", "data_hora_inicio", "direcao", "nome_agente_1", "time_agente_1")}
        if len(t) < TAMANHO_MINIMO:
            reg = {**base, "fonte_classificacao": "auto_curta", "modelo": "", "classificacao": resultado_sem_conteudo(),
                   "uso": {}, "erro": ""}
        else:
            resp, uso, erro = resposta_simulada(t) if args.simular else chamar(url, headers, t)
            if resp is None:
                reg = {**base, "fonte_classificacao": "erro", "modelo": MODELO, "classificacao": resultado_sem_conteudo(),
                       "uso": {}, "erro": erro}
            else:
                reg = {**base, "fonte_classificacao": "simulado" if args.simular else "gpt", "modelo": MODELO,
                       "classificacao": normalizar(resp, lig["direcao"]), "uso": uso, "erro": ""}
        with trava:
            with open(andamento, "a", encoding="utf-8") as f:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            feitos[reg["cd_segmento"]] = reg
            contador[0] += 1
            u = reg["uso"] or {}
            for k in acum:
                acum[k] += u.get(k, 0)
            custo_acum = ((acum["prompt_tokens"] - acum["cached_tokens"]) * pin + acum["cached_tokens"] * pcache
                          + acum["completion_tokens"] * pout) / 1e6
            if not args.silencioso or contador[0] % 25 == 0 or contador[0] == len(pendentes) or reg["erro"]:
                c = reg["classificacao"]
                print(f"  [{contador[0]:>4}/{len(pendentes)}] {reg['cd_segmento'][:8]} {reg['fonte_classificacao']:<11}"
                      f"{c['P1']['desfecho'][:21]:<22}{c['P2']['tipo'][:20]:<21}"
                      f"{fmt(u.get('prompt_tokens', 0)):>8}{fmt(u.get('cached_tokens', 0)):>7}{fmt(u.get('completion_tokens', 0)):>6}"
                      f"   {fmt(acum['prompt_tokens'] + acum['completion_tokens'])} tok ~US$ {custo_acum:.3f}"
                      + (f"  ERRO: {reg['erro'][:70]}" if reg["erro"] else ""))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(processar, pendentes))

    registros = [feitos[l["cd_segmento"]] for l in ligacoes if l["cd_segmento"] in feitos]
    tok = {k: sum(r["uso"].get(k, 0) for r in registros) for k in ("prompt_tokens", "cached_tokens", "completion_tokens")}
    pin, pcache, pout = PRECOS.get(MODELO, (0, 0, 0))
    custo = ((tok["prompt_tokens"] - tok["cached_tokens"]) * pin + tok["cached_tokens"] * pcache
             + tok["completion_tokens"] * pout) / 1e6
    fontes = {f: sum(r["fonte_classificacao"] == f for r in registros) for f in ("gpt", "auto_curta", "erro", "simulado")}

    (saida / f"{nome}.json").write_text(json.dumps({
        "_meta": {"arquivo_entrada": entrada.name, "gerado_em": datetime.now().isoformat(timespec="seconds"),
                  "modelo": MODELO, "ligacoes": len(registros), "por_fonte": fontes, "ignoradas_na_leitura": ignoradas,
                  "tokens": tok, "custo_estimado_usd_preco_publico": round(custo, 4)},
        "ligacoes": registros}, ensure_ascii=False, indent=1), encoding="utf-8")
    linhas = [linha_csv(r) for r in registros]
    with open(saida / f"{nome}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        w.writeheader()
        w.writerows(linhas)

    venda = sum(r["classificacao"]["P1"]["desfecho"] not in ("nao_era_venda", "") for r in registros)
    fechou = sum(r["classificacao"]["P1"]["desfecho"].startswith("fechou") for r in registros)
    resumo = (f"{len(registros)} ligacoes classificadas ({fontes['gpt'] + fontes['simulado']} pelo GPT, "
              f"{fontes['auto_curta']} curtas sem conteudo, {fontes['erro']} com erro)\n"
              f"Venda: {venda} | fecharam: {fechou}\n"
              f"Tokens: {tok['prompt_tokens']:,} entrada ({tok['cached_tokens']:,} do cache), {tok['completion_tokens']:,} saida "
              f"— ~US$ {custo:.2f} (preco publico)\n\nArquivos em {saida}:\n  {nome}.json\n  {nome}.csv")
    if fontes["erro"]:
        resumo += f"\n\n{fontes['erro']} ligacoes deram erro: rode de novo com o mesmo arquivo e pasta para tentar so elas."
    registrar_consumo(saida, nome, entrada.name, registros, fontes, tok, custo)
    resumo += f"\n  consumo_diario.csv (1 linha por dia, para acompanhar o mes)"
    avisar("Classificacao concluida", resumo)


if __name__ == "__main__":
    main()
