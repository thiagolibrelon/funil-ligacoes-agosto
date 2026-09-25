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
TRECHOS (modo foco): p5_trecho_challenger e p3_trechos trazem o trecho REAL da transcricao (fala anterior, a fala,
fala seguinte), localizado a partir de uma ancora literal pedida ao modelo — o texto nunca e o do modelo. Se a ancora
nao existe na transcricao, o trecho fica vazio e a ligacao vai para "revisar". O nome do vendedor vira [VENDEDOR];
nomes de clientes/empresas NAO sao anonimizados — revise antes de usar trechos em apresentacoes.
Na tela aparece cada ligacao (desfecho, tipo, tokens) e o acumulado de tokens e custo; --silencioso mostra
so 1 linha a cada 25.

MODO "foco" (padrao, config C12 do teste): so os 3 temas que o negocio usa hoje — tipo de venda (era venda,
desfecho, tipo de ligacao), comportamento Challenger e problemas (todos os da ligacao, ex: acesso + boleto).
Evidencia antes do veredito, definicoes e regras de fronteira no prompt; qualidade do Challenger calculada
pela regua (dado concreto + conectou a situacao + cliente reagiu). Mude MODO para "completo" para voltar ao C7.

MODO "completo" (config C7 do teste): 11 prompts por ligacao em 1 pedido so
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

MODELO = "gpt-5.4-mini"   # troque para "gpt-4o-mini" se precisar voltar
REASONING_EFFORT = "minimal"  # so para modelos de raciocinio (gpt-5.x): raciocinio e cobrado como SAIDA
MODO = "foco"  # "foco" = C12 (venda, Challenger, problemas) | "completo" = C7 (11 prompts)
CHALLENGER_EXIGE_DADO = True  # Challenger so conta se o vendedor trouxe dado concreto (decisao de 24/09/2026)
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
PRECOS = {"gpt-4o-mini": (0.15, 0.075, 0.60), "gpt-4o": (2.50, 1.25, 10.00),
          "gpt-5.4-mini": (0.75, 0.075, 4.50)}  # 5.4-mini: entrada/saida informadas pelo time; cache = estimativa (10%)

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

# ---------------------------------------------------------------------------
# Prompt C12 (MODO "foco")
# ---------------------------------------------------------------------------
# Codigos de problema. Para acrescentar um codigo novo: uma linha aqui (codigo, nome, definicao/fronteira).
PROBLEMAS = [
    ("P1", "Acesso/sistema", "portal fora do ar, senha, biometria, usuario sem acesso"),
    ("P2", "Faturamento/cobranca", "boleto errado ou atrasado, fatura para terceiro, valor incorreto, cobranca indevida, "
                                  "contestacao de cobranca (avaria, pneu, taxas de encerramento)"),
    ("P3", "Multa/infracao", "multa no condutor errado, prazo de indicacao, contestacao"),
    ("P4", "Manutencao/substituto", "carro em oficina, demora, sem carro substituto"),
    ("P5", "Condutor/cadastro", "cadastro de condutor com ERRO ou bloqueio (CPF bloqueado, habilitacao recusada, acesso de ex-funcionario). "
                               "Pedido de rotina para trocar/incluir condutor, sem erro, NAO e problema"),
    ("P6", "Km excedente", "cobranca de km incorreta, plano de km errado"),
    ("P7", "Disponibilidade", "nao ha carro/categoria disponivel na data ou na loja pedida"),
    ("P8", "Franquia/agencia", "processo diferente ou falha em loja franqueada"),
    ("P9", "Sinistro", "acidente, avaria, franquia de dano"),
    ("P10", "Reserva/sistema", "reserva com erro no sistema: nao aparece, status errado, pendente de aprovacao, erro ao criar/editar. "
                               "Inclui contrato encerrado por engano no sistema. Se o problema e falta de carro, e P7; se a reserva travou por "
                               "limite de credito, e P13; duvida de como usar o portal sem erro nao e problema"),
    ("P12", "Manipulacao de pesquisa/NPS", "vendedor pede nota alta ou dita a nota. So pedir para o cliente avaliar o atendimento NAO e P12"),
    ("P13", "Credito/cadastro PJ", "reserva ou locacao travada por limite de credito insuficiente/comprometido, analise ou "
                                   "ampliacao de credito, cadastro PJ pendente, inativo ou reprovado, exigencia de documento "
                                   "que trava o cadastro (inclui cadastro de campanha)"),
    ("P14", "Preco/competitividade", "cliente reclama que o preco ficou acima da pessoa fisica, da cotacao ou da "
                                     "concorrencia, ou desiste/migra por preco. So informar o preco nao e problema"),
    ("P15", "Tag de pedagio", "FALHA da tag: inativa, nao cobra, erro ao ativar. Pedido de ativacao de rotina NAO e problema"),
    ("P11", "Outro", "problema real que nao cabe em nenhum codigo acima — obrigatorio preencher 'descricao'"),
]
CODIGOS_PROBLEMA = [c for c, _, _ in PROBLEMAS]

SYSTEM_FOCO = f"""Voce e analista de ligacoes comerciais B2B de locacao de frotas corporativas.
{CONTEXTO}

Analise a ligacao enviada pelo usuario e responda 3 perguntas. Nao invente nada que nao esteja na transcricao.
Em cada bloco, preencha PRIMEIRO a "ancora" (COPIA LITERAL de 6 a 12 palavras seguidas da transcricao que provam o
veredito) e SO DEPOIS o veredito. Se nao houver ancora, o veredito e NAO/nenhum.

# 1. VENDA
era_venda = SIM quando ha locacao, contrato ou produto em jogo COM INTERESSE DO CLIENTE: cotacao, reserva, pedido, data,
quantidade, renovacao, prorrogacao. Vendedor oferecer e o cliente nao ter demanda = era_venda NAO (com tentativa_comercial SIM).
desfecho:
- fechou_novo = locacao/contrato novo confirmado NA ligacao (inclui confirmar na ligacao um pedido que o cliente ja tinha feito)
- fechou_renovacao = renovou ou prorrogou contrato existente na ligacao
- fechou_upsell = cliente ativo adicionou carro ou produto na ligacao
- interessou_nao_fechou = cliente tem demanda real (cotacao, reserva, pedido), mas nao concluiu na ligacao
- nao_era_venda = nenhuma locacao em jogo (suporte, cobranca, recado, contato sem demanda)
Use EXATAMENTE um destes 5 valores no desfecho — nunca "nenhum" nem uma frase. Se era_venda=NAO, desfecho=nao_era_venda.
Casos de fronteira:
- cliente quer locar mas depende de credito, aprovacao ou reserva pendente = interessou_nao_fechou
- ligacao so de cobranca/pagamento, sem demanda de locacao = nao_era_venda
- vendedor sonda demanda e o cliente diz que nao tem = nao_era_venda
- desfecho nao_era_venda <=> era_venda NAO
tipo (assunto principal da ligacao):
nova_venda = prospect ou cliente sem contrato buscando locacao | renovacao = renovar/prorrogar contrato |
upsell = mais carros ou produto para cliente ativo | retencao = cliente inativo ou em risco, vendedor tenta retomar |
suporte_operacional = portal, acesso, reserva, cadastro, condutor | pos_venda_sinistro = acidente, avaria |
pos_venda_manutencao = carro em manutencao ou acompanhamento de uma locacao em andamento |
relacionamento_sem_demanda = visita, sondagem ou contato de rotina com cliente ou prospect, sem demanda e sem problema |
cobranca = fatura, boleto, pagamento | onboarding = boas-vindas/ativacao de cliente recem-cadastrado |
duvida_contrato = duvida sobre regras, tarifa ou condicoes | misto = dois assuntos com peso parecido |
sem_conteudo = sem conversa util (caixa postal, recado, transferencia)
Use EXATAMENTE um valor desta lista no tipo.

# 2. CHALLENGER
Challenger = o vendedor ENSINA algo que o cliente nao sabia, ADAPTA a situacao dele e TOMA O CONTROLE da conversa, sem o
cliente ter pedido, e SEMPRE com um DADO CONCRETO (numero, valor, prazo ou regra). Sem dado concreto NAO e Challenger.
NAO e Challenger: oferecer produto, responder duvida, informar preco ou regra quando perguntado, "se precisar me chama",
ser simpatico, conversar sobre frota propria sem mostrar custo ou dado. Na duvida, NAO.
Marque SIM/NAO em cada tipo:
CH1 custo oculto revelado — ex: "se devolver antes do prazo cai em diaria e sai o dobro"
CH2 ROI calculado — ex: "o eletrico custa X a mais, mas economiza Y de combustivel"
CH3 urgencia por regra real — ex: "a readequacao de tarifa e em marco; fechando hoje voce fica fora"
CH4 alternativa/concorrente superado com dado — ex: "frota propria tem IPVA, seguro e manutencao; a locacao elimina isso"
CH5 dor revelada conectada a produto — ex: "voce falou em multa de condutor errado; a telemetria resolve isso"
CH6 antecipacao de problema — ex: "seu contrato vence em 2 semanas; ja deixo renovado para nao cair em diaria"
CH7 descoberta de necessidade nao declarada — ex: vendedor pergunta e descobre frota propria ou caminhoes que o cliente nao citou
Regua (so se houve Challenger): trouxe_dado_concreto (numero, prazo, regra, valor) |
conectou_a_situacao_do_cliente = SIM so se usou algo que o proprio cliente disse NESTA ligacao |
cliente_reagiu = SIM so se o cliente respondeu ao que foi ensinado (aceitou, perguntou mais, disse que vai avaliar);
"ok", "ta bom", "entendi" = NAO

# 3. PROBLEMAS
Problema = algo DEU ERRADO ou esta IMPEDINDO o cliente. Pedido de rotina ou duvida sem erro = nao e problema.
Liste TODOS os problemas da ligacao — uma ligacao pode ter varios (ex: acesso ao portal + boleto atrasado = P1 e P2).
So conte problema que e tema relevante da ligacao, nao mencao de passagem.
""" + "\n".join(f"{c} {n} — {d}" for c, n, d in PROBLEMAS) + """

Responda SOMENTE um objeto JSON valido, sem texto fora dele, neste formato e nesta ordem:
{"venda": {"ancora": "<copia literal>", "era_venda": "SIM|NAO", "tentativa_comercial": "SIM|NAO", "desfecho": "...", "tipo": "..."},
 "challenger": {"ancora": "<copia literal da fala do vendedor ensinando, ou vazio>", "CH1": "SIM|NAO", "CH2": "SIM|NAO",
  "CH3": "SIM|NAO", "CH4": "SIM|NAO", "CH5": "SIM|NAO", "CH6": "SIM|NAO", "CH7": "SIM|NAO",
  "trouxe_dado_concreto": "SIM|NAO", "conectou_a_situacao_do_cliente": "SIM|NAO", "cliente_reagiu": "SIM|NAO"},
 "problemas": {"lista": [{"ancora": "<copia literal>", "codigo": "P..", "descricao": "<so para P11, ate 8 palavras>"}],
  "tem_problema": "SIM|NAO", "principal": "P..|nenhum", "foi_resolvido_na_ligacao": "SIM|NAO|PARCIAL|nenhum"}}
Use sempre os codigos (P1..., CH1...), nunca o nome por extenso. Lista vazia [] se nao houve problema.
"ancora" = COPIA EXATA de 6 a 12 palavras SEGUIDAS de UMA fala da transcricao, como estao escritas (sem corrigir, sem resumir,
sem juntar falas diferentes). E usada para localizar o trecho na ligacao."""

TIPOS_VENDA = ("nova_venda", "renovacao", "upsell")
DESFECHOS = ("fechou_novo", "fechou_renovacao", "fechou_upsell", "interessou_nao_fechou", "nao_era_venda")
TIPOS = ("nova_venda", "renovacao", "upsell", "retencao", "suporte_operacional", "pos_venda_sinistro",
         "pos_venda_manutencao", "relacionamento_sem_demanda", "cobranca", "onboarding", "duvida_contrato", "misto",
         "sem_conteudo")
VAZIOS = ("", "nenhum", "nenhuma", "na", "none", "null")


def _chave(v):
    """'Não era venda' / 'relacionamento sem demanda' -> 'nao_era_venda' / 'relacionamento_sem_demanda'."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(v or "")).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def aplicar_regras_foco(c):
    """Padroniza desfecho/tipo, aplica a regra do Challenger e refaz a lista de 'revisar'. Pode rodar mais de uma vez."""
    p1, p2, p5 = c["P1"], c["P2"], c["P5"]
    revisar = []
    era = p1["era_venda"]

    d = _chave(p1["desfecho"])
    if d not in DESFECHOS:
        if era == "NAO" or d in VAZIOS:
            d = "nao_era_venda" if era == "NAO" else "interessou_nao_fechou"
        else:
            p1["desfecho_original"] = p1["desfecho"]
            d = "interessou_nao_fechou"
            revisar.append("desfecho fora da lista")
    p1["desfecho"] = d
    if p1.get("desfecho_original") and "desfecho fora da lista" not in revisar:
        revisar.append("desfecho fora da lista")  # mantem a marcacao quando as regras rodam de novo
    if (era == "NAO") != (d == "nao_era_venda"):
        revisar.append("era_venda x desfecho")

    t = _chave(p2["tipo"])
    if t in ("relacionamento", "relacionamento_sem_demanda_comercial"):
        t = "relacionamento_sem_demanda"
    if t in VAZIOS:
        t = "sem_conteudo" if era == "NAO" else "misto"
    if t not in TIPOS:
        p2["tipo_original"] = p2.get("tipo_original") or p2["tipo"]
        revisar.append("tipo fora da lista")
    p2["tipo"] = t
    if era == "NAO" and t in TIPOS_VENDA:
        revisar.append("tipo de venda sem venda")

    if CHALLENGER_EXIGE_DADO and p5["teve_challenger"] == "SIM" and p5.get("trouxe_dado_concreto") != "SIM":
        p5["challenger_sem_dado"] = "SIM"  # guardado para auditoria: o modelo marcou, a regra descartou
        p5.update(teve_challenger="NAO", codigos_challenger=[], qualidade="nenhum")
    if p5["teve_challenger"] == "SIM":
        pontos = sum(p5.get(k) == "SIM" for k in ("trouxe_dado_concreto", "conectou_a_situacao_do_cliente", "cliente_reagiu"))
        p5["qualidade"] = "alta" if pontos == 3 else "media" if pontos == 2 else "baixa"

    c["revisar"] = [r for r in c.get("revisar", []) if r == "tem_problema sem codigo" or r.startswith("trecho")] + revisar
    return c


def _sim(v):
    return "SIM" if str(v or "").strip().upper() in ("SIM", "S", "YES", "TRUE") else "NAO"


def normalizar_foco(resp):
    """Resposta do C12 -> mesmo formato de chaves do modo completo (P1, P2, P3, P5) + campos do foco."""
    g = lambda k: resp.get(k) if isinstance(resp.get(k), dict) else {}
    v, ch, pr = g("venda"), g("challenger"), g("problemas")
    revisar = []

    era = _sim(v.get("era_venda"))
    desfecho = _txt(v.get("desfecho"))
    tipo = _txt(v.get("tipo"))

    codigos_ch = [f"CH{i}" for i in range(1, 8) if _sim(ch.get(f"CH{i}")) == "SIM"]
    frase = _txt(ch.get("ancora") or ch.get("frase_vendedor"))
    teve = "SIM" if codigos_ch and frase else "NAO"
    regua = {k: _sim(ch.get(k)) for k in ("trouxe_dado_concreto", "conectou_a_situacao_do_cliente", "cliente_reagiu")}
    pontos = sum(x == "SIM" for x in regua.values())
    qualidade = ("alta" if pontos == 3 else "media" if pontos == 2 else "baixa") if teve == "SIM" else "nenhum"

    itens, vistos = [], set()
    for it in _lista_dicts(pr.get("lista")):
        cod = (_codigos(it.get("codigo"), "P") or [""])[0]
        if cod in CODIGOS_PROBLEMA and cod not in vistos:
            vistos.add(cod)
            itens.append({"codigo": cod, "evidencia": _txt(it.get("ancora") or it.get("evidencia")),
                          "descricao": _txt(it.get("descricao")) if cod == "P11" else ""})
    codigos_pr = [i["codigo"] for i in itens]
    tem = "SIM" if codigos_pr else "NAO"
    if _sim(pr.get("tem_problema")) == "SIM" and not codigos_pr:
        revisar.append("tem_problema sem codigo")
    principal = (_codigos(pr.get("principal"), "P") or [""])[0]
    if principal not in codigos_pr:
        principal = codigos_pr[0] if codigos_pr else "nenhum"

    return aplicar_regras_foco({
        "P1": {"era_venda": era, "desfecho": desfecho, "tentativa_comercial": _sim(v.get("tentativa_comercial")),
               "evidencia_venda": _txt(v.get("ancora") or v.get("evidencia"))},
        "P2": {"tipo": tipo},
        "P3": {"tem_problema": tem, "problemas_identificados": codigos_pr, "problema_principal": principal,
               "foi_resolvido_na_ligacao": _txt(pr.get("foi_resolvido_na_ligacao")).upper() if codigos_pr else "nenhum",
               "itens": itens},
        "P5": {"teve_challenger": teve, "codigos_challenger": codigos_ch if teve == "SIM" else [], "qualidade": qualidade,
               **regua, "frase_vendedor": frase if teve == "SIM" else "", "reacao_cliente": _txt(ch.get("reacao_cliente")) if teve == "SIM" else ""},
        "revisar": revisar,
    })


def resultado_sem_conteudo_foco():
    return {"P1": {"era_venda": "NAO", "desfecho": "nao_era_venda", "tentativa_comercial": "NAO", "evidencia_venda": ""},
            "P2": {"tipo": "sem_conteudo"},
            "P3": {"tem_problema": "NAO", "problemas_identificados": [], "problema_principal": "nenhum",
                   "foi_resolvido_na_ligacao": "nenhum", "itens": []},
            "P5": {"teve_challenger": "NAO", "codigos_challenger": [], "qualidade": "nenhum", "trouxe_dado_concreto": "NAO",
                   "conectou_a_situacao_do_cliente": "NAO", "cliente_reagiu": "NAO", "frase_vendedor": "", "reacao_cliente": ""},
            "revisar": []}


def linha_csv_foco(reg):
    c = reg["classificacao"]
    j = lambda xs: "+".join(xs) if xs else "nenhum"
    p1, p3, p5 = c["P1"], c["P3"], c["P5"]
    return {
        "cd_segmento": reg["cd_segmento"], "data": reg["data"], "data_hora_inicio": reg["data_hora_inicio"],
        "direcao": reg["direcao"], "nome_agente_1": reg["nome_agente_1"], "time_agente_1": reg["time_agente_1"],
        "fonte_classificacao": reg["fonte_classificacao"], "modelo": reg["modelo"],
        "era_venda": p1["era_venda"], "p1_desfecho": p1["desfecho"], "p2_tipo": c["P2"]["tipo"],
        "p1_tentativa_comercial": p1["tentativa_comercial"], "evidencia_venda": p1["evidencia_venda"],
        "p1_trecho_venda": p1.get("trecho", ""),
        "tem_problema": p3["tem_problema"], "p3_problemas_identificados": j(p3["problemas_identificados"]),
        "p3_problema_principal": p3["problema_principal"], "p3_foi_resolvido_na_ligacao": p3["foi_resolvido_na_ligacao"],
        "p3_evidencias": " | ".join(f"{i['codigo']}: {i['evidencia']}" for i in p3["itens"]),
        "p3_trechos": " || ".join(f"{i['codigo']}: {i.get('trecho', '')}" for i in p3["itens"] if i.get("trecho")),
        "p3_trechos_status": " | ".join(f"{i['codigo']}: {i.get('trecho_status', '')}" for i in p3["itens"] if i.get("trecho_status")),
        "p3_outro_descricao": " | ".join(i["descricao"] for i in p3["itens"] if i["codigo"] == "P11"),
        "p5_teve_challenger": p5["teve_challenger"], "p5_codigos_challenger": j(p5["codigos_challenger"]),
        "p5_qualidade": p5["qualidade"], "p5_trouxe_dado": p5["trouxe_dado_concreto"],
        "p5_conectou_situacao": p5["conectou_a_situacao_do_cliente"], "p5_cliente_reagiu": p5["cliente_reagiu"],
        "p5_frase_vendedor": p5["frase_vendedor"], "p5_reacao_cliente": p5["reacao_cliente"],
        "p5_challenger_sem_dado": p5.get("challenger_sem_dado", "NAO"),
        "p5_trecho_challenger": p5.get("trecho", "") if p5["teve_challenger"] == "SIM" else "",
        "p5_trecho_status": p5.get("trecho_status", "") if p5["teve_challenger"] == "SIM" else "",
        "p1_desfecho_original": p1.get("desfecho_original", ""), "p2_tipo_original": c["P2"].get("tipo_original", ""),
        "revisar": "; ".join(c.get("revisar", [])),
        "tokens_entrada": reg["uso"].get("prompt_tokens", 0), "tokens_cache": reg["uso"].get("cached_tokens", 0),
        "tokens_saida": reg["uso"].get("completion_tokens", 0), "tokens_raciocinio": reg["uso"].get("reasoning_tokens", 0),
        "custo_llm_gate": reg["uso"].get("custo_llm_gate") if reg["uso"].get("custo_llm_gate") is not None else "",
        "erro": reg.get("erro", ""),
    }


def _palavras(t):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", t)


def _anonimizar(texto, nome_agente):
    """Troca o nome do vendedor (do cadastro) por [VENDEDOR] e sequencias longas de digitos por [NUM]."""
    nomes = {w for w in re.split(r"[^A-Za-zÀ-ÿ]+", str(nome_agente or "")) if len(w) >= 3}
    for n in sorted(nomes, key=len, reverse=True):
        texto = re.sub(rf"\b{re.escape(n)}\b", "[VENDEDOR]", texto, flags=re.I)
    return re.sub(r"\d[\d .\-/]{5,}\d", "[NUM]", texto)


def localizar_trecho(transcricao_compacta, ancora, nome_agente=""):
    """Acha a fala que contem a ancora e devolve o trecho REAL (fala anterior + fala + seguinte).
    Status: encontrado | aproximado (>=70% das palavras da ancora na mesma fala) | nao_encontrado | sem_ancora."""
    alvo = _palavras(ancora)
    if len(alvo) < 3:
        return "", "sem_ancora"
    turnos = [ln for ln in transcricao_compacta.splitlines() if ln.strip()]
    frase = " ".join(alvo)
    unicas = set(alvo)
    melhor, melhor_i = 0.0, -1
    for i in range(len(turnos)):
        # a fala sozinha e a fala + a seguinte (a ancora pode atravessar a quebra do STT)
        for janela in (turnos[i], turnos[i] + " " + turnos[i + 1] if i + 1 < len(turnos) else None):
            if janela is None:
                continue
            pal = _palavras(janela)
            if frase in " ".join(pal):
                melhor, melhor_i = 1.0, i
                break
            cobertura = len(unicas & set(pal)) / len(unicas)
            if cobertura > melhor:
                melhor, melhor_i = cobertura, i
        if melhor == 1.0:
            break
    if melhor < 0.65:
        return "", "nao_encontrado"
    corte = lambda ln: ln if len(ln) <= 400 else ln[:400] + "..."
    trecho = " / ".join(corte(turnos[j]) for j in range(max(0, melhor_i - 1), min(len(turnos), melhor_i + 2)))
    return _anonimizar(trecho, nome_agente), ("encontrado" if melhor == 1.0 else "aproximado")


def anexar_trechos(c, resp, transcricao_compacta, nome_agente):
    """Guarda os trechos reais de Challenger e de cada problema; marca revisar se a ancora nao existe na transcricao."""
    ch = resp.get("challenger") if isinstance(resp.get("challenger"), dict) else {}
    v = resp.get("venda") if isinstance(resp.get("venda"), dict) else {}
    c["P1"]["trecho"], c["P1"]["trecho_status"] = localizar_trecho(transcricao_compacta, v.get("ancora"), nome_agente)
    if c["P5"]["teve_challenger"] == "SIM":
        c["P5"]["trecho"], c["P5"]["trecho_status"] = localizar_trecho(transcricao_compacta, ch.get("ancora"), nome_agente)
        if c["P5"]["trecho_status"] in ("nao_encontrado", "sem_ancora"):
            c["revisar"].append("trecho do Challenger nao encontrado")
    ancoras = {}
    pr = resp.get("problemas") if isinstance(resp.get("problemas"), dict) else {}
    for it in _lista_dicts(pr.get("lista")):
        cod = (_codigos(it.get("codigo"), "P") or [""])[0]
        ancoras.setdefault(cod, it.get("ancora"))
    faltou = False
    for item in c["P3"]["itens"]:
        item["trecho"], item["trecho_status"] = localizar_trecho(transcricao_compacta, ancoras.get(item["codigo"]), nome_agente)
        faltou |= item["trecho_status"] in ("nao_encontrado", "sem_ancora")
    if faltou:
        c["revisar"].append("trecho de problema nao encontrado")
    return c


def resposta_simulada_foco(transcricao):
    return ({"venda": {"evidencia": "cliente pede cotacao de 2 carros", "era_venda": "SIM", "tentativa_comercial": "SIM",
                       "desfecho": "interessou_nao_fechou", "tipo": "nova_venda"},
             "challenger": {"frase_vendedor": "", "reacao_cliente": "", **{f"CH{i}": "NAO" for i in range(1, 8)},
                            "trouxe_dado_concreto": "NAO", "conectou_a_situacao_do_cliente": "NAO", "cliente_reagiu": "NAO"},
             "problemas": {"lista": [{"evidencia": "nao consegue entrar no portal", "codigo": "P1",
                                      "ancora": " ".join((transcricao.splitlines() or [""])[min(1, len(transcricao.splitlines()) - 1)].split()[1:9])},
                                     {"evidencia": "boleto venceu sem chegar", "codigo": "2", "ancora": "frase que nao existe na ligacao"}],
                           "tem_problema": "SIM", "principal": "P1", "foi_resolvido_na_ligacao": "PARCIAL"}},
            {"prompt_tokens": len(SYSTEM_FOCO + transcricao) // 4, "completion_tokens": 250,
             "cached_tokens": len(SYSTEM_FOCO) // 4}, "")


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


def escolher_arquivos():
    """Um ou varios dias: na janela, Shift+clique (intervalo) ou Ctrl+clique (avulsos)."""
    from tkinter import filedialog
    raiz = _tk()
    caminhos = filedialog.askopenfilenames(
        parent=raiz, title="Escolha o(s) arquivo(s) de ligacoes — Shift+clique para varios dias",
        filetypes=[("Todos os arquivos", "*.*"), ("CSV", "*.csv")])
    raiz.destroy()
    return list(caminhos)


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


INTERATIVO = True  # False quando --entrada e --saida vem pela linha de comando (agendamento): sem janela no fim


def avisar(titulo, texto, erro=False):
    print(texto)
    if not INTERATIVO:
        return
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
    system = SYSTEM_FOCO if MODO == "foco" else SYSTEM
    p = {"model": MODELO, "messages": [{"role": "system", "content": system}, {"role": "user", "content": usuario}],
         "response_format": {"type": "json_object"}}
    if MODELO.startswith(("gpt-5", "o1", "o3", "o4")):
        p["max_completion_tokens"] = MAX_TOKENS_SAIDA * 4
        if REASONING_EFFORT:
            p["reasoning_effort"] = REASONING_EFFORT
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
                custo_gate = ((j.get("cost") or {}).get("token") or {}).get("total")
                uso = {"prompt_tokens": u.get("prompt_tokens") or 0,
                       "completion_tokens": u.get("completion_tokens") or 0,  # inclui o raciocinio
                       "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                       "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                       "custo_llm_gate": custo_gate}
                try:
                    return json.loads(j["choices"][0]["message"]["content"]), uso, ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    erro = f"resposta nao e JSON valido (finish_reason={j['choices'][0].get('finish_reason')})"
            else:
                erro = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code == 400 and "reasoning_effort" in payload and "reasoning" in r.text.lower():
                    # valor nao aceito por este modelo: tenta "low" e, se ainda falhar, sem o parametro
                    payload["reasoning_effort"] = "low" if payload["reasoning_effort"] == "minimal" else None
                    if payload["reasoning_effort"] is None:
                        payload.pop("reasoning_effort")
                    continue
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


def resumo_do_dia(registros):
    """O que o negocio pergunta: vendas, tipos, problemas mais recorrentes (todos os codigos) e Challenger."""
    from collections import Counter
    uteis = [r for r in registros if r["fonte_classificacao"] in ("gpt", "simulado")]
    nomes = {c: n for c, n, _ in PROBLEMAS}
    desf = Counter(r["classificacao"]["P1"]["desfecho"] for r in registros)
    venda = sum(v for k, v in desf.items() if k not in ("nao_era_venda", ""))
    fechou = sum(v for k, v in desf.items() if k.startswith("fechou"))
    tipos = Counter(r["classificacao"]["P2"]["tipo"] for r in uteis).most_common(4)
    com_prob = [r for r in uteis if r["classificacao"]["P3"]["problemas_identificados"]]
    multi = sum(len(r["classificacao"]["P3"]["problemas_identificados"]) > 1 for r in com_prob)
    ranking = Counter(c for r in com_prob for c in r["classificacao"]["P3"]["problemas_identificados"]).most_common(8)
    ch = Counter(r["classificacao"]["P5"]["qualidade"] for r in uteis if r["classificacao"]["P5"]["teve_challenger"] == "SIM")
    revisar = sum(bool(r["classificacao"].get("revisar")) for r in uteis)
    linhas = [
        f"VENDA: {venda} ligacoes com venda em jogo, {fechou} fecharam ({100 * fechou / venda:.0f}% das vendas)" if venda
        else "VENDA: nenhuma ligacao com venda em jogo",
        "Tipos mais comuns: " + ", ".join(f"{t} {n}" for t, n in tipos),
        f"PROBLEMAS: {len(com_prob)} de {len(uteis)} ligacoes com conversa ({multi} com mais de um problema)",
        *[f"  {c} {nomes.get(c, '')}: {n}" for c, n in ranking],
        f"CHALLENGER: {sum(ch.values())} ligacoes (alta {ch.get('alta', 0)}, media {ch.get('media', 0)}, baixa {ch.get('baixa', 0)})",
    ]
    if revisar:
        linhas.append(f"{revisar} ligacoes marcadas para revisar (coluna 'revisar' do CSV)")
    return "\n".join(linhas)


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
def classificar_arquivo(entrada, saida, args, url, headers):
    """Classifica um arquivo (um dia). Devolve (texto do resumo, estatisticas) ou (mensagem de erro, None)."""
    try:
        ligacoes, ignoradas = ler_ligacoes(entrada)
    except Exception as e:
        return f"{entrada.name}: nao consegui ler o arquivo ({e})", None
    if not ligacoes:
        return f"{entrada.name}: nenhuma ligacao valida (ignoradas: {ignoradas})", None

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
    print(f"modo {MODO} ({'C12: venda, Challenger, problemas' if MODO == 'foco' else 'C7: 11 prompts'})")
    print(f"{entrada.name}: {len(ligacoes)} ligacoes validas ({len(feitos)} ja classificadas, {len(pendentes)} a fazer) "
          f"| ignoradas: {ignoradas} | modelo {MODELO}")
    trava = threading.Lock()
    contador = [0]
    vazio = resultado_sem_conteudo_foco if MODO == "foco" else resultado_sem_conteudo
    simulada = resposta_simulada_foco if MODO == "foco" else resposta_simulada
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
            reg = {**base, "fonte_classificacao": "auto_curta", "modelo": "", "classificacao": vazio(),
                   "uso": {}, "erro": ""}
        else:
            resp, uso, erro = simulada(t) if args.simular else chamar(url, headers, t)
            if resp is None:
                reg = {**base, "fonte_classificacao": "erro", "modelo": MODELO, "classificacao": vazio(),
                       "uso": {}, "erro": erro}
            else:
                reg = {**base, "fonte_classificacao": "simulado" if args.simular else "gpt", "modelo": MODELO,
                       "resposta_bruta": resp,
                       "classificacao": (anexar_trechos(normalizar_foco(resp), resp, t, lig["nome_agente_1"])
                                         if MODO == "foco" else normalizar(resp, lig["direcao"])),
                       "uso": uso, "erro": ""}
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
    if MODO == "foco":  # regras novas valem tambem para o que ja estava classificado (sem chamar a API de novo)
        for r in registros:
            if r["fonte_classificacao"] in ("gpt", "simulado") and "venda" not in r["classificacao"]:
                r["classificacao"] = aplicar_regras_foco(r["classificacao"])
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
    linhas = [(linha_csv_foco if MODO == "foco" else linha_csv)(r) for r in registros]
    with open(saida / f"{nome}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        w.writeheader()
        w.writerows(linhas)

    resumo = (f"{len(registros)} ligacoes classificadas ({fontes['gpt'] + fontes['simulado']} pelo GPT, "
              f"{fontes['auto_curta']} curtas sem conteudo, {fontes['erro']} com erro)\n\n"
              + resumo_do_dia(registros) + "\n\n"
              f"Tokens: {tok['prompt_tokens']:,} entrada ({tok['cached_tokens']:,} do cache), {tok['completion_tokens']:,} saida "
              f"— ~US$ {custo:.2f} (preco publico)\n\nArquivos em {saida}:\n  {nome}.json\n  {nome}.csv")
    if fontes["erro"]:
        resumo += f"\n\n{fontes['erro']} ligacoes deram erro: rode de novo com o mesmo arquivo e pasta para tentar so elas."
    registrar_consumo(saida, nome, entrada.name, registros, fontes, tok, custo)
    resumo += f"\n  consumo_diario.csv (1 linha por dia, para acompanhar o mes)"
    return resumo, {"nome": nome, "ligacoes": len(registros), "gpt": fontes["gpt"] + fontes["simulado"],
                    "erro": fontes["erro"], "tokens": tok["prompt_tokens"] + tok["completion_tokens"], "custo": custo}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", nargs="+", help="arquivo(s) do(s) dia(s) (sem isso, abre uma janela)")
    ap.add_argument("--saida", help="pasta de saida (sem isso, abre uma janela)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--simular", action="store_true", help="nao chama a API (teste do script)")
    ap.add_argument("--silencioso", action="store_true", help="mostra so 1 linha a cada 25 ligacoes")
    args = ap.parse_args()

    global INTERATIVO
    INTERATIVO = not (args.entrada and args.saida)
    entradas = args.entrada or escolher_arquivos()
    if not entradas:
        sys.exit("Nenhum arquivo escolhido.")
    saida = args.saida or escolher_pasta()
    if not saida:
        sys.exit("Nenhuma pasta escolhida.")
    entradas, saida = sorted(Path(e) for e in entradas), Path(saida)
    saida.mkdir(parents=True, exist_ok=True)

    url = headers = None
    if not args.simular:
        chave = None if os.getenv("API_KEY") else pedir_chave()
        url, headers = conexao(chave)
        if not headers.get("api_key") and "src.settings" not in sys.modules:
            avisar("Sem chave", "Sem a chave do llm-gate nao da para classificar.", erro=True)
            sys.exit(1)

    resultados = []
    for n, entrada in enumerate(entradas, 1):
        if len(entradas) > 1:
            print(f"\n{'=' * 70}\nARQUIVO {n}/{len(entradas)}: {entrada.name}\n{'=' * 70}")
        texto, stats = classificar_arquivo(entrada, saida, args, url, headers)
        resultados.append((entrada, texto, stats))
        if len(entradas) > 1:
            print(texto)

    if len(entradas) == 1:
        texto, stats = resultados[0][1], resultados[0][2]
        avisar("Classificacao concluida" if stats else "Erro no arquivo", texto, erro=stats is None)
        return

    ok = [r for r in resultados if r[2]]
    linhas = [f"{'dia':<26}{'ligacoes':>9}{'GPT':>6}{'erro':>6}{'tokens':>12}{'US$':>8}"]
    for entrada, texto, st in resultados:
        if st:
            linhas.append(f"{st['nome'].replace('classificacao_', ''):<26}{st['ligacoes']:>9}{st['gpt']:>6}{st['erro']:>6}"
                          f"{st['tokens']:>12,}{st['custo']:>8.2f}")
        else:
            linhas.append(f"{entrada.name:<26}  FALHOU: {texto[:60]}")
    linhas.append(f"{'TOTAL':<26}{sum(s['ligacoes'] for _, _, s in ok):>9}{sum(s['gpt'] for _, _, s in ok):>6}"
                  f"{sum(s['erro'] for _, _, s in ok):>6}{sum(s['tokens'] for _, _, s in ok):>12,}"
                  f"{sum(s['custo'] for _, _, s in ok):>8.2f}")
    fim = (f"{len(ok)} de {len(entradas)} arquivos classificados. Resultados e consumo_diario.csv em {saida}\n\n"
           + "\n".join(linhas))
    if any(s["erro"] for _, _, s in ok):
        fim += "\n\nHouve ligacoes com erro: rode de novo selecionando os mesmos arquivos e a mesma pasta — so elas sao refeitas."
    avisar("Classificacao concluida", fim)


if __name__ == "__main__":
    main()
