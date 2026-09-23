"""Monta amostra_5.json: 5 ligacoes de agosto/2026 com o gabarito do Gemini pago
(gemini-2.5-flash, agosto_grvin/gemini/CONSOLIDADO.csv) nos prompts P1, P2, P3, P5, P6 e P9.

Roda AQUI (maquina local, onde estao chamadas/agosto/). O json gerado contem
transcricao — nao vai pro git (ver .gitignore). Leva-se so ele + teste_gpt.py pra Localiza.

Selecao: 4 ligacoes fixas (uma por quadrante venda x problema, as mesmas da pagina
casos_nao_venda.html) + 1 venda com Challenger de qualidade alta. Todas classificadas
pelo Gemini (nenhuma auto_curta), com P9 preenchido e sem tema eleitoral.
"""
import csv
import json
import re
from pathlib import Path

csv.field_size_limit(10**9)

ROOT = Path(__file__).resolve().parent.parent
CONSOLIDADO = ROOT / "agosto_grvin" / "gemini" / "CONSOLIDADO.csv"
CHAMADAS = ROOT / "chamadas" / "agosto"
SAIDA = Path(__file__).resolve().parent / "amostra_5.json"

FIXAS = {
    "4522d50a": "venda que fechou, sem problema",
    "3bb2a5b8": "venda com problema (reserva presa)",
    "d8cb0363": "nao venda com problema (fatura vencida)",
    "44543c13": "nao venda sem problema (visita de rotina)",
}
ELEITORAL = re.compile(r"elei[cç]|eleitoral|candidat|partido|comit[eê]", re.I)

CAMPOS = {
    "P1": ["desfecho", "intencao_entrada", "tentativa_comercial"] + [f"B{i}" for i in range(1, 11)],
    "P2": ["tipo", "subtipo", "quem_iniciou", "tentativa_comercial"],
    "P3": ["problemas_identificados", "quem_relatou", "foi_resolvido_na_ligacao", "escalou_para"],
    "P5": ["teve_challenger", "codigos_challenger", "resultado_imediato", "qualidade"],
    "P6": ["oportunidades_perdidas", "valor_potencial_R$"],
    "P9": ["tipo_abertura", "usou_nome", "tipo_fechamento", "proximo_passo_claro", "prazo_definido"],
}


def carregar_transcricoes():
    tx = {}
    for arq in sorted(CHAMADAS.iterdir()):
        with open(arq, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tx[row["cd_segmento"]] = row
    return tx


def gabarito(row):
    return {p: {c: row[f"{p.lower()}_{c}"] for c in cols} for p, cols in CAMPOS.items()}


def main():
    with open(CONSOLIDADO, encoding="utf-8", newline="") as f:
        cons = list(csv.DictReader(f))
    tx = carregar_transcricoes()

    escolhidas = []
    for prefixo, perfil in FIXAS.items():
        row = next(r for r in cons if r["cd_segmento"].startswith(prefixo))
        escolhidas.append((row, perfil))

    candidatas = [
        r for r in cons
        if r["fonte_classificacao"] == "gemini" and r["p5_qualidade"] == "alta"
        and r["p1_desfecho"] != "nao_era_venda" and r["p9_tipo_abertura"]
        and not ELEITORAL.search(r["p2_subtipo"] + tx.get(r["cd_segmento"], {}).get("transcricao_limpa", ""))
        and 1500 <= len(tx.get(r["cd_segmento"], {}).get("transcricao_limpa", "")) <= 4000
    ]
    candidatas.sort(key=lambda r: r["cd_segmento"])  # deterministico
    escolhidas.append((candidatas[0], "venda com Challenger de qualidade alta"))

    ligacoes = []
    for row, perfil in escolhidas:
        assert row["fonte_classificacao"] == "gemini", row["cd_segmento"]
        assert row["p9_tipo_abertura"], row["cd_segmento"]
        orig = tx[row["cd_segmento"]]
        ligacoes.append({
            "cd_segmento": row["cd_segmento"],
            "perfil": perfil,
            "data": orig.get("Data", ""),
            "direcao": orig.get("direcao", ""),
            "transcricao_limpa": orig["transcricao_limpa"],
            "gabarito_gemini": gabarito(row),
        })

    SAIDA.write_text(json.dumps({
        "_meta": {"fonte_gabarito": "agosto_grvin/gemini/CONSOLIDADO.csv (gemini-2.5-flash pago, agosto/2026)",
                  "total": len(ligacoes)},
        "ligacoes": ligacoes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for l in ligacoes:
        print(f"{l['cd_segmento'][:8]}  {len(l['transcricao_limpa']):>5} chars  {l['perfil']}")
    print(f"-> {SAIDA}")


if __name__ == "__main__":
    main()
