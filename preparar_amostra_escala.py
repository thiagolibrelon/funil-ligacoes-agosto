"""Monta amostra_100.json, amostra_300.json e amostra_500.json — aninhadas (a de 300 contem a de 100, a de 500
contem a de 300), para o teste escalonado: rodar 100, depois 300 e 500 reaproveita o que ja rodou.

Mesmo formato e mesmo gabarito da amostra_5.json (Gemini pago de agosto, ver preparar_amostra_5.py).
Sorteio aleatorio (seed fixa) entre as ligacoes de agosto classificadas pelo Gemini (nenhuma auto_curta), com
P9 preenchido e sem tema eleitoral — o mix de venda/nao venda/problema segue o do mes. Ficam de fora as 5 do
teste inicial e as ligacoes usadas como exemplo em exemplos_fewshot.json.

Roda AQUI (maquina local). Os json tem transcricao — nao vao para o git.
"""
import json
import random
from pathlib import Path

import preparar_amostra_5 as base

AQUI = Path(__file__).resolve().parent
TAMANHOS = [100, 300, 500]
SEED = 42


def excluidas():
    fora = {l["cd_segmento"] for l in json.loads((AQUI / "amostra_5.json").read_text(encoding="utf-8"))["ligacoes"]}
    prefixos = set()
    exemplos = AQUI / "exemplos_fewshot.json"
    if exemplos.exists():
        for e in json.loads(exemplos.read_text(encoding="utf-8"))["exemplos"]:
            if e["origem"].startswith("agosto "):
                prefixos.add(e["origem"].split()[1])
    return fora, prefixos


def main():
    import csv
    with open(base.CONSOLIDADO, encoding="utf-8", newline="") as f:
        cons = list(csv.DictReader(f))
    with open(base.P11_CSV, encoding="utf-8", newline="") as f:
        p11 = list(csv.DictReader(f))
    sinais = {}
    for s in p11:
        sinais.setdefault(s["cd_segmento"], []).append(
            {k: s[k] for k in ("tipo_sinal", "concorrente_mencionado", "vendedor_explorou")})
    tx = base.carregar_transcricoes()
    fora, prefixos = excluidas()

    elegiveis = [
        r for r in cons
        if r["fonte_classificacao"] == "gemini" and r["p9_tipo_abertura"]
        and r["cd_segmento"] in tx and r["cd_segmento"] not in fora and r["cd_segmento"][:8] not in prefixos
        and not base.ELEITORAL.search(r["p2_subtipo"] + tx[r["cd_segmento"]].get("transcricao_limpa", ""))
    ]
    elegiveis.sort(key=lambda r: r["cd_segmento"])
    random.Random(SEED).shuffle(elegiveis)
    print(f"{len(elegiveis)} ligacoes elegiveis")

    for n in TAMANHOS:
        ligacoes = []
        for row in elegiveis[:n]:
            orig = tx[row["cd_segmento"]]
            ligacoes.append({
                "cd_segmento": row["cd_segmento"], "perfil": "aleatoria",
                "data": orig.get("Data", ""), "direcao": orig.get("direcao", ""),
                "transcricao_limpa": orig["transcricao_limpa"],
                "gabarito_gemini": base.gabarito(row, sinais.get(row["cd_segmento"], [])),
            })
        saida = AQUI / f"amostra_{n}.json"
        saida.write_text(json.dumps({
            "_meta": {"fonte_gabarito": "agosto_grvin/gemini/CONSOLIDADO.csv + P11.csv (gemini-2.5-flash pago, agosto/2026)",
                      "total": n, "seed": SEED, "aninhada": "amostra_100 ⊂ amostra_300 ⊂ amostra_500"},
            "ligacoes": ligacoes,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        venda = sum(l["gabarito_gemini"]["P1"]["desfecho"] != "nao_era_venda" for l in ligacoes)
        prob = sum(l["gabarito_gemini"]["P3"]["problemas_identificados"] not in ("nenhum", "") for l in ligacoes)
        ch = sum(l["gabarito_gemini"]["P5"]["teve_challenger"] == "SIM" for l in ligacoes)
        print(f"{saida.name}: {n} ligacoes | venda {venda} | com problema {prob} | com Challenger {ch} | "
              f"{saida.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
