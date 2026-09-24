"""Diagnostico de um CSV gerado pelo classificar_ligacoes_diario.py (modo foco/C12).

Uso: python diagnosticar_classificacao.py            (abre janela para escolher o classificacao_<dia>.csv)
     python diagnosticar_classificacao.py <arquivo.csv>

Mostra so contagens (nenhum conteudo de ligacao), para colar/printar e analisar:
  - motivos da coluna "revisar"
  - desfecho, era_venda x desfecho, tipo de ligacao — e valores FORA da lista esperada
  - problemas, Challenger (qualidade e regua)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

DESFECHOS = {"fechou_novo", "fechou_renovacao", "fechou_upsell", "interessou_nao_fechou", "nao_era_venda"}
TIPOS = {"nova_venda", "renovacao", "upsell", "retencao", "suporte_operacional", "pos_venda_sinistro",
         "pos_venda_manutencao", "relacionamento_sem_demanda", "cobranca", "onboarding", "duvida_contrato", "misto",
         "sem_conteudo"}


def escolher():
    import tkinter as tk
    from tkinter import filedialog
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(parent=raiz, title="Escolha o classificacao_<dia>.csv",
                                         filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")])
    raiz.destroy()
    return caminho


def tabela(titulo, contagem, esperado=None, limite=25):
    print(f"\n== {titulo}")
    for valor, n in contagem.most_common(limite):
        marca = "   <-- FORA DA LISTA" if esperado is not None and valor not in esperado else ""
        print(f"  {n:>5}  {valor if valor != '' else '(vazio)'}{marca}")


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else escolher()
    if not caminho:
        sys.exit("Nenhum arquivo escolhido.")
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    if not linhas or "era_venda" not in linhas[0]:
        sys.exit("Esse CSV nao e do modo foco (C12) — falta a coluna era_venda.")

    gpt = [r for r in linhas if r["fonte_classificacao"] in ("gpt", "simulado")]
    print(f"Arquivo: {Path(caminho).name}")
    print(f"Ligacoes: {len(linhas)} | pelo GPT: {len(gpt)} | curtas: "
          f"{sum(r['fonte_classificacao'] == 'auto_curta' for r in linhas)} | erro: "
          f"{sum(r['fonte_classificacao'] == 'erro' for r in linhas)}")

    motivos = Counter()
    for r in gpt:
        for m in filter(None, (x.strip() for x in r["revisar"].split(";"))):
            motivos[m] += 1
    print(f"\nMarcadas para revisar: {sum(bool(r['revisar'].strip()) for r in gpt)} de {len(gpt)}")
    tabela("Motivos de revisar", motivos)
    tabela("Desfecho (pelo GPT)", Counter(r["p1_desfecho"] for r in gpt), DESFECHOS)
    tabela("era_venda x desfecho", Counter(f"{r['era_venda']} | {r['p1_desfecho']}" for r in gpt))
    tabela("Tipo de ligacao (pelo GPT)", Counter(r["p2_tipo"] for r in gpt), TIPOS)
    tabela("Tipo de ligacao x era_venda", Counter(f"{r['p2_tipo']} | venda={r['era_venda']}" for r in gpt))

    com_prob = [r for r in gpt if r["tem_problema"] == "SIM"]
    print(f"\n== Problemas: {len(com_prob)} de {len(gpt)} com problema "
          f"({sum('+' in r['p3_problemas_identificados'] for r in com_prob)} com mais de um)")
    tabela("Codigos de problema (todos)", Counter(c for r in com_prob for c in r["p3_problemas_identificados"].split("+")))
    tabela("Problema x tipo de ligacao", Counter(f"{r['p2_tipo']} | problema={r['tem_problema']}" for r in gpt))

    ch = [r for r in gpt if r["p5_teve_challenger"] == "SIM"]
    print(f"\n== Challenger: {len(ch)} de {len(gpt)}")
    tabela("Qualidade", Counter(r["p5_qualidade"] for r in ch))
    tabela("Regua (dado | conectou | reagiu)",
           Counter(f"{r['p5_trouxe_dado']} | {r['p5_conectou_situacao']} | {r['p5_cliente_reagiu']}" for r in ch))
    tabela("Codigos CH", Counter(c for r in ch for c in r["p5_codigos_challenger"].split("+")))
    tabela("Challenger x era_venda", Counter(f"venda={r['era_venda']}" for r in ch))
    if "p5_challenger_sem_dado" in linhas[0]:
        print(f"\nChallenger descartado por falta de dado concreto: {sum(r['p5_challenger_sem_dado'] == 'SIM' for r in gpt)}")


if __name__ == "__main__":
    main()
