"""Compila as transcricoes exportadas (schema bruto Iceberg) em um arquivo por dia: chamadas/<mes>/<DD>.

Feito para rodar TODO DIA: le os exports da pasta de entrada, separa as ligacoes pelo DIA da coluna
data_hora_inicio (convertida de UTC para o horario de Brasilia) e junta no arquivo do dia correspondente,
sem duplicar (chave cd_segmento). Rodar de novo com os mesmos exports nao muda nada; um export que traga
varios dias atualiza cada dia no seu arquivo; um export com ligacoes ja compiladas so as atualiza.

Entrada:  CSV(s) do export bruto (mesmo schema de agosto: 36 colunas, com cd_segmento, data_hora_inicio,
          direcao, nome_agente_1, time_agente_1, transcricao_limpa, status_transcricao...). Separador
          detectado sozinho (virgula ou ponto e virgula).
Saida:    <saida>/<mes>/<DD> — CSV ';', UTF-8, schema canonico de 10 colunas (o mesmo de chamadas/agosto/):
          cd_contato_master;cd_segmento;data_hora_inicio;direcao;nome_agente_1;time_agente_1;
          habilidades_agente_1;transcricao_limpa;Cliente;Data
          + <saida>/<mes>/_controle.json (ligacoes por dia, exports ja lidos) e _log.txt (historico das rodadas).

Filtros (iguais aos de scripts/preparar_agosto_grvin.py): time_agente_1 == LL_GRVIN (troque com --time, ou
--time todos), status_transcricao == success e transcricao com pelo menos 40 caracteres.

Uso:
  python compilar_ligacoes_diarias.py --entrada exportacoes/            # le todos os .csv da pasta
  python compilar_ligacoes_diarias.py --entrada export_2026-09-22.csv   # ou arquivos especificos
  python compilar_ligacoes_diarias.py --entrada exportacoes/ --saida chamadas --time todos

Para agendar (Windows): Agendador de Tarefas > Criar Tarefa Basica > Diariamente > Iniciar um programa:
  programa:   python
  argumentos: compilar_ligacoes_diarias.py --entrada <pasta dos exports> --saida <pasta chamadas>
  iniciar em: <pasta onde esta este script>
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

csv.field_size_limit(10_000_000)

COLS_SAIDA = [
    "cd_contato_master", "cd_segmento", "data_hora_inicio", "direcao",
    "nome_agente_1", "time_agente_1", "habilidades_agente_1",
    "transcricao_limpa", "Cliente", "Data",
]
OBRIGATORIAS = ["cd_segmento", "data_hora_inicio", "transcricao_limpa", "time_agente_1"]
MESES = ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro"]
MIN_CHARS = 40


def ler_export(caminho):
    """Le um CSV do export detectando encoding e separador. Devolve (linhas, colunas)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(caminho, encoding=enc, newline="") as f:
                amostra = f.readline()
                sep = ";" if amostra.count(";") > amostra.count(",") else ","
                f.seek(0)
                leitor = csv.DictReader(f, delimiter=sep)
                return list(leitor), leitor.fieldnames or []
        except UnicodeDecodeError:
            continue
    raise ValueError(f"nao consegui ler {caminho} (encoding)")


def dia_local(valor, fuso_h):
    """'2026-09-22 18:58:38.000000 UTC' -> date no fuso local. Sem 'UTC', assume que ja esta no horario local."""
    v = (valor or "").strip()
    if not v:
        return None
    em_utc = v.upper().endswith("UTC")
    v = v.replace("UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(v[:26], fmt)
            break
        except ValueError:
            continue
    else:
        return None
    if em_utc:
        dt = dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=fuso_h)))
    return dt.date()


def ler_dia_existente(caminho):
    if not caminho.exists():
        return {}
    with open(caminho, encoding="utf-8", newline="") as f:
        return {r["cd_segmento"]: r for r in csv.DictReader(f, delimiter=";")}


def gravar_atomico(caminho, linhas):
    tmp = caminho.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS_SAIDA, delimiter=";")
        w.writeheader()
        w.writerows(linhas)
    os.replace(tmp, caminho)


def arquivos_entrada(entradas):
    arqs = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            arqs += sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in (".csv", ".txt"))
        elif p.is_file():
            arqs.append(p)
        else:
            print(f"aviso: {p} nao existe, ignorado")
    return arqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", nargs="+", required=True, help="pasta(s) ou arquivo(s) CSV do export")
    ap.add_argument("--saida", default=str(Path(__file__).resolve().parent.parent / "chamadas"),
                    help="pasta base de saida (padrao: chamadas/ do projeto)")
    ap.add_argument("--time", default="LL_GRVIN", help='time_agente_1 a manter, ou "todos"')
    ap.add_argument("--fuso", type=int, default=-3, help="fuso em horas para definir o dia (padrao -3, Brasilia)")
    args = ap.parse_args()

    arqs = arquivos_entrada(args.entrada)
    if not arqs:
        sys.exit("Nenhum arquivo de export encontrado na entrada.")

    por_dia = defaultdict(dict)  # date -> {cd_segmento: linha}
    excluidas = defaultdict(int)
    lidas = 0
    for arq in arqs:
        linhas, colunas = ler_export(arq)
        faltando = [c for c in OBRIGATORIAS if c not in colunas]
        if faltando:
            print(f"ERRO: {arq.name} nao tem as colunas {faltando} — arquivo ignorado")
            continue
        for row in linhas:
            lidas += 1
            if args.time != "todos" and (row.get("time_agente_1") or "").strip() != args.time:
                excluidas["outro_time"] += 1
                continue
            if "status_transcricao" in colunas and (row.get("status_transcricao") or "").strip().lower() != "success":
                excluidas["transcricao_nao_success"] += 1
                continue
            transcricao = (row.get("transcricao_limpa") or "").strip()
            if len(transcricao) < MIN_CHARS:
                excluidas["transcricao_curta_ou_vazia"] += 1
                continue
            dia = dia_local(row.get("data_hora_inicio"), args.fuso)
            if dia is None:
                excluidas["data_invalida"] += 1
                continue
            cd = (row.get("cd_segmento") or "").strip()
            por_dia[dia][cd] = {
                "cd_contato_master": row.get("cd_contato_master", ""), "cd_segmento": cd,
                "data_hora_inicio": row.get("data_hora_inicio", ""), "direcao": row.get("direcao", ""),
                "nome_agente_1": row.get("nome_agente_1", ""), "time_agente_1": row.get("time_agente_1", ""),
                "habilidades_agente_1": row.get("habilidades_agente_1", ""), "transcricao_limpa": transcricao,
                "Cliente": row.get("Cliente", "") or row.get("CLIENTE", ""), "Data": dia.isoformat(),
            }

    base = Path(args.saida)
    resumo = []
    controles = {}
    for dia in sorted(por_dia):
        pasta = base / MESES[dia.month - 1]
        pasta.mkdir(parents=True, exist_ok=True)
        caminho = pasta / f"{dia.day:02d}"
        existentes = ler_dia_existente(caminho)
        novas = [cd for cd in por_dia[dia] if cd not in existentes]
        atualizadas = [cd for cd in por_dia[dia] if cd in existentes and existentes[cd] != por_dia[dia][cd]]
        if novas or atualizadas or not caminho.exists():
            existentes.update(por_dia[dia])
            gravar_atomico(caminho, sorted(existentes.values(), key=lambda r: r["data_hora_inicio"]))
        resumo.append((dia, len(existentes), len(novas), len(atualizadas), caminho))
        ctrl = controles.setdefault(pasta, json.loads((pasta / "_controle.json").read_text(encoding="utf-8"))
                                    if (pasta / "_controle.json").exists() else {"dias": {}, "exports_lidos": {}})
        ctrl["dias"][f"{dia.day:02d}"] = {"data": dia.isoformat(), "ligacoes": len(existentes),
                                          "atualizado_em": datetime.now().isoformat(timespec="seconds")}

    agora = datetime.now().isoformat(timespec="seconds")
    for pasta, ctrl in controles.items():
        for arq in arqs:
            ctrl["exports_lidos"][arq.name] = {"lido_em": agora, "tamanho_bytes": arq.stat().st_size}
        (pasta / "_controle.json").write_text(json.dumps(ctrl, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(arqs)} export(s), {lidas} linhas lidas | excluidas: {dict(excluidas) or 'nenhuma'}")
    print(f"{'dia':<12}{'no arquivo':>11}{'novas':>7}{'atualiz.':>9}  arquivo")
    for dia, total, n, a, caminho in resumo:
        print(f"{dia.isoformat():<12}{total:>11}{n:>7}{a:>9}  {caminho}")
    for pasta in controles:
        with open(pasta / "_log.txt", "a", encoding="utf-8") as log:
            dias = ", ".join(f"{d.day:02d}(+{n})" for d, _, n, _, c in resumo if c.parent == pasta)
            log.write(f"{agora} | exports: {', '.join(a.name for a in arqs)} | dias: {dias}\n")


if __name__ == "__main__":
    main()
