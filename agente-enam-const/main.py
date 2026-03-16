"""
Main runner – processa as questões do CSV, gera justificativas via RAG+Claude
e persiste em banco de dados + CSV.

Salva o CSV a cada 10 questões processadas.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from ingest import run_ingestion
from agent import EnamAgent
from database import init_db, upsert_questao, get_processed_enunciados

BASE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, "questoes extraidas - CONSTITUCIONAL - maritaca.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "questoes extraidas - CONSTITUCIONAL - maritaca - justificadas.csv")

SAVE_EVERY = 10
START_FROM = 1    # processa a partir da questão 1 (1-indexado)
MAX_QUESTIONS = 0  # 0 = sem limite (processa todas)


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def save_csv(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, sep=";", index=False, encoding="utf-8")
    print(f"  [CSV] Salvo em: {os.path.basename(OUTPUT_CSV)}")


def main():
    print("=" * 60)
    print(" AGENTE ENAM – JUSTIFICADOR DE QUESTÕES CONSTITUCIONAIS")
    print("=" * 60)

    # 1. Garante que os documentos estão indexados
    print("\n[PASSO 1] Verificando/indexando documentos...\n")
    run_ingestion()

    # 2. Inicializa banco de dados
    print("\n[PASSO 2] Inicializando banco de dados...")
    init_db()

    # 3. Carrega CSV de entrada
    print("\n[PASSO 3] Carregando CSV de questões...")
    df = load_csv()
    print(f"  Total de questões: {len(df)}")

    # 4. Adiciona coluna de justificativa se não existir
    if "justificativa" not in df.columns:
        df["justificativa"] = ""
    if "fontes_usadas" not in df.columns:
        df["fontes_usadas"] = ""

    # 5. Descobre questões já processadas
    already_done = get_processed_enunciados()
    pending = [
        i for i in df[~df["enunciado"].isin(already_done)].index.tolist()
        if i >= START_FROM - 1
    ]
    if MAX_QUESTIONS > 0:
        pending = pending[:MAX_QUESTIONS]
    print(f"  Já processadas: {len(already_done)}")
    print(f"  Pendentes:      {len(pending)}")

    if not pending:
        print("\nTodas as questões já foram processadas!")
        # Garante que o CSV de saída existe com os dados do DB
        _fill_from_db(df)
        save_csv(df)
        return

    # 6. Inicializa agente
    print("\n[PASSO 4] Inicializando agente RAG...")
    agent = EnamAgent()

    # 7. Preenche justificativas já existentes no DB
    _fill_from_db(df)

    # 8. Processa questões pendentes
    print(f"\n[PASSO 5] Processando {len(pending)} questões...\n")
    counter = 0

    for idx in pending:
        row = df.loc[idx].to_dict()
        num = idx + 1
        print(f"  [{num}/{len(df)}] {str(row.get('enunciado', ''))[:80]}...")

        try:
            justificativa, fontes = agent.justify(row)
        except Exception as e:
            print(f"    ERRO: {e}")
            justificativa = f"ERRO ao processar: {e}"
            fontes = "erro"

        df.at[idx, "justificativa"] = justificativa
        df.at[idx, "fontes_usadas"] = fontes

        upsert_questao(row, justificativa, fontes)

        counter += 1
        if counter % SAVE_EVERY == 0:
            print(f"\n  [{counter} questões processadas] Salvando CSV...\n")
            save_csv(df)

    # 9. Salva CSV final
    print("\n[PASSO 6] Salvando CSV final...")
    save_csv(df)
    print(f"\n✓ Concluído! {counter} questões novas justificadas.")
    print(f"  Output: {OUTPUT_CSV}")


def _fill_from_db(df: pd.DataFrame):
    """Preenche o DataFrame com justificativas já salvas no banco."""
    import sqlite3
    from database import DB_PATH

    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT enunciado, justificativa, fontes_usadas FROM questoes_justificadas"
    ).fetchall()
    conn.close()

    db_map = {r[0]: (r[1], r[2]) for r in rows}

    for idx, row in df.iterrows():
        enunciado = row.get("enunciado")
        if enunciado in db_map:
            just, fontes = db_map[enunciado]
            if just:
                df.at[idx, "justificativa"] = just
                df.at[idx, "fontes_usadas"] = fontes or ""


if __name__ == "__main__":
    main()
