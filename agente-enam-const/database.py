import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "questoes.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questoes_justificadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enunciado TEXT NOT NULL,
            materia TEXT,
            topico_enam TEXT,
            alternativas TEXT,
            alternativa_correta TEXT,
            prova TEXT,
            ano TEXT,
            banca TEXT,
            gabarito TEXT,
            justificativa TEXT,
            fontes_usadas TEXT,
            processado_em TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_enunciado ON questoes_justificadas(enunciado)
    """)
    conn.commit()
    conn.close()


def upsert_questao(row: dict, justificativa: str, fontes: str):
    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM questoes_justificadas WHERE enunciado = ?",
        (row["enunciado"],)
    ).fetchone()

    now = datetime.now().isoformat()

    if existing:
        conn.execute("""
            UPDATE questoes_justificadas
            SET justificativa = ?, fontes_usadas = ?, processado_em = ?
            WHERE enunciado = ?
        """, (justificativa, fontes, now, row["enunciado"]))
    else:
        conn.execute("""
            INSERT INTO questoes_justificadas
            (enunciado, materia, topico_enam, alternativas, alternativa_correta,
             prova, ano, banca, gabarito, justificativa, fontes_usadas, processado_em)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("enunciado"), row.get("materia"), row.get("topico_enam"),
            row.get("alternativas"), row.get("alternativa_correta"),
            row.get("prova"), row.get("ano"), row.get("banca"),
            row.get("gabarito"), justificativa, fontes, now
        ))

    conn.commit()
    conn.close()


def get_processed_enunciados() -> set:
    conn = get_connection()
    rows = conn.execute(
        "SELECT enunciado FROM questoes_justificadas WHERE justificativa IS NOT NULL"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}
