"""
Pipeline de classificação de questões jurídicas por tópico do edital ENAM.

Arquitetura:
  Fase 1 — Parsing direto do edital.md (sem LLM) → edital_topicos.json
  Fase 2 — Indexação dos tópicos com BGE-M3 → ChromaDB (./chroma_edital)
  Fase 3 — Classificação das questões (embedding primeiro, Groq como fallback)

Uso:
  python pipeline.py              # roda as 3 fases
  python pipeline.py --fase 1     # só extração do edital
  python pipeline.py --fase 2     # só indexação (requer edital_topicos.json)
  python pipeline.py --fase 3     # só classificação (requer chroma_edital)
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuração de caminhos
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR.parent / "agente-enam-const" / ".env"
load_dotenv(ENV_PATH)

EDITAL_PATH      = BASE_DIR / "edital.md"
INPUT_CSV        = BASE_DIR / "questoes_para_categorizacao.csv"
OUTPUT_CSV       = BASE_DIR / "questoes_classificadas_pipeline.csv"
TOPICOS_JSON     = BASE_DIR / "edital_topicos.json"
CHROMA_DIR       = BASE_DIR / "chroma_edital"
PROGRESSO_JSON   = BASE_DIR / "progresso.json"

CSV_DELIMITER    = ";"
SCORE_THRESHOLD  = 0.72
GROQ_MODEL       = "llama-3.3-70b-versatile"
BGE_PREFIX_TOPIC = "Represente este tópico jurídico de concurso público: "
BGE_PREFIX_QUEST = "Questão de concurso público sobre direito: "
SAVE_EVERY       = 10          # salvar CSV parcial a cada N questões
GROQ_SLEEP       = 2           # segundos entre chamadas Groq
RATE_LIMIT_WAIT  = 60          # segundos ao receber 429
MAX_RETRIES      = 3


# ---------------------------------------------------------------------------
# Fase 1 — Parsing direto do edital.md (sem LLM)
# ---------------------------------------------------------------------------

# Mapa de prefixo romano → sigla para gerar topico_id
_ROMAN_TO_SLUG = {
    "I":    "dir_const",
    "II":   "dir_adm",
    "III":  "noc_ger",
    "IV":   "dir_hum",
    "V":    "dir_proc_civ",
    "VI":   "dir_civ",
    "VII":  "dir_emp",
    "VIII": "dir_pen",
    "IX":   "dir_proc_pen",
    "X":    "dir_trab",
}


def _parse_edital_md(path: Path) -> list[dict]:
    """
    Lê edital.md e retorna lista de dicts com disciplina, topico (texto
    completo e fiel), topico_id — sem nenhuma chamada a LLM.
    """
    import re

    content = path.read_text(encoding="utf-8")
    items: list[dict] = []

    current_disciplina: str | None = None
    current_slug: str | None = None
    current_num: str | None = None
    current_parts: list[str] = []

    # Cabeçalho de disciplina: romano + ponto/espaço opcionais + nome em maiúsculas
    # Aceita tanto "I. DIREITO" quanto "II.DIREITO" (sem espaço)
    DISC_RE = re.compile(
        r'^(VIII|VII|VI|V|IV|III|II|IX|X|I)\.?\s*'
        r'([A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN][A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN\s]+)$'
    )
    # Subtópico inline dentro de uma linha: " 10. Palavra..."
    INLINE_RE = re.compile(r'(?<!\d)\s+(\d{1,2})\.\s+([A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN\w])')

    def flush():
        nonlocal current_num, current_parts
        if current_disciplina and current_num and current_parts:
            texto = " ".join(current_parts).strip()
            # remove espaços múltiplos e hifens de quebra de linha
            texto = re.sub(r'\s+', ' ', texto)
            slug = current_slug or "topico"
            topico_id = f"{slug}_{int(current_num):03d}"
            items.append({
                "disciplina": current_disciplina,
                "topico": texto,
                "topico_id": topico_id,
            })
        current_num = None
        current_parts = []

    def process_segment(seg: str):
        nonlocal current_num, current_parts
        m = re.match(r'^\s*(\d{1,2})\.\s+(.+)', seg, re.DOTALL)
        if m:
            flush()
            current_num = m.group(1)
            current_parts = [m.group(2).strip()]
        elif current_num and seg.strip():
            current_parts.append(seg.strip())

    for raw_line in content.split("\n"):
        stripped = raw_line.strip()

        # Detecta cabeçalho de disciplina
        disc_match = DISC_RE.match(stripped)
        if disc_match:
            flush()
            roman = disc_match.group(1).strip()
            nome  = disc_match.group(2).strip().rstrip(".")
            current_disciplina = nome
            current_slug = _ROMAN_TO_SLUG.get(roman, roman.lower())
            continue

        if current_disciplina is None:
            continue

        # Divide linha em segmentos por subtópicos inline
        segments = INLINE_RE.split(raw_line)
        if len(segments) == 1:
            process_segment(raw_line)
        else:
            if segments[0].strip():
                process_segment(segments[0])
            i = 1
            while i + 2 <= len(segments):
                num   = segments[i]
                letra = segments[i + 1]
                resto = segments[i + 2] if i + 2 < len(segments) else ""
                process_segment(f"{num}. {letra}{resto}")
                i += 3

    flush()
    return items


def fase1_extrair_topicos():
    if TOPICOS_JSON.exists():
        print(f"[Fase 1] {TOPICOS_JSON.name} já existe — pulando extração.")
        return

    print("[Fase 1] Parseando edital.md diretamente (sem LLM) ...")
    topicos = _parse_edital_md(EDITAL_PATH)

    if not topicos:
        print("ERRO: nenhum tópico encontrado em edital.md", file=sys.stderr)
        sys.exit(1)

    TOPICOS_JSON.write_text(json.dumps(topicos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Fase 1] {len(topicos)} tópicos extraídos e salvos em {TOPICOS_JSON.name}")


# ---------------------------------------------------------------------------
# Fase 2 — Indexação dos tópicos no ChromaDB com BGE-M3
# ---------------------------------------------------------------------------

def fase2_indexar_topicos():
    import chromadb
    from sentence_transformers import SentenceTransformer

    if not TOPICOS_JSON.exists():
        print("ERRO: edital_topicos.json não encontrado. Execute a Fase 1 primeiro.", file=sys.stderr)
        sys.exit(1)

    topicos = json.loads(TOPICOS_JSON.read_text(encoding="utf-8"))

    # Verifica se ChromaDB já tem dados
    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client_chroma.get_or_create_collection(
        name="edital_topicos",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() >= len(topicos):
        print(f"[Fase 2] ChromaDB já indexado ({collection.count()} entradas) — pulando.")
        return

    print("[Fase 2] Carregando modelo BGE-M3 (pode demorar na primeira vez) ...")
    model = SentenceTransformer("BAAI/bge-m3")

    textos   = [BGE_PREFIX_TOPIC + t["topico"] for t in topicos]
    ids      = [t["topico_id"] for t in topicos]
    metas    = [{"disciplina": t["disciplina"], "topico": t["topico"], "topico_id": t["topico_id"]} for t in topicos]
    docs     = [t["topico"] for t in topicos]

    print(f"[Fase 2] Gerando embeddings para {len(topicos)} tópicos ...")
    embeddings = model.encode(textos, show_progress_bar=True, normalize_embeddings=True).tolist()

    # Insere em lotes
    BATCH = 100
    for i in range(0, len(topicos), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
            documents=docs[i:i+BATCH],
        )

    print(f"[Fase 2] {len(topicos)} tópicos indexados em {CHROMA_DIR}/")


# ---------------------------------------------------------------------------
# Fase 3 — Classificação das questões
# ---------------------------------------------------------------------------

def _groq_classificar(client_groq, enunciado: str, alternativas: str, candidatos: list[dict]) -> dict:
    """Desempate via Groq quando score < threshold."""
    candidatos_txt = "\n".join(
        f"{i+1}. [{c['disciplina']}] {c['topico']} (score={c['score']:.2f})"
        for i, c in enumerate(candidatos)
    )

    prompt = (
        "Você é um classificador de questões jurídicas de concurso público brasileiro.\n\n"
        f"Questão:\n{enunciado}\n{alternativas}\n\n"
        f"Tópicos candidatos do edital:\n{candidatos_txt}\n\n"
        "Escolha o tópico mais adequado.\n"
        'Responda APENAS em JSON válido:\n'
        '{"disciplina": "...", "topico": "...", "topico_id": "...", '
        '"justificativa": "...", "confianca": "alta|media|baixa"}'
    )

    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            resp = client_groq.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            if "429" in str(e) and tentativa < MAX_RETRIES:
                print(f"  Rate limit (429). Aguardando {RATE_LIMIT_WAIT}s ...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                raise
    raise RuntimeError("Groq falhou após todas as tentativas.")


def fase3_classificar():
    import chromadb
    from sentence_transformers import SentenceTransformer
    from groq import Groq

    # Validações iniciais
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERRO: GROQ_API_KEY não definida.", file=sys.stderr)
        sys.exit(1)
    if not CHROMA_DIR.exists():
        print("ERRO: chroma_edital não encontrado. Execute a Fase 2 primeiro.", file=sys.stderr)
        sys.exit(1)

    # Carrega progresso anterior
    progresso: set[int] = set()
    if PROGRESSO_JSON.exists():
        progresso = set(json.loads(PROGRESSO_JSON.read_text(encoding="utf-8")))
    print(f"[Fase 3] {len(progresso)} questões já processadas anteriormente.")

    # Lê CSV de entrada
    print(f"[Fase 3] Lendo {INPUT_CSV.name} ...")
    df = pd.read_csv(INPUT_CSV, sep=CSV_DELIMITER, encoding="utf-8-sig", dtype=str)
    df = df.fillna("")
    total = len(df)
    print(f"[Fase 3] {total} questões no CSV.")

    # Prepara CSV de saída
    novas_colunas = ["disciplina", "topico", "topico_id",
                     "score_similaridade", "metodo_classificacao", "confianca"]
    colunas_saida = list(df.columns) + [c for c in novas_colunas if c not in df.columns]

    resultados: list[dict] = []

    # Carrega CSV parcial se existir
    if OUTPUT_CSV.exists():
        df_out = pd.read_csv(OUTPUT_CSV, sep=CSV_DELIMITER, encoding="utf-8-sig", dtype=str)
        resultados = df_out.to_dict("records")

    # Modelos
    print("[Fase 3] Carregando BGE-M3 ...")
    bge = SentenceTransformer("BAAI/bge-m3")

    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client_chroma.get_collection("edital_topicos")

    client_groq = Groq(api_key=api_key)

    pendentes = [i for i in range(total) if i not in progresso]
    print(f"[Fase 3] {len(pendentes)} questões a classificar.\n")

    novos_desde_ultimo_save = 0

    for idx_pos, idx in enumerate(pendentes, 1):
        row = df.iloc[idx]
        codigo    = row.get("Código", str(idx))
        enunciado = row.get("enunciado", "")
        alternativas = row.get("alternativas", "")

        texto_questao = BGE_PREFIX_QUEST + enunciado + " " + alternativas

        # Embedding
        emb = bge.encode([texto_questao], normalize_embeddings=True).tolist()

        # Busca top-5
        results = collection.query(
            query_embeddings=emb,
            n_results=5,
            include=["metadatas", "distances"],
        )

        metadatas = results["metadatas"][0]
        distances  = results["distances"][0]   # distância cosseno ∈ [0, 2]
        scores     = [round(1 - d, 4) for d in distances]

        candidatos = [
            {**m, "score": s}
            for m, s in zip(metadatas, scores)
        ]

        top = candidatos[0]
        score_top = scores[0]

        if score_top >= SCORE_THRESHOLD:
            disciplina  = top["disciplina"]
            topico      = top["topico"]
            topico_id   = top["topico_id"]
            metodo      = "embedding"
            confianca   = "alta" if score_top >= 0.85 else "media"
            print(
                f"[{idx_pos}/{len(pendentes)}] {codigo} | embedding | "
                f"{disciplina} | score={score_top:.2f}"
            )
        else:
            # Fallback Groq
            print(
                f"[{idx_pos}/{len(pendentes)}] {codigo} | groq (score={score_top:.2f} < {SCORE_THRESHOLD}) ...",
                end=" ", flush=True,
            )
            try:
                groq_result = _groq_classificar(client_groq, enunciado, alternativas, candidatos)
                disciplina  = groq_result.get("disciplina", "sem aderência")
                topico      = groq_result.get("topico", "sem aderência")
                topico_id   = groq_result.get("topico_id", "sem aderência")
                confianca   = groq_result.get("confianca", "baixa")
                metodo      = "groq"
                print(f"{disciplina}")
            except Exception as e:
                print(f"ERRO: {e}")
                disciplina = topico = topico_id = "erro"
                confianca  = "baixa"
                metodo     = "groq"

            time.sleep(GROQ_SLEEP)

        out_row = row.to_dict()
        out_row["disciplina"]           = disciplina
        out_row["topico"]               = topico
        out_row["topico_id"]            = topico_id
        out_row["score_similaridade"]   = f"{score_top:.2f}"
        out_row["metodo_classificacao"] = metodo
        out_row["confianca"]            = confianca
        resultados.append(out_row)

        progresso.add(idx)
        novos_desde_ultimo_save += 1

        # Salva incrementalmente
        if novos_desde_ultimo_save >= SAVE_EVERY:
            _salvar_parcial(resultados, colunas_saida)
            PROGRESSO_JSON.write_text(json.dumps(list(progresso)), encoding="utf-8")
            novos_desde_ultimo_save = 0

    # Salva resultado final
    _salvar_parcial(resultados, colunas_saida)
    PROGRESSO_JSON.write_text(json.dumps(list(progresso)), encoding="utf-8")
    print(f"\n[Fase 3] Concluído! {len(resultados)} questões salvas em {OUTPUT_CSV.name}")


def _salvar_parcial(resultados: list[dict], colunas: list[str]):
    df_out = pd.DataFrame(resultados)
    for col in colunas:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out = df_out[colunas]
    df_out.to_csv(OUTPUT_CSV, sep=CSV_DELIMITER, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline de classificação de questões jurídicas")
    parser.add_argument(
        "--fase", type=int, choices=[1, 2, 3],
        help="Roda apenas a fase indicada (1, 2 ou 3). Sem argumento, roda as 3 fases."
    )
    args = parser.parse_args()

    if args.fase == 1 or args.fase is None:
        fase1_extrair_topicos()
    if args.fase == 2 or args.fase is None:
        fase2_indexar_topicos()
    if args.fase == 3 or args.fase is None:
        fase3_classificar()


if __name__ == "__main__":
    main()
