"""
Agente classificador de questões por matéria e subtópico do edital ENAM.

- Lê questoes_para_categorizacao.csv
- Classifica cada questão conforme edital.md
- Grava incrementalmente em questoes_classificadas.csv
"""

import os
import csv
import re
import json
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
EDITAL_PATH = BASE_DIR / "edital.md"
INPUT_CSV = BASE_DIR / "questoes_para_categorizacao.csv"
OUTPUT_CSV = BASE_DIR / "questoes_classificadas.csv"
ENV_PATH = BASE_DIR.parent / "agente-enam-const" / ".env"

load_dotenv(ENV_PATH)

MARITACA_MODEL = "sabiazinho-3"
MARITACA_BASE_URL = "https://chat.maritaca.ai/api"

INPUT_DELIMITER = ";"
OUTPUT_DELIMITER = ";"


# ---------------------------------------------------------------------------
# Parsing do edital
# ---------------------------------------------------------------------------

def parse_edital(path: Path) -> list[dict]:
    """
    Retorna lista de dicts:
      {"materia": "I. DIREITO CONSTITUCIONAL", "num": "1", "texto": "Teoria da Constituição..."}

    Lida com subtópicos concatenados na mesma linha, ex:
      "...texto anterior. 10. Próximo subtópico. Conteúdo..."
    """
    content = path.read_text(encoding="utf-8")
    items = []
    current_materia = None
    current_num = None
    current_texto: list[str] = []

    MATERIA_RE = re.compile(
        r'^((?:VIII|VII|VI|V|IV|III|II|IX|X|I)\.?\s*'
        r'[A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN][A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN\s]+)$'
    )
    # Detecta início de subtópico inline: " 10. Texto..."
    INLINE_NUM_RE = re.compile(r'(?<!\d)\s+(\d{1,2})\.\s+([A-ZÁÉÍÓÚN])')

    def flush():
        nonlocal current_num, current_texto
        if current_materia and current_num and current_texto:
            items.append({
                "materia": current_materia,
                "num": current_num,
                "texto": " ".join(current_texto).strip(),
            })
        current_num = None
        current_texto = []

    def process_segment(text: str):
        """Processa um segmento de texto já sem cabeçalho de matéria."""
        nonlocal current_num, current_texto
        # Tenta detectar início de subtópico no início do segmento
        num_match = re.match(r'^\s*(\d+)\.\s+(.+)', text)
        if num_match:
            flush()
            current_num = num_match.group(1)
            current_texto = [num_match.group(2).strip()]
        elif current_num and text.strip():
            current_texto.append(text.strip())

    for raw_line in content.split("\n"):
        stripped = raw_line.strip()

        # Detecta cabeçalho de matéria
        mat_match = MATERIA_RE.match(stripped)
        if mat_match:
            flush()
            current_materia = stripped.rstrip(".")
            continue

        if current_materia is None:
            continue

        # Divide a linha em segmentos separados por subtópicos inline
        # Ex: "...texto. 10. Próximo subtópico..."
        segments = INLINE_NUM_RE.split(raw_line)
        # split com grupos capturados produz: [antes, num, primeira_letra, ...]
        # grupos: [texto_antes, num1, letra1, texto1, num2, letra2, texto2, ...]

        if len(segments) == 1:
            # Nenhum subtópico inline — processa normalmente
            process_segment(raw_line)
        else:
            # Primeiro segmento: continuação do subtópico atual
            if segments[0].strip():
                process_segment(segments[0])
            # Segmentos seguintes: grupos de 3 (num, letra, resto)
            i = 1
            while i + 2 <= len(segments):
                num = segments[i]
                letra = segments[i + 1]
                resto = segments[i + 2] if i + 2 < len(segments) else ""
                fake_line = f"{num}. {letra}{resto}"
                process_segment(fake_line)
                i += 3

    flush()
    return items


# ---------------------------------------------------------------------------
# Classificação via API
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Você é um classificador jurídico especializado no edital do ENAM (Exame Nacional da Magistratura).

Sua tarefa é classificar uma questão de prova conforme o edital fornecido.

INSTRUÇÕES:
1. Leia o edital abaixo com as matérias (identificadas por algarismo romano) e seus subtópicos (identificados por número inteiro).
2. Identifique a matéria e o subtópico mais aderentes à questão.
3. Identifique um segundo subtópico se houver aderência secundária (pode ser de outra matéria).
4. Se não houver aderência alguma, use "sem aderência".

FORMATO DE RESPOSTA — responda APENAS com JSON válido, sem texto adicional:
{"materia": "<ALGARISMO_ROMANO>", "subtopico1": "<NUMERO>", "subtopico2": "<NUMERO_OU_sem_aderencia>"}

Exemplos válidos:
{"materia": "VI", "subtopico1": "7", "subtopico2": "sem aderência"}
{"materia": "I", "subtopico1": "15", "subtopico2": "6"}
{"materia": "sem aderência", "subtopico1": "sem aderência", "subtopico2": "sem aderência"}

IMPORTANTE: use APENAS os algarismos romanos e números que aparecem no edital abaixo. Não invente."""


def build_edital_summary(items: list[dict]) -> str:
    """Constrói resumo compacto do edital para enviar ao modelo."""
    lines = []
    current_materia = None
    for item in items:
        if item["materia"] != current_materia:
            current_materia = item["materia"]
            # Extrai só o algarismo romano para o modelo usar como chave
            roman = current_materia.split(".")[0].strip()
            lines.append(f"\n{roman}. {current_materia.split('.', 1)[-1].strip()}")
        texto = item["texto"][:150]
        lines.append(f"  {item['num']}. {texto}")
    return "\n".join(lines)


def _lookup(items: list[dict], roman: str, num: str) -> str:
    """Reconstrói o label completo a partir do algarismo romano e número."""
    roman = roman.strip().upper()
    num = num.strip()
    for item in items:
        item_roman = item["materia"].split(".")[0].strip().upper()
        if item_roman == roman and item["num"] == num:
            titulo = item["texto"].split(".")[0][:80]
            return f"{num}. {titulo}"
    return f"{num}. (subtópico não encontrado)"


def classify_question(
    client: OpenAI,
    edital_summary: str,
    edital_items: list[dict],
    enunciado: str,
    alternativas: str,
) -> dict:
    """Chama a API e retorna dict com Matéria, Subtópico 1, Subtópico 2."""
    user_msg = (
        f"EDITAL:\n{edital_summary}\n\n"
        f"QUESTÃO:\n{enunciado}\n\n"
        f"ALTERNATIVAS:\n{alternativas}\n\n"
        "Classifique esta questão conforme o edital acima. Responda apenas com JSON."
    )

    response = client.chat.completions.create(
        model=MARITACA_MODEL,
        max_tokens=100,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content.strip()

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "Matéria": "sem aderência",
            "Subtópico 1": "sem aderência",
            "Subtópico 2": "sem aderência",
        }

    roman = result.get("materia", "sem aderência").strip()
    sub1 = result.get("subtopico1", "sem aderência").strip()
    sub2 = result.get("subtopico2", "sem aderência").strip()

    if roman.lower() == "sem aderência":
        return {
            "Matéria": "sem aderência",
            "Subtópico 1": "sem aderência",
            "Subtópico 2": "sem aderência",
        }

    # Reconstrói matéria completa a partir do algarismo romano
    materia_label = "sem aderência"
    for item in edital_items:
        item_roman = item["materia"].split(".")[0].strip().upper()
        if item_roman == roman.upper():
            materia_label = item["materia"]
            break

    sub1_label = "sem aderência" if sub1.lower() == "sem aderência" else _lookup(edital_items, roman, sub1)
    sub2_label = "sem aderência" if sub2.lower() == "sem aderência" else _lookup(edital_items, roman, sub2)

    return {
        "Matéria": materia_label,
        "Subtópico 1": sub1_label,
        "Subtópico 2": sub2_label,
    }


# ---------------------------------------------------------------------------
# CSV incremental
# ---------------------------------------------------------------------------

def load_already_classified(output_path: Path) -> set[str]:
    """Retorna conjunto de códigos já classificados no CSV de saída."""
    if not output_path.exists():
        return set()
    classified = set()
    with open(output_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=OUTPUT_DELIMITER)
        for row in reader:
            code = row.get("Código", "").strip()
            if code:
                classified.add(code)
    return classified


def ensure_output_csv(output_path: Path, fieldnames: list[str]):
    """Cria o CSV de saída com cabeçalho se ainda não existir."""
    if not output_path.exists():
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=OUTPUT_DELIMITER)
            writer.writeheader()


def append_row(output_path: Path, fieldnames: list[str], row: dict):
    """Acrescenta uma linha no CSV de saída."""
    with open(output_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=OUTPUT_DELIMITER)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("MARITACA_API_KEY")
    if not api_key:
        print("ERRO: variável MARITACA_API_KEY não definida.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=MARITACA_BASE_URL, api_key=api_key)

    print("Parseando edital...")
    edital_items = parse_edital(EDITAL_PATH)
    if not edital_items:
        print("ERRO: nenhum item encontrado no edital.md", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(edital_items)} subtópicos encontrados no edital.")

    edital_summary = build_edital_summary(edital_items)

    print("Carregando questões...")
    with open(INPUT_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=INPUT_DELIMITER)
        input_fieldnames = reader.fieldnames or []
        questions = list(reader)
    print(f"  {len(questions)} questões carregadas.")

    output_fieldnames = list(input_fieldnames) + ["Matéria", "Subtópico 1", "Subtópico 2"]
    ensure_output_csv(OUTPUT_CSV, output_fieldnames)

    already_classified = load_already_classified(OUTPUT_CSV)
    print(f"  {len(already_classified)} questões já classificadas (serão puladas).")

    pending = [q for q in questions if q.get("Código", "").strip() not in already_classified]
    total = len(pending)
    print(f"  {total} questões a classificar.\n")

    for idx, row in enumerate(pending, 1):
        codigo = row.get("Código", "").strip()
        enunciado = row.get("enunciado", "").strip()
        alternativas = row.get("alternativas", "").strip()

        print(f"[{idx}/{total}] {codigo} ...", end=" ", flush=True)

        try:
            classification = classify_question(client, edital_summary, edital_items, enunciado, alternativas)
        except Exception as e:
            print(f"ERRO: {e}")
            classification = {
                "Matéria": "erro",
                "Subtópico 1": "erro",
                "Subtópico 2": "erro",
            }

        out_row = {**row, **classification}
        append_row(OUTPUT_CSV, output_fieldnames, out_row)

        print(f"{classification['Matéria']} | {classification['Subtópico 1']}")

    print(f"\nConcluído! CSV salvo em: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
