"""
Classifica 10 questões aleatórias usando OpenAI (gpt-4o-mini) e Anthropic (claude-haiku-4-5).
Exibe resultado comparativo lado a lado e grava CSV comparativo.
"""

import os
import csv
import random
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR.parent / "agente-enam-const" / ".env"
load_dotenv(ENV_PATH)

INPUT_CSV = BASE_DIR / "questoes_para_categorizacao.csv"
EDITAL_PATH = BASE_DIR / "edital.md"
OUTPUT_CSV = BASE_DIR / "comparativo_apis.csv"

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Reutiliza funções do agent.py
# ---------------------------------------------------------------------------

def parse_edital(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    items = []
    current_materia = None
    current_num = None
    current_texto: list[str] = []

    MATERIA_RE = re.compile(
        r'^((?:VIII|VII|VI|V|IV|III|II|IX|X|I)\.?\s*'
        r'[A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN][A-ZÁÉÍÓÚÀÂÊÎÔÛÃÕÇN\s]+)$'
    )
    INLINE_NUM_RE = re.compile(r'(?<!\d)\s+(\d{1,2})\.\s+([A-ZÁÉÍÓÚN])')

    def flush():
        nonlocal current_num, current_texto
        if current_materia and current_num and current_texto:
            items.append({"materia": current_materia, "num": current_num,
                          "texto": " ".join(current_texto).strip()})
        current_num = None
        current_texto = []

    def process_segment(text: str):
        nonlocal current_num, current_texto
        num_match = re.match(r'^\s*(\d+)\.\s+(.+)', text)
        if num_match:
            flush()
            current_num = num_match.group(1)
            current_texto = [num_match.group(2).strip()]
        elif current_num and text.strip():
            current_texto.append(text.strip())

    for raw_line in content.split("\n"):
        stripped = raw_line.strip()
        mat_match = MATERIA_RE.match(stripped)
        if mat_match:
            flush()
            current_materia = stripped.rstrip(".")
            continue
        if current_materia is None:
            continue
        segments = INLINE_NUM_RE.split(raw_line)
        if len(segments) == 1:
            process_segment(raw_line)
        else:
            if segments[0].strip():
                process_segment(segments[0])
            i = 1
            while i + 2 <= len(segments):
                num = segments[i]
                letra = segments[i + 1]
                resto = segments[i + 2] if i + 2 < len(segments) else ""
                process_segment(f"{num}. {letra}{resto}")
                i += 3

    flush()
    return items


def build_edital_summary(items: list[dict]) -> str:
    lines = []
    current_materia = None
    for item in items:
        if item["materia"] != current_materia:
            current_materia = item["materia"]
            roman = current_materia.split(".")[0].strip()
            lines.append(f"\n{roman}. {current_materia.split('.', 1)[-1].strip()}")
        lines.append(f"  {item['num']}. {item['texto'][:150]}")
    return "\n".join(lines)


def _lookup(items: list[dict], roman: str, num: str) -> str:
    roman = roman.strip().upper()
    num = num.strip()
    for item in items:
        item_roman = item["materia"].split(".")[0].strip().upper()
        if item_roman == roman and item["num"] == num:
            titulo = item["texto"].split(".")[0][:80]
            return f"{num}. {titulo}"
    return f"{num}. (não encontrado)"


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


def parse_response(raw: str, edital_items: list[dict]) -> dict:
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"Matéria": "erro", "Subtópico 1": "erro", "Subtópico 2": "erro"}

    roman = result.get("materia", "sem aderência").strip()
    sub1 = result.get("subtopico1", "sem aderência").strip()
    sub2 = result.get("subtopico2", "sem aderência").strip()

    if roman.lower() == "sem aderência":
        return {"Matéria": "sem aderência", "Subtópico 1": "sem aderência", "Subtópico 2": "sem aderência"}

    materia_label = "sem aderência"
    for item in edital_items:
        if item["materia"].split(".")[0].strip().upper() == roman.upper():
            materia_label = item["materia"]
            break

    sub1_label = "sem aderência" if sub1.lower() == "sem aderência" else _lookup(edital_items, roman, sub1)
    sub2_label = "sem aderência" if sub2.lower() == "sem aderência" else _lookup(edital_items, roman, sub2)

    return {"Matéria": materia_label, "Subtópico 1": sub1_label, "Subtópico 2": sub2_label}


def classify_openai(client: OpenAI, edital_summary: str, edital_items: list[dict],
                    enunciado: str, alternativas: str) -> dict:
    user_msg = (f"EDITAL:\n{edital_summary}\n\nQUESTÃO:\n{enunciado}\n\n"
                f"ALTERNATIVAS:\n{alternativas}\n\nClassifique esta questão conforme o edital. Responda apenas com JSON.")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=100,
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
    )
    return parse_response(response.choices[0].message.content.strip(), edital_items)


def classify_anthropic(client: anthropic.Anthropic, edital_summary: str, edital_items: list[dict],
                       enunciado: str, alternativas: str) -> dict:
    user_msg = (f"EDITAL:\n{edital_summary}\n\nQUESTÃO:\n{enunciado}\n\n"
                f"ALTERNATIVAS:\n{alternativas}\n\nClassifique esta questão conforme o edital. Responda apenas com JSON.")
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return parse_response(response.content[0].text.strip(), edital_items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    openai_key = os.environ.get("OPENAI_API_KEY")
    claude_key = os.environ.get("CLAUDE_API_KEY")

    if not openai_key:
        print("ERRO: OPENAI_API_KEY não definida.")
        return
    if not claude_key:
        print("ERRO: CLAUDE_API_KEY não definida.")
        return

    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = anthropic.Anthropic(api_key=claude_key)

    print("Parseando edital...")
    edital_items = parse_edital(EDITAL_PATH)
    edital_summary = build_edital_summary(edital_items)
    print(f"  {len(edital_items)} subtópicos carregados.\n")

    print("Carregando questões...")
    with open(INPUT_CSV, encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f, delimiter=";"))

    sample = random.sample(questions, 10)
    print(f"  10 questões aleatórias selecionadas.\n")
    print("=" * 100)

    csv_rows = []
    fieldnames = [
        "Código", "enunciado", "alternativas",
        f"{OPENAI_MODEL}_Matéria", f"{OPENAI_MODEL}_Subtópico1", f"{OPENAI_MODEL}_Subtópico2",
        f"{ANTHROPIC_MODEL}_Matéria", f"{ANTHROPIC_MODEL}_Subtópico1", f"{ANTHROPIC_MODEL}_Subtópico2",
        "concordam_materia",
    ]

    for i, row in enumerate(sample, 1):
        codigo = row.get("Código", "").strip()
        enunciado = row.get("enunciado", "").strip()
        alternativas = row.get("alternativas", "").strip()

        enunciado_curto = enunciado[:120] + "..." if len(enunciado) > 120 else enunciado

        print(f"\n[{i}/10] {codigo}")
        print(f"  Enunciado: {enunciado_curto}")

        try:
            r_openai = classify_openai(openai_client, edital_summary, edital_items, enunciado, alternativas)
        except Exception as e:
            r_openai = {"Matéria": f"ERRO: {e}", "Subtópico 1": "", "Subtópico 2": ""}

        try:
            r_anthropic = classify_anthropic(anthropic_client, edital_summary, edital_items, enunciado, alternativas)
        except Exception as e:
            r_anthropic = {"Matéria": f"ERRO: {e}", "Subtópico 1": "", "Subtópico 2": ""}

        concordam = "SIM" if r_openai["Matéria"] == r_anthropic["Matéria"] else "NÃO"

        print(f"\n  {'Modelo':<20} {'Matéria':<40} {'Subtópico 1':<45} {'Subtópico 2'}")
        print(f"  {'-'*18:<20} {'-'*38:<40} {'-'*43:<45} {'-'*30}")
        print(f"  {OPENAI_MODEL:<20} {r_openai['Matéria']:<40} {r_openai['Subtópico 1']:<45} {r_openai['Subtópico 2']}")
        print(f"  {ANTHROPIC_MODEL:<20} {r_anthropic['Matéria']:<40} {r_anthropic['Subtópico 1']:<45} {r_anthropic['Subtópico 2']}")
        print(f"  Concordam na matéria: {concordam}")
        print("-" * 100)

        csv_rows.append({
            "Código": codigo,
            "enunciado": enunciado,
            "alternativas": alternativas,
            f"{OPENAI_MODEL}_Matéria": r_openai["Matéria"],
            f"{OPENAI_MODEL}_Subtópico1": r_openai["Subtópico 1"],
            f"{OPENAI_MODEL}_Subtópico2": r_openai["Subtópico 2"],
            f"{ANTHROPIC_MODEL}_Matéria": r_anthropic["Matéria"],
            f"{ANTHROPIC_MODEL}_Subtópico1": r_anthropic["Subtópico 1"],
            f"{ANTHROPIC_MODEL}_Subtópico2": r_anthropic["Subtópico 2"],
            "concordam_materia": concordam,
        })

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)

    concordancias = sum(1 for r in csv_rows if r["concordam_materia"] == "SIM")
    print(f"\nCSV salvo em: {OUTPUT_CSV}")
    print(f"Concordância na matéria: {concordancias}/10 questões")


if __name__ == "__main__":
    main()
