"""
RAG Agent – consulta ChromaDB na ordem de prioridade definida nas instruções
e gera justificativas via Claude API.

Ordem de busca:
  1. constituicao
  2. repercussao_geral
  3. sumulas_vinculantes
  4. doutrina
  5. informativos
"""

import os
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ingest import CHROMA_DIR, EMBEDDING_MODEL

TOP_K = 4          # chunks por coleção
MAX_CONTEXT_CHARS = 12_000  # limite total de contexto enviado ao modelo

OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

COLLECTION_ORDER = [
    ("constituicao",        "Constituição Federal"),
    ("repercussao_geral",   "Repercussão Geral"),
    ("sumulas_vinculantes", "Súmulas Vinculantes"),
    ("doutrina",            "Doutrina - Direito Constitucional"),
    ("informativos",        "Informativos Temáticos"),
]

SYSTEM_PROMPT = """Você é um agente especialista em Direito Constitucional para concursos públicos (ENAM/magistratura).

Seu papel é justificar a alternativa correta de questões objetivas exclusivamente com base nos trechos dos documentos fornecidos no contexto (Constituição Federal, Repercussão Geral, Súmulas Vinculantes, Doutrina e Informativos).

REGRAS OBRIGATÓRIAS:
- Use SOMENTE as informações presentes nos trechos fornecidos. Nunca use seu próprio conhecimento ou a internet.
- Explique apenas por que a alternativa correta está certa, fundamentando-a nos trechos.
- NÃO explique por que as demais alternativas estão erradas.
- Cite a fonte (artigo da CF, número da súmula, tema de repercussão geral, etc.) sempre que disponível nos trechos.
- Seja objetivo, claro e direto. Escreva em português jurídico formal.

EXEMPLO DO FORMATO DE RESPOSTA ESPERADO:
A alternativa correta é a letra C. De acordo com art. 17, § 8º da CF: "§ 8º O montante do Fundo Especial de Financiamento de Campanha e da parcela do fundo partidário destinada a campanhas eleitorais, bem como o tempo de propaganda gratuita no rádio e na televisão a ser distribuído pelos partidos às respectivas candidatas, deverão ser de no mínimo 30% (trinta por cento), proporcional ao número de candidatas, e a distribuição deverá ser realizada conforme critérios definidos pelos respectivos órgãos de direção e pelas normas estatutárias, considerados a autonomia e o interesse partidário."

Explique somente o motivo e a justificativa fundamenta da questão correta. Não responda o motivo das demais questões estarem erradas."""


def _get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _get_ef():
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


class EnamAgent:
    def __init__(self):
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.chroma = _get_chroma_client()
        self.ef = _get_ef()
        self._collections: dict = {}

    def _get_collection(self, name: str):
        if name not in self._collections:
            try:
                self._collections[name] = self.chroma.get_collection(
                    name=name, embedding_function=self.ef
                )
            except Exception:
                self._collections[name] = None
        return self._collections[name]

    def _query_collection(self, name: str, query: str, k: int = TOP_K) -> list[dict]:
        col = self._get_collection(name)
        if col is None or col.count() == 0:
            return []
        results = col.query(query_texts=[query], n_results=min(k, col.count()))
        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"text": doc, "source": meta.get("source", name)})
        return chunks

    def _build_context(self, question: str) -> tuple[str, list[str]]:
        """Query all collections in priority order and build context string."""
        all_chunks = []
        sources_used = []
        total_chars = 0

        for col_name, label in COLLECTION_ORDER:
            chunks = self._query_collection(col_name, question)
            for chunk in chunks:
                if total_chars + len(chunk["text"]) > MAX_CONTEXT_CHARS:
                    break
                all_chunks.append(chunk)
                total_chars += len(chunk["text"])
                src = chunk["source"]
                if src not in sources_used:
                    sources_used.append(src)

        context_parts = []
        for i, chunk in enumerate(all_chunks, 1):
            context_parts.append(f"[Trecho {i} – {chunk['source']}]\n{chunk['text']}")

        return "\n\n---\n\n".join(context_parts), sources_used

    def justify(self, row: dict) -> tuple[str, str]:
        """
        Generate justification for a question row.

        Returns:
            (justificativa, fontes_usadas_str)
        """
        enunciado = row.get("enunciado", "")
        alternativas = row.get("alternativas", "")
        gabarito = row.get("gabarito", "")
        topico = row.get("topico_enam", "")

        query = f"{topico} {enunciado[:400]}"
        context, sources = self._build_context(query)

        if not context.strip():
            return (
                "Não foi possível encontrar trechos relevantes nos documentos indexados.",
                "nenhuma"
            )

        user_message = (
            f"QUESTÃO:\n{enunciado}\n\n"
            f"ALTERNATIVAS:\n{alternativas}\n\n"
            f"ALTERNATIVA CORRETA: {gabarito}\n\n"
            f"TRECHOS DOS DOCUMENTOS:\n{context}\n\n"
            "Com base exclusivamente nos trechos acima, justifique por que a alternativa "
            f"correta ({gabarito}) está correta."
        )

        response = self.client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        justificativa = response.choices[0].message.content.strip()
        fontes_str = " | ".join(sources)

        return justificativa, fontes_str
