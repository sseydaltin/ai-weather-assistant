# src/rag_helpers.py
"""
RAG sistemi için yardımcı fonksiyonlar
"""

from typing import List
from langchain_core.documents import Document


def format_docs_for_llm(docs: List[Document]) -> str:
    """
    Dökümanları LLM'e göndermek için formatla

    Args:
        docs: Document listesi

    Returns:
        Formatlanmış string
    """
    if not docs:
        return "İlgili döküman bulunamadı."

    formatted = []
    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get('similarity_score', 0)
        source = doc.metadata.get('source', 'Unknown')

        formatted.append(
            f"[Döküman {i}] (Relevance: {score:.2f})\n"
            f"Kaynak: {source}\n\n"
            f"{doc.page_content}"
        )

    return "\n\n" + "=" * 70 + "\n\n".join(formatted)


def print_search_results(docs: List[Document], query: str):
    """
    Arama sonuçlarını güzel formatta yazdır

    Args:
        docs: Document listesi
        query: Arama sorgusu
    """
    print(f"\n{'=' * 70}")
    print(f"🔍 Sorgu: {query}")
    print(f"{'=' * 70}\n")

    if not docs:
        print("❌ Sonuç bulunamadı\n")
        return

    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get('similarity_score', 0)
        chunk_id = doc.metadata.get('chunk_id', 'N/A')

        print(f"📄 Sonuç {i}")
        print(f"   Score: {score:.3f}")
        print(f"   Chunk ID: {chunk_id}")
        print(f"   {'─' * 70}")
        print(f"   {doc.page_content[:200]}...")
        print()


def calculate_rag_metrics(docs: List[Document]) -> dict:
    """
    RAG sonuçları için metrikler hesapla

    Args:
        docs: Document listesi

    Returns:
        Metrik dictionary'si
    """
    if not docs:
        return {
            "num_results": 0,
            "avg_score": 0,
            "min_score": 0,
            "max_score": 0,
            "avg_length": 0
        }

    scores = [doc.metadata.get('similarity_score', 0) for doc in docs]
    lengths = [len(doc.page_content) for doc in docs]

    return {
        "num_results": len(docs),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "avg_length": sum(lengths) / len(lengths)
    }