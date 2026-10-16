"""Simple deterministic retrievers for the retrieval example.

Documents are hardcoded for reproducibility. Relevance is graded 0-3.
"""

from __future__ import annotations

# Tiny corpus: id -> text
CORPUS = {
    "doc1": "introduction to machine learning and neural networks",
    "doc2": "machine learning basics tutorial for beginners",
    "doc3": "python for data science and pandas",
    "doc4": "data analysis with python",
    "doc5": "supervised learning overview",
    "doc6": "deep learning neural networks tutorial",
    "doc7": "javascript web development fundamentals",
    "doc8": "frontend javascript and html",
    "doc9": "sql database queries tutorial",
    "doc10": "relational databases explained",
    "doc11": "aws cloud computing services",
    "doc12": "introduction to cloud infrastructure",
    "doc13": "android mobile app development",
    "doc14": "encryption and security best practices",
    "doc15": "cryptography basics",
    "doc16": "devops ci cd pipelines",
    "doc17": "continuous integration tools",
    "doc18": "natural language processing techniques",
    "doc19": "computer vision with opencv",
    "doc20": "reinforcement learning algorithms",
    "doc21": "react frontend framework",
    "doc22": "nodejs backend development",
    "doc23": "data visualization with matplotlib",
    "doc24": "kubernetes container orchestration",
    "doc25": "git version control tutorial",
    "doc26": "rest api design principles",
    "doc27": "pytest testing framework",
    "doc28": "contributing to open source projects",
}


def _score(query: str, doc: str) -> int:
    q_terms = set(query.lower().split())
    d_terms = set(doc.lower().split())
    return len(q_terms & d_terms)


def good_retriever(query: str) -> dict:
    """Rank all documents by term overlap with the query."""
    ranked = sorted(CORPUS.items(), key=lambda item: -_score(query, item[1]))
    retrieved = [doc_id for doc_id, _ in ranked[:10]]
    return {"output": f"Retrieved {len(retrieved)} docs", "retrieved": retrieved}


def noisy_retriever(query: str) -> dict:
    """good_retriever with a regression: on even-length queries the top hit drops to rank 5."""
    ranked = sorted(CORPUS.items(), key=lambda item: -_score(query, item[1]))
    retrieved = [doc_id for doc_id, _ in ranked[:10]]
    # Deterministic "noise": if query length even, move first doc to position 5
    if len(query) % 2 == 0 and len(retrieved) > 5:
        top = retrieved.pop(0)
        retrieved.insert(4, top)
    return {"output": f"Retrieved {len(retrieved)} docs (noisy)", "retrieved": retrieved}
