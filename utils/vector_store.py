# utils/vector_store.py
import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Tuple

# embedders
def get_embedding_function():
    """
    Retorna uma função embedding(text) -> vector (np.array).
    Usa OpenAI embeddings se OPENAI_API_KEY estiver setado, senão usa sentence-transformers.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        import openai
        openai.api_key = openai_key

        def embed_openai(text: str) -> np.ndarray:
            # usando text-embedding-3-small ou text-embedding-3-large dependendo do uso
            # Atenção: custo de uso em OpenAI
            resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
            vec = np.array(resp["data"][0]["embedding"], dtype="float32")
            return vec
        return embed_openai
    else:
        # fallback: sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_st(text: str) -> np.ndarray:
            vec = model.encode(text, show_progress_bar=False)
            return np.array(vec, dtype="float32")
        return embed_st

class FaissVectorStore:
    def __init__(self, dim: int, index_path: str = "faiss.index", meta_path: str = "faiss_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        # if existing index on disk, try to load
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.metadatas = meta["metadatas"]
                    self.ids = meta["ids"]
                print(f"Loaded existing FAISS index ({len(self.ids)} vectors).")
            except Exception as e:
                print("Could not load existing FAISS index:", e)

    def add(self, vectors: List[np.ndarray], metadatas: List[Dict[str, Any]], ids: List[str]=None):
        """
        vectors: list of np arrays shape (dim,)
        """
        if ids is None:
            ids = [f"id_{len(self.ids) + i}" for i in range(len(vectors))]
        arr = np.stack(vectors).astype("float32")
        self.index.add(arr)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str,Any]]]:
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(np.array([query_vector]).astype("float32"), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append((float(dist), self.metadatas[idx]))
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"metadatas": self.metadatas, "ids": self.ids}, f)
