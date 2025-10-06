import os
import openai
from typing import List, Dict, Any
import numpy as np
from .vector_store import get_embedding_function, FaissVectorStore
import pandas as pd
import io, base64
import matplotlib.pyplot as plt

# LLM answer
def call_openai_chat(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 512) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Please set it or modify call to another LLM.")
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

def build_context_from_results(results: List[Dict[str, Any]]) -> str:
    """
    Concatena os trechos recuperados pra montar o contexto no prompt.
    """
    parts = []
    for i, meta in enumerate(results):
        txt = meta.get("text", "")
        # sometimes only meta saved; include text if present
        summary = meta.get("summary")
        if summary:
            parts.append(f"CHUNK {i} (rows {meta.get('start_row')} - {meta.get('end_row')}):\n{summary}")
        else:
            parts.append(f"CHUNK {i} (rows {meta.get('start_row')} - {meta.get('end_row')}):\n{meta.get('text','')[:2000]}")
    return "\n\n".join(parts)

class CSVQASystem:
    def __init__(self, vector_store: FaissVectorStore, embed_fn=None):
        self.vs = vector_store
        self.embed_fn = embed_fn or get_embedding_function()

    def answer_question(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        qvec = self.embed_fn(question)
        results = self.vs.search(qvec, k=top_k)

        # results are tuples (dist, metadata). We need the full text for context.
        # In our implementation metadata should include 'text'
        contexts = []
        for dist, meta in results:
            contexts.append(meta)

        context_text = build_context_from_results(contexts) if contexts else "No context found."

        system_prompt = (
            "Você é um assistente que responde perguntas sobre dados tabulares. "
            "Use apenas as informações do contexto fornecido e seja conciso. "
            "Se a pergunta requer cálculos simples, faça-os e explique brevemente."
        )

        user_prompt = f"Contexto:\n{context_text}\n\nPergunta: {question}\n\nResponda de forma clara e objetiva."

        try:
            answer = call_openai_chat(system_prompt, user_prompt)
        except Exception as e:
            answer = f"Erro ao chamar LLM: {e}"

        # tentativa simples de detectar pedido de gráfico
        wants_plot = any(tok in question.lower() for tok in ["gráfico", "plot", "plotar", "histograma", "dispersão", "grafico", "plot"])
        plot_base64 = None
        plot_info = None

        if wants_plot:
            if contexts:
                meta0 = contexts[0]
                file = meta0.get("file")
                # try to read file and create a plot for the most relevant numeric column
                try:
                    df = pd.read_csv(file, dtype=str)  # read full small file; warn if too big
                    # try to coerce numeric columns
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    if numeric_cols:
                        col = numeric_cols[0]
                        plt.figure(figsize=(6,4))
                        df[col].dropna().hist(bins=30)
                        plt.title(f"Histograma: {col}")
                        plt.xlabel(col)
                        plt.ylabel("Frequência")
                        buf = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        plt.close()
                        plot_info = {"column": col, "type": "histogram"}
                except Exception as e:
                    # se falhar em carregar o arquivo (muito grande), não quebrar
                    plot_info = {"error": str(e)}

        return {
            "answer": answer,
            "contexts": contexts,
            "plot_base64": plot_base64,
            "plot_info": plot_info
        }
