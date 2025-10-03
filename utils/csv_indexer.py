# utils/csv_indexer.py
import pandas as pd
import math
from typing import Iterable, Dict, Any, List
from tqdm import tqdm

def row_to_text(row: pd.Series) -> str:
    """
    Converte uma linha (Series) para texto simples.
    """
    parts = []
    for col, val in row.items():
        parts.append(f"{col}: {val}")
    return " | ".join(parts)

def stream_csv_to_docs(filepath: str, chunksize: int = 10000, rows_per_doc: int = 200) -> Iterable[Dict[str, Any]]:
    """
    Lê CSV em streaming (chunksize linhas por leitura do disco),
    agrupa rows_per_doc linhas em um documento textual e yield cada documento.

    Cada documento tem:
      - 'text': str
      - 'meta': {'file': filename, 'start_row': int, 'end_row': int}

    Esse design evita carregar tudo em memória.
    """
    start_row = 0
    chunk_iter = pd.read_csv(filepath, chunksize=chunksize, dtype=str, keep_default_na=False)
    doc_id = 0

    buffer_rows = []
    for chunk in chunk_iter:
        # chunk é DataFrame com até chunksize linhas
        for _, row in chunk.iterrows():
            buffer_rows.append(row_to_text(row))
            if len(buffer_rows) >= rows_per_doc:
                text = "\n".join(buffer_rows)
                end_row = start_row + len(buffer_rows) - 1
                yield {
                    "id": f"doc_{doc_id}",
                    "text": text,
                    "meta": {"file": filepath, "start_row": start_row, "end_row": end_row}
                }
                doc_id += 1
                start_row = end_row + 1
                buffer_rows = []
        # continue reading next chunk

    # flush remaining rows
    if buffer_rows:
        end_row = start_row + len(buffer_rows) - 1
        yield {
            "id": f"doc_{doc_id}",
            "text": "\n".join(buffer_rows),
            "meta": {"file": filepath, "start_row": start_row, "end_row": end_row}
        }
