import openai
import tiktoken
import pypdf

from itertools import islice
import numpy as np
import pandas as pd

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"
EMBEDDING_CACHE_PATH = "cache/embedding_cache.pkl"

def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


# def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
#     """
#     Read the document embeddings and their keys from a CSV.
    
#     fname is the path to a CSV with exactly these named columns: 
#         "title", "heading", "0", "1", ... up to the length of the embedding vectors.
#     """
    
#     df = pd.read_csv(fname, header=0)
#     max_dim = max([int(c) for c in df.columns if c != "Page" and c != "Content"])
#     return {
#            (r["Page"], r["Content"]): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
#     }


def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

def pdf_to_txt(file_path):
    pdfFileObj = open(file_path, 'rb')
    pdfReader = pypdf.PdfReader(pdfFileObj)
    number_of_pages = len(pdfReader.pages)
    arr = []
    print("Converting PDF to DF ....")
    for i in range(number_of_pages):
        page = pdfReader.pages[i]
        text = page.extract_text().replace('\n', " ")
        arr.append([i+1, text])
    df = pd.DataFrame(arr, columns = ['Page','Content'])
    return df