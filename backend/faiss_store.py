import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedder = OpenAIEmbeddings()

def embed_chunks(chunk_dict):
    chunk_titles = []
    chunk_vectors = []
    for title, chunk_list in chunk_dict.items():
        for i, chunk in enumerate(chunk_list):
            key = f"{title} - Part {i+1}" if len(chunk_list) > 1 else title
            chunk_titles.append(key)
            vec = embedder.embed_query(chunk)
            chunk_vectors.append(vec)
    return chunk_titles, np.array(chunk_vectors)

def create_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search_faiss_index(index, query, chunk_titles, chunk_dict):
    query_vec = embedder.embed_query(query)
    D, I = index.search(np.array([query_vec]), k=4)  # top 4 results
    results = []
    all_chunks = []
    for title, chunk_list in chunk_dict.items():
        for i, chunk in enumerate(chunk_list):
            key = f"{title} - Part {i+1}" if len(chunk_list) > 1 else title
            all_chunks.append((key, chunk))
    for idx in I[0]:
        results.append(all_chunks[idx])
    return results
