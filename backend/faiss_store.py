import os
from typing import Dict, List, Tuple, Any, Optional, Set
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings globally.
# This assumes OPENAI_API_KEY is set in the environment.
embedder = OpenAIEmbeddings()

def embed_chunks(chunk_dict: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    """
    Embeds text chunks using OpenAI Embeddings in batch mode for efficiency.

    Args:
        chunk_dict: A dictionary where keys are section titles and values are lists of text chunks.

    Returns:
        A tuple containing:
        - List of section titles/keys corresponding to each vector (aligned with the index).
        - Numpy array of embedding vectors (float32), ready for FAISS.
    """
    chunk_titles = []
    chunk_texts = []

    # Flatten the dictionary to lists for batch processing
    # The order here defines the ID mapping for the FAISS index (0, 1, 2, ...)
    for title, chunk_list in chunk_dict.items():
        for i, chunk in enumerate(chunk_list):
            key = f"{title} - Part {i+1}" if len(chunk_list) > 1 else title
            chunk_texts.append(chunk)

    if not chunk_texts:
        # Return empty structures if no text
        return [], np.array([], dtype=np.float32)

    # Use embed_documents for batching (significantly faster than loop)
    # This may fail if chunk_texts is too large for a single API call limit, 
    # but langchain usually handles chunking internally or implementation is manageable for typical PDF sizes.
    embeddings = embedder.embed_documents(chunk_texts)
    
    # FAISS requires float32 vectors
    return chunk_titles, np.array(embeddings, dtype=np.float32)

def create_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Creates a FAISS IndexFlatL2 from the given vectors.

    Args:
        vectors: Numpy array of vectors (float32).

    Returns:
        A FAISS index containing the vectors.
    """
    if vectors.size == 0:
        # Create a dummy index with a default dimension if vectors are empty to avoid crash,
        # though this case should be handled upstream. 
        # OpenAI embeddings are 1536 dimensions.
        return faiss.IndexFlatL2(1536)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search_faiss_index(
    index: faiss.Index, 
    query: str, 
    chunk_titles: List[str], 
    chunk_dict: Dict[str, List[str]], 
    k: int = 4
) -> List[Tuple[str, str]]:
    """
    Searches the FAISS index for the query and retrieves the corresponding text chunks.

    Args:
        index: The FAISS index to search.
        query: The search query string.
        chunk_titles: List of titles corresponding to index IDs. (kept for signature compatibility, unused in optimized logic)
        chunk_dict: Dictionary of text chunks to retrieve content from.
        k: Number of results to return.

    Returns:
        List of tuples (title_key, content_text) for the top k matches.
    """
    # Create query vector
    query_vec = embedder.embed_query(query)
    
    # Prepare query array for FAISS (1, dim), float32
    query_arr = np.array([query_vec], dtype=np.float32)
    
    # Search index
    D, I = index.search(query_arr, k)
    
    # Indices of nearest neighbors
    neighbor_indices = I[0]
    
    # We need to map these indices (0..N) back to (title, content) from chunk_dict.
    # Instead of flattening the entire dictionary every time (O(N) memory & formatted string creation), 
    # we iterate to find only the targets.
    
    target_indices_set = set(idx for idx in neighbor_indices if idx >= 0)
    found_items: Dict[int, Tuple[str, str]] = {}
    
    if not target_indices_set:
        return []

    current_idx = 0
    # Stop early if we found all we need
    # We iterate in the exact same order as embed_chunks to ensure index alignment
    for title, chunk_list in chunk_dict.items():
        if len(found_items) == len(target_indices_set):
            break
            
        for i, chunk in enumerate(chunk_list):
            if current_idx in target_indices_set:
                key = f"{title} - Part {i+1}" if len(chunk_list) > 1 else title
                found_items[current_idx] = (key, chunk)
            
            current_idx += 1
            
            if len(found_items) == len(target_indices_set):
                break
                
    # Construct results preserving the order returned by FAISS search
    results = []
    for idx in neighbor_indices:
        if idx in found_items:
            results.append(found_items[idx])
            
    return results
