import os
import sys
import json
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../credentials')))
from credentials import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

from pymilvus import (
    MilvusClient,
    DataType,
    connections,
    CollectionSchema,
    FieldSchema,
    Collection
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename = 'pipeline.log', filemode = 'a')

# --- Configuration ---
MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'
MAX_TOKENS = 8192
VECTOR_DIMENSION = 768 # For a 1000 token chunk size, a 200 token overlap is considered
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200

SEARCH_DOCUMENT_PREFIX = "search_document: "

MILVUS_HOST = MILVUS_HOST
MILVUS_PORT = MILVUS_PORT
COLLECTION_NAME = COLLECTION_NAME

logging.info("Loading embedding model and tokenizer...")
try:
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception as e:
    logging.error(f"Failed to load model or tokenizer: {e}")
    raise e

# --- Text Chunking and Token Management ---
def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    (Internal helper function)
    Splits a large text into smaller chunks with a specified overlap.
    """
    logging.info(f"Splitting text into chunks of size {chunk_size} with overlap of {chunk_overlap}.")
    
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    chunks = []
    
    # Check if the entire text already fits within a single chunk
    if token_count <= chunk_size:
        return [text]

    # Use a sliding window to create chunks with overlap
    for i in range(0, token_count, chunk_size - chunk_overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

    logging.info(f"Original text split into {len(chunks)} chunks.")
    return chunks

# --- Main Vectorization & Insertion Function ---
def convert_to_vectors(key: str, filename: str, full_text: str):
    """
    Takes a file key, filename, and the full text of a document,
    chunks it, converts the chunks to vectors, and inserts them into Milvus.
    """
    # Defensive check for valid inputs
    if not key or not filename or not full_text:
        logging.warning("Skipping vectorization and insertion. Missing key, filename, or text input.")
        return

    # Chunk the raw text to ensure each piece fits the model's token limit
    chunks = _chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    milvus_payload = []

    for i, chunk in enumerate(chunks):
        # Apply the task-specific prefix for RAG
        prefixed_chunk = SEARCH_DOCUMENT_PREFIX + chunk
        
        # Generate the vector embedding
        try:
            vector = model.encode(prefixed_chunk).tolist()
        except Exception as e:
            logging.error(f"Failed to encode chunk for {filename}, chunk {i}: {e}")
            continue

        # Prepare the payload for insertion into Milvus
        payload_entry = {
            "ETag": key,
            "filename": filename,
            "chunk_id": i,
            "text": chunk, 
            "vector": vector,
        }
        milvus_payload.append(payload_entry)
        
    logging.info(f"Created a total of {len(milvus_payload)} vectors for file '{filename}'.")

    # Connect to Milvus and insert the payload
    if not milvus_payload:
        logging.info("Payload is empty. Skipping Milvus insertion.")
        return

    try:
        logging.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
        client = MilvusClient(
            uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        )
        
        logging.info(f"Inserting {len(milvus_payload)} vectors into collection '{COLLECTION_NAME}'...")
        client.insert(
            collection_name=COLLECTION_NAME,
            data=milvus_payload
        )
        
        logging.info("Data insertion successful.")
        
    except Exception as e:
        logging.error(f"An error occurred during Milvus insertion: {e}")
        raise e

# --- Example Usage (for local testing) ---
if __name__ == "__main__":
    example_key = "oci_file_key_12345"
    example_filename = "sample_document.txt"
    long_text = """
    The project began with a comprehensive planning phase, outlining key milestones and resource allocation. The team identified potential challenges, including integration with legacy systems and data migration complexities. As a result, a phased approach was adopted, prioritizing core functionalities.

    During the development cycle, continuous feedback loops were established with stakeholders. This agile methodology ensured that the project remained aligned with evolving business needs, leading to several key feature enhancements. The team leveraged OCI's scalable infrastructure to handle increasing data loads, maintaining a high level of performance and reliability.

    The final deployment was executed flawlessly, with minimal downtime. The user acceptance testing phase concluded successfully, and the system went live ahead of schedule. The post-launch review highlighted significant improvements in operational efficiency, a testament to the team's meticulous planning and execution.
    """ * 10 # Create a long document for demonstration purposes

    # --- Test case with valid data ---
    logging.info("--- Running with valid data ---")
    try:
        vectorize_and_insert(example_key, example_filename, long_text)
        print("\n--- Workflow completed successfully. ---")
    except Exception as e:
        print(f"Workflow failed: {e}")
    
    # --- Test case with missing data ---
    print("\n" + "="*50 + "\n")
    logging.info("--- Running with invalid data ---")
    try:
        # Intentionally missing the key and filename
        vectorize_and_insert(None, None, long_text)
        print("\n--- Workflow completed successfully. ---")
    except Exception as e:
        print(f"Workflow failed: {e}")