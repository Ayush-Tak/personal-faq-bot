import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Config
DATA_PATH = "data"
VECTOR_STORE_PATH = "vector_store/faiss_index"

# Embeddings model selector
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_store():
  """
  Loads documents, splits them into chunks, and generates embeddings,
  and saves them to FAISS vector store.
  """
  print("Loading documents and creating vector store...")

  # Check if gpu_isavailable
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")

  # Load documents
  # DirectoryLoader can handle multiple file types
  loader = DirectoryLoader(
    DATA_PATH,
    glob="*.md",
    loader_cls=UnstructuredMarkdownLoader,
    show_progress=True,
    use_multithreading=True
  )
  documents = loader.load()
  if not documents:
    print("No documents found in the specified directory.\n exiting...")
    return

  print(f"Loaded {len(documents)} documents.")

  # Chunking
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
  ) #Define splitter
  texts = text_splitter.split_documents(documents)
  print(f"Split into {len(texts)} chunks.")

  # Create embeddings
  embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
  )

  # Create FAISS vector store from thus made chunks
  print("Creating FAISS vector store...(May take a while)")
  db = FAISS.from_documents(
    texts,
    embeddings
  )

  # Save the vector store to disk
  db.save_local(VECTOR_STORE_PATH)
  print(f"Vector store saved to {VECTOR_STORE_PATH}")

if __name__ == "__main__":
  if not os.path.exists(DATA_PATH):
    print(f"Data path '{DATA_PATH}' does not exist. Please check the path and try again.")
  else:
    create_vector_store()