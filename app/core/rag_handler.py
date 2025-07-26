import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# --- IMPORT THE GOOGLE CHAT MODEL ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import traceback

# Constants
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RAG Handler using device: {DEVICE}")

qa_chain = None

def initialize_rag_pipeline():
    global qa_chain
    print("Initializing RAG pipeline with Google Gemini...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    prompt_template = """
    You are a helpful AI assistant specialized in answering questions based only on the provided context.

    Follow these rules:
    - Use only the information from the context to answer the question.
    - If the answer is not found in the context, respond with "I don't know."
    - Be concise, factual, and avoid speculation.

    Context:
    {context}

    Question:
    {question}

    Answer (based only on the context above):
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.1)
    # ----------------------------------

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("RAG pipeline with Google Gemini initialized successfully!")


def answer_question(query: str) -> dict:
    if not qa_chain:
        return {"error": "RAG pipeline not initialized."}

    print(f"Processing query: {query}")
    try:
        result = qa_chain.invoke({"query": query})
        response = {
            "answer": result.get("result"),
            "source_documents": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("source_documents", [])
            ],
        }
        return response
    except Exception as e:
        print("--- AN ERROR OCCURRED IN RAG HANDLER ---")
        print(traceback.format_exc())
        print("-----------------------------------------")
        return {"error": f"An exception occurred: {e}"}