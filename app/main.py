from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from contextlib import asynccontextmanager
from .core import rag_handler
from .core.config import settings

# Model (pydantic)
class AskRequest(BaseModel):
    query: str


# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager to initialize the RAG pipeline.
    """
    print("Application startup...")
    try:
        rag_handler.initialize_rag_pipeline()
    except Exception as e:
        print(f"❌ Error during RAG pipeline init: {e}")
        raise RuntimeError("Failed to initialize RAG pipeline.")

    yield

    print("Application shutdown.")

# FastAPI app
app = FastAPI(
  title="Personal FAQ Bot API",
  description="A RAG-based chatbot for personal FAQs and documents.",
  version="1.0.0",
  lifespan=lifespan
)

# Endpoints
@app.get("/", tags=["Status"])
async def root():
    """
    Health check endpoint.
    """
    return {"status": "Personal FAQ Bot API is running!"}

@app.post("/ask", tags=["RAG"])
async def ask(request: AskRequest):
    """
    Endpoint to ask a question using the RAG pipeline.

    Args:
        request (AskRequest): The request body containing the query.

    Returns:
        dict: The answer to the query.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = rag_handler.answer_question(request.query)
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
