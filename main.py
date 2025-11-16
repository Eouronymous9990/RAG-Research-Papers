from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

app = FastAPI(title="RAG PDF System")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

pdf_store = {
    "chunks": [],
    "embeddings": None,
    "index": None
}

class Question(BaseModel):
    question: str

def extract_text_from_pdf(pdf_file):

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=500):

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks  #--->[set  of chunks1 , chunk2, chunk3,...]

def create_embeddings(chunks):
    """Create embeddings for chunks"""
    embeddings = embedder.encode(chunks)
    return np.array(embeddings).astype('float32')

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_similar(query, top_k=3):

    if pdf_store["index"] is None:
        return []
    
    query_embedding = np.array(embedder.encode([query])).astype('float32')
    
    _, indices = pdf_store["index"].search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(pdf_store["chunks"]):
            results.append(pdf_store["chunks"][idx])
    
    return results

def ask_llm(question, context):

    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Based on the following information, answer the question in detail:

Information:
{context}

Question: {question}

Answer:"""
    
    data = {
        "model": "openai/gpt-3.5-turbo", 
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    #opnroutr api partt
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error connecting to model: {str(e)}"
    except KeyError as e:
        return f"Error parsing response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
#fastapi section
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be PDF")
    
    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        chunks = split_text(text)
        
        embeddings = create_embeddings(chunks)
        
        index = create_faiss_index(embeddings)
        #temporary memory store#
        pdf_store["chunks"] = chunks
        pdf_store["embeddings"] = embeddings
        pdf_store["index"] = index
        
        return JSONResponse({
            "message": "File uploaded and processed successfully",
            "num_chunks": len(chunks),
            "total_characters": len(text)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/ask/")
async def ask_question(question: Question):
    """Ask a question about the document"""
    if pdf_store["index"] is None:
        raise HTTPException(status_code=400, detail="Must upload PDF first")
    
    try:

        relevant_chunks = search_similar(question.question, top_k=3)
        
        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant information found")
        

        context = "\n\n".join(relevant_chunks)
        

        answer = ask_llm(question.question, context)
        
        return JSONResponse({
            "question": question.question,
            "answer": answer,
            "sources_used": len(relevant_chunks)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    """Home page"""
    return {
        "message": "RAG System for PDF Processing",
        "endpoints": {
            "/upload-pdf/": "POST - Upload PDF file",
            "/ask/": "POST - Ask a question",
            "/docs": "Swagger UI for testing"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
