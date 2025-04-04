from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

app = FastAPI(title="Fraud Detection RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt_model = pipeline("text-classification", model="microsoft/deberta-v3-base")

# Load and index the fraud data
def load_fraud_data():
    data_path = Path("data/sample_fraud_data.json")
    if not data_path.exists():
        raise FileNotFoundError("Fraud data not found. Please run the data generation script first.")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Create embeddings for the descriptions
    descriptions = [item['description'] for item in data]
    embeddings = bert_model.encode(descriptions)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return data, index, embeddings

# Initialize data and index
try:
    fraud_data, faiss_index, embeddings = load_fraud_data()
except Exception as e:
    print(f"Error loading data: {e}")
    fraud_data, faiss_index, embeddings = None, None, None

class FraudRequest(BaseModel):
    text: str
    amount: float
    merchant_category: str

class FraudResponse(BaseModel):
    is_fraud: bool
    confidence: float
    model: str
    processing_time: float
    similar_transactions: List[Dict[str, Any]]

@app.post("/api/detect-fraud", response_model=Dict[str, Any])
async def detect_fraud(request: FraudRequest):
    if fraud_data is None:
        raise HTTPException(status_code=500, detail="Fraud detection system not initialized")
    
    # Prepare input text
    input_text = f"{request.text} - Amount: ${request.amount:.2f} - Category: {request.merchant_category}"
    
    # Get BERT results
    start_time = time.time()
    bert_embedding = bert_model.encode([input_text])[0]
    D, I = faiss_index.search(bert_embedding.reshape(1, -1).astype('float32'), k=5)
    
    # Get similar transactions
    similar_transactions = []
    for idx in I[0]:
        similar_transactions.append({
            'description': fraud_data[idx]['description'],
            'amount': fraud_data[idx]['amount'],
            'is_fraud': fraud_data[idx]['is_fraud']
        })
    
    # Get GPT results
    gpt_result = gpt_model(input_text)[0]
    
    # Calculate confidence scores
    bert_confidence = 1 - (D[0][0] / 100)  # Normalize distance to confidence
    gpt_confidence = gpt_result['score']
    
    # Combine results
    is_fraud = (bert_confidence > 0.7) or (gpt_confidence > 0.7)
    processing_time = time.time() - start_time
    
    return {
        'is_fraud': is_fraud,
        'bert_confidence': float(bert_confidence),
        'gpt_confidence': float(gpt_confidence),
        'processing_time': processing_time,
        'similar_transactions': similar_transactions
    }

@app.get("/api/metrics")
async def get_metrics():
    if fraud_data is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Calculate basic metrics
    total_transactions = len(fraud_data)
    fraud_transactions = sum(1 for item in fraud_data if item['is_fraud'])
    fraud_rate = fraud_transactions / total_transactions
    
    # Calculate average amounts
    fraud_amounts = [item['amount'] for item in fraud_data if item['is_fraud']]
    non_fraud_amounts = [item['amount'] for item in fraud_data if not item['is_fraud']]
    
    return {
        'total_transactions': total_transactions,
        'fraud_transactions': fraud_transactions,
        'fraud_rate': fraud_rate,
        'avg_fraud_amount': sum(fraud_amounts) / len(fraud_amounts) if fraud_amounts else 0,
        'avg_non_fraud_amount': sum(non_fraud_amounts) / len(non_fraud_amounts) if non_fraud_amounts else 0
    }

@app.get("/api/compare-models")
async def compare_models():
    if fraud_data is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Sample some transactions for comparison
    sample_size = min(100, len(fraud_data))
    sample_indices = np.random.choice(len(fraud_data), sample_size, replace=False)
    
    results = {
        'bert_accuracy': 0,
        'gpt_accuracy': 0,
        'bert_processing_time': 0,
        'gpt_processing_time': 0
    }
    
    for idx in sample_indices:
        transaction = fraud_data[idx]
        input_text = transaction['description']
        
        # BERT processing
        start_time = time.time()
        bert_embedding = bert_model.encode([input_text])[0]
        D, I = faiss_index.search(bert_embedding.reshape(1, -1).astype('float32'), k=1)
        bert_confidence = 1 - (D[0][0] / 100)
        bert_prediction = bert_confidence > 0.7
        results['bert_processing_time'] += time.time() - start_time
        
        # GPT processing
        start_time = time.time()
        gpt_result = gpt_model(input_text)[0]
        gpt_prediction = gpt_result['score'] > 0.7
        results['gpt_processing_time'] += time.time() - start_time
        
        # Calculate accuracy
        if bert_prediction == transaction['is_fraud']:
            results['bert_accuracy'] += 1
        if gpt_prediction == transaction['is_fraud']:
            results['gpt_accuracy'] += 1
    
    # Calculate final metrics
    results['bert_accuracy'] /= sample_size
    results['gpt_accuracy'] /= sample_size
    results['bert_processing_time'] /= sample_size
    results['gpt_processing_time'] /= sample_size
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 