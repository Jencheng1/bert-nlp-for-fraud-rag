# Fraud Detection RAG System with BERT vs GPT Comparison

This project implements a fraud detection system using RAG (Retrieval Augmented Generation) architecture, comparing the performance of BERT and GPT models. The system includes a Streamlit frontend and FastAPI backend for real-time fraud detection and model comparison.

## Current Progress

✅ Project Structure Created
✅ Synthetic Data Generation Script
✅ FastAPI Backend Implementation
✅ Streamlit Frontend Implementation
✅ Model Integration (BERT and GPT)
✅ Evaluation Metrics
✅ Interactive Dashboard
✅ Model Comparison Features

## Architecture Overview

```mermaid
graph TD
    A[Streamlit Frontend] --> B[FastAPI Backend]
    B --> C[Document Store]
    B --> D[Embedding Models]
    D --> E[BERT Model]
    D --> F[GPT Model]
    B --> G[Evaluation Metrics]
    G --> H[Performance Comparison]
```

## System Components

### High-Level Architecture

```mermaid
graph TD
    subgraph Frontend[Streamlit Frontend]
        UI[User Interface]
        Forms[Input Forms]
        Viz[Visualizations]
    end

    subgraph Backend[FastAPI Backend]
        API[API Endpoints]
        Models[ML Models]
        Store[Document Store]
    end

    subgraph Models[ML Models]
        BERT[BERT Model]
        GPT[GPT Model]
        FAISS[FAISS Index]
    end

    subgraph Data[Data Layer]
        DB[(Transaction Data)]
        Cache[(Embedding Cache)]
    end

    UI --> Forms
    Forms --> API
    API --> Models
    Models --> BERT
    Models --> GPT
    Models --> FAISS
    BERT --> Store
    GPT --> Store
    Store --> DB
    Store --> Cache
```

### Detailed Processing Flow

```mermaid
graph LR
    subgraph Input[Input Processing]
        A[Raw Transaction] --> B[Text Preprocessing]
        B --> C[Feature Extraction]
    end

    subgraph BERT_Flow[BERT Processing]
        C --> D[Embedding Generation]
        D --> E[FAISS Search]
        E --> F[Similarity Scoring]
    end

    subgraph GPT_Flow[GPT Processing]
        C --> G[Token Generation]
        G --> H[Context Analysis]
        H --> I[Confidence Scoring]
    end

    subgraph Results[Results Aggregation]
        F --> J[Score Combination]
        I --> J
        J --> K[Final Decision]
        K --> L[Performance Metrics]
    end

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style BERT_Flow fill:#bbf,stroke:#333,stroke-width:2px
    style GPT_Flow fill:#bfb,stroke:#333,stroke-width:2px
    style Results fill:#fbb,stroke:#333,stroke-width:2px
```

## Model Comparison Flow

```mermaid
graph TD
    A[Input Query] --> B[Parallel Processing]
    B --> C[BERT Processing]
    B --> D[GPT Processing]
    C --> E[Embedding Generation]
    D --> F[Token Generation]
    E --> G[Similarity Search]
    F --> H[Context Generation]
    G --> I[Results]
    H --> I
    I --> J[Metrics Calculation]
```

## Evaluation Metrics

```mermaid
graph LR
    A[Model Outputs] --> B[Accuracy]
    A --> C[Precision]
    A --> D[Recall]
    A --> E[F1 Score]
    A --> F[Response Time]
    B --> G[Comparison Dashboard]
    C --> G
    D --> G
    E --> G
    F --> G
```

## Project Structure

```
fraud-rag/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   ├── services/
│   │   └── utils/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── generate_fraud_data.py
│   └── sample_fraud_data.json
└── README.md
```

## Features

- Real-time fraud detection using RAG architecture
- Parallel processing with BERT and GPT models
- Comprehensive evaluation metrics
- Interactive Streamlit dashboard
- FastAPI backend with async support
- Document similarity search
- Performance comparison visualization
- Synthetic data generation for testing

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fraud-rag
   ```

2. Generate synthetic data:
   ```bash
   cd data
   python generate_fraud_data.py
   cd ..
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

5. Run the backend:
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

6. Run the frontend:
   ```bash
   cd frontend
   streamlit run app.py
   ```

## API Endpoints

- POST `/api/detect-fraud`: Fraud detection endpoint
- GET `/api/compare-models`: Model comparison endpoint
- GET `/api/metrics`: Performance metrics endpoint

## Model Comparison Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Response Time
- Context Relevance
- False Positive Rate
- False Negative Rate

## Frontend Features

1. Dashboard
   - Transaction distribution
   - Key metrics visualization
   - Real-time statistics

2. Fraud Detection
   - Transaction input form
   - Real-time analysis
   - Similar transactions display
   - Confidence scores

3. Model Comparison
   - Accuracy comparison
   - Processing time analysis
   - Detailed metrics table
   - Interactive visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 