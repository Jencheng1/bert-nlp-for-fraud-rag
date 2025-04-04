import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configure the page
st.set_page_config(
    page_title="Fraud Detection RAG System",
    page_icon="🔍",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"
MERCHANT_CATEGORIES = [
    "retail", "food", "travel", "entertainment", "utilities"
]

def fetch_metrics():
    try:
        response = requests.get(f"{API_URL}/api/metrics")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return None

def fetch_model_comparison():
    try:
        response = requests.get(f"{API_URL}/api/compare-models")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching model comparison: {e}")
        return None

def detect_fraud(text, amount, merchant_category):
    try:
        response = requests.post(
            f"{API_URL}/api/detect-fraud",
            json={
                "text": text,
                "amount": amount,
                "merchant_category": merchant_category
            }
        )
        return response.json()
    except Exception as e:
        st.error(f"Error detecting fraud: {e}")
        return None

# Main title
st.title("🔍 Fraud Detection RAG System")
st.markdown("""
This system uses BERT and GPT models to detect fraudulent transactions using RAG (Retrieval Augmented Generation) architecture.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Fraud Detection", "Model Comparison"]
)

if page == "Dashboard":
    # Dashboard metrics
    metrics = fetch_metrics()
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Transactions",
                metrics["total_transactions"],
                f"Fraud Rate: {metrics['fraud_rate']:.2%}"
            )
        
        with col2:
            st.metric(
                "Fraud Transactions",
                metrics["fraud_transactions"],
                f"Avg Amount: ${metrics['avg_fraud_amount']:.2f}"
            )
        
        with col3:
            st.metric(
                "Non-Fraud Transactions",
                metrics["total_transactions"] - metrics["fraud_transactions"],
                f"Avg Amount: ${metrics['avg_non_fraud_amount']:.2f}"
            )
        
        # Transaction distribution
        st.subheader("Transaction Distribution")
        fig = px.pie(
            values=[metrics["fraud_transactions"], metrics["total_transactions"] - metrics["fraud_transactions"]],
            names=["Fraud", "Non-Fraud"],
            title="Transaction Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Fraud Detection":
    st.subheader("Fraud Detection")
    
    # Input form
    with st.form("fraud_detection_form"):
        text = st.text_area("Transaction Description", placeholder="Enter transaction description...")
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
        merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
        
        submitted = st.form_submit_button("Detect Fraud")
        
        if submitted:
            if not text:
                st.warning("Please enter a transaction description")
            else:
                with st.spinner("Analyzing transaction..."):
                    result = detect_fraud(text, amount, merchant_category)
                    
                    if result:
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Fraud Detection Result",
                                "Fraudulent" if result["is_fraud"] else "Legitimate",
                                f"BERT Confidence: {result['bert_confidence']:.2%}"
                            )
                        
                        with col2:
                            st.metric(
                                "Processing Time",
                                f"{result['processing_time']:.2f}s",
                                f"GPT Confidence: {result['gpt_confidence']:.2%}"
                            )
                        
                        # Similar transactions
                        st.subheader("Similar Transactions")
                        similar_df = pd.DataFrame(result["similar_transactions"])
                        st.dataframe(similar_df)

elif page == "Model Comparison":
    st.subheader("Model Performance Comparison")
    
    comparison = fetch_model_comparison()
    if comparison:
        # Accuracy comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="BERT",
            x=["Accuracy"],
            y=[comparison["bert_accuracy"]],
            text=[f"{comparison['bert_accuracy']:.2%}"],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="GPT",
            x=["Accuracy"],
            y=[comparison["gpt_accuracy"]],
            text=[f"{comparison['gpt_accuracy']:.2%}"],
            textposition="auto",
        ))
        fig.update_layout(title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Processing time comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="BERT",
            x=["Processing Time (s)"],
            y=[comparison["bert_processing_time"]],
            text=[f"{comparison['bert_processing_time']:.3f}s"],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="GPT",
            x=["Processing Time (s)"],
            y=[comparison["gpt_processing_time"]],
            text=[f"{comparison['gpt_processing_time']:.3f}s"],
            textposition="auto",
        ))
        fig.update_layout(title="Model Processing Time Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Processing Time (s)"],
            "BERT": [
                f"{comparison['bert_accuracy']:.2%}",
                f"{comparison['bert_processing_time']:.3f}"
            ],
            "GPT": [
                f"{comparison['gpt_accuracy']:.2%}",
                f"{comparison['gpt_processing_time']:.3f}"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using FastAPI and Streamlit</p>
    <p>© 2024 Fraud Detection RAG System</p>
</div>
""", unsafe_allow_html=True) 