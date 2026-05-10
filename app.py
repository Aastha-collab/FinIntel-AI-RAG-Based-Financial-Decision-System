import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="FinIntel AI",
    layout="wide"
)

st.title("💳 FinIntel AI")
st.subheader("RAG-Based Financial Fraud Intelligence System")

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("creditcard_sample.csv")

# Feature engineering
df['Hour'] = df['Time'] // 3600

# -----------------------------------
# LOAD MODEL
# -----------------------------------

rf_model = joblib.load("rf_model.pkl")

# -----------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# -----------------------------------
# CREATE FRAUD KNOWLEDGE BASE
# -----------------------------------

fraud_df = df[df['Class'] == 1]

fraud_df = fraud_df[
    ['Time', 'Amount', 'V14', 'V10', 'Class']
]

documents = []

for _, row in fraud_df.iterrows():

    text = f"""
Fraud Transaction Summary

Transaction Amount: {row['Amount']}
Transaction Time: {row['Time']}
Suspicious V14 Score: {row['V14']}
Suspicious V10 Score: {row['V10']}

This transaction shows abnormal fraud-related behavior.
"""

    documents.append(text)

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

embeddings = embedding_model.encode(documents)

# -----------------------------------
# CHROMADB
# -----------------------------------

client_db = chromadb.Client()

collection = client_db.create_collection(
    name="fraud_rag"
)

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(documents))]
)

# -----------------------------------
# NVIDIA NIM API
# -----------------------------------

NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# -----------------------------------
# USER INPUT
# -----------------------------------

st.sidebar.header("Transaction Selection")

row_number = st.sidebar.number_input(
    "Enter Transaction Row Number",
    min_value=0,
    max_value=len(df)-1,
    value=0
)

# -----------------------------------
# PROCESS TRANSACTION
# -----------------------------------

sample_transaction = df.drop(
    'Class',
    axis=1
).iloc[[row_number]]

prediction = rf_model.predict(
    sample_transaction
)

# -----------------------------------
# LABEL
# -----------------------------------

if prediction[0] == 1:
    prediction_label = "Fraudulent Transaction"

else:
    prediction_label = "Normal Transaction"

# -----------------------------------
# SHOW PREDICTION
# -----------------------------------

st.subheader("Prediction Result")

st.success(prediction_label)

# -----------------------------------
# TRANSACTION TEXT
# -----------------------------------

transaction_text = f"""
Transaction Summary

Transaction Amount: {sample_transaction['Amount'].values[0]}
Transaction Time: {sample_transaction['Time'].values[0]}
Suspicious V14 Score: {sample_transaction['V14'].values[0]}
Suspicious V10 Score: {sample_transaction['V10'].values[0]}
"""

# -----------------------------------
# QUERY EMBEDDING
# -----------------------------------

query_embedding = embedding_model.encode(
    [transaction_text]
)

# -----------------------------------
# RETRIEVAL
# -----------------------------------

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

retrieved_cases = "\n\n".join(
    results['documents'][0]
)

# -----------------------------------
# PROMPT
# -----------------------------------

prompt = f"""
You are an expert financial fraud analyst.

Prediction Result:
{prediction_label}

Transaction Details:
{transaction_text}

Retrieved Similar Fraud Cases:
{retrieved_cases}

Generate:

1. Fraud Reason
2. Risk Level
3. Suggested Action
4. Investigation Summary

Keep the response concise and professional.
"""

# -----------------------------------
# GENERATE AI RESPONSE
# -----------------------------------

if st.button("Generate AI Fraud Analysis"):

    with st.spinner("Generating AI insights..."):

        response = client.chat.completions.create(

            model="meta/llama-3.1-70b-instruct",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,
            max_tokens=300
        )

        ai_output = response.choices[0].message.content

        st.subheader("AI Fraud Intelligence Report")

        st.write(ai_output)
