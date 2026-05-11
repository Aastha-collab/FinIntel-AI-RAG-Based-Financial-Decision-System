import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="FinIntel AI",
    layout="wide"
)

# -----------------------------------
# CUSTOM DARK THEME
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3, h4 {
    color: #00FFAA;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    height: 3em;
    width: 100%;
}

div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #00FFAA;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown("""
# 💳 FinIntel AI
### AI-Powered Fraud Detection & Financial Decision Intelligence System

This system combines:
- Machine Learning Fraud Detection
- Retrieval-Augmented Generation (RAG)
- NVIDIA LLM APIs
- AI-Based Financial Reasoning

to identify suspicious financial transactions and generate intelligent fraud investigation reports.
""")

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("creditcard_sample.csv")

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
# CREATE KNOWLEDGE BASE
# -----------------------------------

fraud_df = df[df['Class'] == 1]

normal_df = df[df['Class'] == 0]

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

collection = client_db.get_or_create_collection(
    name="fraud_rag"
)

# Avoid duplicate insertion

if collection.count() == 0:

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(documents))]
    )

# -----------------------------------
# NVIDIA CLIENT
# -----------------------------------

NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.header("🔍 Transaction Selection")

st.sidebar.markdown("""
Select a transaction row to:
- Predict fraud risk
- Retrieve similar fraud cases
- Generate AI investigation report
""")

# DROPDOWN

row_number = st.sidebar.selectbox(
    "Select Transaction Row",
    options=df.index.tolist()
)

# -----------------------------------
# METRICS
# -----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Fraud Transactions",
        len(fraud_df)
    )

with col2:
    st.metric(
        "Normal Transactions",
        len(normal_df)
    )

with col3:
    st.metric(
        "Fraud Knowledge Base",
        len(documents)
    )

# -----------------------------------
# DYNAMIC PIE CHART
# -----------------------------------

st.subheader("📊 Dataset Distribution")

fig, ax = plt.subplots()

labels = ['Normal', 'Fraud']

sizes = [
    len(normal_df),
    len(fraud_df)
]

ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

st.pyplot(fig)

# -----------------------------------
# TRANSACTION
# -----------------------------------

sample_transaction = df.drop(
    'Class',
    axis=1
).iloc[[row_number]]

prediction = rf_model.predict(
    sample_transaction
)

# -----------------------------------
# PREDICTION LABEL
# -----------------------------------

if prediction[0] == 1:

    prediction_label = "Fraudulent Transaction"

else:

    prediction_label = "Normal Transaction"

# -----------------------------------
# SHOW PREDICTION
# -----------------------------------

st.subheader("🎯 Prediction Result")

if prediction_label == "Fraudulent Transaction":

    st.error(f"🚨 {prediction_label}")

else:

    st.success(f"✅ {prediction_label}")

# -----------------------------------
# TRANSACTION DETAILS
# -----------------------------------

with st.expander("📄 Transaction Details"):

    st.write(sample_transaction)

# -----------------------------------
# DYNAMIC TRANSACTION GRAPH
# -----------------------------------

st.subheader("📈 Transaction Feature Visualization")

features = ['Amount', 'V14', 'V10']

values = [
    sample_transaction['Amount'].values[0],
    sample_transaction['V14'].values[0],
    sample_transaction['V10'].values[0]
]

fig2, ax2 = plt.subplots()

ax2.bar(features, values)

st.pyplot(fig2)

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
# FRAUD / NORMAL LOGIC
# -----------------------------------

if prediction_label == "Fraudulent Transaction":

    query_embedding = embedding_model.encode(
        [transaction_text]
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    retrieved_cases = "\n\n".join(
        results['documents'][0]
    )

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
"""

else:

    prompt = f"""
You are a financial transaction analyst.

Prediction Result:
{prediction_label}

Transaction Details:
{transaction_text}

Explain briefly:

1. Why transaction appears normal
2. Risk level
3. Why transaction is low risk
4. Final summary
"""

# -----------------------------------
# AI BUTTON
# -----------------------------------

if st.button("🚀 Generate AI Fraud Analysis"):

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

        st.subheader("🤖 AI Fraud Intelligence Report")

        st.markdown(f"""
<div style="
background-color:#1E1E1E;
padding:20px;
border-radius:15px;
border:2px solid #00FFAA;
">

{ai_output}

</div>
""", unsafe_allow_html=True)

# -----------------------------------
# BUSINESS IMPACT
# -----------------------------------

st.markdown("""
---
## Business Impact

FinIntel AI helps financial institutions:

- Detect suspicious transactions faster
- Reduce fraud investigation time
- Improve explainability in fraud detection
- Generate AI-powered fraud intelligence reports
- Enhance financial risk monitoring systems

This project demonstrates:
Machine Learning + RAG + Vector Databases + LLM APIs.
""")
