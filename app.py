import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FinIntel AI",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# PREMIUM UI / CSS
# =========================================================

st.markdown(
    """  
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #F9FAFF 0%,
            #EEF2FF 50%,
            #F5F3FF 100%
        );
        color: #1E293B;
        font-family: 'Segoe UI', sans-serif;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    h1 {
        color: #5B21B6 !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }

    h2, h3, h4 {
        color: #6D28D9 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #6D28D9 0%,
            #7C3AED 50%,
            #9333EA 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* ============================= */
/* METRIC CARDS */
/* ============================= */

div[data-testid="metric-container"] {

    background: white !important;

    border-radius: 20px !important;

    padding: 20px !important;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.08) !important;

    border-left: 8px solid #8B5CF6 !important;
}

/* FORCE METRIC TEXT VISIBLE */

div[data-testid="metric-container"] * {

    color: black !important;
}

    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    }

    .stButton>button {
        background: linear-gradient(
            to right,
            #EC4899,
            #F472B6
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        padding: 12px 18px !important;
        width: 100% !important;
        transition: 0.3s !important;
        box-shadow: 0px 4px 15px rgba(236,72,153,0.35);
    }

    .stButton>button:hover {
        transform: scale(1.02);

        background: linear-gradient(
            to right,
            #DB2777,
            #EC4899
        ) !important;
    }

    [data-testid="stDownloadButton"] button {
        background: white !important;
        color: #EC4899 !important;
        border: 2px solid #F9A8D4 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #FCE7F3 !important;
        color: #BE185D !important;
    }

    .streamlit-expanderHeader {
        background: white !important;
        border-radius: 12px !important;
        color: #1E293B !important;
    }

    .streamlit-expanderContent {
        background-color: white !important;
        border-radius: 12px !important;
        padding: 10px !important;
        color: #1E293B !important;
    }

    .block-container {
        padding-top: 2rem;
    }

/* DATAFRAME FIX */

[data-testid="stDataFrame"] {

    background-color: white !important;

    border-radius: 15px !important;

    padding: 10px !important;
}

/* DATAFRAME TEXT */

[data-testid="stDataFrame"] * {

    color: #1E293B !important;
}    

/* SIDEBAR DROPDOWN FIX */

[data-testid="stSidebar"] .stSelectbox > div > div {

    background-color: white !important;

    color: white !important;

    border-radius: 12px !important;

    border: 2px solid #C084FC !important;

    box-shadow: 0px 2px 8px rgba(0,0,0,0.08) !important;
}

/* DROPDOWN TEXT */

[data-testid="stSidebar"] .stSelectbox * {

    color: black !important;

    font-weight: 600 !important;
}

/* DROPDOWN ARROW */

[data-testid="stSidebar"] svg {

    fill: #7C3AED !important;
}

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
# 💳 FinIntel AI

### AI-Powered Fraud Detection & Financial Decision Intelligence Dashboard

FinIntel AI combines:

- Machine Learning Fraud Detection
- Retrieval-Augmented Generation (RAG)
- NVIDIA LLM APIs
- AI-Based Financial Reasoning

to identify suspicious transactions and generate intelligent fraud investigation reports in real time.
""")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("creditcard_sample.csv")

df['Hour'] = df['Time'] // 3600

# =========================================================
# LOAD MODEL
# =========================================================

rf_model = joblib.load("rf_model.pkl")

# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# =========================================================
# KNOWLEDGE BASE
# =========================================================

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

embeddings = embedding_model.encode(documents)

# =========================================================
# CHROMADB
# =========================================================

client_db = chromadb.Client()

collection = client_db.get_or_create_collection(
    name="fraud_rag"
)

if collection.count() == 0:

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(documents))]
    )

# =========================================================
# NVIDIA CLIENT
# =========================================================

NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🔍 Transaction Explorer")

st.sidebar.markdown("""
Select a transaction to:

✔ Predict fraud risk  
✔ Retrieve similar fraud cases  
✔ Generate AI investigation report  
✔ Analyze suspicious patterns
""")

row_number = st.sidebar.selectbox(
    "Choose Transaction Row",
    options=df.index.tolist()
)

# =========================================================
# METRICS
# =========================================================
st.markdown("""
<h2 style="
color:#4C1D95;
font-weight:800;
margin-bottom:20px;
">
📊 System Analytics
</h2>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

cards = [

    ("Total Transactions", len(df), "#6366F1"),

    ("Fraud Cases", len(fraud_df), "#EC4899"),

    ("Normal Cases", len(normal_df), "#10B981"),

    ("Fraud Knowledge Base", len(documents), "#8B5CF6")
]

for col, (title, value, color) in zip(
    [col1, col2, col3, col4],
    cards
):

    col.markdown(f"""
    <div style="
    background:white;
    padding:22px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    border-top:6px solid {color};
    text-align:center;
    ">

    <h3 style="
    color:#6B7280;
    font-size:16px;
    margin-bottom:10px;
    ">
    {title}
    </h3>

    <h1 style="
    color:#111827;
    font-size:38px;
    font-weight:800;
    ">
    {value}
    </h1>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TRANSACTION
# =========================================================

sample_transaction = df.drop(
    'Class',
    axis=1
).iloc[[row_number]]

prediction = rf_model.predict(
    sample_transaction
)

# =========================================================
# LABEL
# =========================================================

if prediction[0] == 1:

    prediction_label = "Fraudulent Transaction"

else:

    prediction_label = "Normal Transaction"

# =========================================================
# RESULT
# =========================================================

st.markdown("## 🎯 Prediction Result")

if prediction_label == "Fraudulent Transaction":

    st.error(f"🚨 {prediction_label}")

else:

    st.success(f"✅ {prediction_label}")

# =========================================================
# TRANSACTION DETAILS
# =========================================================

with st.expander("📄 View Transaction Details"):

    st.dataframe(

        sample_transaction,

        use_container_width=True
    )

# =========================================================
# TRANSACTION TEXT
# =========================================================

transaction_text = f"""
Transaction Summary

Transaction Amount: {sample_transaction['Amount'].values[0]}
Transaction Time: {sample_transaction['Time'].values[0]}
Suspicious V14 Score: {sample_transaction['V14'].values[0]}
Suspicious V10 Score: {sample_transaction['V10'].values[0]}
"""

# =========================================================
# FRAUD / NORMAL LOGIC
# =========================================================

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

# =========================================================
# AI BUTTON
# =========================================================

if st.button("✨ Generate AI Fraud Analysis"):

    with st.spinner("Generating AI financial intelligence..."):

        response = client.chat.completions.create(

            model="meta/llama-3.1-70b-instruct",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,

            top_p=0.9,

            max_tokens=700
        )

        ai_output = response.choices[0].message.content

        st.markdown(f"""
<div class="report-box">

{ai_output}

</div>
""", unsafe_allow_html=True)
        
        # DOWNLOAD REPORT

        st.download_button(

            label="📥 Download AI Report",

            data=ai_output,

            file_name="fraud_analysis_report.txt",

            mime="text/plain"
        )


# =========================================================
# PREMIUM AI FRAUD CHATBOT
# =========================================================

st.markdown("""
<div style="
background:linear-gradient(
135deg,
#FFF1F7 0%,
#FFE4F1 100%
);
padding:30px;
border-radius:28px;
border-left:7px solid #EC4899;
box-shadow:0px 8px 25px rgba(236,72,153,0.12);
margin-top:35px;
">

<h2 style="
color:#BE185D;
font-weight:800;
margin-bottom:10px;
">
🤖 AI Fraud Chatbot
</h2>

<p style="
color:#4B5563;
font-size:16px;
line-height:1.9;
margin-bottom:0px;
">

Ask questions related to:

• Fraud Prediction  
• Risk Level  
• Transaction Behavior  
• Fraud Patterns  
• AI Investigation  
• Similar Fraud Cases  
• Model Logic  
• Financial Security  
• Banking Fraud Prevention  
• Dataset Insights  

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# CHAT INPUT STYLE
# =========================================================

st.markdown("""
<style>

.stTextInput > div > div > input {

    background-color: #FFFFFF !important;

    color: #374151 !important;

    border: 2px solid #F9A8D4 !important;

    border-radius: 16px !important;

    padding: 14px !important;

    font-size: 16px !important;

    box-shadow: 0px 4px 15px rgba(236,72,153,0.08) !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# USER INPUT
# =========================================================

chat_query = st.text_input(
    "💬 Ask your question here..."
)

# =========================================================
# CHATBOT RESPONSE
# =========================================================

if chat_query:

    with st.spinner("Generating AI response..."):

        # SAFE FRAUD CASES

        if prediction_label == "Fraudulent Transaction":

            chatbot_cases = retrieved_cases

        else:

            chatbot_cases = "No similar fraud cases retrieved."

        # SAFE AI REPORT

        if 'ai_output' in locals():

            chatbot_report = ai_output

        else:

            chatbot_report = "No AI fraud report generated yet."

        # =====================================================
        # PROMPT
        # =====================================================

        chatbot_prompt = f"""
You are an expert AI Financial Fraud Investigation Assistant.

PROJECT:
FinIntel AI - RAG Based Financial Fraud Detection System

Prediction Result:
{prediction_label}

Transaction Details:
{transaction_text}

Retrieved Fraud Cases:
{chatbot_cases}

AI Generated Fraud Report:
{chatbot_report}

User Question:
{chat_query}

Instructions:
- Answer accurately
- Keep explanations simple
- Answer only project-related questions
- Explain fraud logic clearly
- Explain risk behavior clearly
- Keep formatting clean and professional
"""

        # =====================================================
        # NVIDIA RESPONSE
        # =====================================================

        response = client.chat.completions.create(

            model="meta/llama-3.1-70b-instruct",

            messages=[
                {
                    "role": "user",
                    "content": chatbot_prompt
                }
            ],

            temperature=0.2,

            max_tokens=220
        )

        final_answer = response.choices[0].message.content

        # =====================================================
        # DISPLAY RESPONSE
        # =====================================================

        st.markdown(f"""
<div style="
background:#FFFFFF;
padding:28px;
border-radius:24px;
border-left:6px solid #EC4899;
box-shadow:0px 6px 18px rgba(236,72,153,0.10);
margin-top:20px;
">

<h3 style="
color:#BE185D;
font-weight:800;
margin-bottom:18px;
">
💡 AI Assistant Response
</h3>

<div style="
color:#374151;
font-size:16px;
line-height:1.9;
white-space:pre-wrap;
">

{final_answer}

</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# BAR GRAPH
# =========================================================

st.markdown("## 📊 Transaction Feature Visualization")

features = ['Amount', 'V14', 'V10']

values = [
    sample_transaction['Amount'].values[0],
    sample_transaction['V14'].values[0],
    sample_transaction['V10'].values[0]
]

fig2, ax2 = plt.subplots(figsize=(5,3))

colors = ['#F472B6', '#A78BFA', '#60A5FA']

ax2.bar(
    features,
    values,
    color=colors,
    width=0.45
)

fig2.patch.set_facecolor('#F8FAFF')

ax2.set_facecolor('#FFFFFF')

ax2.tick_params(
    axis='x',
    labelsize=8
)

ax2.tick_params(
    axis='y',
    labelsize=8
)

ax2.set_title(
    "Transaction Feature Scores",
    fontsize=10
)

ax2.set_ylabel(
    "Values",
    fontsize=8
)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

graph_container2 = st.container(border=True)

with graph_container2:
    st.pyplot(fig2)

# =========================================================
# PIE CHART
# =========================================================

st.markdown("## 📈 Dataset Distribution")

fig, ax = plt.subplots(figsize=(4,4))

colors = ['#A78BFA', '#F472B6']

ax.pie(
    [
        len(normal_df),
        len(fraud_df)
    ],
    labels=['Normal', 'Fraud'],
    autopct='%1.1f%%',
    colors=colors,
    textprops={'fontsize': 10}
)

fig.patch.set_facecolor('#F8FAFF')

graph_container = st.container(border=True)

with graph_container:
    st.pyplot(fig)

# =========================================================
# BUSINESS IMPACT
# =========================================================

st.markdown("""
---

## 💼 Business Impact

FinIntel AI helps financial institutions:

- Detect suspicious transactions faster
- Reduce fraud investigation time
- Improve fraud explainability
- Generate AI-powered investigation reports
- Enhance financial risk monitoring systems

""")
