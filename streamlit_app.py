import streamlit as st
import os
import datetime
import numpy as np
import psutil
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


# =========================
# CONFIGURATION
# =========================

st.set_page_config(
    page_title="Enterprise AI IT Support Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

KNOWLEDGE_DIR = "knowledge"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "chat_log.txt")

EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:1b"


# =========================
# CUSTOM ENTERPRISE CSS
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

.header-title {
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(90deg,#4f46e5,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.user-msg {
    background: linear-gradient(90deg,#4f46e5,#6366f1);
    color: white;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
}

.assistant-msg {
    background: white;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    margin-bottom: 8px;
}

.source-badge {
    background: #e0f2fe;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

embed_model = load_embed_model()


# =========================
# LOAD KNOWLEDGE BASE
# =========================

@st.cache_resource
def load_knowledge():

    docs = []
    names = []

    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)

    for file in os.listdir(KNOWLEDGE_DIR):

        with open(os.path.join(KNOWLEDGE_DIR, file), "r", encoding="utf-8") as f:

            docs.append(f.read())
            names.append(file)

    return docs, names


documents, filenames = load_knowledge()


@st.cache_resource
def create_embeddings(docs):

    if len(docs) == 0:
        return None

    return embed_model.encode(docs)


doc_embeddings = create_embeddings(documents)


# =========================
# AI FALLBACK
# =========================

def generate_ai_response(query):

    response = ollama.chat(

        model=OLLAMA_MODEL,

        messages=[
            {
                "role": "system",
                "content": "You are a professional enterprise IT support assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    return response["message"]["content"]


# =========================
# SEARCH FUNCTION
# =========================

def search(query):

    if doc_embeddings is None:
        return generate_ai_response(query), "AI", 0

    q_embed = embed_model.encode([query])

    similarity = cosine_similarity(q_embed, doc_embeddings)

    index = np.argmax(similarity)

    confidence = similarity[0][index]

    if confidence >= 0.4:

        return documents[index], filenames[index], confidence

    return generate_ai_response(query), "AI", confidence


# =========================
# LOGGING
# =========================

def log(user, response):

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    with open(LOG_FILE, "a", encoding="utf-8") as f:

        f.write(
            f"{datetime.datetime.now()} | {user} | {response}\n"
        )


# =========================
# HEADER
# =========================

col1, col2 = st.columns([1,6])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=70)

with col2:
    st.markdown('<div class="header-title">Enterprise AI IT Support Assistant</div>', unsafe_allow_html=True)
    st.write("Hybrid AI system with semantic search and conversational intelligence.")


# =========================
# METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

cpu = psutil.cpu_percent()
memory = psutil.virtual_memory().percent

col1.markdown(f'<div class="metric-card"><h4>CPU</h4><h2>{cpu}%</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><h4>Memory</h4><h2>{memory}%</h2></div>', unsafe_allow_html=True)
col3.markdown('<div class="metric-card"><h4>AI Status</h4><h2 style="color:green">Online</h2></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-card"><h4>Time</h4><h2>{datetime.datetime.now().strftime("%H:%M:%S")}</h2></div>', unsafe_allow_html=True)

st.divider()


# =========================
# CHAT
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?", "source": ""}
    ]


# Display messages
for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    else:

        st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)

        if msg["source"]:
            st.markdown(f'<span class="source-badge">Source: {msg["source"]}</span>', unsafe_allow_html=True)


# =========================
# INPUT
# =========================

query = st.chat_input("Ask IT support question...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query, "source": ""}
    )

    with st.spinner("Thinking..."):

        response, source, confidence = search(query)

        log(query, response)

    # Streaming effect
    placeholder = st.empty()

    full = ""

    for char in response:

        full += char

        placeholder.markdown(f'<div class="assistant-msg">{full}</div>', unsafe_allow_html=True)

        time.sleep(0.005)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "source": source}
    )

    st.rerun()


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("Control Panel")

    st.success("Embedding Model Loaded")
    st.success("Conversational AI Ready")

    if os.path.exists(LOG_FILE):

        with open(LOG_FILE) as f:
            logs = f.readlines()

        st.metric("Total Queries", len(logs))

    st.divider()

    st.write("System Monitoring")

    st.write(f"CPU: {cpu}%")
    st.write(f"Memory: {memory}%")
