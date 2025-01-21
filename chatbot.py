import streamlit as st
from streamlit_chat import message
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
from datetime import datetime, timedelta
import time
from groq import Groq
import tensorflow as tf

# Suppress TensorFlow warnings
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

# Pinecone-configuratie
PINECONE_API_KEY = "pcsk_4v8YTF_6Sgtnwh2Tm38koMurffJJLUp4eyHncT843KmkeKN3GVbjqSNbFPtzjBiDTfkF6V"
INDEX_NAME = "projectvito"

# API-sleutels en configuratie
GEMINI_API_KEY = "AIzaSyBZxIoL3X8wyzcKFAC_7Hw5Fz6FkIFWLfQ"
GROQ_API_KEY = "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"
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "mixtral-8x7b-32768"

# Initialiseer Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Embedder configuratie
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_embeddings" not in st.session_state:
    st.session_state.chat_embeddings = []
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = datetime.min
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "message_counter" not in st.session_state:
    st.session_state.message_counter = len(st.session_state.messages) * 2  # Initialize based on existing messages

# Migrate existing messages to include keys if they don't have them
for i, msg in enumerate(st.session_state.messages):
    if "key" not in msg:
        msg["key"] = f"msg_{i}"
        st.session_state.message_counter = max(st.session_state.message_counter, i + 1)

def check_rate_limit():
    current_time = datetime.now()
    if current_time - st.session_state.last_request_time > timedelta(minutes=1):
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    if st.session_state.request_count >= 60:
        time_to_wait = 60 - (current_time - st.session_state.last_request_time).seconds
        return False, time_to_wait
    st.session_state.request_count += 1
    st.session_state.last_request_time = current_time
    return True, 0

def limit_tokens(text, max_tokens=500):
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + "... (antwoord ingekort)"
    return text

def limit_context(text, max_tokens=2000):
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + "..."
    return text

def generate_with_groq(prompt):
    try:
        limited_prompt = limit_context(prompt, max_tokens=4000)
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Je bent een behulpzame AI-assistent."},
                {"role": "user", "content": limited_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        if completion and completion.choices:
            return completion.choices[0].message.content
        else:
            return "Geen geldig antwoord ontvangen van de Groq API."
    except Exception as e:
        st.error(f"Fout bij Groq API: {e}")
        return "Er is een fout opgetreden met Groq. Probeer het later opnieuw."

def generate_response(prompt):
    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
        if response and response.text:
            return limit_tokens(response.text, max_tokens=500)
    except Exception:
        return generate_with_groq(prompt)

def ask_question(question):
    with st.spinner("Bezig met het genereren van een antwoord..."):
        try:
            can_proceed, wait_time = check_rate_limit()
            if not can_proceed:
                st.warning(f"Wacht {wait_time} seconden voordat je een nieuwe vraag stelt.")
                return

            query_embedding = embedder.encode(question).tolist()
            st.session_state.chat_embeddings.append(query_embedding)

            results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            context = ""
            if results["matches"]:
                for match in results["matches"]:
                    context += match["metadata"].get("text", "") + "\n"

            limited_context = limit_context(context, max_tokens=2000)
            prompt = f"""
Context:
{limited_context}

Vraag: {question}

Geef een duidelijk en behulpzaam antwoord gebaseerd op de context.
            """
            answer = generate_response(prompt)
            if not answer:
                answer = "Ik kon geen relevante informatie vinden. Probeer opnieuw."

            # Add messages with unique keys
            st.session_state.message_counter += 2
            st.session_state.messages.append({
                "role": "user",
                "content": question,
                "key": f"msg_{st.session_state.message_counter - 1}"
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "key": f"msg_{st.session_state.message_counter}"
            })

        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")

def handle_input():
    if st.session_state.user_input and st.session_state.user_input.strip():
        question = st.session_state.user_input.strip()
        st.session_state.user_input = ""  # Reset input
        ask_question(question)

# Create chat container
chat_container = st.container()

# Display chat history with proper key handling
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        if "key" not in msg:
            msg["key"] = f"legacy_msg_{i}"
        message(
            msg["content"],
            is_user=(msg["role"] == "user"),
            key=msg["key"]
        )



# Place user input below chat history with automatic sending
st.text_input(
    "Typ je vraag:",
    key="user_input",
    on_change=handle_input
)