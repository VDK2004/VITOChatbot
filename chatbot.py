import streamlit as st
from streamlit_chat import message  # Voor een interactieve chatstijl
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
from datetime import datetime, timedelta
import time
from groq import Groq

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

# Initialiseer chatgeschiedenis en andere sessie-variabelen
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_embeddings" not in st.session_state:
    st.session_state.chat_embeddings = []
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = datetime.min
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

# Rate limiting
def check_rate_limit():
    """Controleer of verzoeken binnen de toegestane limiet vallen."""
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

# Limiteer tokens
def limit_tokens(text, max_tokens=500):
    """Beperk het aantal tokens (woorden) in tekst."""
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + "... (antwoord ingekort)"
    return text

# Limiteer context
def limit_context(text, max_tokens=2000):
    """Beperk context tot een maximum aantal tokens."""
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + "..."
    return text

# Genereer antwoord met Groq
def generate_with_groq(prompt):
    """Genereer antwoord met de Groq API."""
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

# Genereer antwoord met fallback naar Groq
def generate_response(prompt):
    """Probeer eerst Gemini, gebruik Groq als fallback."""
    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
        if response and response.text:
            return limit_tokens(response.text, max_tokens=500)
    except Exception:
        return generate_with_groq(prompt)

# Vraagverwerking
def ask_question(question):
    """Verwerk de vraag en genereer een antwoord."""
    with st.spinner("Bezig met het genereren van een antwoord..."):
        try:
            can_proceed, wait_time = check_rate_limit()
            if not can_proceed:
                st.warning(f"Wacht {wait_time} seconden voordat je een nieuwe vraag stelt.")
                return

            # Genereer embedding voor de vraag
            query_embedding = embedder.encode(question).tolist()
            st.session_state.chat_embeddings.append(query_embedding)

            # Zoek in Pinecone
            results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            context = ""
            if results["matches"]:
                for match in results["matches"]:
                    context += match["metadata"].get("text", "") + "\n"

            # Bereid prompt voor
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

            # Update chatgeschiedenis
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Er is een fout opgetreden: {e}")


# Gebruikersinvoer
user_input = st.text_input("Typ je vraag:")
if st.button("Verstuur") and user_input:
    ask_question(user_input)

# Toon chatgeschiedenis
for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True)
    elif msg["role"] == "assistant":
        message(msg["content"], is_user=False)
