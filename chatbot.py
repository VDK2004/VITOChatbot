import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
from datetime import datetime, timedelta
from groq import Groq

# Pinecone-configuratie
PINECONE_API_KEY = "pcsk_4v8YTF_6Sgtnwh2Tm38koMurffJJLUp4eyHncT843KmkeKN3GVbjqSNbFPtzjBiDTfkF6V"
INDEX_NAME = "projectvito"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Embedder configuratie
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini API-configuratie
GEMINI_API_KEY = "AIzaSyBZxIoL3X8wyzcKFAC_7Hw5Fz6FkIFWLfQ"
GROQ_API_KEY = "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"  # Vul je Groq API key in
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Configureer de modellen
gemini_model = genai.GenerativeModel('gemini-pro')
GROQ_MODEL = "mixtral-8x7b-32768"

# LSTM Model voor chat context
def create_lstm_model():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(None, 384)),  # Aangepast naar 384 dimensies
        LSTM(64),
        Dense(384, activation='relu')  # Output dimensie ook aangepast naar 384
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialiseer LSTM model
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = create_lstm_model()

# Chat geschiedenis met embeddings
if 'chat_embeddings' not in st.session_state:
    st.session_state.chat_embeddings = []

# Rate limiting configuratie
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = datetime.min
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0

def check_rate_limit():
    """Controleer en beheer rate limiting."""
    current_time = datetime.now()
    # Reset teller als er een minuut voorbij is
    if current_time - st.session_state.last_request_time > timedelta(minutes=1):
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    # Check of we onder de limiet zitten (60 verzoeken per minuut)
    if st.session_state.request_count >= 60:
        time_to_wait = 60 - (current_time - st.session_state.last_request_time).seconds
        if time_to_wait > 0:
            return False, time_to_wait
    
    st.session_state.request_count += 1
    st.session_state.last_request_time = current_time
    return True, 0

def process_chat_history(embeddings):
    if len(embeddings) < 2:
        return None
    # Converteer embeddings naar numpy array voor LSTM
    X = np.array(embeddings[-5:])  # Neem laatste 5 berichten
    X = np.expand_dims(X, axis=0)  # Voeg batch dimensie toe
    return X

def limit_tokens(text, max_tokens=500):
    """Limiteer tekst tot een maximum aantal tokens (woorden)."""
    words = text.split()
    if len(words) > max_tokens:
        limited_text = ' '.join(words[:max_tokens])
        return limited_text + "... (antwoord ingekort)"
    return text

def limit_context(text, max_tokens=2000):
    """Limiteer context tot een maximum aantal tokens."""
    words = text.split()
    if len(words) > max_tokens:
        limited_text = ' '.join(words[:max_tokens])
        return limited_text + "..."
    return text

def generate_with_groq(prompt):
    """Genereer antwoord met Groq API."""
    try:
        # Limiteer de prompt grootte voor Groq
        limited_prompt = limit_context(prompt, max_tokens=4000)  # Houd ruimte voor system message en response
        
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Je bent een behulpzame AI-assistent."},
                {"role": "user", "content": limited_prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        
        if completion and completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            st.warning("Geen geldig antwoord van Groq API")
            raise Exception("Geen geldig antwoord van Groq API")
    except Exception as e:
        st.warning(f"Groq API fout: {str(e)}")
        raise Exception(f"Groq API fout: {str(e)}")

def generate_response(prompt, use_groq=False):
    """Genereer antwoord met fallback naar Groq."""
    if use_groq:
        try:
            response = generate_with_groq(prompt)
            if response:
                return response
            return "Bezig met het genereren van een antwoord..."
        except Exception as e:
            # Probeer Gemini als fallback
            try:
                response = gemini_model.generate_content(prompt)
                if response and response.text:
                    return response.text
                return "Bezig met het genereren van een antwoord..."
            except:
                return "Bezig met het genereren van een antwoord..."
    
    try:
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text
        return "Bezig met het genereren van een antwoord..."
    except Exception as e:
        if "500" in str(e) or "429" in str(e):  # Internal error of rate limit
            return generate_with_groq(prompt)
        return "Bezig met het genereren van een antwoord..."

def ask_question(question):
    with st.spinner('Bezig met het genereren van een antwoord...'):
        try:
            # Check rate limit voordat we het verzoek doen
            can_proceed, wait_time = check_rate_limit()
            use_groq = not can_proceed

            # Genereer embedding voor de vraag
            query_embedding = embedder.encode(question).tolist()
            
            # Voeg embedding toe aan chat geschiedenis
            if 'chat_embeddings' not in st.session_state:
                st.session_state.chat_embeddings = []
            st.session_state.chat_embeddings.append(query_embedding)
            
            # Verwerk chat geschiedenis met LSTM als er genoeg context is
            chat_context = process_chat_history(st.session_state.chat_embeddings)
            if chat_context is not None:
                try:
                    lstm_context = st.session_state.lstm_model.predict(chat_context)
                except:
                    lstm_context = query_embedding
            else:
                lstm_context = query_embedding

            # Zoek in Pinecone
            results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
            context = ""

            if results["matches"]:
                for match in results["matches"]:
                    context += match["metadata"].get("text", "") + "\n"

            # Voeg de vraag toe aan de chatgeschiedenis
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})

            if context:
                # Limiteer de context en chat geschiedenis
                limited_context = limit_context(context, max_tokens=2000)
                chat_history = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.messages[-3:] if msg
                ])
                limited_history = limit_context(chat_history, max_tokens=1000)
                
                prompt = f"""Chat geschiedenis:
{limited_history}

Context uit kennisbank:
{limited_context}

Vraag: {question}

Geef een behulpzaam antwoord dat rekening houdt met zowel de chat geschiedenis als de context uit de kennisbank. 
Houd je antwoord beknopt en to-the-point."""

                max_retries = 3
                answer = None
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            time.sleep(2 * attempt)
                        
                        answer = limit_tokens(generate_response(prompt, use_groq=use_groq), max_tokens=500)
                        if answer and answer != "Bezig met het genereren van een antwoord...":
                            break
                    except:
                        continue

                if not answer or answer == "Bezig met het genereren van een antwoord...":
                    answer = "Ik werk momenteel aan een antwoord. Probeer je vraag over een moment opnieuw te stellen."
            else:
                answer = "Ik kon geen relevante informatie in de kennisbank vinden, maar ik zal proberen te helpen op basis van onze chat geschiedenis."

            # Voeg het antwoord toe aan de chatgeschiedenis
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Voeg embedding van antwoord toe aan chat geschiedenis
            try:
                answer_embedding = embedder.encode(answer).tolist()
                st.session_state.chat_embeddings.append(answer_embedding)
            except:
                pass  # Negeer fouten bij het maken van embeddings

        except Exception as e:
            # Als er toch een onverwachte fout optreedt
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Ik werk momenteel aan een antwoord. Probeer je vraag over een moment opnieuw te stellen."})

# Verbeterde chat interface met geschiedenis
st.title("Chatbot met Knowledge Base")
st.write("Stel vragen aan de chatbot. Hij onthoudt het gesprek en gebruikt de kennisbank om te antwoorden.")

# Initialiseer chatgeschiedenis als het nog niet bestaat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat interface
user_input = st.text_input("Stel je vraag:")
if st.button("Verstuur") and user_input:
    ask_question(user_input)

# Toon de chatgeschiedenis in een netter formaat
st.write("### Chat Geschiedenis")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"🧑 **Jij:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"🤖 **Chatbot:** {msg['content']}")
