"""
============================================================
LLaMA Chatbot — Streamlit + Ollama
============================================================

Description:
------------
This application is a lightweight web-based chatbot built using
Streamlit and Ollama. It runs the LLaMA 3.2 model locally and
provides an interactive chat interface without relying on any
external APIs or cloud services.

The chatbot communicates with Ollama via its local REST API
and maintains chat history using Streamlit session state.

------------------------------------------------------------
Prerequisites:
--------------
1. Python 3.9 or higher
2. Ollama installed and running locally
3. LLaMA 3.2 model pulled in Ollama

Installation:
-------------
1. Install dependencies:
   pip install -r requirements.txt

2. Start Ollama and pull the model (if not already done):
   ollama run llama3.2

3. Run the Streamlit application:
   streamlit run app.py

------------------------------------------------------------
How It Works:
-------------
- Streamlit handles the UI and session management.
- User prompts are sent to Ollama's local REST endpoint.
- The LLaMA model generates responses locally.
- Chat history is preserved for the duration of the session.

------------------------------------------------------------
Notes:
------
- Ollama must be running on http://localhost:11434
- No API keys are required
- Chat history resets when the session restarts

License:
--------
MIT License
============================================================
"""

import streamlit as st
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def chat_with_llama(prompt):
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "response" in data:
            return data["response"]
        elif "error" in data:
            return f"Ollama Error: {data['error']}"
        else:
            return f"Unexpected response format: {data}"

    except requests.exceptions.RequestException as e:
        return f"Connection Error: {e}"


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="LLaMA Chatbot", page_icon="🦙")

st.title("🦙 LLaMA Chatbot (Ollama + Streamlit)")
st.caption("Local LLaMA 3.2 running via Ollama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get LLaMA response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = chat_with_llama(user_input)
            st.markdown(reply)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": reply})
