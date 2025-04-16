import streamlit as st
from chatbot import ask_chatbot, recognize_speech, text_to_speech

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask me anything!</p>", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear button
if st.button("🧹 Clear Chat", type="primary"):
    st.session_state.chat_history = []

# Display chat history
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.chat_message("user").write(entry["text"])
    else:
        st.chat_message("assistant").write(entry["text"])

# Input layout: row with textbox + mic button
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Type your question here...", key="input", label_visibility="collapsed")
with col2:
    if st.button("🎙️ Speak Now"):
        user_input = recognize_speech()
        st.session_state.input = user_input

# Process input
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    response = ask_chatbot(user_input)
    st.session_state.chat_history.append({"role": "assistant", "text": response})
    text_to_speech(response)
    st.rerun()
