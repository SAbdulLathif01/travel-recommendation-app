import streamlit as st
from chatbot import ask_chatbot, recognize_speech, text_to_speech

# ✅ Ensure `set_page_config` is at the top
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Chatbot UI
st.title("🤖 AI Chatbot")
st.write("Ask me anything!")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🔹 Voice Input Button
col1, col2 = st.columns([3, 1])  # Creates a layout with text input & mic button
with col1:
    user_query = st.chat_input("Type your question here...")
with col2:
    if st.button("🎤 Speak Now"):
        user_query = recognize_speech()  # Get voice input
        st.write(f"🗣️ You: {user_query}")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Thinking..."):
        chatbot_response = ask_chatbot(user_query)

    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
    
    with st.chat_message("assistant"):
        st.markdown(chatbot_response)

    # 🔊 Speak chatbot response
    text_to_speech(chatbot_response)
