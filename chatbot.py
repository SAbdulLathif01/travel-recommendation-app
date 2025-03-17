from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os

MODEL_NAME = "facebook/blenderbot-400M-distill"  # Smaller, faster model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def ask_chatbot(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# 🔹 Speech-to-Text (Voice Input)
def recognize_speech():
    """Capture voice input and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Error: Speech service unavailable."

# 🔹 Text-to-Speech (Voice Output)
def text_to_speech(text):
    """Convert text response to voice output."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
    # OR using gTTS (Google TTS)
    # tts = gTTS(text)
    # tts.save("response.mp3")
    # os.system("start response.mp3")  # Windows
