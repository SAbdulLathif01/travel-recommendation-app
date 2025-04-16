from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import speech_recognition as sr
import pyttsx3

# Load Facebook BlenderBot model
MODEL_NAME = "facebook/blenderbot-1B-distill"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# Global history for context-aware responses
conversation_history = []

# Chatbot response with context
def ask_chatbot(user_input):
    global conversation_history
    conversation_history.append(f"User: {user_input}")
    prompt = "\n".join(conversation_history)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    conversation_history.append(f"Bot: {response}")
    return response.strip()

# Speech-to-text
def recognize_speech():
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
            return "Speech recognition service is unavailable."

# Text-to-speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
