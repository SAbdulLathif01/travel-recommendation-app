🌍 Travel Assistant - AI Chatbot with Voice Assistant 🎙️🤖


📌 Project Overview
The Travel Assistant is a Streamlit-powered web application that integrates multiple travel-related services into a single platform. This app allows users to:

🔍 Search Flights using an API to fetch real-time airline prices.
🏨 Get Hotel Recommendations based on sentiment analysis and similarity matching.
🌤️ Check Weather Updates for any travel destination.
🤖 Chat with an AI Chatbot to get answers to general queries.
🎙️ Use a Voice Assistant to interact with the chatbot using voice commands.
This end-to-end travel companion simplifies trip planning and ensures a seamless experience for travelers.

🎯 Features & Functionality
✈️ Flight Search
🚀 Search real-time flights using the Skyscanner API. Users can enter:

Origin & Destination Airport Codes
Departure Date
Get real-time flight details & prices

🏨 Hotel Recommendations
🔍 Find the best hotels based on:

User-selected city
Travel theme (Adventure, Luxury, Budget, etc.)
Hotel reviews analyzed with sentiment analysis
TF-IDF & cosine similarity ensure the best matches

🌤️ Weather Forecast
☀️ Get real-time weather updates for any travel destination:

Uses OpenWeather API
Displays temperature, humidity, and conditions
Helps plan your trip accordingly
🤖 AI Chatbot
💬 Chatbot powered by Hugging Face's facebook/blenderbot-400M-distill

Users can ask any general question
Provides quick and accurate responses
Integrated directly into Streamlit UI

🎙️ Voice Assistant
🎤 Speak to the AI Chatbot instead of typing!

Uses Speech-to-Text (SpeechRecognition)
Uses Text-to-Speech (pyttsx3)
Helps users interact hands-free

🛠️ Tech Stack & Libraries
Feature	Library/Technology Used
Web Framework	Streamlit
Flight Search API	requests (Skyscanner API)
Hotel Recommendation	pandas, scikit-learn (TF-IDF, cosine similarity)
Weather API	requests (OpenWeather API)
AI Chatbot	transformers, torch (Hugging Face blenderbot-400M-distill)
Voice Assistant	SpeechRecognition (STT), pyttsx3 (TTS)
Database (Authentication)	MongoDB (pymongo)
📦 Installation Guide
1️⃣ Clone the Repository
sh
Copy
Edit
git clone https://github.com/yourusername/travel-assistant.git
cd travel-assistant
2️⃣ Create a Virtual Environment
🔹 Windows
sh
Copy
Edit
python -m venv venv
venv\Scripts\activate
🔹 macOS / Linux
sh
Copy
Edit
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the Application
sh
Copy
Edit
streamlit run app.py
📜 API Keys Setup
For this project to work, you need API keys from:

Skyscanner API - Get it here
OpenWeather API - Get it here
Once you have the keys, add them to the app.py file:

python
Copy
Edit
FLIGHT_API_KEY = "your_skyscanner_api_key"
WEATHER_API_KEY = "your_openweather_api_key"
