import streamlit as st
st.set_page_config(page_title="Travel Assistant", page_icon="✈️", layout="wide")
import pandas as pd
import requests
from pymongo import MongoClient
import bcrypt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flights import get_flight_prices 
from chatbot import ask_chatbot 
from chatbot import ask_chatbot, recognize_speech, text_to_speech


# 🔹 API Keys
FLIGHT_API_KEY = "ed0c810a50msh1e3b82d8580b49dp15bd57jsnf558c7d483ff"
WEATHER_API_KEY = "dc8df4de7be0108b91ae7a6769ca8713"


# 🔹 Path to Dataset
dataset_path = r"C:\Users\lathif\Downloads\archive (21)\Hotel_Reviews.csv"

# 🔹 MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["travel_db"]
users_collection = db["users"]

# 🔹 Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv(dataset_path).sample(n=20000, random_state=42)
    text_cols = ["Hotel_Name", "Positive_Review", "Negative_Review", "Hotel_Address"]
    numeric_cols = ["Reviewer_Score", "Review_Total_Positive_Word_Counts", "Review_Total_Negative_Word_Counts"]

    for col in text_cols:
        df[col] = df[col].fillna("Unknown")
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    return df

df = load_data()

# 🔹 Sentiment Analysis
@st.cache_data
def compute_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["Positive_Sentiment"] = df["Positive_Review"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["Negative_Sentiment"] = df["Negative_Review"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

df = compute_sentiment(df)

# 🔹 User Authentication
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False
    hashed_pw = hash_password(password)
    users_collection.insert_one({"username": username, "password": hashed_pw})
    return True

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    return user and check_password(password, user["password"])

# 🔹 Improved Recommendation System
@st.cache_data
def train_vectorizer():
    vectorizer = TfidfVectorizer(stop_words='english')
    df["combined_features"] = (
        df["Hotel_Name"] + " " +
        df["Positive_Review"] + " " +
        df["Negative_Review"] + " " +
        df["Hotel_Address"] + " " +
        df["Reviewer_Score"].astype(str)
    )
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = train_vectorizer()



def get_hotel_image(hotel_name):
    import hashlib
    import random

    # Sample keywords that reflect hotel vibes
    sample_keywords = [
        "luxury-hotel", "beach-resort", "urban-hotel", "city-view-room",
        "suite-interior", "tropical-hotel", "europe-hotel",
        "skyline-room", "mountain-resort", "vintage-hotel"
    ]

    # Create a consistent index using a hash of the hotel name
    idx = int(hashlib.sha256(hotel_name.encode('utf-8')).hexdigest(), 16) % len(sample_keywords)
    keyword = sample_keywords[idx]

    # Use picsum.photos with seed for consistent but different images
    seed = hashlib.md5(hotel_name.encode()).hexdigest()
    return f"https://picsum.photos/seed/{seed}/600/400"


import hashlib
import streamlit as st
def get_flight_image(airline_name):
    airline_images = {
        "hawaiian airlines": "https://upload.wikimedia.org/wikipedia/commons/6/6c/Hawaiian_Airlines_A330_N380HA.jpg",
        "alaska airlines": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Alaska_Airlines_Boeing_737-800.jpg",
        "jetblue": "https://upload.wikimedia.org/wikipedia/commons/1/1c/JetBlue_Airways_Airbus_A320_N595JB.jpg",
        "delta air lines": "https://upload.wikimedia.org/wikipedia/commons/f/fb/Delta_Air_Lines_Boeing_757_N67171.jpg",
        "united airlines": "https://upload.wikimedia.org/wikipedia/commons/6/6f/United_Airlines_Boeing_787.jpg",
        "american airlines": "https://upload.wikimedia.org/wikipedia/commons/2/24/American_Airlines_Boeing_737-800_N973AN.jpg",
        "southwest airlines": "https://upload.wikimedia.org/wikipedia/commons/7/7d/Southwest_Airlines_Boeing_737.jpg",
        "air canada": "https://upload.wikimedia.org/wikipedia/commons/6/6a/Air_Canada_Boeing_787.jpg",
        "british airways": "https://upload.wikimedia.org/wikipedia/commons/2/2e/British_Airways_A380_G-XLEA.jpg",
        "emirates": "https://upload.wikimedia.org/wikipedia/commons/0/07/Emirates_A380.jpg"
    }

    normalized_name = airline_name.strip().lower()
    return airline_images.get(normalized_name, "https://upload.wikimedia.org/wikipedia/commons/e/e0/Airplane_icon.png")




def display_flights(flights):
    for flight in flights:
        airline_name = flight['Airline']
        img_url = get_flight_image(airline_name)

        st.image(img_url, width=600, caption=f"🛫 {airline_name}")

        st.markdown(
            f"""
            <div style="padding:12px; background-color:#111; border-radius:10px; margin-top:-10px; color:#eee;">
                <p>🕒 <strong>Departure:</strong> {flight['Departure Time']}</p>
                <p>💸 <strong>Price:</strong> {flight['Price']}</p>
            </div>
            """, unsafe_allow_html=True
        )


def recommend_hotels(city, theme):
    city_data = df[df["Hotel_Address"].str.contains(city, case=False, na=False)].copy()
    
    if city_data.empty:
        st.error(f"❌ No data found for city: {city}")
        return None

    theme_vector = vectorizer.transform([theme])
    similarity_scores = cosine_similarity(theme_vector, tfidf_matrix[df.index.isin(city_data.index)]).flatten()

    city_data["Similarity_Score"] = similarity_scores
    city_data["Sentiment_Score"] = city_data["Positive_Sentiment"] - abs(city_data["Negative_Sentiment"])
    city_data["Final_Score"] = city_data["Similarity_Score"] + (city_data["Sentiment_Score"] * 0.5)

    recommended_hotels = city_data.nlargest(10, "Final_Score")[["Hotel_Name", "Reviewer_Score", "Hotel_Address"]].drop_duplicates()

    if recommended_hotels.empty:
        st.warning("⚠️ No recommendations available for this theme!")
        return None

    return recommended_hotels

# 🔹 API Integration
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather_desc = data["weather"][0]["description"].title()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]

        weather_report = (
            f"🌤️ **Weather in {city}**\n\n"
            f"**Condition:** {weather_desc}\n"
            f"**Temperature:** {temp}°C (feels like {feels_like}°C)\n"
            f"**Humidity:** {humidity}%\n"
            f"**Wind Speed:** {wind_speed} m/s"
        )

        return weather_report
    else:
        return "❌ Couldn't fetch weather data. Please check the city name."



st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1146/1146890.png", width=100)
st.sidebar.title("🌍 Travel Assistant")
menu = st.sidebar.radio("Navigation", ["Login", "Register", "Dashboard","AI Chatbot"], key="navigation_radio")

if menu == "AI Chatbot":
    st.switch_page("pages/Chatbot.py")

# 🔹 Login Page
if menu == "Login":
    st.title("🔐 Login to Travel Assistant")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid Credentials")

# 🔹 Register Page
elif menu == "Register":
    st.title("📝 Register a New Account")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    
    if st.button("Create Account"):
        if register_user(new_username, new_password):
            st.success("✅ Account Created! Please Login.")
        else:
            st.error("❌ Username already exists!")

# 🔹 Dashboard Page (Contains Flight Search)
elif menu == "Dashboard":
    if "authenticated" in st.session_state:
        st.title("🏠 Travel Assistant Dashboard")

        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("🌆 Select a City", df["Hotel_Address"].str.split().str[-1].unique())
        with col2:
            theme = st.selectbox("🎭 Select a Theme", ["Adventure", "Food", "Luxury", "Budget"])
            
    if "show_flights" not in st.session_state:
        st.session_state["show_flights"] = False

# Button 1: Recommendations
    if st.button("🔎 Get Recommendations"):
        recommendations = recommend_hotels(city, theme)
        if recommendations is not None:
            st.success("✅ Here are the best hotels for your selection!")
            for idx, row in recommendations.iterrows():
                col1, col2 = st.columns([1, 2])
                with col1:
                    image_url = get_hotel_image(row['Hotel_Name'])
                    st.image(image_url, use_container_width=True)
                    with col2:
                        st.markdown(
                            f"""<div style="padding: 0.5rem; background-color: #1c1c1c; border-radius: 10px;">
                            <h4 style="color:#fafafa;">🏨 {row['Hotel_Name']}</h4>
                            <p style="color:#cccccc;">
                            📍 <strong>Location:</strong> {row['Hotel_Address']}<br>
                            ⭐ <strong>Reviewer Score:</strong> {row['Reviewer_Score']}
                            </p></div>""",
                            unsafe_allow_html=True
                            )

# Button 2: Weather
    if st.button("☁️ Get Weather"):
        weather_data = get_weather(city)
        if weather_data:
            st.success(f"✅ Weather in {city}")
            st.markdown(
                f"""<div style="padding: 1rem; background-color: #1e1e2f; border-radius: 10px; color: #f5f5f5;">
                <h4 style="margin-bottom: 0.8rem;">🌤️ <u>Current Weather Summary</u></h4>
                <pre style="white-space: pre-wrap;">{weather_data}</pre></div>""",
                unsafe_allow_html=True
                )
        # 🔄 Set flag to show flight section
            st.session_state["show_flights"] = True
            
            
    if st.session_state["show_flights"]:
        st.subheader("✈️ Check Flight Prices")
        origin = st.text_input("📍 Enter Origin Airport Code (e.g., JFK)")
        origin_id = st.text_input("Enter Origin ID", "27537542")
        destination = st.text_input("🏙️ Enter Destination Airport Code (e.g., LAX)")
        destination_id = st.text_input("Enter Destination ID", "95673827")
        date = st.date_input("📅 Select Departure Date").strftime("%Y-%m-%d")
        
        if st.button("🛫 Find Flights"):
            flights = get_flight_prices(origin, origin_id, destination, destination_id, date)
            
            if isinstance(flights, dict) and "error" in flights:
                st.error(f"❌ {flights['error']}")
            elif isinstance(flights, list):
                display_flights(flights)
                
            else:
                st.warning("⚠️ No flight data available.")
    
    
    
    
   
        
