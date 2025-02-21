import os
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

###############################################################################
# API Keys & Configurations
###############################################################################
WEATHER_API_KEY = "bf1c55bcdac77159cd0d3062f2b90613"
FLIGHT_API_KEY = "b07898df29ca04fb0df91f9d64765a2f"
UNSPLASH_API_KEY = "Wb2t4pUrurt5nqyv6OnDz3zfEztDKzirlbThtaRa9b8"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")

###############################################################################
# Load Dataset for Hotel Recommendations
###############################################################################
file_path = "Hotel_Reviews.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path)

# Rename columns for consistency
df.rename(columns={"Hotel_Name": "Hotel", "Positive_Review": "Review", "Reviewer_Score": "Rating"}, inplace=True)

# Extract Country from "Hotel_Address"
df["Country"] = df["Hotel_Address"].apply(lambda x: x.split()[-1] if isinstance(x, str) else "N/A")

# Derive Theme based on "Tags"
def derive_theme(tags):
    tags = str(tags).lower()
    if "luxury" in tags:
        return "luxury"
    elif "budget" in tags or "cheap" in tags:
        return "budget"
    elif "adventure" in tags or "explore" in tags:
        return "adventure"
    else:
        return "general"
df["Theme"] = df["Tags"].apply(derive_theme)

# Sample 5000 rows for performance
df = df.sample(n=5000, random_state=42)
df.dropna(subset=["Review", "Rating", "Country"], inplace=True)
df.reset_index(drop=True, inplace=True)
df["rec_id"] = df.index

# Generate price estimates based on rating
df["Price"] = df["Rating"].apply(lambda r: f"${100 + int(r) * 10} (Approx)")

# Compute TF-IDF similarity for reviews
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["Review"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

###############################################################################
# API Functions: Weather, Flight Prices, Hotel Images
###############################################################################
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        r = requests.get(url)
        data = r.json()
        return f"{data['main']['temp']}°C, {data['weather'][0]['description']}" if "main" in data else "N/A"
    except Exception:
        return "N/A"

def get_flight_prices(origin, destination, date):
    url = f"https://api.flightapi.io/onewaytrip/{FLIGHT_API_KEY}/{origin}/{destination}/{date}/1/0/0/Economy/USD"
    try:
        r = requests.get(url)
        data = r.json()
        return f"${data['data'][0]['price']}" if "data" in data else "N/A"
    except Exception:
        return "N/A"

def get_hotel_image(hotel_name):
    url = f"https://api.unsplash.com/search/photos?query={hotel_name}&client_id={UNSPLASH_API_KEY}&per_page=1"
    try:
        r = requests.get(url)
        data = r.json()
        return data["results"][0]["urls"]["small"] if "results" in data and len(data["results"]) > 0 else None
    except Exception:
        return None

###############################################################################
# Hotel Recommendation Function (Filter by Country & Theme)
###############################################################################
def get_best_hotels(country, theme, top_n=5):
    filtered_df = df[(df["Country"].str.contains(country, case=False, na=False)) & (df["Theme"] == theme)]
    if filtered_df.empty:
        filtered_df = df[df["Country"].str.contains(country, case=False, na=False)]
    filtered_df = filtered_df.sort_values(by="Rating", ascending=False).head(top_n)
    
    recommendations = []
    for _, row in filtered_df.iterrows():
        recommendations.append({
            "Hotel": row["Hotel"],
            "Rating": float(row["Rating"]),
            "Review": row["Review"],
            "Price": row["Price"],
            "Country": row["Country"],
            "Theme": row["Theme"],
            "Image": get_hotel_image(row["Hotel"]),
            "Weather": get_weather(row["Country"]),
            "Flight": get_flight_prices("NYC", row["Country"], "2025-12-01")
        })
    return recommendations

###############################################################################
# Flask App Setup
###############################################################################
app = Flask(_name_)

###############################################################################
# RAG Chatbot Setup
###############################################################################
wikipedia_data = load_dataset("wikipedia", "20220301.simple", split="train")
wiki_subset = wikipedia_data.select(range(1000))
wiki_docs = [entry["text"][:500] for entry in wiki_subset]

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_faiss = FAISS.from_texts(texts=wiki_docs, embedding=embeddings_model)
retriever = db_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

###############################################################################
# API Endpoints
###############################################################################
@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    data = request.json or {}
    country = data.get("country", "United Kingdom")
    theme = data.get("theme", "luxury")
    results = get_best_hotels(country, theme, top_n=5)
    return jsonify(results)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json or {}
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"answer": "No query provided."})
    response = qa_chain.invoke(user_query)
    return jsonify({"answer": str(response)})

###############################################################################
# UI Endpoint
###############################################################################
@app.route("/")
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <title>Travel Recommendation App</title>
    </head>
    <body>
        <h1>Search Hotels</h1>
        <label>Country:</label>
        <input type='text' id='country' value='United Kingdom'><br>
        <label>Theme:</label>
        <select id='theme'>
            <option value='luxury'>Luxury</option>
            <option value='budget'>Budget</option>
            <option value='adventure'>Adventure</option>
            <option value='general'>General</option>
        </select>
        <button onclick='fetchRecommendations()'>Search</button>
        <div id='recommendations'></div>

        <h2>Chatbot</h2>
        <input type='text' id='query' placeholder='Ask something...'>
        <button onclick='fetchChat()'>Ask</button>
        <pre id='response'></pre>

        <script>
        async function fetchRecommendations() {
            let country = document.getElementById('country').value;
            let theme = document.getElementById('theme').value;
            let res = await fetch('/recommend', { method: 'POST', headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify({ country, theme })});
            let data = await res.json();
            document.getElementById("recommendations").innerText = JSON.stringify(data, null, 2);
        }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if _name_ == '_main_':
    app.run(debug=True)
