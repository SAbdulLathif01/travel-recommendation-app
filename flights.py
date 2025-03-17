import requests
import json

# Define API credentials
FLIGHT_API_KEY = "ed0c810a50msh1e3b82d8580b49dp15bd57jsnf558c7d483ff"  # Replace with your actual API key
API_URL ="https://skyscanner89.p.rapidapi.com/flights/one-way/list"



# OpenAI API Key (Replace with your actual key)
OPENAI_API_KEY = "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"


# Headers for authentication
headers = {
    "X-RapidAPI-Key": FLIGHT_API_KEY,
    "X-RapidAPI-Host": "skyscanner89.p.rapidapi.com",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

import requests

def get_flight_prices(origin, origin_id, destination, destination_id, date):
    url = "https://skyscanner89.p.rapidapi.com/flights/one-way/list"

    querystring = {
        "date": date,  
        "origin": origin,  
        "originId": origin_id,  
        "destination": destination,  
        "destinationId": destination_id
    }

    headers = {
        "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
        "X-RapidAPI-Host": "skyscanner89.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    try:
        data = response.json()
        print("🔹 API Response:", data)  # Debugging step
        
        if "data" in data and "results" in data["data"]:
            return data["data"]["results"]
        elif "flights" in data:
            return data["flights"]  # Sometimes API returns "flights"
        else:
            print("❌ Unexpected response format:", data)
            return None
    except requests.exceptions.JSONDecodeError:
        print("❌ API returned invalid JSON:", response.text)
        return None
