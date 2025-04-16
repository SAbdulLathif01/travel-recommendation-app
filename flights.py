import requests

# API credentials
FLIGHT_API_KEY = "ed0c810a50msh1e3b82d8580b49dp15bd57jsnf558c7d483ff"
API_HOST = "skyscanner89.p.rapidapi.com"

# Headers
HEADERS = {
    "X-RapidAPI-Key": FLIGHT_API_KEY,
    "X-RapidAPI-Host": API_HOST
}

def get_flight_prices(origin, origin_id, destination, destination_id, date):
    """Get one-way flight prices from Skyscanner API"""
    url = f"https://{API_HOST}/flights/one-way/list"
    querystring = {
        "origin": origin,
        "originId": origin_id,
        "destination": destination,
        "destinationId": destination_id,
        "date": date
    }

    try:
        response = requests.get(url, headers=HEADERS, params=querystring)
        response.raise_for_status()
        data = response.json()

        # Debug output
        print("🟦 API Response (short):", list(data.get("data", {}).keys()))

        itineraries = data.get("data", {}).get("itineraries", {}).get("buckets", [])
        if not itineraries:
            return {"error": "No itineraries found"}

        flights = []
        for bucket in itineraries:
            for item in bucket.get("items", [])[:5]:
                leg = item.get("legs", [])[0]
                airline = leg.get("carriers", {}).get("marketing", [{}])[0].get("name", "Unknown")
                departure = leg.get("departure", "Unknown")
                price = item.get("price", {}).get("formatted", "N/A")

                flights.append({
                    "Airline": airline,
                    "Departure Time": departure,
                    "Price": price
                })

        return flights if flights else {"error": "No flight options found."}

    except Exception as e:
        return {"error": str(e)}
