import requests
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (set in .env file)
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Override with provided key for testing
OPENWEATHER_API_KEY = 'fd636018ff4e12c8ba105497ffba6bba'

# Mock AQI data for demonstration (since WAQI demo token only returns Shanghai data)
MOCK_AQI_DATA = {
    'delhi': {
        'aqi': 120,
        'pollutants': {'pm25': {'v': 85}, 'pm10': {'v': 120}, 'o3': {'v': 45}, 'no2': {'v': 30}},
        'city': 'Delhi, India'
    },
    'mumbai': {
        'aqi': 95,
        'pollutants': {'pm25': {'v': 65}, 'pm10': {'v': 95}, 'o3': {'v': 35}, 'no2': {'v': 25}},
        'city': 'Mumbai, India'
    },
    'newyork': {
        'aqi': 45,
        'pollutants': {'pm25': {'v': 15}, 'pm10': {'v': 25}, 'o3': {'v': 20}, 'no2': {'v': 10}},
        'city': 'New York, USA'
    },
    'london': {
        'aqi': 55,
        'pollutants': {'pm25': {'v': 20}, 'pm10': {'v': 30}, 'o3': {'v': 25}, 'no2': {'v': 15}},
        'city': 'London, UK'
    },
    'tokyo': {
        'aqi': 40,
        'pollutants': {'pm25': {'v': 12}, 'pm10': {'v': 20}, 'o3': {'v': 18}, 'no2': {'v': 8}},
        'city': 'Tokyo, Japan'
    },
    'beijing': {
        'aqi': 150,
        'pollutants': {'pm25': {'v': 110}, 'pm10': {'v': 150}, 'o3': {'v': 50}, 'no2': {'v': 40}},
        'city': 'Beijing, China'
    },
    'paris': {
        'aqi': 50,
        'pollutants': {'pm25': {'v': 18}, 'pm10': {'v': 28}, 'o3': {'v': 22}, 'no2': {'v': 12}},
        'city': 'Paris, France'
    },
    'losangeles': {
        'aqi': 60,
        'pollutants': {'pm25': {'v': 25}, 'pm10': {'v': 35}, 'o3': {'v': 30}, 'no2': {'v': 18}},
        'city': 'Los Angeles, USA'
    },
    'shanghai': {
        'aqi': 53,
        'pollutants': {'pm25': {'v': 20}, 'pm10': {'v': 30}, 'o3': {'v': 25}, 'no2': {'v': 15}},
        'city': 'Shanghai, China'
    }
}

def search_city(city):
    """
    Search for a city station using WAQI API.
    """
    url = f"https://api.waqi.info/search/?keyword={city}&token={WAQI_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'ok' and data['data']:
            # Find the best match by checking if city name contains the input
            for station in data['data']:
                station_name = station['station']['name'].lower()
                if city.lower() in station_name and 'bangalore' not in station_name:
                    return station['uid']
            # If no exact match, return the first one that doesn't contain 'bangalore'
            for station in data['data']:
                if 'bangalore' not in station['station']['name'].lower():
                    return station['uid']
            # Fallback to first one
            return data['data'][0]['uid']
    return None

def get_aqi_data(city):
    """
    Fetch AQI data for a given city using mock data for demonstration.
    In production, replace with real API calls.
    """
    city_lower = city.lower().replace(' ', '')
    if city_lower in MOCK_AQI_DATA:
        return MOCK_AQI_DATA[city_lower]
    else:
        # Default to Shanghai if city not found
        return MOCK_AQI_DATA['shanghai']

def get_weather_data(city):
    """
    Fetch weather data for a given city using OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description'],
            'city': data['name']
        }
    else:
        return None

def fetch_data(city):
    """
    Fetch both AQI and weather data for a city.
    """
    aqi_data = get_aqi_data(city)
    weather_data = get_weather_data(city)
    if aqi_data and weather_data:
        return {**aqi_data, **weather_data}
    else:
        return None
