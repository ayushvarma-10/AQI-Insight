import requests
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

load_dotenv()

# API Keys (set in .env file)
WAQI_API_KEY = os.getenv('WAQI_API_KEY') or 'demo'
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY') or 'fd636018ff4e12c8ba105497ffba6bba'

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
    'bangalore': {
        'aqi': 75,
        'pollutants': {'pm25': {'v': 45}, 'pm10': {'v': 75}, 'o3': {'v': 25}, 'no2': {'v': 20}},
        'city': 'Bangalore, India'
    },
    'chennai': {
        'aqi': 85,
        'pollutants': {'pm25': {'v': 55}, 'pm10': {'v': 85}, 'o3': {'v': 30}, 'no2': {'v': 22}},
        'city': 'Chennai, India'
    },
    'kolkata': {
        'aqi': 110,
        'pollutants': {'pm25': {'v': 80}, 'pm10': {'v': 110}, 'o3': {'v': 40}, 'no2': {'v': 28}},
        'city': 'Kolkata, India'
    },
    'hyderabad': {
        'aqi': 90,
        'pollutants': {'pm25': {'v': 60}, 'pm10': {'v': 90}, 'o3': {'v': 32}, 'no2': {'v': 24}},
        'city': 'Hyderabad, India'
    },
    'pune': {
        'aqi': 80,
        'pollutants': {'pm25': {'v': 50}, 'pm10': {'v': 80}, 'o3': {'v': 28}, 'no2': {'v': 21}},
        'city': 'Pune, India'
    },
    'newyork': {
        'aqi': 45,
        'pollutants': {'pm25': {'v': 15}, 'pm10': {'v': 25}, 'o3': {'v': 20}, 'no2': {'v': 10}},
        'city': 'New York, USA'
    },
    'losangeles': {
        'aqi': 60,
        'pollutants': {'pm25': {'v': 25}, 'pm10': {'v': 35}, 'o3': {'v': 30}, 'no2': {'v': 18}},
        'city': 'Los Angeles, USA'
    },
    'chicago': {
        'aqi': 50,
        'pollutants': {'pm25': {'v': 18}, 'pm10': {'v': 28}, 'o3': {'v': 22}, 'no2': {'v': 12}},
        'city': 'Chicago, USA'
    },
    'houston': {
        'aqi': 55,
        'pollutants': {'pm25': {'v': 20}, 'pm10': {'v': 30}, 'o3': {'v': 25}, 'no2': {'v': 15}},
        'city': 'Houston, USA'
    },
    'london': {
        'aqi': 55,
        'pollutants': {'pm25': {'v': 20}, 'pm10': {'v': 30}, 'o3': {'v': 25}, 'no2': {'v': 15}},
        'city': 'London, UK'
    },
    'manchester': {
        'aqi': 48,
        'pollutants': {'pm25': {'v': 16}, 'pm10': {'v': 26}, 'o3': {'v': 21}, 'no2': {'v': 11}},
        'city': 'Manchester, UK'
    },
    'tokyo': {
        'aqi': 40,
        'pollutants': {'pm25': {'v': 12}, 'pm10': {'v': 20}, 'o3': {'v': 18}, 'no2': {'v': 8}},
        'city': 'Tokyo, Japan'
    },
    'osaka': {
        'aqi': 42,
        'pollutants': {'pm25': {'v': 14}, 'pm10': {'v': 22}, 'o3': {'v': 19}, 'no2': {'v': 9}},
        'city': 'Osaka, Japan'
    },
    'beijing': {
        'aqi': 150,
        'pollutants': {'pm25': {'v': 110}, 'pm10': {'v': 150}, 'o3': {'v': 50}, 'no2': {'v': 40}},
        'city': 'Beijing, China'
    },
    'shanghai': {
        'aqi': 53,
        'pollutants': {'pm25': {'v': 20}, 'pm10': {'v': 30}, 'o3': {'v': 25}, 'no2': {'v': 15}},
        'city': 'Shanghai, China'
    },
    'guangzhou': {
        'aqi': 65,
        'pollutants': {'pm25': {'v': 30}, 'pm10': {'v': 40}, 'o3': {'v': 32}, 'no2': {'v': 20}},
        'city': 'Guangzhou, China'
    },
    'paris': {
        'aqi': 50,
        'pollutants': {'pm25': {'v': 18}, 'pm10': {'v': 28}, 'o3': {'v': 22}, 'no2': {'v': 12}},
        'city': 'Paris, France'
    },
    'lyon': {
        'aqi': 45,
        'pollutants': {'pm25': {'v': 15}, 'pm10': {'v': 25}, 'o3': {'v': 20}, 'no2': {'v': 10}},
        'city': 'Lyon, France'
    },
    'berlin': {
        'aqi': 52,
        'pollutants': {'pm25': {'v': 19}, 'pm10': {'v': 29}, 'o3': {'v': 23}, 'no2': {'v': 13}},
        'city': 'Berlin, Germany'
    },
    'munich': {
        'aqi': 48,
        'pollutants': {'pm25': {'v': 16}, 'pm10': {'v': 26}, 'o3': {'v': 21}, 'no2': {'v': 11}},
        'city': 'Munich, Germany'
    },
    'sydney': {
        'aqi': 35,
        'pollutants': {'pm25': {'v': 10}, 'pm10': {'v': 15}, 'o3': {'v': 15}, 'no2': {'v': 5}},
        'city': 'Sydney, Australia'
    },
    'melbourne': {
        'aqi': 38,
        'pollutants': {'pm25': {'v': 11}, 'pm10': {'v': 16}, 'o3': {'v': 16}, 'no2': {'v': 6}},
        'city': 'Melbourne, Australia'
    },
    'moscow': {
        'aqi': 70,
        'pollutants': {'pm25': {'v': 35}, 'pm10': {'v': 45}, 'o3': {'v': 28}, 'no2': {'v': 25}},
        'city': 'Moscow, Russia'
    },
    'saintpetersburg': {
        'aqi': 65,
        'pollutants': {'pm25': {'v': 30}, 'pm10': {'v': 40}, 'o3': {'v': 26}, 'no2': {'v': 22}},
        'city': 'Saint Petersburg, Russia'
    }
}

# Mock weather data for fallback
MOCK_WEATHER_DATA = {
    'delhi': {
        'temperature': 30.5,
        'humidity': 65,
        'wind_speed': 3.2,
        'description': 'haze',
        'city': 'Delhi'
    },
    'mumbai': {
        'temperature': 28.0,
        'humidity': 75,
        'wind_speed': 4.1,
        'description': 'clear sky',
        'city': 'Mumbai'
    },
    'bangalore': {
        'temperature': 25.8,
        'humidity': 70,
        'wind_speed': 2.5,
        'description': 'partly cloudy',
        'city': 'Bangalore'
    },
    'chennai': {
        'temperature': 32.1,
        'humidity': 78,
        'wind_speed': 3.8,
        'description': 'hot and humid',
        'city': 'Chennai'
    },
    'kolkata': {
        'temperature': 29.4,
        'humidity': 82,
        'wind_speed': 2.9,
        'description': 'humid',
        'city': 'Kolkata'
    },
    'hyderabad': {
        'temperature': 31.2,
        'humidity': 68,
        'wind_speed': 3.1,
        'description': 'sunny',
        'city': 'Hyderabad'
    },
    'pune': {
        'temperature': 26.7,
        'humidity': 72,
        'wind_speed': 2.8,
        'description': 'pleasant',
        'city': 'Pune'
    },
    'newyork': {
        'temperature': 15.2,
        'humidity': 55,
        'wind_speed': 2.8,
        'description': 'few clouds',
        'city': 'New York'
    },
    'losangeles': {
        'temperature': 24.1,
        'humidity': 50,
        'wind_speed': 1.5,
        'description': 'clear sky',
        'city': 'Los Angeles'
    },
    'chicago': {
        'temperature': 8.9,
        'humidity': 62,
        'wind_speed': 4.2,
        'description': 'cold and windy',
        'city': 'Chicago'
    },
    'houston': {
        'temperature': 26.3,
        'humidity': 74,
        'wind_speed': 2.1,
        'description': 'warm',
        'city': 'Houston'
    },
    'london': {
        'temperature': 12.8,
        'humidity': 70,
        'wind_speed': 3.5,
        'description': 'light rain',
        'city': 'London'
    },
    'manchester': {
        'temperature': 10.5,
        'humidity': 75,
        'wind_speed': 4.1,
        'description': 'rainy',
        'city': 'Manchester'
    },
    'tokyo': {
        'temperature': 18.5,
        'humidity': 60,
        'wind_speed': 2.1,
        'description': 'overcast clouds',
        'city': 'Tokyo'
    },
    'osaka': {
        'temperature': 19.8,
        'humidity': 65,
        'wind_speed': 1.8,
        'description': 'cloudy',
        'city': 'Osaka'
    },
    'beijing': {
        'temperature': 22.3,
        'humidity': 45,
        'wind_speed': 1.9,
        'description': 'smoke',
        'city': 'Beijing'
    },
    'shanghai': {
        'temperature': 20.8,
        'humidity': 72,
        'wind_speed': 2.7,
        'description': 'mist',
        'city': 'Shanghai'
    },
    'guangzhou': {
        'temperature': 25.6,
        'humidity': 80,
        'wind_speed': 1.2,
        'description': 'humid',
        'city': 'Guangzhou'
    },
    'paris': {
        'temperature': 16.7,
        'humidity': 68,
        'wind_speed': 2.4,
        'description': 'scattered clouds',
        'city': 'Paris'
    },
    'lyon': {
        'temperature': 14.3,
        'humidity': 71,
        'wind_speed': 2.9,
        'description': 'mild',
        'city': 'Lyon'
    },
    'berlin': {
        'temperature': 11.2,
        'humidity': 69,
        'wind_speed': 3.7,
        'description': 'cool',
        'city': 'Berlin'
    },
    'munich': {
        'temperature': 9.8,
        'humidity': 73,
        'wind_speed': 2.6,
        'description': 'chilly',
        'city': 'Munich'
    },
    'sydney': {
        'temperature': 22.4,
        'humidity': 58,
        'wind_speed': 3.2,
        'description': 'sunny',
        'city': 'Sydney'
    },
    'melbourne': {
        'temperature': 18.9,
        'humidity': 61,
        'wind_speed': 2.8,
        'description': 'mild',
        'city': 'Melbourne'
    },
    'moscow': {
        'temperature': -5.2,
        'humidity': 85,
        'wind_speed': 3.4,
        'description': 'snow',
        'city': 'Moscow'
    },
    'saintpetersburg': {
        'temperature': -3.8,
        'humidity': 88,
        'wind_speed': 4.1,
        'description': 'freezing',
        'city': 'Saint Petersburg'
    }
}

def search_city(city):
    """
    Search for a city station using WAQI API.
    """
    url = f"https://api.waqi.info/search/?keyword={city}&token={WAQI_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
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
    except requests.exceptions.RequestException:
        pass
    return None

def get_aqi_data(city):
    """
    Fetch AQI data for a given city using WAQI API.
    Falls back to mock data if API fails.
    """
    if WAQI_API_KEY and WAQI_API_KEY != 'demo':
        station_id = search_city(city)
        if station_id:
            url = f"https://api.waqi.info/feed/@{station_id}/?token={WAQI_API_KEY}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'ok':
                        aqi_data = data['data']
                        pollutants = aqi_data.get('iaqi', {})
                        return {
                            'aqi': aqi_data.get('aqi', 0),
                            'pollutants': pollutants,
                            'city': aqi_data.get('city', {}).get('name', city)
                        }
            except requests.exceptions.RequestException:
                pass
    # Fallback to mock data
    city_lower = city.lower().replace(' ', '')
    if city_lower in MOCK_AQI_DATA:
        return MOCK_AQI_DATA[city_lower]
    else:
        # Default to Shanghai if city not found
        return MOCK_AQI_DATA['shanghai']

def get_weather_data(city):
    """
    Fetch weather data for a given city using OpenWeatherMap API.
    Falls back to mock data if API fails.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
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
            # Fallback to mock data
            city_lower = city.lower().replace(' ', '')
            if city_lower in MOCK_WEATHER_DATA:
                return MOCK_WEATHER_DATA[city_lower]
            else:
                return MOCK_WEATHER_DATA['shanghai']
    except requests.exceptions.RequestException:
        # Fallback to mock data
        city_lower = city.lower().replace(' ', '')
        if city_lower in MOCK_WEATHER_DATA:
            return MOCK_WEATHER_DATA[city_lower]
        else:
            return MOCK_WEATHER_DATA['shanghai']

def get_historical_aqi_data(city, days=30):
    """
    Fetch historical AQI data for training the ML model.
    Falls back to generated mock data if API fails.
    """
    if WAQI_API_KEY and WAQI_API_KEY != 'demo':
        station_id = search_city(city)
        if station_id:
            historical_data = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                url = f"https://api.waqi.info/feed/@{station_id}/{date_str}/?token={WAQI_API_KEY}"
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data['status'] == 'ok' and 'data' in data:
                            day_data = data['data']
                            if day_data and 'aqi' in day_data:
                                historical_data.append(day_data['aqi'])
                except requests.exceptions.RequestException:
                    pass
                current_date += timedelta(days=1)

            if len(historical_data) >= 24:  # Need minimum data for training
                return historical_data

    # Fallback: Generate mock historical data based on current AQI
    current_aqi = get_aqi_data(city)['aqi']
    np.random.seed(42)  # For reproducible results

    # Generate time series with some noise and trends
    base_aqi = current_aqi
    historical_data = []

    for i in range(days * 24):  # Hourly data for the period
        # Add some random variation and daily patterns
        noise = np.random.normal(0, base_aqi * 0.1)
        daily_pattern = np.sin(2 * np.pi * (i % 24) / 24) * (base_aqi * 0.05)
        trend = np.random.choice([-1, 0, 1]) * (base_aqi * 0.02)

        aqi_value = max(0, base_aqi + noise + daily_pattern + trend)
        historical_data.append(int(aqi_value))

        # Slowly drift the base AQI
        base_aqi += np.random.normal(0, 0.5)

    return historical_data

def fetch_data(city):
    """
    Fetch both AQI and weather data for a city.
    Always returns data using mock data as fallback.
    """
    aqi_data = get_aqi_data(city)
    weather_data = get_weather_data(city)
    # Always return data since we have mock fallbacks
    return {**aqi_data, **weather_data}
