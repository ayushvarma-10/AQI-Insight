# Air Quality Index (AQI) Dashboard

A data science project that provides an interactive dashboard to search for any city or region and display its Air Quality Index (AQI) along with weather information. The app uses real-time data from APIs and includes impressive visualizations for better understanding.

## Features

- **City Search**: Enter any city name to fetch AQI and weather data.
- **AQI Display**: Shows AQI value with category (Good, Moderate, etc.) and a gauge chart.
- **Weather Information**: Displays temperature, humidity, wind speed, and description.
- **Visualizations**: Interactive charts for weather metrics and pollutant levels.
- **Data Science Elements**: Uses Pandas for data handling and Plotly for visualizations.

## Technologies Used

- **Python**: Core language.
- **Streamlit**: For building the web app.
- **Requests**: For API calls.
- **Pandas**: For data manipulation.
- **Plotly**: For interactive charts.
- **APIs**: WAQI for AQI data, OpenWeatherMap for weather data.

## Setup Instructions

1. **Clone or Download the Project**:
   - Ensure you have Python installed (version 3.7+ recommended).

2. **Install Dependencies**:
   - Run `pip install -r requirements.txt`

3. **Obtain API Keys**:
   - **WAQI API**: Sign up at [WAQI](https://aqicn.org/api/) to get a free API token.
   - **OpenWeatherMap API**: Sign up at [OpenWeatherMap](https://openweathermap.org/api) to get a free API key.

4. **Create a `.env` File**:
   - In the project root directory, create a file named `.env`.
   - Add your API keys:
     ```
     WAQI_API_KEY=your_waqi_api_key_here
     OPENWEATHER_API_KEY=your_openweather_api_key_here
     ```

5. **Run the App**:
   - Execute `streamlit run app.py`
   - Open the provided local URL in your browser.

## Usage

- Enter a city name in the search bar and click "Search".
- View the AQI gauge, weather metrics, and charts.
- Explore pollutant levels if available.

## Data Sources

- AQI Data: [World Air Quality Index Project (WAQI)](https://aqicn.org/)
- Weather Data: [OpenWeatherMap](https://openweathermap.org/)

## Contributing

Feel free to fork the repository and submit pull requests for improvements.

## License

This project is open-source. Use it as per your needs.
