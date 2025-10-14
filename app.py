import streamlit as st
import pandas as pd
from data_fetch import fetch_data
from analysis import create_aqi_gauge, create_weather_bar_chart, analyze_pollutants, get_aqi_category

st.set_page_config(page_title="Air Quality Index Dashboard", page_icon="🌍", layout="wide")

st.title("🌍 Air Quality Index (AQI) Dashboard")
st.markdown("Search for any city or region to view its Air Quality Index and weather information.")

# Search bar
city = st.text_input("Enter City Name:", placeholder="e.g., Delhi, New York, Tokyo")

if st.button("Search"):
    if city:
        with st.spinner("Fetching data..."):
            data = fetch_data(city)
        if data:
            st.success(f"Data for {data['city']} retrieved successfully!")

            # Display AQI and category
            aqi = data['aqi']
            category, color = get_aqi_category(aqi)
            st.subheader(f"AQI: {aqi} - {category}")
            st.markdown(f"<h3 style='color:{color};'>Category: {category}</h3>", unsafe_allow_html=True)

            # AQI Gauge
            st.plotly_chart(create_aqi_gauge(aqi), use_container_width=True)

            # Weather Information
            st.subheader("Weather Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature", f"{data['temperature']} °C")
            with col2:
                st.metric("Humidity", f"{data['humidity']} %")
            with col3:
                st.metric("Wind Speed", f"{data['wind_speed']} m/s")

            st.write(f"**Description:** {data['description'].capitalize()}")

            # Weather Bar Chart
            st.plotly_chart(create_weather_bar_chart(data['temperature'], data['humidity'], data['wind_speed']), use_container_width=True)

            # Pollutant Analysis
            if 'pollutants' in data and data['pollutants']:
                st.subheader("Pollutant Levels")
                poll_fig = analyze_pollutants(data['pollutants'])
                if poll_fig:
                    st.plotly_chart(poll_fig, use_container_width=True)
                else:
                    st.write("No pollutant data available.")
        else:
            st.error("Unable to fetch data for the entered city. Please check the city name or try again later.")
    else:
        st.warning("Please enter a city name.")

st.markdown("---")
st.markdown("**Note:** This app uses real-time data from WAQI and OpenWeatherMap APIs. Ensure you have valid API keys set in a `.env` file.")
