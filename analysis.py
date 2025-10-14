import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_aqi_gauge(aqi_value):
    """
    Create a gauge chart for AQI value.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={'text': "Air Quality Index (AQI)"},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "orange"},
                {'range': [150, 200], 'color': "red"},
                {'range': [200, 300], 'color': "purple"},
                {'range': [300, 500], 'color': "maroon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    return fig

def create_weather_bar_chart(temperature, humidity, wind_speed):
    """
    Create a bar chart for weather metrics.
    """
    data = {
        'Metric': ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)'],
        'Value': [temperature, humidity, wind_speed]
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Metric', y='Value', title="Weather Metrics", color='Metric')
    return fig

def analyze_pollutants(pollutants):
    """
    Analyze and visualize pollutant levels.
    """
    if pollutants:
        poll_data = {k: v['v'] for k, v in pollutants.items() if 'v' in v}
        df = pd.DataFrame(list(poll_data.items()), columns=['Pollutant', 'Level'])
        fig = px.bar(df, x='Pollutant', y='Level', title="Pollutant Levels", color='Pollutant')
        return fig
    return None

def get_aqi_category(aqi):
    """
    Get AQI category based on value.
    """
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"
