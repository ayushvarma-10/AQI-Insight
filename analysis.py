import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

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

def create_aqi_prediction_chart(actual_aqi, predicted_aqi, city_name):
    """
    Create a line chart showing actual vs predicted AQI values.
    """
    # Create time indices for the next 24 hours
    current_time = datetime.now()
    time_labels = [(current_time + timedelta(hours=i)).strftime('%H:%M') for i in range(25)]

    # Create the figure
    fig = go.Figure()

    # Add actual AQI point
    fig.add_trace(go.Scatter(
        x=[time_labels[0]],
        y=[actual_aqi],
        mode='markers+text',
        name='Current AQI',
        text=[f'Current: {actual_aqi}'],
        textposition="top center",
        marker=dict(size=12, color='red', symbol='diamond'),
        showlegend=True
    ))

    # Add predicted AQI line
    fig.add_trace(go.Scatter(
        x=time_labels[1:],
        y=predicted_aqi,
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='blue', width=3, dash='dash'),
        marker=dict(size=6, color='blue'),
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        title=f"AQI Forecast for {city_name} (Next 24 Hours)",
        xaxis_title="Time (Hours)",
        yaxis_title="AQI Value",
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[0, max(max(predicted_aqi), actual_aqi) * 1.2]),
        template="plotly_white",
        height=400
    )

    # Add AQI category zones
    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="Good")
    fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0, annotation_text="Moderate")
    fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, layer="below", line_width=0, annotation_text="Unhealthy for Sensitive")
    fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Unhealthy")
    fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, layer="below", line_width=0, annotation_text="Very Unhealthy")
    fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, layer="below", line_width=0, annotation_text="Hazardous")

    return fig

def create_prediction_accuracy_chart(actual_history, predicted_history):
    """
    Create a chart showing prediction accuracy over time.
    """
    if len(actual_history) != len(predicted_history):
        return None

    # Calculate errors
    errors = np.abs(np.array(actual_history) - np.array(predicted_history))

    # Create time labels (last N hours)
    time_labels = [f'T-{len(errors)-i}' for i in range(len(errors))]

    fig = go.Figure()

    # Add actual vs predicted lines
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=actual_history,
        mode='lines+markers',
        name='Actual AQI',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=time_labels,
        y=predicted_history,
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    # Add error bars
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=errors,
        mode='lines',
        name='Prediction Error',
        line=dict(color='red', width=1),
        yaxis="y2"
    ))

    # Update layout with secondary y-axis
    fig.update_layout(
        title="Model Prediction Accuracy (Last 24 Hours)",
        xaxis_title="Time (Hours ago)",
        yaxis_title="AQI Value",
        yaxis2=dict(
            title="Prediction Error",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        height=400,
        legend=dict(x=0.02, y=0.98)
    )

    return fig

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
