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

def create_health_impact_recommendations(aqi_value, category):
    """
    Generate health impact recommendations based on AQI.
    """
    recommendations = {
        "Good": {
            "impact": "Air quality is satisfactory, and air pollution poses little or no risk.",
            "advice": "No special precautions needed. Enjoy outdoor activities!",
            "groups": "General population"
        },
        "Moderate": {
            "impact": "Air quality is acceptable; however, there may be a risk for some people.",
            "advice": "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "groups": "People with respiratory or heart conditions, children, and older adults"
        },
        "Unhealthy for Sensitive Groups": {
            "impact": "Members of sensitive groups may experience health effects.",
            "advice": "Reduce prolonged or heavy outdoor exertion. Consider rescheduling outdoor activities.",
            "groups": "Children, older adults, and people with heart or lung conditions"
        },
        "Unhealthy": {
            "impact": "Everyone may begin to experience health effects.",
            "advice": "Avoid prolonged outdoor exertion. Reschedule outdoor activities to times when air quality is better.",
            "groups": "General population"
        },
        "Very Unhealthy": {
            "impact": "Health alert: everyone may experience more serious health effects.",
            "advice": "Avoid all physical activity outdoors. Stay indoors and keep activity levels low.",
            "groups": "General population, especially children and people with respiratory conditions"
        },
        "Hazardous": {
            "impact": "Health warning of emergency conditions. The entire population is more likely to be affected.",
            "advice": "Remain indoors and keep doors and windows closed. Use air purifiers if available.",
            "groups": "Everyone"
        }
    }

    return recommendations.get(category, recommendations["Good"])

def create_seasonal_trends_chart(historical_data, city_name):
    """
    Create a chart showing seasonal AQI trends.
    """
    if not historical_data or len(historical_data) < 7:
        return None

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(historical_data)
    df['date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
    df['month'] = df['date'].dt.month_name()
    df['aqi'] = df['aqi'].astype(float)

    # Group by month and calculate statistics
    monthly_stats = df.groupby('month')['aqi'].agg(['mean', 'min', 'max']).reset_index()

    # Reorder months chronologically
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_stats['month'] = pd.Categorical(monthly_stats['month'], categories=month_order, ordered=True)
    monthly_stats = monthly_stats.sort_values('month')

    fig = go.Figure()

    # Add mean line
    fig.add_trace(go.Scatter(
        x=monthly_stats['month'],
        y=monthly_stats['mean'],
        mode='lines+markers',
        name='Average AQI',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    # Add range area
    fig.add_trace(go.Scatter(
        x=monthly_stats['month'],
        y=monthly_stats['max'],
        mode='lines',
        name='Max AQI',
        line=dict(color='red', width=1, dash='dot'),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=monthly_stats['month'],
        y=monthly_stats['min'],
        mode='lines',
        name='Min AQI',
        line=dict(color='green', width=1, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Seasonal AQI Trends for {city_name}",
        xaxis_title="Month",
        yaxis_title="AQI Value",
        template="plotly_white",
        height=400,
        xaxis=dict(tickangle=45)
    )

    return fig

def create_comparative_analysis_chart(cities_data):
    """
    Create a comparative chart showing AQI across multiple cities.
    """
    if not cities_data:
        return None

    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, (city, data) in enumerate(cities_data.items()):
        fig.add_trace(go.Bar(
            name=city,
            x=['AQI', 'PM2.5', 'PM10', 'O3', 'NO2'],
            y=[
                data.get('aqi', 0),
                data.get('pollutants', {}).get('pm25', {}).get('v', 0),
                data.get('pollutants', {}).get('pm10', {}).get('v', 0),
                data.get('pollutants', {}).get('o3', {}).get('v', 0),
                data.get('pollutants', {}).get('no2', {}).get('v', 0)
            ],
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        title="Comparative Air Quality Analysis",
        xaxis_title="Pollutant",
        yaxis_title="Concentration",
        barmode='group',
        template="plotly_white",
        height=400
    )

    return fig

def create_prediction_confidence_chart(predictions, lower_bounds, upper_bounds, city_name):
    """
    Create a chart showing predictions with confidence intervals.
    """
    # Create time indices for the next 24 hours
    current_time = datetime.now()
    time_labels = [(current_time + timedelta(hours=i)).strftime('%H:%M') for i in range(1, len(predictions)+1)]

    fig = go.Figure()

    # Add confidence interval area
    fig.add_trace(go.Scatter(
        x=time_labels + time_labels[::-1],
        y=list(upper_bounds) + list(lower_bounds)[::-1],
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))

    # Add mean prediction line
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=predictions,
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='blue', width=3),
        marker=dict(size=6, color='blue')
    ))

    # Update layout
    fig.update_layout(
        title=f"AQI Forecast with Confidence Intervals for {city_name}",
        xaxis_title="Time (Hours)",
        yaxis_title="AQI Value",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        height=400
    )

    # Add AQI category zones
    fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, layer="below", line_width=0)

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
