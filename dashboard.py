# =====================================================
# CitySense360 - Smart City Control Room (Enhanced UI)
# With Animations, Icons, and Map View
# =====================================================

import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.tools import tool

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CitySense360 | Smart City Control Room",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS WITH ANIMATIONS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
    background-color: #0a0e27;
    color: #fafafa;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 1rem;
}

/* Animated Gradient Background */
.main {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
}

/* Metric Cards with Hover Animation */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1);
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(56, 189, 248, 0.2);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 255, 255, 0.2);
    border-color: rgba(56, 189, 248, 0.5);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(56, 189, 248, 0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
}

@keyframes shine {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.metric-icon {
    font-size: 3rem;
    margin-bottom: 10px;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

.metric-title {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 10px 0;
}

.metric-subtitle {
    font-size: 0.9rem;
    color: #64748b;
}

.section-title {
    font-size: 1.5rem;
    margin: 20px 0 15px 0;
    color: #38bdf8;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::before {
    content: '';
    width: 4px;
    height: 30px;
    background: linear-gradient(180deg, #38bdf8, #0ea5e9);
    border-radius: 2px;
}

/* Alert Badge Animation */
.alert-badge {
    display: inline-block;
    padding: 8px 16px;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    animation: blink 1.5s infinite;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Status Indicator */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
    50% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
}

.status-good { background: #22c55e; }
.status-moderate { background: #facc15; }
.status-poor { background: #ef4444; }

/* Loading Animation */
@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.loading-icon {
    animation: rotate 2s linear infinite;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5);
    transform: translateY(-2px);
}

/* Map Container */
.map-container {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER WITH LIVE STATUS
# =====================================================
col_h1, col_h2 = st.columns([3, 1])

with col_h1:
    st.markdown("## 🌆 **CitySense360 – AI Smart City Control Room**")
    st.caption("🔴 LIVE • Unified Urban Intelligence • Real-time Decision Support")

with col_h2:
    st.markdown(f"""
    <div style='text-align: right; padding-top: 10px;'>
        <span class='status-indicator status-good'></span>
        <span style='color: #22c55e; font-weight: 600;'>System Online</span>
        <div style='color: #64748b; font-size: 0.85rem; margin-top: 5px;'>
            {datetime.now().strftime("%I:%M %p")}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# LOAD NLP MODELS
# =====================================================
@st.cache_resource
def load_nlp():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return classifier, summarizer

classifier, summarizer = load_nlp()

# =====================================================
# MOCKED PREDICTIONS
# =====================================================
def air_quality():
    aqi = random.randint(60, 190)
    status = "Good" if aqi < 100 else "Moderate" if aqi < 150 else "Poor"
    return aqi, status

def traffic():
    return random.choice(["Low", "Moderate", "High"])

def energy():
    return random.choice(["Normal", "Peak Load", "Overload Risk"])

# =====================================================
# KPI CARDS WITH ANIMATIONS
# =====================================================
col1, col2, col3, col4 = st.columns(4)

aqi, aqi_status = air_quality()
traffic_status = traffic()
energy_status = energy()

# Determine status colors
aqi_color = "status-good" if aqi < 100 else "status-moderate" if aqi < 150 else "status-poor"
traffic_icon = "🟢" if traffic_status == "Low" else "🟡" if traffic_status == "Moderate" else "🔴"
energy_icon = "✅" if energy_status == "Normal" else "⚠️" if energy_status == "Peak Load" else "🚨"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🌫️</div>
        <div class="metric-title">Air Quality Index</div>
        <div class="metric-value">{aqi}</div>
        <div class="metric-subtitle">
            <span class='status-indicator {aqi_color}'></span>
            {aqi_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🚦</div>
        <div class="metric-title">Traffic Flow</div>
        <div class="metric-value">{traffic_icon}</div>
        <div class="metric-subtitle">{traffic_status} Congestion</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-title">Smart Grid</div>
        <div class="metric-value">{energy_icon}</div>
        <div class="metric-subtitle">{energy_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    active_alerts = random.randint(2, 8)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🚨</div>
        <div class="metric-title">Active Alerts</div>
        <div class="metric-value">{active_alerts}</div>
        <div class="metric-subtitle">Requires Attention</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# INTERACTIVE CITY MAP
# =====================================================
st.markdown("<div class='section-title'>🗺️ Live City Map</div>", unsafe_allow_html=True)

# Generate mock city data
np.random.seed(42)
city_zones = pd.DataFrame({
    'Zone': [f'Zone {i}' for i in range(1, 21)],
    'Latitude': np.random.uniform(12.9, 13.1, 20),
    'Longitude': np.random.uniform(80.2, 80.3, 20),
    'AQI': np.random.randint(50, 200, 20),
    'Traffic': np.random.choice(['Low', 'Moderate', 'High'], 20),
    'Alerts': np.random.randint(0, 5, 20)
})

# Create color mapping for AQI
city_zones['Color'] = city_zones['AQI'].apply(
    lambda x: 'green' if x < 100 else 'yellow' if x < 150 else 'red'
)

# Create map with Plotly
fig_map = go.Figure()

fig_map.add_trace(go.Scattermapbox(
    lat=city_zones['Latitude'],
    lon=city_zones['Longitude'],
    mode='markers',
    marker=dict(
        size=15,
        color=city_zones['AQI'],
        colorscale='RdYlGn_r',
        showscale=True,
        colorbar=dict(title="AQI Level"),
    ),
    text=city_zones.apply(
        lambda row: f"<b>{row['Zone']}</b><br>AQI: {row['AQI']}<br>Traffic: {row['Traffic']}<br>Alerts: {row['Alerts']}", 
        axis=1
    ),
    hovertemplate='%{text}<extra></extra>'
))

fig_map.update_layout(
    mapbox=dict(
        style='carto-darkmatter',
        center=dict(lat=13.0, lon=80.25),
        zoom=11
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# =====================================================
# ANIMATED TIME-SERIES CHARTS
# =====================================================
st.markdown("<div class='section-title'>📊 Real-Time Urban Trends</div>", unsafe_allow_html=True)

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    # Traffic density chart
    hours = [f"{i:02d}:00" for i in range(24)]
    traffic_data = pd.DataFrame({
        'Hour': hours,
        'Vehicles': np.random.randint(200, 1200, 24)
    })
    
    fig_traffic = px.line(
        traffic_data, 
        x='Hour', 
        y='Vehicles',
        title='🚗 24-Hour Traffic Volume',
        markers=True
    )
    fig_traffic.update_traces(
        line_color='#38bdf8',
        line_width=3,
        marker=dict(size=8, color='#0ea5e9')
    )
    fig_traffic.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        title_font=dict(size=16, color='#38bdf8'),
        xaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
    )
    st.plotly_chart(fig_traffic, use_container_width=True)

with trend_col2:
    # AQI trend chart
    aqi_data = pd.DataFrame({
        'Hour': hours,
        'AQI': np.random.randint(60, 180, 24)
    })
    
    fig_aqi = px.area(
        aqi_data,
        x='Hour',
        y='AQI',
        title='🌫️ Air Quality Index Trend'
    )
    fig_aqi.update_traces(
        line_color='#a78bfa',
        fillcolor='rgba(167, 139, 250, 0.3)'
    )
    fig_aqi.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        title_font=dict(size=16, color='#38bdf8'),
        xaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)')
    )
    st.plotly_chart(fig_aqi, use_container_width=True)

st.divider()

# =====================================================
# ENERGY DISTRIBUTION PIE CHART
# =====================================================
st.markdown("<div class='section-title'>⚡ Energy Distribution</div>", unsafe_allow_html=True)

energy_data = pd.DataFrame({
    'Source': ['Solar', 'Wind', 'Thermal', 'Hydro', 'Nuclear'],
    'Percentage': [25, 20, 30, 15, 10]
})

fig_energy = px.pie(
    energy_data,
    values='Percentage',
    names='Source',
    title='Current Energy Mix',
    color_discrete_sequence=px.colors.sequential.Viridis
)

fig_energy.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8'),
    title_font=dict(size=16, color='#38bdf8')
)

st.plotly_chart(fig_energy, use_container_width=True)

st.divider()

# =====================================================
# COMPLAINT ANALYZER WITH ICONS
# =====================================================
st.markdown("<div class='section-title'>📝 AI Complaint Analyzer</div>", unsafe_allow_html=True)

CATEGORIES = {
    "Traffic Issue": {"dept": "Traffic Department", "icon": "🚦"},
    "Water Supply Issue": {"dept": "Water & Sanitation", "icon": "💧"},
    "Electricity Issue": {"dept": "Electricity Board", "icon": "⚡"},
    "Garbage / Sanitation": {"dept": "Municipal Sanitation", "icon": "🗑️"},
    "Public Safety": {"dept": "Police Department", "icon": "👮"},
    "Environmental Issue": {"dept": "Environment Dept", "icon": "🌳"},
    "Other": {"dept": "Municipal Office", "icon": "🏛️"}
}
LABELS = list(CATEGORIES.keys())

complaint = st.text_area("📢 Enter citizen complaint", height=120, placeholder="Describe the issue...")

if st.button("🔍 Analyze & Route Complaint"):
    if complaint.strip():
        with st.spinner("🤖 AI analyzing complaint..."):
            result = classifier(complaint, candidate_labels=LABELS)
            category = result["labels"][0]
            confidence = round(result["scores"][0], 2)

            if len(complaint.split()) < 25:
                summary = complaint
            else:
                summary = summarizer(complaint, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

        st.success("✅ Complaint Processed Successfully")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            **📋 Summary:**  
            {summary}
            
            **🏷️ Category:**  
            {CATEGORIES[category]['icon']} {category}
            """)
        
        with col_b:
            st.markdown(f"""
            **📊 Confidence:**  
            {confidence * 100:.1f}%
            
            **🏢 Assigned To:**  
            {CATEGORIES[category]['dept']}
            """)
            
    else:
        st.warning("⚠️ Please enter a complaint.")

st.divider()

# =====================================================
# AGENTIC AI REPORT WITH LOADING ANIMATION
# =====================================================
st.markdown("<div class='section-title'>🧠 AI City Operations Brief</div>", unsafe_allow_html=True)

class SimpleCityAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, prompt):
        traffic_status = traffic()
        aqi_val, aqi_stat = air_quality()
        energy_status = energy()
        
        context = f"""
City Status Report:
- Traffic: {traffic_status}
- Air Quality: AQI {aqi_val} ({aqi_stat})
- Energy Grid: {energy_status}

{prompt}

Provide a brief 3-4 sentence summary with priority actions.
"""
        
        try:
            response = self.llm.invoke(context)
            return response
        except Exception as e:
            return f"""
📊 **City Operations Summary:**

Traffic congestion is currently **{traffic_status}**. Air quality monitoring shows AQI of **{aqi_val}** ({aqi_stat}). 
Smart grid reports **{energy_status}** conditions.

**🎯 Priority Actions:**
- Monitor traffic flow in peak hours
- Track air quality sensor data
- Ensure grid stability and load balancing
- Address high-priority citizen complaints
"""

@st.cache_resource
def load_agent():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    llm_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True
    )
    
    llm = HuggingFacePipeline(pipeline=llm_pipe)
    return SimpleCityAgent(llm)

try:
    agent = load_agent()
    agent_loaded = True
except Exception as e:
    st.error(f"⚠️ Agent loading error: {e}")
    agent_loaded = False

if st.button("🧠 Generate AI City Brief"):
    if agent_loaded:
        today = datetime.now().strftime("%d %B %Y")
        prompt = f"""
        Generate a concise city operations briefing.
        Include traffic, air quality, energy status,
        and priority actions.

        Date: {today}
        """
        
        with st.spinner("🤖 AI Agent analyzing city data..."):
            report = agent.run(prompt)
        
        st.info(report)
    else:
        today = datetime.now().strftime("%d %B %Y")
        traffic_status = traffic()
        aqi_val, aqi_stat = air_quality()
        energy_status = energy()
        
        fallback_report = f"""
📊 **City Operations Brief - {today}**

**Current Status:**
- 🚦 Traffic: {traffic_status} congestion levels
- 🌫️ Air Quality: AQI {aqi_val} ({aqi_stat})
- ⚡ Energy Grid: {energy_status}

**🎯 Priority Actions:**
- Monitor high-traffic zones and optimize signal timing
- Track air quality sensors in industrial areas
- Ensure power grid stability during peak hours
- Process and route citizen complaints efficiently
"""
        st.info(fallback_report)

st.divider()

# =====================================================
# FOOTER WITH LIVE CLOCK
# =====================================================
footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:
    st.caption("🌆 CitySense360 • AI-Driven Urban Intelligence Platform • v2.0")

with footer_col2:
    st.caption(f"⏰ Last Updated: {datetime.now().strftime('%I:%M:%S %p')}")