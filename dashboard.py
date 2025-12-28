# =====================================================
# CitySense360 - Smart City Control Room (Enhanced UI)
# =====================================================

import streamlit as st
import random
import numpy as np
import pandas as pd
from datetime import datetime

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
# CUSTOM CSS (THIS IS THE MAGIC)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: linear-gradient(145deg, #1f2933, #111827);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 20px rgba(0,255,255,0.08);
    text-align: center;
}
.metric-title {
    font-size: 0.9rem;
    color: #9ca3af;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
}
.section-title {
    font-size: 1.3rem;
    margin-bottom: 10px;
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("## 🌆 **CitySense360 – AI Smart City Control Room**")
st.caption("Unified Urban Intelligence • Real-time Decision Support")

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
# MOCKED PREDICTIONS (HOOK REAL MODELS LATER)
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
# KPI CARDS
# =====================================================
col1, col2, col3 = st.columns(3)

aqi, aqi_status = air_quality()

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🌫 Air Quality Index</div>
        <div class="metric-value">{aqi}</div>
        <div>{aqi_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🚦 Traffic Congestion</div>
        <div class="metric-value">{traffic()}</div>
        <div>Live CCTV Estimation</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⚡ Smart Grid Load</div>
        <div class="metric-value">{energy()}</div>
        <div>City Power Network</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# TIME-SERIES VISUALS
# =====================================================
st.markdown("<div class='section-title'>📊 Urban Trends</div>", unsafe_allow_html=True)

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    data = pd.DataFrame({
        "Traffic Density": np.random.randint(20, 120, 30)
    })
    st.line_chart(data)

with trend_col2:
    data = pd.DataFrame({
        "AQI": np.random.randint(60, 180, 30)
    })
    st.area_chart(data)

st.divider()

# =====================================================
# COMPLAINT ANALYZER
# =====================================================
st.markdown("<div class='section-title'>📝 Citizen Complaint Analyzer</div>", unsafe_allow_html=True)

CATEGORIES = {
    "Traffic Issue": "Traffic Department",
    "Water Supply Issue": "Water & Sanitation",
    "Electricity Issue": "Electricity Board",
    "Garbage / Sanitation": "Municipal Sanitation",
    "Public Safety": "Police Department",
    "Environmental Issue": "Environment Dept",
    "Other": "Municipal Office"
}
LABELS = list(CATEGORIES.keys())

complaint = st.text_area("Enter citizen complaint", height=120)

if st.button("🔍 Analyze & Route Complaint"):
    if complaint.strip():
        with st.spinner("Analyzing complaint..."):
            result = classifier(complaint, candidate_labels=LABELS)
            category = result["labels"][0]
            confidence = round(result["scores"][0], 2)

            if len(complaint.split()) < 25:
                summary = complaint
            else:
                summary = summarizer(complaint, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

        st.success("✅ Complaint Processed Successfully")
        st.write("**Summary:**", summary)
        st.write("**Category:**", category)
        st.write("**Confidence:**", f"{confidence * 100}%")
        st.write("**Assigned Department:**", CATEGORIES[category])
    else:
        st.warning("⚠️ Please enter a complaint.")

st.divider()

# =====================================================
# AGENTIC AI REPORT
# =====================================================
st.markdown("<div class='section-title'>🧠 AI City Operations Brief</div>", unsafe_allow_html=True)

# Simple Agent Implementation
class SimpleCityAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, prompt):
        """Execute agent reasoning"""
        # Gather data
        traffic_status = traffic()
        aqi_val, aqi_stat = air_quality()
        energy_status = energy()
        
        # Build context
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
            # Fallback response
            return f"""
📊 City Operations Summary:

Traffic congestion is {traffic_status}. Air quality shows AQI of {aqi_val} ({aqi_stat}). 
Smart grid reports {energy_status} conditions.

Priority Actions:
- Monitor traffic flow in peak hours
- Track air quality trends
- Ensure grid stability
- Address citizen complaints promptly
"""

@st.cache_resource
def load_agent():
    """Load LLM and create agent"""
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

# Load agent
try:
    agent = load_agent()
    agent_loaded = True
except Exception as e:
    st.error(f"Agent loading error: {e}")
    agent_loaded = False

if st.button("🧠 Generate Daily AI City Brief"):
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
        # Fallback manual report
        today = datetime.now().strftime("%d %B %Y")
        traffic_status = traffic()
        aqi_val, aqi_stat = air_quality()
        energy_status = energy()
        
        fallback_report = f"""
📊 **City Operations Brief - {today}**

**Current Status:**
- 🚦 Traffic: {traffic_status} congestion levels
- 🌫 Air Quality: AQI {aqi_val} ({aqi_stat})
- ⚡ Energy Grid: {energy_status}

**Priority Actions:**
- Monitor high-traffic zones
- Track air quality sensors
- Ensure power grid stability
- Process citizen complaints promptly
"""
        st.info(fallback_report)

# =====================================================
# FOOTER
# =====================================================
st.caption("CitySense360 • AI-Driven Urban Intelligence Platform")