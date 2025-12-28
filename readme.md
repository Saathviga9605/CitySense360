# 🌆 CitySense360  
### AI-Powered Smart City Intelligence & Public Infrastructure Automation

CitySense360 is a **unified AI-driven smart city platform** that integrates **machine learning, deep learning, NLP, LLMs, agentic AI, and interactive visualization** to enable real-time monitoring, prediction, and decision support for urban infrastructure.

The system is designed as a **smart city control room**, transforming raw data from sensors, CCTV feeds, and citizen complaints into **actionable insights** for city administrators.

---

## 🚀 Key Features

- 🌫 **Air Quality Prediction** using LSTM-based time-series forecasting  
- ⚡ **Smart Grid Energy Forecasting** with electrical & FFT signal features  
- 🚦 **Traffic Congestion Prediction** from CCTV video analytics + LSTM  
- 📝 **Citizen Complaint Analyzer** (NLP + LLM) with automatic routing  
- 🧠 **Agentic AI City Brain** for multi-domain reasoning & reporting  
- 🗺 **Interactive City Map** with zone-level insights  
- 📊 **Real-time Dashboard** with animated KPIs and trends  

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
├─────────────────────────────────────────────────────────────┤
│  • CCTV Videos (Traffic)                                    │
│  • IoT Sensors (AQI, Energy)                                │
│  • Citizen Complaints (Text)                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      AI Models                              │
├─────────────────────────────────────────────────────────────┤
│  • LSTM (Traffic Forecasting)                               │
│  • LSTM / GRU (Smart Grid Load)                             │
│  • LSTM (Air Quality Prediction)                            │
│  • Transformer NLP (Complaint Analysis)                     │
│  • LLM (Summarization & Reasoning)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agentic AI Layer                          │
├─────────────────────────────────────────────────────────────┤
│  • Tool-based Reasoning                                     │
│  • Priority Action Planning                                 │
│  • City Operations Report Generation                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Unified Frontend (Streamlit Dashboard)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Modules Implemented

### 🌫 Air Quality Prediction
- **Model:** LSTM (Long Short-Term Memory)
- **Input:** Pollutants (CO, NO₂, NOx), Temperature, Humidity  
- **Output:** AQI prediction + health status classification
- **Metrics:** MAE, RMSE  
- **Use Case:** Early warning system for air pollution events

---

### ⚡ Smart Grid Energy Monitoring
- **Model:** LSTM with signal processing
- **Input:** Voltage, Current, Frequency, FFT features  
- **Output:** Future power usage forecasting (15-min to 24-hr ahead)
- **Use Case:** Load balancing, peak demand prediction, grid optimization

---

### 🚦 Traffic Congestion Prediction
- **Computer Vision:** YOLO-based vehicle detection from CCTV videos  
- **Time-Series Model:** LSTM for traffic flow prediction
- **Output:** Traffic density heatmaps & congestion level (Low/Medium/High)
- **Use Case:** Dynamic traffic signal control, route optimization

---

### 📝 Citizen Complaint Analyzer (NLP + LLM)
- **Classification:** Zero-shot transformer (facebook/bart-large-mnli)
- **Summarization:** Abstractive LLM with hallucination safeguards  
- **Routing:** Automatic department assignment  

**Example:**

```
Input: "Street lights are flickering in my area for the past 3 days."

Output:
  Category: Electricity Issue
  Department: Electricity Board
  Summary: Street lights malfunctioning in residential area
  Priority: Medium
```

---

### 🧠 Agentic AI City Brain
- **Framework:** LLM-based reasoning agent with tool integration
- **Capabilities:**
  - Aggregates traffic, AQI, energy, and complaint insights
  - Generates daily city operations reports
  - Suggests priority actions based on severity
  - Multi-domain decision support
- **Design:** Tool-based, explainable, fault-tolerant, and auditable

---

## 🖥 Frontend – Smart City Control Room

Built with **Streamlit + Plotly** featuring:

- Dark, futuristic **control-room UI** design
- Animated KPI cards with real-time updates
- Interactive city map with zone-level AQI & alerts
- Real-time charts for traffic, AQI, and energy consumption
- Integrated complaint analysis dashboard
- AI-generated city operations reports
- Mobile-responsive design

> The frontend abstracts complex AI models into a **human-centered decision interface** for city administrators.

---

## 🛠 Tech Stack

### Machine Learning & AI
- Python 3.8+
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- LSTM, CNN, RNN, GRU architectures

### NLP & LLMs
- Hugging Face Transformers
- BERT, BART, FLAN-T5
- Zero-shot classification
- Abstractive summarization
- Prompt engineering

### Agentic AI
- Tool-based reasoning
- Multi-domain orchestration
- Autonomous report generation
- Decision tree logic

### Computer Vision
- OpenCV
- YOLO (You Only Look Once)
- Video stream processing

### Frontend & Visualization
- Streamlit
- Plotly
- Custom CSS animations
- Folium (Interactive maps)

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended)
- GPU support optional (for faster model inference)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/citysense360.git
cd citysense360
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Core dependencies:**
```bash
pip install streamlit tensorflow torch transformers sentence-transformers \
    plotly opencv-python pandas numpy scikit-learn folium
```

### Step 4: Download Pre-trained Models (Optional)
```bash
python scripts/download_models.py
```

---

## ▶️ Usage

### Running the Dashboard

```bash
streamlit run citysense360_dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Running Individual Modules

**Air Quality Prediction:**
```bash
python modules/air_quality_predictor.py
```

**Traffic Analysis:**
```bash
python modules/traffic_analyzer.py --video path/to/video.mp4
```

**Complaint Analysis:**
```bash
python modules/complaint_analyzer.py --input complaints.csv
```

### API Mode (if implemented)

```bash
uvicorn api.main:app --reload
```

---



## 📊 Evaluation Metrics

### Deep Learning Models
- **MAE (Mean Absolute Error):** Average prediction error magnitude
- **RMSE (Root Mean Squared Error):** Penalizes larger errors
- **Time-series Accuracy:** Forecast horizon performance (1-hr, 6-hr, 24-hr)

### NLP & LLM
- **Classification Confidence:** Zero-shot prediction certainty
- **Summarization Quality:** ROUGE scores, human evaluation
- **Hallucination Detection:** Fact-checking against source text
- **Response Time:** Latency for real-time processing

### System Performance
- **Dashboard Responsiveness:** Page load time < 2s
- **Model Inference Time:** < 500ms per prediction
- **Modularity:** Independent module testing
- **Explainability:** Human-interpretable outputs

---
