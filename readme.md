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



Data Sources
│
├── CCTV Videos (Traffic)
├── IoT Sensors (AQI, Energy)
├── Citizen Complaints (Text)
│
▼
AI Models
│
├── LSTM (Traffic Forecasting)
├── LSTM / GRU (Smart Grid Load)
├── LSTM (Air Quality Prediction)
├── Transformer NLP (Complaint Analysis)
├── LLM (Summarization & Reasoning)
│
▼
Agentic AI Layer
│
├── Tool-based Reasoning
├── Priority Action Planning
├── City Operations Report Generation
│
▼
Unified Frontend (Streamlit Dashboard)


---

## 🧩 Modules Implemented

### 🌫 Air Quality Prediction
- **Model:** LSTM  
- **Input:** Pollutants (CO, NO₂, NOx), Temperature, Humidity  
- **Output:** AQI prediction + status  
- **Metrics:** MAE, RMSE  

---

### ⚡ Smart Grid Energy Monitoring
- **Model:** LSTM  
- **Input:** Voltage, Current, Frequency, FFT features  
- **Output:** Future power usage forecasting  
- **Use Case:** Load balancing & peak demand prediction  

---

### 🚦 Traffic Congestion Prediction
- **Computer Vision:** YOLO-based vehicle detection from CCTV videos  
- **Time-Series Model:** LSTM  
- **Output:** Traffic density & congestion level  

---

### 📝 Citizen Complaint Analyzer (NLP + LLM)
- **Classification:** Zero-shot transformer (BART-MNLI)  
- **Summarization:** Abstractive LLM with hallucination safeguards  
- **Routing:** Automatic department assignment  

Example:


Input: "Street lights are flickering in my area."
Output:

Category: Electricity Issue

Department: Electricity Board

Summary: Street lights are malfunctioning in the area.


---

### 🧠 Agentic AI City Brain
- **Framework:** LLM-based reasoning agent  
- **Capabilities:**
  - Aggregates traffic, AQI, energy, and complaint insights
  - Generates daily city operations reports
  - Suggests priority actions  
- **Design:** Tool-based, explainable, and fault-tolerant  

---

## 🖥 Frontend – Smart City Control Room

- Built with **Streamlit + Plotly**
- Dark, futuristic **control-room UI**
- Animated KPI cards
- Interactive city map (zone-level AQI & alerts)
- Real-time charts for traffic, AQI, and energy
- Integrated complaint analysis & AI reports  

> The frontend abstracts complex AI models into a **human-centered decision interface** for city administrators.

---

## 🛠 Tech Stack

**Machine Learning & AI**
- Python
- TensorFlow / Keras
- PyTorch
- Transformers (BERT, BART, FLAN-T5)
- LSTM, CNN, RNN, GRU

**NLP & LLMs**
- Hugging Face Transformers
- Zero-shot classification
- Abstractive summarization
- Prompt-based reasoning

**Agentic AI**
- Tool-based reasoning
- Multi-domain orchestration
- Autonomous report generation

**Frontend & Visualization**
- Streamlit
- Plotly
- Custom CSS animations
- Interactive maps

---

## 📊 Evaluation Metrics

**Deep Learning**
- MAE, RMSE
- Time-series forecasting accuracy  

**NLP & LLM**
- Classification confidence
- Summarization quality
- Hallucination safeguards  

**System Metrics**
- Responsiveness
- Modularity
- Explainability

---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install streamlit tensorflow torch transformers sentence-transformers plotly opencv-python

2️⃣ Run the dashboard
streamlit run citysense360_dashboard_sexy.py

📌 Project Highlights

End-to-end AI system integration

Realistic smart city use cases

Explainable & modular design

Strong focus on decision support, not just prediction

Demo-ready frontend

🔮 Future Enhancements

Replace mock data with live IoT feeds

FastAPI backend for model serving

Multi-agent specialization (Traffic Agent, Energy Agent, etc.)

Vector database for long-term city memory

Cloud deployment (AWS / GCP / Azure)

👤 Author

CitySense360
AI-Powered Smart City Intelligence Platform
