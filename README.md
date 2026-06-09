# 🇰🇪 Huduma Centre Demand Prediction System
### Makandara Huduma Centre · Service By Appointment (SBA) · Quarterly Forecasting Dashboard

A data science project that forecasts quarterly demand for e-government services at Makandara Huduma Centre using time series forecasting models — ARIMA, Facebook Prophet, and LSTM. The system is deployed as an interactive Streamlit web dashboard enabling Huduma Centre managers to generate demand forecasts and receive operational planning alerts.

---

## 📊 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://demand-prediction-ycmm8eb8smd8dzuigzgvkt.streamlit.app/)

> **📄 Published Paper:** [Read on ResearchGate](https://www.researchgate.net/publication/387456579)  


## 🧭 Project Overview

Public service delivery points in Kenya have witnessed long queues which leave citizens unsatisfied. This system addresses that challenge by predicting quarterly demand for the top 5 highest-demand services at Makandara Huduma Centre, enabling proactive resource allocation and staffing decisions before queues form.

The system analyses 426,491 booking records from the SBA (Service By Appointment) system spanning 2020 Q2 to 2023 Q4 across 43 unique government services.

### Top 5 Services Forecasted

| Service | Total Bookings |
|---------|---------------|
| NRB — Application for Duplicate ID | 130,789 |
| DCI — Police Clearance | 50,970 |
| CRD — Birth Certificate Enquiries | 48,327 |
| NHIF — NHIF Services | 42,622 |
| NRB — Collection of ID | 38,116 |

---

## 🏆 Model Performance (MAPE)

| Service | ARIMA | Prophet | LSTM | Best Model |
|---------|-------|---------|------|------------|
| NRB Duplicate ID | 154.0% | 148.0% | 146.1% | LSTM |
| DCI Police Clearance | 114.9% | 70.4% | 698.2% | Prophet |
| CRD Birth Certificate | 3645.0% | 10393.1% | 6222.8% | — (unforecastable) |
| NHIF Services | 42.6% | 32.5% | **30.4%** | LSTM ✅ |
| NRB Collection of ID | 60.1% | 57.0% | 62.1% | Prophet |

> **Key finding:** Prophet is the most consistently reliable model overall. NHIF Services is the most forecastable service with a best MAPE of 30.4%.

---

## ✨ Features

- **📁 CSV Upload** — Upload the SBA booking dataset directly through the dashboard sidebar
- **📈 Historical Demand Chart** — Interactive line chart showing quarterly demand trends for all top 5 services from 2020 to 2023
- **🔮 Demand Forecast** — Prophet-powered quarterly forecast for any selected service, 1 to 4 quarters ahead
- **📊 Confidence Intervals** — 80% confidence range displayed alongside every forecast
- **🚦 Traffic Light Alert** — Operational demand alert (🟢 Normal / 🟡 Elevated / 🔴 High Demand) to guide staffing decisions
- **🏆 Model Comparison Table** — ARIMA vs Prophet vs LSTM performance comparison for all 5 services
- **📊 Booking Breakdown** — Booking type split and gender distribution visualisations

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.x |
| Dashboard | Streamlit |
| Forecasting | Facebook Prophet |
| Classical Model | ARIMA / SARIMA (statsmodels) |
| Deep Learning | LSTM (TensorFlow / Keras) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| ML Utilities | Scikit-learn |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/angelamawia/demand-prediction.git
cd demand-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app locally

```bash
streamlit run app_demand.py
```

The app will open automatically at `http://localhost:8501`

### 4. Upload your dataset

In the sidebar, upload the `transformed_data.csv` file from the Huduma Centre SBA system to load the full dashboard.

---

---

## 📓 Notebook

The `demand-prediction.ipynb` file contains the complete step-by-step analysis including:

- Data loading and exploration
- Data cleaning and preprocessing
- Time series engineering
- Stationarity testing (ADF test)
- ARIMA model building and evaluation
- Prophet model building and evaluation
- LSTM model building and evaluation
- Model comparison and business impact analysis
- Production considerations

---

## 🏛️ Dataset

The dataset originates from the **Huduma Kenya SBA (Service By Appointment)** system — a government booking platform used at all Huduma Centres in Kenya. The data used in this project is from **Makandara Huduma Centre** and covers booking records from 2020 to 2023.

| Column | Description |
|--------|-------------|
| `type` | Booking method (walk-in, online-booking, cc-booking) |
| `placeofbirth` | Citizen's place of birth |
| `gender` | Citizen's gender (M/F) |
| `quarter` | Quarter of the booking (Q1–Q4) |
| `servicename` | Government service booked |
| `year` | Year of booking (2020–2023) |

> **Note:** Upload your copy of `transformed_data.csv` from SBA system in Makandara Huduma Center via the dashboard sidebar.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```
3. Commit your changes:
```bash
git commit -m "Add: describe your feature"
```
4. Push the branch:
```bash
git push origin feature/your-feature-name
```
5. Open a Pull Request

---

## 👩‍💻 Author

**Angela Mawia**  
BSc Computer Science — University of Nairobi  
Fourth Year Project · Department of Computing and Informatics

---

## 📄 License

This project is submitted as an academic project in partial fulfilment of the requirements for the award of the degree of Bachelor of Science in Computer Science at the University of Nairobi.

---

*Built with ❤️ for better public service delivery in Kenya 🇰🇪*
