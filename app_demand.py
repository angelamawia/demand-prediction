import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Huduma Centre Demand Predictor",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #F7F9FC; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .app-header {
        background: linear-gradient(135deg, #006600 0%, #009900 50%, #CC0000 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .app-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .app-header p  { font-size: 1rem; margin: 0.4rem 0 0 0; opacity: 0.9; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #006600;
        margin-bottom: 1rem;
    }
    .metric-card.red   { border-left-color: #CC0000; }
    .metric-card.amber { border-left-color: #E6A817; }
    .metric-card.green { border-left-color: #006600; }

    .metric-label {
        font-size: 0.75rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.8px;
        color: #6B7280; margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.9rem; font-weight: 700;
        color: #111827; font-family: 'DM Mono', monospace;
    }
    .metric-sub { font-size: 0.8rem; color: #9CA3AF; margin-top: 0.2rem; }

    .badge { display: inline-block; padding: 0.35rem 1rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
    .badge-green  { background: #D1FAE5; color: #065F46; }
    .badge-amber  { background: #FEF3C7; color: #92400E; }
    .badge-red    { background: #FEE2E2; color: #991B1B; }

    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #111827;
        margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.4rem;
        border-bottom: 2px solid #E5E7EB;
    }

    .info-box {
        background: #EFF6FF; border: 1px solid #BFDBFE;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.88rem; color: #1E40AF; margin: 1rem 0;
    }
    .warn-box {
        background: #FFFBEB; border: 1px solid #FCD34D;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.88rem; color: #92400E; margin: 1rem 0;
    }

    section[data-testid="stSidebar"] { background: #111827; }
    section[data-testid="stSidebar"] * { color: #F9FAFB !important; }

    .styled-table {
        width: 100%; border-collapse: collapse; font-size: 0.88rem;
        background: white; border-radius: 10px; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .styled-table th {
        background: #111827; color: white; padding: 0.7rem 1rem;
        text-align: left; font-weight: 600; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .styled-table td { padding: 0.65rem 1rem; border-bottom: 1px solid #F3F4F6; color: #374151; }
    .styled-table tr:last-child td { border-bottom: none; }
    .styled-table tr:hover td { background: #F9FAFB; }

    .footer {
        text-align: center; font-size: 0.78rem; color: #9CA3AF;
        margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data[data['year'].between(2020, 2023)]
    data['year'] = data['year'].astype(int)
    data = data[data['servicename'].notna()]
    data = data[data['type'] != 'mashinani']
    data = data[~data['quarter'].isin(['er','es','on','te','ct','ID','ds','s)','nt','ce'])]
    data = data[data['gender'] != 'gender']
    data['period'] = data['year'].astype(str) + '-' + data['quarter'].astype(str)
    period_order = [
        '2020-Q2','2020-Q3','2020-Q4',
        '2021-Q1','2021-Q2','2021-Q3','2021-Q4',
        '2022-Q1','2022-Q2','2022-Q3','2022-Q4',
        '2023-Q1','2023-Q2','2023-Q3','2023-Q4'
    ]
    period_map = {p: i+1 for i, p in enumerate(period_order)}
    data['numeric'] = data['period'].map(period_map)
    return data


@st.cache_data
def build_top5_df(data):
    from itertools import product as iproduct

    top5 = [
        'NRB -  Application for Duplicate ID',
        'DCI - Police Clearance',
        'CRD - Birth Certificate Enquiries',
        'NHIF - NHIF Services',
        'NRB - Collection of ID'
    ]

    period_to_date = {
        '2020-Q2':'2020-04-01','2020-Q3':'2020-07-01','2020-Q4':'2020-10-01',
        '2021-Q1':'2021-01-01','2021-Q2':'2021-04-01','2021-Q3':'2021-07-01','2021-Q4':'2021-10-01',
        '2022-Q1':'2022-01-01','2022-Q2':'2022-04-01','2022-Q3':'2022-07-01','2022-Q4':'2022-10-01',
        '2023-Q1':'2023-01-01','2023-Q2':'2023-04-01','2023-Q3':'2023-07-01','2023-Q4':'2023-10-01'
    }

    total_bookings = data.groupby(['servicename','period']).size().reset_index(name='demand_count')
    top5_df = total_bookings[total_bookings['servicename'].isin(top5)]
    all_periods = sorted(total_bookings['period'].unique())
    full_index = pd.DataFrame(list(iproduct(top5, all_periods)), columns=['servicename','period'])
    top5_df = full_index.merge(top5_df, on=['servicename','period'], how='left')
    top5_df['demand_count'] = top5_df['demand_count'].fillna(0).astype(int)
    top5_df['ds'] = pd.to_datetime(top5_df['period'].map(period_to_date))

    return top5_df, top5


@st.cache_data
def run_prophet(service_name, top5_df_json, forecast_quarters):
    from io import StringIO
    top5_df = pd.read_json(StringIO(top5_df_json))
    top5_df['ds'] = pd.to_datetime(top5_df['ds'], unit='ms')

    service_df = (top5_df[top5_df['servicename'] == service_name]
                  [['ds','demand_count']]
                  .rename(columns={'demand_count':'y'})
                  .sort_values('ds')
                  .reset_index(drop=True))

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive'
    )
    model.fit(service_df)

    future   = model.make_future_dataframe(periods=forecast_quarters, freq='QS')
    forecast = model.predict(future)

    return forecast, service_df


def get_traffic_light(predicted, hist_mean, hist_std):
    if predicted > hist_mean + hist_std:
        return "🔴 HIGH DEMAND",  "red",   "Surge expected — increase staffing and stock materials in advance."
    elif predicted > hist_mean:
        return "🟡 ELEVATED",     "amber", "Above average demand — prepare additional resources."
    else:
        return "🟢 NORMAL",       "green", "Demand within normal range — standard staffing should suffice."


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇰🇪 Huduma Centre")
    st.markdown("**Demand Prediction System**")
    st.markdown("---")
    st.markdown("#### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("---")
    st.markdown("#### ⚙️ Forecast Settings")
    forecast_quarters = st.slider("Quarters to forecast ahead", min_value=1, max_value=4, value=2)
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    Forecasts quarterly demand for the **top 5 services**
    at Makandara Huduma Centre using Facebook Prophet.

    **Best performing service:** NHIF (MAPE ~30%)

    **Data range:** 2020 Q2 — 2023 Q4
    """)
    st.caption("Research prototype | Makandara Huduma Centre SBA Data")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🇰🇪 Huduma Centre Demand Prediction System</h1>
    <p>Makandara Huduma Centre &nbsp;·&nbsp; Service By Appointment (SBA) &nbsp;·&nbsp; Quarterly Forecasting Dashboard</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GATE: require upload
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        📁 <strong>Getting Started:</strong> Upload <code>transformed_data.csv</code> file from Github in demand-prediction repository located in data folder
        using the sidebar panel on the left to load the full dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What this dashboard does</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 Overview**\n\nShows total bookings, unique services, and historical demand trends across all top 5 services.")
    with col2:
        st.markdown("**🔮 Forecast**\n\nUses Facebook Prophet to predict demand for the next 1–4 quarters with confidence intervals.")
    with col3:
        st.markdown("**🚦 Traffic Light**\n\nAlerts managers to normal, elevated, or surge demand levels to guide staffing decisions.")
    st.stop()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading and cleaning data..."):
    data = load_and_clean_data(uploaded_file)
    top5_df, top5 = build_top5_df(data)


# ─────────────────────────────────────────────
# OVERVIEW METRICS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card green">
        <div class="metric-label">Total Bookings</div>
        <div class="metric-value">{len(data):,}</div>
        <div class="metric-sub">All services combined</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Unique Services</div>
        <div class="metric-value">{data['servicename'].nunique()}</div>
        <div class="metric-sub">Across all agencies</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card amber">
        <div class="metric-label">Data Range</div>
        <div class="metric-value" style="font-size:1.2rem">{data['year'].min()} Q2 — {data['year'].max()} Q4</div>
        <div class="metric-sub">15 quarters of data</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card red">
        <div class="metric-label">Highest Demand Service</div>
        <div class="metric-value" style="font-size:0.95rem; padding-top:0.3rem">Duplicate ID</div>
        <div class="metric-sub">130,789 total bookings</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HISTORICAL DEMAND CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Historical Quarterly Demand — Top 5 Services</div>', unsafe_allow_html=True)

pivot_df = top5_df.pivot(index='period', columns='servicename', values='demand_count').sort_index()

short_names = {
    'NRB -  Application for Duplicate ID': 'NRB Duplicate ID',
    'DCI - Police Clearance':              'DCI Police Clearance',
    'CRD - Birth Certificate Enquiries':   'CRD Birth Certificate',
    'NHIF - NHIF Services':                'NHIF Services',
    'NRB - Collection of ID':              'NRB Collection of ID'
}
chart_colors = ['#006600','#CC0000','#E6A817','#1D4ED8','#7C3AED']

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('#F7F9FC')
ax.set_facecolor('#F7F9FC')

for i, col in enumerate(pivot_df.columns):
    ax.plot(pivot_df.index, pivot_df[col],
            marker='o', markersize=5, linewidth=2,
            color=chart_colors[i % len(chart_colors)],
            label=short_names.get(col, col))

ax.set_xlabel('Quarter', fontsize=10, color='#6B7280')
ax.set_ylabel('Number of Bookings', fontsize=10, color='#6B7280')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#E5E7EB')
ax.spines['bottom'].set_color('#E5E7EB')
ax.grid(axis='y', color='#E5E7EB', linewidth=0.8)
plt.tight_layout()
st.pyplot(fig)
plt.close()


# ─────────────────────────────────────────────
# FORECAST SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🔮 Demand Forecast — Select a Service</div>', unsafe_allow_html=True)

display_to_raw = {v: k for k, v in short_names.items()}
selected_display = st.selectbox(
    "Choose a service to forecast:",
    options=list(short_names.values()),
    index=3,   # NHIF default — best performer
    help="NHIF Services has the best forecast accuracy (MAPE ~30%)"
)
selected_service = display_to_raw[selected_display]

with st.spinner(f"Training Prophet model for {selected_display}..."):
    forecast, service_df = run_prophet(selected_service, top5_df.to_json(), forecast_quarters)

last_known_date  = service_df['ds'].max()
future_forecast  = forecast[forecast['ds'] > last_known_date].head(forecast_quarters)
hist_mean        = service_df['y'].mean()
hist_std         = service_df['y'].std()
next_q_pred      = max(0, future_forecast['yhat'].iloc[0])
label, color, advice = get_traffic_light(next_q_pred, hist_mean, hist_std)

# ── Metric row ──
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">Next Quarter Forecast</div>
        <div class="metric-value">{int(next_q_pred):,}</div>
        <div class="metric-sub">Predicted bookings</div>
    </div>""", unsafe_allow_html=True)

with col2:
    low_v  = max(0, int(future_forecast['yhat_lower'].iloc[0]))
    high_v = max(0, int(future_forecast['yhat_upper'].iloc[0]))
    st.markdown(f"""
    <div class="metric-card amber">
        <div class="metric-label">Confidence Range</div>
        <div class="metric-value" style="font-size:1.3rem">{low_v:,} — {high_v:,}</div>
        <div class="metric-sub">80% confidence interval</div>
    </div>""", unsafe_allow_html=True)

with col3:
    mape_lookup = {
        'NRB -  Application for Duplicate ID': 148.0,
        'DCI - Police Clearance':              70.37,
        'CRD - Birth Certificate Enquiries':   10393.05,
        'NHIF - NHIF Services':                32.50,
        'NRB - Collection of ID':              57.02
    }
    mape_val   = mape_lookup.get(selected_service, 0)
    mape_color = "green" if mape_val < 50 else "amber" if mape_val < 150 else "red"
    st.markdown(f"""
    <div class="metric-card {mape_color}">
        <div class="metric-label">Model Accuracy (MAPE)</div>
        <div class="metric-value">{mape_val:.1f}%</div>
        <div class="metric-sub">Lower is better. &lt;50% = reliable</div>
    </div>""", unsafe_allow_html=True)

# ── Traffic light advice ──
box_cls = "warn-box" if color in ["amber","red"] else "info-box"
st.markdown(f'<div class="{box_cls}"><strong>{label}</strong> — {advice}</div>', unsafe_allow_html=True)

# ── Forecast chart ──
fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2.patch.set_facecolor('#F7F9FC')
ax2.set_facecolor('#F7F9FC')

ax2.plot(service_df['ds'], service_df['y'],
         marker='o', markersize=5, linewidth=2,
         color='#111827', label='Historical Demand', zorder=3)

forecast_plot = forecast[forecast['ds'] >= service_df['ds'].min()]
ax2.plot(forecast_plot['ds'], forecast_plot['yhat'].clip(lower=0),
         linestyle='--', linewidth=2, color='#006600', label='Prophet Forecast')

ax2.fill_between(forecast_plot['ds'],
                 forecast_plot['yhat_lower'].clip(lower=0),
                 forecast_plot['yhat_upper'].clip(lower=0),
                 alpha=0.15, color='#006600', label='Confidence Interval')

ax2.axvline(x=last_known_date, color='#CC0000', linestyle=':', linewidth=1.5, label='Forecast Start')

ax2.set_xlabel('Quarter', fontsize=10, color='#6B7280')
ax2.set_ylabel('Number of Bookings', fontsize=10, color='#6B7280')
ax2.set_title(f'{selected_display} — Demand Forecast', fontsize=12, fontweight='bold', color='#111827')
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.tick_params(axis='y', labelsize=8)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#E5E7EB')
ax2.spines['bottom'].set_color('#E5E7EB')
ax2.grid(axis='y', color='#E5E7EB', linewidth=0.8)
plt.tight_layout()
st.pyplot(fig2)
plt.close()


# ─────────────────────────────────────────────
# FORECAST TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Forecast Details</div>', unsafe_allow_html=True)

# Map month to quarter correctly
month_to_quarter = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:3, 8:3, 9:3, 10:4, 11:4, 12:4}

last_year = int(last_known_date.strftime('%Y'))
last_q    = month_to_quarter[int(last_known_date.strftime('%m'))]

q_labels = []
for _ in range(forecast_quarters):
    last_q += 1
    if last_q > 4:
        last_q = 1
        last_year += 1
    q_labels.append(f"{last_year}-Q{last_q}")

rows_html = ""
for i, (_, row) in enumerate(future_forecast.iterrows()):
    pred   = max(0, int(row['yhat']))
    lo     = max(0, int(row['yhat_lower']))
    hi     = max(0, int(row['yhat_upper']))
    qlabel = q_labels[i] if i < len(q_labels) else f"Q+{i+1}"
    lbl, clr, _ = get_traffic_light(pred, hist_mean, hist_std)
    rows_html += f"""
    <tr>
        <td><strong>{qlabel}</strong></td>
        <td style="font-family:'DM Mono',monospace">{pred:,}</td>
        <td style="font-family:'DM Mono',monospace">{lo:,} — {hi:,}</td>
        <td><span class="badge badge-{clr}">{lbl}</span></td>
    </tr>"""

st.markdown(f"""
<table class="styled-table">
    <thead><tr>
        <th>Quarter</th><th>Predicted Bookings</th>
        <th>Confidence Range</th><th>Demand Level</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL PERFORMANCE TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Model Performance Summary (All Services)</div>', unsafe_allow_html=True)

perf = [
    ('NRB Duplicate ID',      '154.0%', '148.0%', '146.1%', 'LSTM',    'Partial'),
    ('DCI Police Clearance',  '114.9%',  '70.4%', '698.2%', 'Prophet', 'Yes'),
    ('CRD Birth Certificate','3645.0%','10393.1%','6222.8%', 'ARIMA*',  'No'),
    ('NHIF Services',          '42.6%',  '32.5%',  '30.4%', 'LSTM',    'Yes'),
    ('NRB Collection of ID',   '60.1%',  '57.0%',  '62.1%', 'Prophet', 'Partial'),
]

rows2 = ""
for row in perf:
    svc, ar, pr, ls, best, fc = row
    badge_cls = 'badge-green' if fc == 'Yes' else 'badge-amber' if fc == 'Partial' else 'badge-red'
    hl = ' style="background:#F0FDF4"' if svc == 'NHIF Services' else ''
    rows2 += f"""
    <tr{hl}>
        <td><strong>{svc}</strong></td>
        <td>{ar}</td><td>{pr}</td><td>{ls}</td>
        <td><strong>{best}</strong></td>
        <td><span class="badge {badge_cls}">{fc}</span></td>
    </tr>"""

st.markdown(f"""
<table class="styled-table">
    <thead><tr>
        <th>Service</th><th>ARIMA MAPE</th><th>Prophet MAPE</th>
        <th>LSTM MAPE</th><th>Best Model</th><th>Forecastable</th>
    </tr></thead>
    <tbody>{rows2}</tbody>
</table>
<p style="font-size:0.78rem;color:#9CA3AF;margin-top:0.5rem">
* CRD experienced a service disruption in Q2 2023 (29 bookings vs 3,108 in Q1). All models failed.
  ARIMA listed as best only because it failed least catastrophically.
</p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BOOKING TYPE & GENDER
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Booking Breakdown</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    type_counts = data['type'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig3.patch.set_facecolor('#F7F9FC')
    ax3.set_facecolor('#F7F9FC')
    ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
            colors=['#006600','#CC0000','#E6A817','#1D4ED8'][:len(type_counts)],
            startangle=90, pctdistance=0.85)
    ax3.set_title('Booking Type Split', fontsize=11, fontweight='bold', color='#111827')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col2:
    gender_counts = data['gender'].value_counts()
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    fig4.patch.set_facecolor('#F7F9FC')
    ax4.set_facecolor('#F7F9FC')
    bars = ax4.bar(['Male','Female'],
                   [gender_counts.get('M',0), gender_counts.get('F',0)],
                   color=['#1D4ED8','#CC0000'], width=0.5, edgecolor='white')
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1000,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom',
                 fontsize=9, color='#374151')
    ax4.set_title('Gender Distribution', fontsize=11, fontweight='bold', color='#111827')
    ax4.set_ylabel('Number of Bookings', fontsize=9, color='#6B7280')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#E5E7EB')
    ax4.spines['bottom'].set_color('#E5E7EB')
    ax4.grid(axis='y', color='#E5E7EB', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Huduma Centre Demand Prediction System &nbsp;·&nbsp; Makandara Huduma Centre &nbsp;·&nbsp;
    Built using Facebook Prophet &nbsp;·&nbsp; Data: Service By Appointment (SBA) System<br>
    <em>Research prototype. Forecasts should be used as planning guidance alongside operational judgment.</em>
</div>
""", unsafe_allow_html=True)
