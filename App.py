import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import date
import os
if not os.path.exists('models/best_model.pkl'):
    from setup import train_and_save
    train_and_save()

st.set_page_config(page_title="CTA Load Predictor", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding: 2rem 3rem; }
    
    .top-bar {
        background: #1a1a2e;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border-left: 5px solid #e94560;
    }
    .top-bar h1 {
        color: #ffffff;
        font-size: 28px;
        font-weight: 600;
        margin: 0 0 6px 0;
    }
    .top-bar p {
        color: #a0a0b0;
        font-size: 14px;
        margin: 0;
    }
    
    .stat-box {
        background: #16213e;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #0f3460;
        text-align: center;
    }
    .stat-box .val {
        font-size: 22px;
        font-weight: 600;
        color: #e94560;
        margin: 0;
    }
    .stat-box .lbl {
        font-size: 12px;
        color: #a0a0b0;
        margin: 4px 0 0 0;
    }
    
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    
    .result-high {
        background: #2d1b1b;
        border: 1px solid #e94560;
        border-left: 4px solid #e94560;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 16px;
    }
    .result-normal {
        background: #1b2d1f;
        border: 1px solid #2ecc71;
        border-left: 4px solid #2ecc71;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 16px;
    }
    .result-high h3 { color: #e94560; margin: 0 0 4px 0; font-size: 18px; }
    .result-normal h3 { color: #2ecc71; margin: 0 0 4px 0; font-size: 18px; }
    .result-high p, .result-normal p { color: #a0a0b0; margin: 0; font-size: 13px; }
    
    .tag {
        display: inline-block;
        background: #0f3460;
        color: #a0c4ff;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        margin: 3px 3px 3px 0;
    }
    
    div[data-testid="stButton"] button {
        background: #e94560 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px !important;
        font-size: 15px !important;
    }
    div[data-testid="stButton"] button:hover {
        background: #c73652 !important;
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="top-bar">
    <h1>🚌 Chicago Transit Load Predictor</h1>
    <p>Enter a bus route, date, and weather conditions to predict whether that route will experience high passenger load.</p>
</div>
""", unsafe_allow_html=True)

# stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><p class="val">87%</p><p class="lbl">Model Accuracy</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><p class="val">824K</p><p class="lbl">Training Rows</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><p class="val">18 yrs</p><p class="lbl">Data Range</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><p class="val">16</p><p class="lbl">Features</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# load model and data
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/merged_features.csv')
    df['route_encoded'] = df['route'].astype('category').cat.codes
    return df

model = load_model()
df_data = load_data()

# main layout
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown('<p class="section-title">Route & Date</p>', unsafe_allow_html=True)
    route = st.number_input("Bus Route Number", min_value=1, max_value=999, value=49, label_visibility="collapsed")
    st.caption("Bus route number (e.g. 49, 8, 22)")
    selected_date = st.date_input("Select date", value=date.today())

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Weather Conditions</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        tmax = st.number_input("Max Temp (°C)", value=20, min_value=-30, max_value=45)
        prcp = st.number_input("Rain (mm)", value=0.0, min_value=0.0, max_value=50.0)
        awnd = st.number_input("Wind (km/h)", value=15.0, min_value=0.0, max_value=80.0)
    with col_b:
        tmin = st.number_input("Min Temp (°C)", value=10, min_value=-30, max_value=40)
        snow = st.number_input("Snow (mm)", value=0.0, min_value=0.0, max_value=50.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Load →", use_container_width=True)

    # tags
    month = selected_date.month
    day_of_week = selected_date.weekday()
    is_weekend = day_of_week >= 5
    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    st.markdown("<br>", unsafe_allow_html=True)
    tags = f"""
    <span class="tag">{days[day_of_week]}</span>
    <span class="tag">{season_map[month]}</span>
    <span class="tag">{'Weekend' if is_weekend else 'Weekday'}</span>
    <span class="tag">{'🌧 Rainy' if prcp > 0 else '❄️ Snowy' if snow > 0 else '☀️ Clear'}</span>
    """
    st.markdown(tags, unsafe_allow_html=True)

with right:
    st.markdown('<p class="section-title">Historical Rides — Route ' + str(route) + '</p>', unsafe_allow_html=True)

    route_data = df_data[df_data['route'] == str(route)].copy()
    if len(route_data) > 0:
        route_data['date'] = pd.to_datetime(route_data['date'])
        monthly = route_data.groupby(route_data['date'].dt.to_period('M'))['rides'].mean().reset_index()
        monthly['date'] = monthly['date'].astype(str)
        monthly = monthly.tail(48)

        fig, ax = plt.subplots(figsize=(9, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#16213e')
        ax.plot(range(len(monthly)), monthly['rides'], color='#e94560', linewidth=2, zorder=3)
        ax.fill_between(range(len(monthly)), monthly['rides'],
                        alpha=0.15, color='#e94560', zorder=2)
        ax.scatter(range(len(monthly)), monthly['rides'],
                   color='#e94560', s=20, zorder=4)
        step = max(1, len(monthly) // 8)
        ax.set_xticks(range(0, len(monthly), step))
        ax.set_xticklabels(monthly['date'].iloc[::step],
                           rotation=30, color='#a0a0b0', fontsize=8, ha='right')
        ax.tick_params(colors='#a0a0b0', length=0)
        for spine in ax.spines.values():
            spine.set_color('#0f3460')
        ax.set_ylabel('Avg Daily Rides', color='#a0a0b0', fontsize=9)
        ax.yaxis.label.set_color('#a0a0b0')
        ax.grid(axis='y', color='#0f3460', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info(f"No historical data found for route {route}.")

    # prediction
    if predict_btn:
        day = selected_date.day
        is_rainy = 1 if prcp > 0 else 0
        is_snowy = 1 if snow > 0 else 0
        season_enc = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[month]

        route_map = df_data[['route','route_encoded']].drop_duplicates()
        re = route_map[route_map['route'] == str(route)]['route_encoded']
        route_encoded = int(re.values[0]) if len(re) > 0 else 0

        features = pd.DataFrame([[
            month, day, day_of_week, int(is_weekend),
            tmax, tmin, prcp, snow, awnd,
            is_rainy, is_snowy, route_encoded, season_enc,
            5000, 5000, 5000
        ]], columns=[
            'month','day','day_of_week','is_weekend',
            'TMAX','TMIN','PRCP','SNOW','AWND',
            'is_rainy','is_snowy','route_encoded','season',
            'rides_lag_1','rides_lag_7','rides_rolling_7'
        ])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        confidence = round(probability * 100) if prediction == 1 else round((1 - probability) * 100)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <h3>🔴 High Load Expected</h3>
                <p>This route is likely to be crowded on {selected_date.strftime('%B %d, %Y')}. 
                Confidence: <strong style="color:#e94560">{confidence}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                <h3>🟢 Normal Load Expected</h3>
                <p>This route should operate normally on {selected_date.strftime('%B %d, %Y')}. 
                Confidence: <strong style="color:#2ecc71">{confidence}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        # confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(9, 0.6))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        bar_color = '#e94560' if prediction == 1 else '#2ecc71'
        ax2.barh([0], [confidence], color=bar_color, height=0.4, zorder=3)
        ax2.barh([0], [100-confidence], left=[confidence], color='#16213e', height=0.4, zorder=2)
        ax2.set_xlim(0, 100)
        ax2.set_yticks([])
        ax2.set_xlabel('Confidence %', color='#a0a0b0', fontsize=9)
        ax2.tick_params(colors='#a0a0b0', length=0)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.text(confidence/2, 0, f'{confidence}%',
                 ha='center', va='center', color='white',
                 fontweight='600', fontsize=10)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()