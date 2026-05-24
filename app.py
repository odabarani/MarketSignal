import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.data_loader import get_stock_data
from src.features import add_features
from src.model import train_model
from src.signals import generate_signal
from src.backtest import backtest

FEATURES = [
    'MA_5', 'MA_20', 'Momentum', 'Volatility',
    'Volume_Change', 'Volume_Ratio',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
    'VIX', 'RS_SPY',
    'Body_Size', 'Upper_Wick', 'Lower_Wick', 'Gap',
    'PE_Ratio', 'Profit_Margin', 'Revenue_Growth', 'Debt_Equity'
]

st.set_page_config(
    page_title="MarketSignal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0a0f;
        color: #e0e0e0;
    }
    .main { background-color: #0a0a0f; padding: 2rem 3rem; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; font-size: 2rem !important; color: #ffffff !important; }

    .metric-card {
        background: #12121a;
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.7rem;
        color: #555;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 0.3rem;
    }
    .metric-value.up   { color: #00e676; }
    .metric-value.down { color: #ff1744; }

    .signal-banner {
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .signal-buy  { background:#003300; border:1px solid #00e676; color:#00e676; }
    .signal-sell { background:#1a0000; border:1px solid #ff1744; color:#ff1744; }
    .signal-hold { background:#1a1200; border:1px solid #ffab00; color:#ffab00; }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #1e1e2e;
        padding-bottom: 0.4rem;
    }
    .chart-caption {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #555;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stTextInput"] input {
        background: #12121a !important;
        border: 1px solid #1e1e2e !important;
        color: #ffffff !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1rem !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
    }
    div[data-testid="stButton"] button {
        background: #ffffff !important;
        color: #000000 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button:hover { background: #00e676 !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("# 📈 Stock ML Predictor")
st.markdown('<p style="color:#444; font-family:IBM Plex Mono; font-size:0.75rem; letter-spacing:2px;">5-DAY DIRECTION PREDICTION · TECHNICAL + FUNDAMENTAL SIGNALS</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    ticker = st.text_input("", placeholder="Enter ticker — AAPL, TSLA, NVDA...", label_visibility="collapsed").upper()
with col2:
    run = st.button("RUN →")

st.markdown("<br>", unsafe_allow_html=True)

# --- Cached pipeline ---
@st.cache_resource(show_spinner=False)
def run_pipeline(ticker):
    df    = get_stock_data(ticker)
    df    = add_features(df)
    model, cv_accuracy = train_model(df, FEATURES)
    return df, model, cv_accuracy

if run and ticker:
    with st.spinner(f"Pulling data and training model for {ticker}..."):
        df, model, cv_accuracy = run_pipeline(ticker)

    latest     = df[FEATURES].iloc[-1].values.reshape(1, -1)
    prediction = int(model.predict(latest)[0])
    proba      = model.predict_proba(latest)[0]
    confidence = round(float(max(proba)) * 100, 1)
    direction  = "📈 UP" if prediction == 1 else "📉 DOWN"
    signal     = generate_signal(prediction, max(proba))
    last_close = round(float(df['Close'].iloc[-1].item()), 2)
    sig_class  = signal.lower()
    dir_class  = "up" if prediction == 1 else "down"

    # --- Metrics ---
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">5-Day Direction</div>
            <div class="metric-value {dir_class}">{direction}</div>
            <div class="metric-sub">Expected move over next 5 trading days</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence}%</div>
            <div class="metric-sub">Above 65% triggers a signal</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{round(cv_accuracy * 100, 1)}%</div>
            <div class="metric-sub">Tested on unseen historical data</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Close</div>
            <div class="metric-value">${last_close:,.2f}</div>
            <div class="metric-sub">Most recent closing price</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="signal-banner signal-{sig_class}">⚡ SIGNAL: {signal}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Price Chart + Projection ---
    st.markdown('<div class="section-label">Price History & 30-Day Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-caption">Blue = closing price &nbsp;|&nbsp; Orange = 5-day MA &nbsp;|&nbsp; Green = 20-day MA &nbsp;|&nbsp; Shaded = projected range</div>', unsafe_allow_html=True)

    df_recent    = df.tail(180)
    last_price   = float(df_recent['Close'].iloc[-1].item())
    daily_vol    = float(df_recent['Close'].pct_change().std().item())
    trend        = 1 if prediction == 1 else -1
    days_ahead   = 30
    last_date    = df_recent.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=days_ahead + 1)[1:]

    projected  = [last_price]
    upper_band = [last_price]
    lower_band = [last_price]

    for i in range(1, days_ahead + 1):
        drift      = trend * daily_vol * 0.5
        next_p     = projected[-1] * (1 + drift)
        band_width = last_price * daily_vol * np.sqrt(i) * 1.5
        projected.append(round(next_p, 2))
        upper_band.append(round(next_p + band_width, 2))
        lower_band.append(round(next_p - band_width, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Close'].squeeze().round(2), name='Close Price', line=dict(color='#2979ff', width=2)))
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MA_5'].squeeze().round(2), name='5-Day MA', line=dict(color='#ffab00', width=1.2, dash='dot')))
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['MA_20'].squeeze().round(2), name='20-Day MA', line=dict(color='#00e676', width=1.2, dash='dash')))
    fig.add_trace(go.Scatter(x=list(future_dates), y=upper_band[1:], line=dict(color='rgba(0,0,0,0)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=list(future_dates), y=lower_band[1:], name='Projected Range', fill='tonexty', fillcolor='rgba(255,171,0,0.12)', line=dict(color='rgba(0,0,0,0)', width=0)))
    fig.add_trace(go.Scatter(x=[df_recent.index[-1]] + list(future_dates), y=projected, name='Projected Path', line=dict(color='#ffab00', width=1.5, dash='dot')))
    fig.update_layout(template='plotly_dark', paper_bgcolor='#12121a', plot_bgcolor='#12121a', hovermode='x unified', height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02), font=dict(family='IBM Plex Mono'), xaxis_title="Date", yaxis_title="Price (USD)", yaxis_tickformat="$,.2f")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- RSI ---
    st.markdown('<div class="section-label">RSI — Market Momentum</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-caption">Above 70 = overheated (pullback likely) &nbsp;|&nbsp; Below 30 = oversold (bounce likely)</div>', unsafe_allow_html=True)

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['RSI'].squeeze().round(1), name='RSI', line=dict(color='#e040fb', width=1.5), fill='tozeroy', fillcolor='rgba(224,64,251,0.05)'))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", line_width=1, annotation_text="Overbought", annotation_position="top right", annotation_font_color="#ff1744")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="#00e676", line_width=1, annotation_text="Oversold", annotation_position="bottom right", annotation_font_color="#00e676")
    rsi_fig.update_layout(template='plotly_dark', paper_bgcolor='#12121a', plot_bgcolor='#12121a', height=200, margin=dict(l=10, r=10, t=10, b=10), font=dict(family='IBM Plex Mono'), xaxis_title="Date", yaxis_title="RSI", yaxis=dict(range=[0, 100], tickvals=[0, 30, 50, 70, 100]))
    st.plotly_chart(rsi_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Backtest ---
    st.markdown("---")
    st.subheader("Backtest — How Would This Strategy Have Performed?")
    st.caption("Simulates trading $10,000 using model signals. Trained on 70% of data, tested on remaining 30% to prevent data leakage.")
    
    from src.backtest import calculate_metrics

    portfolio    = backtest(df, model, FEATURES)
    final_value  = round(portfolio[-1], 2)
    total_return = round(((final_value - 10000) / 10000) * 100, 1)
    peak         = round(max(portfolio), 2)
    ret_class    = "up" if total_return > 0 else "down"
    sharpe, max_dd = calculate_metrics(portfolio)
    sharpe_class   = "up" if sharpe > 1 else "down" if sharpe < 0 else ""
    dd_class       = "down" if max_dd < -20 else ""

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Final Value</div><div class="metric-value">${final_value:,.0f}</div><div class="metric-sub">Starting from $10,000</div></div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Return</div><div class="metric-value {ret_class}">{total_return}%</div><div class="metric-sub">Over full test period</div></div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Peak Value</div><div class="metric-value">${peak:,.0f}</div><div class="metric-sub">Highest point reached</div></div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value {sharpe_class}">{sharpe}</div><div class="metric-sub">Above 1.0 is good</div></div>""", unsafe_allow_html=True)
    with b5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value {dd_class}">{max_dd}%</div><div class="metric-sub">Worst peak-to-trough drop</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    bt_dates = df.index[50:50 + len(portfolio)]
    bt_fig   = go.Figure()
    bt_fig.add_trace(go.Scatter(x=bt_dates, y=[round(v, 2) for v in portfolio], mode='lines', name='Portfolio Value', line=dict(color='#00e676', width=2), fill='tozeroy', fillcolor='rgba(0,230,118,0.05)'))
    bt_fig.add_hline(y=10000, line_dash="dash", line_color="#666", line_width=1, annotation_text="Starting Capital ($10,000)", annotation_font_color="#666")
    bt_fig.update_layout(template='plotly_dark', paper_bgcolor='#12121a', plot_bgcolor='#12121a', height=250, margin=dict(l=10, r=10, t=10, b=10), font=dict(family='IBM Plex Mono'), xaxis_title="Date", yaxis_title="Portfolio Value (USD)", yaxis_tickformat="$,.0f", hovermode='x unified')
    st.plotly_chart(bt_fig, use_container_width=True)

elif run and not ticker:
    st.warning("Enter a ticker symbol first.")
