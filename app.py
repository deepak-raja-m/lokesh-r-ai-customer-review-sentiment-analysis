import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import random

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Sentix AI | Customer Review Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark modern FinTech/SaaS aesthetic styling
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 10%, #031B28 0%, #020C14 45%, #05131E 100%) !important;
        color: #E2E8F0;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        gap: 12px;
        background: rgba(8, 28, 44, 0.7);
        padding: 6px 14px;
        border-radius: 35px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    div[role="radiogroup"] > label {
        padding: 5px 12px;
        margin-bottom: 0px !important;
        font-weight: 500;
        cursor: pointer;
    }
    .user-pill {
        color: #38BDF8;
        font-weight: 600;
        font-size: 0.9rem;
        background: rgba(56, 189, 248, 0.12);
        padding: 6px 16px;
        border-radius: 25px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        display: inline-block;
    }
    .points-pill {
        background: linear-gradient(135deg, #10B981, #059669);
        color: #FFFFFF;
        padding: 6px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .fin-card {
        background: rgba(8, 24, 38, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic dataset if not present
@st.cache_data
def get_synthetic_reviews():
    sample_reviews = [
        ("The product quality is absolutely outstanding! Exceeded all my expectations.", 5, "Electronics"),
        ("Terrible delivery service. Arrived two weeks late and damaged.", 1, "Shipping"),
        ("Decent build quality, but battery backup could be better.", 3, "Electronics"),
        ("Customer support team was remarkably polite and resolved my query in 5 minutes.", 5, "Customer Support"),
        ("Worst experience ever. The application continuously crashes on checkout.", 1, "App Experience"),
        ("Value for money! Clean interface and very user-friendly navigation.", 4, "App Experience"),
        ("Highly disappointed with the pricing. Not worth the subscription fee.", 2, "Pricing"),
        ("Seamless connectivity and sleek design. Five stars all the way!", 5, "Electronics"),
        ("Refund policy is confusing and unhelpful.", 2, "Customer Support"),
        ("Average product. Neither too impressive nor too bad.", 3, "General")
    ]
    
    data = []
    base_time = datetime.now() - timedelta(days=60)
    for i in range(250):
        text, stars, dept = random.choice(sample_reviews)
        dt = base_time + timedelta(days=random.randint(0, 60), hours=random.randint(8, 22))
        data.append({
            "review_id": f"REV-{1000 + i}",
            "date": dt.strftime('%Y-%m-%d %H:%M:%S'),
            "department": dept,
            "rating": stars,
            "review_text": text
        })
    return pd.DataFrame(data)

# Sentiment scoring logic using VADER
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    else:
        return "Neutral", compound

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# --- LOGIN SCREEN ---
if not st.session_state.authenticated:
    _, col_mid, _ = st.columns([1, 1.6, 1])
    with col_mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="fin-card" style="text-align: center;">
            <div style="font-size: 3rem;">💬</div>
            <h2 style="color: #F8FAFC; margin-top: 5px;">Sentix AI Portal</h2>
            <p style="color: #94A3B8; font-size: 0.95rem;">AI-Powered Customer Sentiment Intelligence Engine</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            user_input = st.text_input("Username", value="", placeholder="Enter operator ID / username")
            pass_input = st.text_input("Security Key", type="password", value="", placeholder="Enter password")
            btn_sub = st.form_submit_button("Access Workspace")
            
            if btn_sub:
                if user_input.strip() != "":
                    st.session_state.authenticated = True
                    st.session_state.user_name = user_input
                    st.rerun()
                else:
                    st.error("Please provide a valid username.")
    st.stop()

# --- TOP NAVIGATION BAR ---
h1, h2, h3, h4 = st.columns([1.8, 4.4, 1.2, 0.8])
with h1:
    st.markdown('<h3 style="margin:0; color:#38BDF8; font-weight:800;">💬 Sentix AI</h3>', unsafe_allow_html=True)
with h2:
    selected_tab = st.radio(
        label="Navigation Menu",
        options=["🏠 Home", "📊 Sentiment Dashboard", "⚡ Live Review Tester", "⚙️ Model Details"],
        horizontal=True,
        label_visibility="collapsed"
    )
with h3:
    st.markdown('<div class="points-pill">⭐ 250 pts</div>', unsafe_allow_html=True)
with h4:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_name = ""
        st.rerun()

st.markdown(
    f'<div style="margin-top:-6px; margin-bottom:18px;"><span class="user-pill">Logged in as: {st.session_state.user_name}</span></div>',
    unsafe_allow_html=True
)
st.divider()

# Load and score dataset
df = get_synthetic_reviews()
df[['sentiment', 'polarity']] = df['review_text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# --- TAB 1: HOME ---
if selected_tab == "🏠 Home":
    st.markdown("## 🎯 AI Customer Review Sentiment Analysis")
    st.caption("Natural Language Processing system that automatically extracts polarity, emotional tone, and sentiment metrics from customer feedback.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="fin-card">
            <h4>💡 Automated Sentiment Scoring</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Eliminates manual review scanning by automatically tagging customer feedback as Positive, Neutral, or Negative in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="fin-card">
            <h4>📈 Departmental Heatmaps</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Pinpoints which domains (Shipping, Support, App Experience) have the highest negative sentiment for immediate operational fixes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="fin-card">
            <h4>⚡ Real-Time Testing Sandbox</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">
                Interactive sentiment classifier where reviewers can type custom sentences to inspect sentiment confidence scores instantly.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2: SENTIMENT DASHBOARD ---
elif selected_tab == "📊 Sentiment Dashboard":
    st.markdown("## 📊 Customer Sentiment Analytics & KPIs")
    
    pos_count = (df['sentiment'] == 'Positive').sum()
    neg_count = (df['sentiment'] == 'Negative').sum()
    neu_count = (df['sentiment'] == 'Neutral').sum()
    nps_proxy = round(((pos_count - neg_count) / len(df)) * 100, 1)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews Analyzed", len(df))
    k2.metric("Positive Feedback", f"{pos_count} ({pos_count/len(df)*100:.1f}%)")
    k3.metric("Negative Criticals", f"{neg_count} ({neg_count/len(df)*100:.1f}%)")
    k4.metric("Sentiment Health Score", f"{nps_proxy}%")
    
    st.divider()
    
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Sentiment Distribution")
        fig_pie = px.pie(
            df, names="sentiment",
            color="sentiment",
            color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#F59E0B"},
            hole=0.45
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E2E8F0")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g2:
        st.markdown("#### Department-Wise Negative Sentiment Breakdown")
        dept_neg = df[df['sentiment'] == 'Negative'].groupby('department').size().reset_index(name='count')
        fig_bar = px.bar(
            dept_neg, x='department', y='count',
            color='count', color_continuous_scale='Reds',
            labels={'count': 'Negative Reviews', 'department': 'Domain'}
        )
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E2E8F0")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.subheader("📋 Filtered Review Telemetry")
    dept_choice = st.selectbox("Filter by Category", ["All"] + sorted(df['department'].unique().tolist()))
    if dept_choice != "All":
        view_df = df[df['department'] == dept_choice]
    else:
        view_df = df
    st.dataframe(view_df[['date', 'department', 'rating', 'sentiment', 'polarity', 'review_text']], use_container_width=True)

# --- TAB 3: LIVE TESTER ---
elif selected_tab == "⚡ Live Review Tester":
    st.markdown("## ⚡ Live Feedback Sentiment Screener")
    st.caption("Type any custom feedback below to check how the NLP model evaluates customer polarity.")
    
    test_text = st.text_area("Customer Review Input", "The packaging was slightly damaged, but the product works like a dream! Very happy with it.")
    
    if st.button("Evaluate Sentiment"):
        label, score = analyze_sentiment(test_text)
        
        if label == "Positive":
            st.success(f"✅ POSITIVE SENTIMENT (Compound Score: {score:.3f})")
        elif label == "Negative":
            st.error(f"🚨 NEGATIVE CRITICAL FEEDBACK (Compound Score: {score:.3f})")
        else:
            st.warning(f"⚖️ NEUTRAL SENTIMENT (Compound Score: {score:.3f})")
            
        full_scores = sia.polarity_scores(test_text)
        m1, m2, m3 = st.columns(3)
        m1.metric("Positive Metric", f"{full_scores['pos']*100:.1f}%")
        m2.metric("Neutral Metric", f"{full_scores['neu']*100:.1f}%")
        m3.metric("Negative Metric", f"{full_scores['neg']*100:.1f}%")

# --- TAB 4: MODEL SPECS ---
elif selected_tab == "⚙️ Model Details":
    st.markdown("## ⚙️ Model Architecture & Technical Pipeline")
    st.markdown("""
    * **Algorithm:** VADER (Valence Aware Dictionary and sEntiment Reasoner) Lexicon-Based NLP
    * **Preprocessing:** Lemmatization, Punctuation emphasis scoring, and capitalization weight adjustment
    * **Evaluation Metric:** Compound Polarity Score range $[-1.0, +1.0]$
    * **Deployment:** Streamlit Cloud with Interactive Plotly Visualizations
    """)