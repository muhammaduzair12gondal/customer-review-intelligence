import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Page Configuration
st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Native Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: #3B82F6;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    h1 {
        color: #1E293B;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

def sentiment_gauge(score, label):
    """Create a refined sentiment gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': f"Sentiment: {label.upper()}", 'font': {'size': 18, 'color': '#475569'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 0},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "#F1F5F9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "#FEE2E2"},
                {'range': [40, 60], 'color': "#FEF3C7"},
                {'range': [60, 100], 'color': "#DCFCE7"}
            ],
        },
        number={'suffix': "%", 'font': {'color': '#1E293B', 'size': 48}}
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def fake_probability_gauge(prob):
    """Create a refined fake probability gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Authenticity Risk", 'font': {'size': 18, 'color': '#475569'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 0},
            'bar': {'color': "#F59E0B"},
            'bgcolor': "#F1F5F9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': "#DCFCE7"},
                {'range': [50, 100], 'color': "#FEE2E2"}
            ],
        },
        number={'suffix': "%", 'font': {'color': '#1E293B', 'size': 48}}
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Header Section
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Customer Review Intelligence")
    st.markdown("Advanced analytics platform for sentiment, fake detection, and topic discovery.")
with col_h2:
    st.info("System Status: Operational")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["Analyze Review", "Batch Processing", "Global Insights"])

with tab1:
    col_in, col_out = st.columns([1, 1.2])
    
    with col_in:
        with st.container(border=True):
            st.subheader("Input Data")
            review_text = st.text_area(
                "Customer Feedback Text",
                placeholder="Type or paste the review here to begin analysis...",
                height=180
            )
            
            if st.button("Analyze Intelligence"):
                if not review_text.strip():
                    st.warning("Please provide input text.")
                else:
                    with st.spinner("Executing Neural Pipeline..."):
                        try:
                            resp = requests.post(f"{API_URL}/analyze", json={"text": review_text}, timeout=60)
                            if resp.status_code == 200:
                                st.session_state['res'] = resp.json()
                            else:
                                st.error(f"Backend Error: {resp.status_code}")
                        except Exception as e:
                            st.error(f"Network Failure: {str(e)}")
    
    with col_out:
        with st.container(border=True):
            if 'res' in st.session_state:
                res = st.session_state['res']
                st.subheader("Intelligence Overview")
                
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.plotly_chart(sentiment_gauge(res['sentiment_confidence'], res['sentiment']), use_container_width=True)
                with g_col2:
                    st.plotly_chart(fake_probability_gauge(res['is_fake_probability']), use_container_width=True)
            else:
                st.write("Awaiting input data...")

    if 'res' in st.session_state:
        res = st.session_state['res']
        
        col_asp, col_top = st.columns(2)
        
        with col_asp:
            with st.container(border=True):
                st.subheader("Product Aspects")
                st.markdown(
                    "<div style='font-size: 14px; color: #64748B; margin-bottom: 12px;'>"
                    "<b>How to read this:</b> This tool breaks down the review to identify specific features (like 'quality' or 'delivery') "
                    "and calculates whether the customer felt positive (green) or negative (red) about that specific feature."
                    "</div>", unsafe_allow_html=True
                )
                if res['aspects']:
                    asp_df = pd.DataFrame(res['aspects'])
                    # Map strings to numbers so the bar chart actually renders length
                    asp_df['score'] = asp_df['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
                    
                    fig_asp = px.bar(
                        asp_df, x='score', y='aspect', orientation='h',
                        color='score', color_continuous_scale='RdYlGn',
                        range_color=[-1, 1], range_x=[-1.2, 1.2]
                    )
                    fig_asp.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_asp, use_container_width=True)
                else:
                    st.info("No distinct aspects identified in this review.")
                
        with col_top:
            with st.container(border=True):
                st.subheader("Topic Classification")
                topics = [t.replace("_", " ").title() for t in res['top_topics']]
                if topics:
                    for t in topics:
                        st.success(f"📌 {t}")
                else:
                    st.info("No major topic identified for this text.")

with tab2:
    with st.container(border=True):
        st.subheader("Batch Dataset Processing")
        st.markdown("Upload a CSV file with a 'Text' column to process feedback in bulk.")
        batch_file = st.file_uploader("Select File", type=["csv"])
        if batch_file:
            st.success("File uploaded successfully.")
            st.button("Run Batch Pipeline")

with tab3:
    st.subheader("Historical Dataset Overview")
    data_path = Path("data/reviews_processed.parquet")
    if data_path.exists():
        df_p = pd.read_parquet(data_path)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Database Entries", f"{len(df_p):,}")
        m_col2.metric("Mean Rating", f"{df_p['Score'].mean():.2f}")
        m_col3.metric("Anomalous Reviews", f"{df_p['is_suspicious'].sum():,}")
        
        st.divider()
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            with st.container(border=True):
                st.subheader("Score Distribution")
                fig_hist = px.histogram(df_p, x='Score', nbins=5, color_discrete_sequence=['#3B82F6'])
                st.plotly_chart(fig_hist, use_container_width=True)
        with d_col2:
            with st.container(border=True):
                st.subheader("Sentiment Composition")
                counts = df_p['sentiment_name'].value_counts()
                fig_pie = px.pie(values=counts.values, names=counts.index, 
                                 color_discrete_sequence=['#10B981', '#EF4444', '#F59E0B'])
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Processed data cache not found.")

# Footer
st.divider()
st.caption("Intelligence System powered by Fine Food Reviews Dataset | 2026")
