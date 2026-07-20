import streamlit as st
from PIL import Image
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="VIRAL-Net Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    h1 {
        color: #FF4B4B;
        font-weight: bold;
    }
    h2 {
        color: #1F77B4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        height: 100%
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("📊 VIRAL-Net")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigasi",
        ["Home", "Data & Preprocessing", "Model Forecasting", "Analisis Pendukung"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **VIRAL-Net** adalah sistem forecasting frekuensi topik, hashtag, dan kata
        yang dilengkapi dengan analisis pendukung klasifikasi tingkat interaksi tweet.
        Sistem ini digunakan untuk analisis data tweet dan
        optimalisasi strategi marketing.
        
        **Fitur:**
        - Input Data & Preprocessing
        - Forecasting Topik, Hashtag, dan Kata
        - Analisis Pendukung Klasifikasi Tingkat Interaksi Tweet
        """
    )
    
    # Routing halaman
    if page == "Home":
        show_home()
    elif page == "Data & Preprocessing":
        show_data_preprocessing()
    elif page == "Model Forecasting":
        show_forecasting()
    elif page == "Analisis Pendukung":
        show_classification()

def show_home():
    """Halaman Home"""
    # Header
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    st.title("VIRAL-Net Dashboard")
    st.markdown("### *Social Media Marketing Campaign Optimization System*")
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ## Selamat Datang di VIRAL-Net!
    
    VIRAL-Net adalah sistem berbasis Machine Learning dan Deep Learning yang dirancang untuk:

    1. **Forecasting Frekuensi Topik, Hashtag, dan Kata** 
       - Memprediksi tren topik, hashtag, dan kata di masa depan
       - Membantu strategi konten media sosial

    2. **Analisis Pendukung Tingkat Interaksi Tweet** 
       - Menganalisis potensi viral sebuah tweet sebelum dipublikasikan
        - Memprediksi tingkat interaksi atau engagement yang akan didapatkan oleh tweet
       - Mempertimbangkan fitur-fitur penting yang mencakup aspek konten tweet serta karakteristik akun pengguna
    """)
    
    # Features
    st.markdown("---")
    st.markdown("## Fitur Utama")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>Analisis Data</h3>
        <p>Preprocessing dan eksplorasi data Twitter yang comprehensive</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>Forecasting</h3>
        <p>Prediksi tren topik, hashtag, dan kata dengan time series</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>Analisis Pendukung</h3>
        <p>Prediksi Viralitas Tweet dan Engagement yang Didapatkan</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to Use
    st.markdown("---")
    st.markdown("## Cara Menggunakan")
    
    st.markdown("""
    1. **Data & Preprocessing**: Upload dan lihat proses pembersihan data
    2. **Model Forecasting**: Analisis prediksi tren topik, hashtag, dan kata
    3. **Analisis Pendukung**: 
        - Prediksi viralitas pada tweet baru 
        - Prediksi engagement yang akan didapatkan
       - Analisis fitur-fitur penting untuk mengetahui karakteristik viralitas pada tweet
    """)
    
    # =========================
    # Model Architecture (Metric Style)
    # =========================
    st.markdown("---")
    st.markdown("## Model Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>IndoBERTweet</h3>
            <p>Text encoder untuk mengekstraksi representasi semantik dari teks tweet berbahasa Indonesia.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>BERTopic</h3>
            <p>Topic modeling untuk mengidentifikasi topik utama dari setiap tweet.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>RoBERTa</h3>
            <p>Model analisis sentimen untuk menghasilkan fitur polaritas emosi pada tweet.</p>
        </div>
        """, unsafe_allow_html=True)

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>LightGBM</h3>
            <p>
            Model time series forecasting untuk memprediksi tren topik, hashtag, dan kata
            berdasarkan data historis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="metric-card">
            <h3>Multi-Layer Perceptron (MLP)</h3>
            <p>
            Model fusi fitur yang menggabungkan embedding teks, sentimen, dan metadata akun
            untuk melakukan klasifikasi viralitas.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Developed by Deannisa SP | VIRAL-Net System © 2026</p>
    </div>
    """, unsafe_allow_html=True)

def show_data_preprocessing():
    """Import halaman data preprocessing"""
    from modules import data_preprocessing
    data_preprocessing.show()

def show_classification():
    """Import halaman klasifikasi"""
    from modules import classification
    classification.show()

def show_forecasting():
    """Import halaman forecasting"""
    from modules import forecasting
    forecasting.show()

if __name__ == "__main__":
    main()
