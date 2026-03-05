import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime
from scipy import stats

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DataPulse Pro", layout="wide", initial_sidebar_state="expanded")

# CSS ile Arayüzü Güzelleştirme
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- FONKSİYONLAR ---
def veritabani_kaydet(df, tablo_adi):
    conn = sqlite3.connect('analiz_merkezi.db')
    df['kayit_tarihi'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_sql(tablo_adi, conn, if_exists='append', index=False)
    conn.close()

# --- SIDEBAR (YAN MENÜ) ---
st.sidebar.title("🚀 DataPulse Pro")
st.sidebar.info("Ham veriyi yükleyin, derinlemesine analiz edin ve kaydedin.")
uploaded_file = st.sidebar.file_uploader("Dosya Seçin (CSV veya Excel)", type=["csv", "xlsx"])

# --- ANA EKRAN ---
if uploaded_file:
    # Veri Yükleme
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.title("🔍 Veri Analiz Raporu")
    
    # Üst Özet Kartları
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Satır Sayısı", len(df))
    col2.metric("Sütun Sayısı", len(df.columns))
    col3.metric("Eksik Veri", df.isnull().sum().sum())
    col4.metric("Sayısal Sütunlar", len(df.select_dtypes(include=[np.number]).columns))

    # --- SEKMELİ ANALİZ ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Veri Seti", "🔬 Detaylı Analiz", "🔗 İlişki Matrisi", "💾 Kayıt Yönetimi"])

    with tab1:
        st.subheader("Ham Veri Önizleme")
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("İstatistiksel Özet")
        st.write(df.describe())

    with tab2:
        st.subheader("🔬 Derinlemesine İstatistik ve Aykırı Değerler")
        sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if sayisal_sutunlar:
            secilen_s = st.selectbox("Analiz edilecek sütun:", sayisal_sutunlar)
            
            c1, c2 = st.columns(2)
            # Histogram ve Dağılım
            fig_hist = px.histogram(df, x=secilen_s, marginal="box", title=f"{secilen_s} Dağılımı ve Kutu Grafiği", color_discrete_sequence=['#636EFA'])
            c1.plotly_chart(fig_hist, use_container_width=True)
            
            # Aykırı Değer Tespiti (Z-Score)
            z_scores = np.abs(stats.zscore(df[secilen_s].dropna()))
            outliers = df[z_scores > 3]
            
            with c2:
                st.write(f"**{secilen_s}** için Aykırı Değer Analizi:")
                if not outliers.empty:
                    st.error(f"⚠️ {len(outliers)} adet aykırı değer saptandı!")
                    st.dataframe(outliers)
                else:
                    st.success("✅ Veri temiz görünüyor (Aykırı değer saptanmadı).")

    with tab3:
        st.subheader("🔗 Değişkenler Arası İlişkiler (Korelasyon)")
        if len(sayisal_sutunlar) > 1:
            corr = df[sayisal_sutunlar].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Korelasyon Isı Haritası")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("💡 Not: 1.0'a yakın değerler güçlü pozitif ilişkiyi, -1.0'a yakın değerler güçlü negatif ilişkiyi gösterir.")
        else:
            st.warning("İlişki analizi için en az 2 sayısal sütun gereklidir.")

    with tab4:
        st.subheader("💾 Verileri Veritabanına Aktar")
        tablo_ismi = st.text_input("Tablo Adı Belirleyin:", "analiz_tablosu")
        if st.button("Veritabanına Kaydet ve Kilitle"):
            veritabani_kaydet(df, tablo_ismi)
            st.balloons()
            st.success(f"'{tablo_ismi}' tablosu başarıyla 'analiz_merkezi.db' dosyasına kaydedildi.")

else:
    # Karşılama Ekranı
    st.header("Hoş Geldiniz!")
    st.write("Lütfen sol taraftaki menüyü kullanarak bir **CSV** veya **Excel** dosyası yükleyin.")
