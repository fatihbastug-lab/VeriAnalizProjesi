import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3

# Sayfa Yapılandırması
st.set_page_config(page_title="Veri Analiz Motoru", layout="wide")
st.title("📊 Ham Veri Analiz ve Kayıt Sistemi")

# --- DOSYA YÜKLEME BÖLÜMÜ ---
st.sidebar.header("1. Adım: Veri Yükle")
uploaded_file = st.sidebar.file_uploader("CSV veya Excel dosyasını seçin", type=["csv", "xlsx"])

if uploaded_file:
    # Veriyi tanıma
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"'{uploaded_file.name}' başarıyla yüklendi!")

    # --- ANALİZ BÖLÜMÜ ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📋 Veri Özeti")
        st.write(df.describe()) # İstatistiksel özet
        
        if st.button("💾 Veritabanına Kaydet"):
            conn = sqlite3.connect('veritabani.db')
            df.to_sql('analiz_sonuclari', conn, if_exists='append', index=False)
            conn.close()
            st.toast("Veri başarıyla SQLite'a kaydedildi!", icon='✅')

    with col2:
        st.subheader("📈 Hızlı Görselleştirme")
        # Sayısal sütunları bulalım
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            secilen_sutun = st.selectbox("Grafik için bir sütun seçin:", numeric_cols)
            fig = px.histogram(df, x=secilen_sutun, title=f"{secilen_sutun} Dağılım Grafiği")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Görselleştirme için sayısal veri bulunamadı.")

    # --- TABLO GÖSTERİMİ ---
    st.divider()
    st.subheader("🔍 Tüm Veriler")
    st.dataframe(df)

else:
    st.info("Lütfen sol taraftaki menüden bir dosya yükleyerek başlayın.")
