import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from datetime import datetime

# Sayfa Ayarları
st.set_page_config(page_title="DataPulse Analiz", layout="wide")
st.title("🚀 DataPulse: Ham Veri Analiz Motoru")
st.markdown("---")

# 1. Veri Yükleme Alanı
uploaded_file = st.file_uploader("Analiz edilecek dosyayı sürükleyin (CSV veya Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Veriyi Tanıma ve Okuma
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ {uploaded_file.name} yüklendi. Toplam {len(df)} satır veri bulundu.")

        # Sekmeli Görünüm (Düzenli bir arayüz için)
        tab1, tab2, tab3 = st.tabs(["📊 Veri Önizleme", "📈 Grafiksel Analiz", "💾 Veritabanı İşlemleri"])

        with tab1:
            st.subheader("Ham Veri Tablosu")
            st.dataframe(df, use_container_width=True)
            st.subheader("İstatistiksel Özet")
            st.write(df.describe())

        with tab2:
            st.subheader("Hızlı Görselleştirme")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                secilen_sutun = st.selectbox("Analiz edilecek sütunu seçin:", numeric_cols)
                fig = px.histogram(df, x=secilen_sutun, marginal="box", title=f"{secilen_sutun} Dağılımı", color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Grafik çizmek için sayısal bir sütun bulunamadı.")

        with tab3:
            st.subheader("Veriyi Sisteme Kaydet")
            proje_adi = st.text_input("Proje Adı/Notu:", "Genel Analiz")
            if st.button("Veritabanına Aktar"):
                conn = sqlite3.connect('veritabani.db')
                df['kayit_tarihi'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df['proje_notu'] = proje_adi
                df.to_sql('analiz_kayitlari', conn, if_exists='append', index=False)
                conn.close()
                st.balloons()
                st.success("Veriler SQLite veritabanına başarıyla işlendi!")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

else:
    st.info("💡 Başlamak için lütfen bir dosya yükleyin.")
