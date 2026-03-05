import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sqlite3
from datetime import datetime

# --- ARAYÜZ YAPILANDIRMASI ---
st.set_page_config(page_title="DataScience Pro Hub", layout="wide", initial_sidebar_state="expanded")

# Kurumsal Görünüm İçin CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border-radius: 10px; background-color: white; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# --- SIDEBAR (KONTROL PANELİ) ---
st.sidebar.title("🛠️ Veri İşleme Merkezi")
uploaded_file = st.sidebar.file_uploader("Dosyanızı buraya bırakın", type=["csv", "xlsx"])

if uploaded_file:
    raw_df = load_data(uploaded_file)
    
    # Dinamik Filtreleme Katmanı
    st.sidebar.subheader("🎯 Akıllı Filtreler")
    all_cols = raw_df.columns.tolist()
    filter_col = st.sidebar.selectbox("Filtre Uygulanacak Sütun", ["Filtresiz"] + all_cols)
    
    if filter_col != "Filtresiz":
        selected_values = st.sidebar.multiselect(f"{filter_col} Değerleri", raw_df[filter_col].unique())
        if selected_values:
            df = raw_df[raw_df[filter_col].isin(selected_values)]
        else:
            df = raw_df.copy()
    else:
        df = raw_df.copy()

    # --- ANA PANEL ---
    st.title("🔬 Veri Analiz Laboratuvarı")
    
    # Üst Özet Kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Kayıt", len(df))
    c2.metric("Sayısal Değişkenler", len(df.select_dtypes(include=np.number).columns))
    c3.metric("Eksik Veri Hücresi", df.isnull().sum().sum())
    c4.metric("Kopya Satır", df.duplicated().sum())

    # SEKMELİ ANALİZ YAPISI
    tab_data, tab_stats, tab_pivot, tab_viz = st.tabs([
        "📋 Veri Gezgini", "🧠 İleri İstatistik", "🧮 Dinamik Pivot", "🎨 Grafik Laboratuvarı"
    ])

    # 1. SEKME: VERİ GEZGİNİ
    with tab_data:
        st.subheader("Veri Kümesi Önizlemesi")
        st.dataframe(df.head(100), use_container_width=True)
        
        col_inf1, col_inf2 = st.columns(2)
        with col_inf1:
            st.write("### Sütun Tipleri")
            st.write(df.dtypes)
        with col_inf2:
            st.write("### Eksik Veri Analizi")
            st.write(df.isnull().sum())

    # 2. SEKME: İLERİ İSTATİSTİK VE TEMİZLİK
    with tab_stats:
        st.header("🧠 Derinlemesine Analiz")
        num_df = df.select_dtypes(include=np.number)
        
        if not num_df.empty:
            st.subheader("🔗 Korelasyon Matrisi (İlişki Gücü)")
            corr = num_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.divider()
            
            st.subheader("🚨 Aykırı Değer (Outlier) Tespiti")
            target_col = st.selectbox("Sütun Seçin:", num_df.columns)
            z_scores = np.abs(stats.zscore(df[target_col].dropna()))
            outliers = df[z_scores > 3]
            
            c_out1, c_out2 = st.columns([1, 2])
            with c_out1:
                st.warning(f"Saptanan Aykırı Değer: {len(outliers)}")
                if st.button("Aykırı Değerleri Veriden At"):
                    df = df[z_scores <= 3]
                    st.success("Veri temizlendi!")
            with c_out2:
                fig_box = px.box(df, y=target_col, title=f"{target_col} Dağılım ve Uç Değerler")
                st.plotly_chart(fig_box)
        else:
            st.error("Analiz için sayısal veri bulunamadı.")

    # 3. SEKME: PİVOT ANALİZ
    with tab_pivot:
        st.header("🧮 Esnek Veri Özetleme (Pivot)")
        col_row = st.selectbox("Satır (Grup):", all_cols)
        col_val = st.selectbox("Değer (Sayı):", num_df.columns)
        agg_func = st.radio("İşlem:", ["mean", "sum", "count", "max", "min"], horizontal=True)
        
        pivot_res = df.groupby(col_row)[col_val].agg(agg_func).reset_index()
        st.dataframe(pivot_res, use_container_width=True)
        
        fig_pivot = px.bar(pivot_res, x=col_row, y=col_val, title=f"{col_row} bazında {agg_func} {col_val}")
        st.plotly_chart(fig_pivot, use_container_width=True)

    # 4. SEKME: GRAFİK LABORATUVARI
    with tab_viz:
        st.header("🎨 Özelleştirilmiş Görselleştirme")
        g1, g2 = st.columns([1, 3])
        
        with g1:
            chart_type = st.selectbox("Grafik Türü", ["Dağılım (Scatter)", "Çizgi (Line)", "Alan (Area)"])
            x_ax = st.selectbox("X Ekseni", all_cols)
            y_ax = st.selectbox("Y Ekseni", num_df.columns)
            color_ax = st.selectbox("Renk Ayrımı", ["Yok"] + all_cols)
            trend = st.checkbox("Trend Çizgisi Ekle (Regresyon)")

        with g2:
            color_param = None if color_ax == "Yok" else color_ax
            trend_param = "ols" if trend else None
            
            if chart_type == "Dağılım (Scatter)":
                fig = px.scatter(df, x=x_ax, y=y_ax, color=color_param, trendline=trend_param)
            elif chart_type == "Çizgi (Line)":
                fig = px.line(df, x=x_ax, y=y_ax, color=color_param)
            else:
                fig = px.area(df, x=x_ax, y=y_ax, color=color_param)
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Hoş geldiniz! Lütfen analiz etmek istediğiniz ham datayı (CSV veya Excel) soldaki panelden yükleyin.")
