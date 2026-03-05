import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# --- 1. SAYFA AYARLARI VE TASARIM ---
st.set_page_config(page_title="Karne Analiz Pro", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { border: 1px solid #d1d1d1; padding: 10px; border-radius: 5px; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AKILLI VERİ OKUYUCU (Data Cleaner) ---
def smart_load(file):
    # Dosyanın başındaki boş satırları ve "ANA SAYFA" yazılarını otomatik atlar
    if file.name.endswith('.csv'):
        # İlk 50 satırı kontrol et, başlığı (header) otomatik bul
        df_test = pd.read_csv(file, nrows=50, header=None)
        # En çok doluluk oranına sahip satırı başlık olarak seç (Tipik Karne Dosyası Yapısı)
        header_row = df_test.dropna(thresh=3).index[0] 
        file.seek(0)
        df = pd.read_csv(file, skiprows=header_row)
    else:
        df = pd.read_excel(file)
        # Excel için de benzer temizlik (Eğer ilk satırlar boşsa)
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # "Unnamed" sütunlarını ve tamamen boş sütunları temizle
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

# --- 3. SIDEBAR: KONTROL PANELİ ---
st.sidebar.title("🛠️ Analiz Ayarları")
uploaded_file = st.sidebar.file_uploader("Karne/Ham Data Dosyasını Yükle", type=["csv", "xlsx"])

if uploaded_file:
    df = smart_load(uploaded_file)
    
    # Veri setindeki tüm sütunları filtreleme için hazırla
    st.sidebar.subheader("🎯 Dinamik Filtreler")
    filter_cols = st.sidebar.multiselect("Filtrelemek istediğiniz sütunları seçin:", df.columns)
    
    for col in filter_cols:
        unique_vals = df[col].unique()
        selected = st.sidebar.multiselect(f"{col} Seçimi", unique_vals, default=unique_vals)
        df = df[df[col].isin(selected)]

    # --- 4. ANA PANEL: DETAYLI ANALİZ ---
    st.title("📊 Gelişmiş Operasyonel Analiz")
    
    # Üst Bilgi Kartları
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Satır", len(df))
    m2.metric("Sütun Sayısı", len(df.columns))
    m3.metric("Benzersiz Personel", df['Sicil'].nunique() if 'Sicil' in df.columns else "N/A")
    m4.metric("Hata Oranı (%)", round((df.isnull().sum().sum() / df.size) * 100, 2))

    # SEKMELER (Daha derin analiz için)
    t_data, t_pivot, t_stat, t_viz = st.tabs(["📋 Ham Veri", "🧮 Pivot Tablo", "🔬 İstatistiksel Test", "🎨 Grafik Lab"])

    with t_data:
        st.subheader("Temizlenmiş Veri Önizlemesi")
        st.dataframe(df, use_container_width=True)

    with t_pivot:
        st.subheader("🧮 Dinamik Özetleyici (Pivot)")
        col_group = st.selectbox("Gruplandırma (Satır):", df.columns, index=0)
        col_calc = st.selectbox("Hesaplanacak Sütun (Sayı):", df.select_dtypes(include=np.number).columns)
        
        pivot_table = df.groupby(col_group)[col_calc].agg(['mean', 'sum', 'count']).reset_index()
        st.write(pivot_table)
        
        fig_p = px.bar(pivot_table, x=col_group, y='mean', color='count', title=f"{col_group} Bazında Ortalama Performans")
        st.plotly_chart(fig_p, use_container_width=True)

    with t_stat:
        st.subheader("🔬 Derin İstatistik: Aykırı Değerler")
        num_col = st.selectbox("Analiz Sütunu:", df.select_dtypes(include=np.number).columns)
        
        # Z-Score ile Aykırı Değer Tespiti
        z_scores = np.abs(stats.zscore(df[num_col].dropna()))
        df_outliers = df.iloc[np.where(z_scores > 3)]
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.error(f"⚠️ Saptanan Uç Değer Sayısı: {len(df_outliers)}")
            st.dataframe(df_outliers)
        with col_s2:
            fig_box = px.box(df, y=num_col, points="all", title="Veri Dağılımı ve Sapmalar")
            st.plotly_chart(fig_box)

    with t_viz:
        st.subheader("🎨 Özel Grafik Oluşturucu")
        c1, c2, c3 = st.columns(3)
        x_ax = c1.selectbox("X Ekseni:", df.columns, key="x")
        y_ax = c2.selectbox("Y Ekseni:", df.select_dtypes(include=np.number).columns, key="y")
        color_ax = c3.selectbox("Renk Ayrımı (Kategori):", [None] + list(df.columns), key="c")
        
        fig_custom = px.scatter(df, x=x_ax, y=y_ax, color=color_ax, trendline="ols", 
                                title="Değişkenler Arası İlişki ve Trend")
        st.plotly_chart(fig_custom, use_container_width=True)

else:
    st.header("📂 Başlamak için Karne Dosyasını Yükleyin")
    st.info("Bu program, 'KARNE ÇALIŞMA GÜNCEL' dosyalarındaki karmaşık yapıyı otomatik olarak çözer.")
