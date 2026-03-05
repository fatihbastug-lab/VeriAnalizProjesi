import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------------------------
# 0) SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(page_title="Karne Analiz Pro", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { border: 1px solid #d1d1d1; padding: 10px; border-radius: 8px; background: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 1) YARDIMCI: KOLON TEMİZLEME
# ------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Unnamed ve tamamen boş kolonları sil
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df = df.dropna(how="all", axis=1)
    return df


# ------------------------------------------------------------
# 2) YARDIMCI: CSV HEADER TESPİTİ (GÜVENLİ)
# ------------------------------------------------------------
def detect_header_row_csv(file_like, nrows=60, min_non_na=3) -> int:
    """
    İlk nrows satırı header=None okur.
    Doluluk oranı yüksek olan satırı header kabul eder.
    Hiç aday yoksa 0 döner.
    """
    test = pd.read_csv(file_like, nrows=nrows, header=None, dtype=str, engine="python")
    filled = test.notna().sum(axis=1)
    candidates = filled[filled >= min_non_na]
    if candidates.empty:
        return 0
    return int(candidates.idxmax())


# ------------------------------------------------------------
# 3) DOSYA OKUMA (CACHE'LI)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def smart_load(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import io

    bio = io.BytesIO(file_bytes)
    bio.name = filename

    if filename.lower().endswith(".csv"):
        bio.seek(0)
        header_row = detect_header_row_csv(bio)
        bio.seek(0)
        df = pd.read_csv(bio, skiprows=header_row, engine="python")
    else:
        bio.seek(0)
        df = pd.read_excel(bio)
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    df = normalize_columns(df)
    return df


# ------------------------------------------------------------
# 4) YARDIMCI: PERSONEL KOLONU TAHMİNİ
# ------------------------------------------------------------
def guess_person_col(df: pd.DataFrame) -> str | None:
    candidates = ["Sicil", "SİCİL", "Personel", "Personel No", "Employee", "ID"]
    for c in candidates:
        if c in df.columns:
            return c

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return None

    nun = {c: df[c].nunique(dropna=True) for c in obj_cols}
    return max(nun, key=nun.get) if nun else None


# ------------------------------------------------------------
# 5) YARDIMCI: TARİH BENZERLİĞİ TESPİTİ
# ------------------------------------------------------------
def is_date_like(series: pd.Series) -> bool:
    if np.issubdtype(series.dtype, np.datetime64):
        return True
    if series.dtype != "object":
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() > 0.6


# ------------------------------------------------------------
# 6) YARDIMCI: Z-SCORE (SciPy'siz)
# ------------------------------------------------------------
def zscore_abs(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return ((x - mu) / sd).abs()


# ------------------------------------------------------------
# 7) SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("🛠️ Analiz Ayarları")
uploaded_file = st.sidebar.file_uploader("Karne/Ham Data Dosyasını Yükle", type=["csv", "xlsx"])

if not uploaded_file:
    st.header("📂 Başlamak için Karne Dosyasını Yükleyin")
    st.info("Bu uygulama, karne/ham veri dosyalarını otomatik temizler ve analiz eder.")
    st.stop()

# Dosya okuma (hata yakala)
try:
    file_bytes = uploaded_file.getvalue()
    df_raw = smart_load(file_bytes, uploaded_file.name)
except Exception as e:
    st.error("Dosya okunurken hata oluştu. Format beklenenden farklı olabilir.")
    st.exception(e)
    st.stop()

# ------------------------------------------------------------
# 8) DİNAMİK FİLTRELER
# ------------------------------------------------------------
df = df_raw.copy()
st.sidebar.subheader("🎯 Dinamik Filtreler")

filter_cols = st.sidebar.multiselect("Filtrelemek istediğiniz sütunları seçin:", df.columns)

for col in filter_cols:
    s = df[col]

    # Tarih
    if is_date_like(s):
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        min_d, max_d = dt.min(), dt.max()
        if pd.isna(min_d) or pd.isna(max_d):
            st.sidebar.caption(f"{col}: Tarih filtrelemek için yeterli veri yok.")
            continue

        start, end = st.sidebar.date_input(
            f"📅 {col} tarih aralığı",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
        df = df[(dt.dt.date >= start) & (dt.dt.date <= end)]
        continue

    # Sayısal
    if np.issubdtype(s.dtype, np.number):
        if s.dropna().empty:
            st.sidebar.caption(f"{col}: Boş sayısal sütun.")
            continue
        min_v = float(np.nanmin(s.values))
        max_v = float(np.nanmax(s.values))
        if min_v == max_v:
            st.sidebar.caption(f"{col}: Tek değer ({min_v}).")
            continue
        vmin, vmax = st.sidebar.slider(f"🔢 {col} aralığı", min_v, max_v, (min_v, max_v))
        df = df[(df[col] >= vmin) & (df[col] <= vmax)]
        continue

    # Kategorik
    unique_vals = df[col].dropna().unique().tolist()
    unique_vals = sorted(unique_vals, key=lambda x: str(x))
    default_vals = unique_vals if len(unique_vals) <= 200 else unique_vals[:200]
    selected = st.sidebar.multiselect(f"🏷️ {col} seçimi", unique_vals, default=default_vals)
    if selected:
        df = df[df[col].isin(selected)]

# ------------------------------------------------------------
# 9) ÜST KPI'LAR
# ------------------------------------------------------------
st.title("📊 Karne Analiz Pro (Yeniden Yazılmış)")

person_col = guess_person_col(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Toplam Satır", f"{len(df):,}")
m2.metric("Sütun Sayısı", f"{len(df.columns):,}")
m3.metric("Benzersiz Personel", df[person_col].nunique() if person_col else "N/A")
m4.metric("Hata Oranı (%)", round((df.isnull().sum().sum() / max(df.size, 1)) * 100, 2))

# Büyük veri için örnekleme (grafikler hızlansın)
MAX_VIZ = 30000
df_viz = df.sample(MAX_VIZ, random_state=42) if len(df) > MAX_VIZ else df

# ------------------------------------------------------------
# 10) SEKME YAPISI
# ------------------------------------------------------------
t_data, t_pivot, t_stat, t_viz = st.tabs(
    ["📋 Ham Veri", "🧮 Pivot Tablo", "🔬 Aykırı Değer", "🎨 Grafik Lab"]
)

# ------------------ Ham Veri ------------------
with t_data:
    st.subheader("Temizlenmiş Veri Önizlemesi")
    st.dataframe(df, use_container_width=True)

# ------------------ Pivot ------------------
with t_pivot:
    st.subheader("🧮 Dinamik Pivot")

    group_col = st.selectbox("Gruplandırma (Satır):", df.columns, index=0)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.warning("Pivot için sayısal sütun yok. (Örn: skor, adet, süre)")
    else:
        calc_col = st.selectbox("Hesaplanacak Sayısal Sütun:", num_cols)
        agg_choice = st.multiselect(
            "Agregasyonlar:", ["mean", "sum", "count", "median", "min", "max"],
            default=["mean", "sum", "count"]
        )

        pivot = df.groupby(group_col)[calc_col].agg(agg_choice).reset_index()
        st.dataframe(pivot, use_container_width=True)

        # Basit görselleştirme
        y_col = "mean" if "mean" in pivot.columns else agg_choice[0]
        fig = px.bar(pivot, x=group_col, y=y_col, title=f"{group_col} Bazında {y_col}({calc_col})")
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Aykırı Değer ------------------
with t_stat:
    st.subheader("🔬 Aykırı Değer Analizi")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.warning("Aykırı değer analizi için sayısal sütun yok.")
    else:
        num_col = st.selectbox("Analiz Sütunu:", num_cols)
        method = st.radio("Yöntem:", ["Z-Score (|z|>3)", "IQR (1.5x)"], horizontal=True)

        s = df[num_col]

        if method.startswith("Z-Score"):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if clean.nunique() <= 1:
                st.info("Bu sütunda varyans yok; z-score anlamlı değil.")
                df_out = df.iloc[0:0]
            else:
                z = zscore_abs(clean)
                out_idx = z[z > 3].index
                df_out = df.loc[out_idx]
        else:
            x = pd.to_numeric(s, errors="coerce")
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df_out = df[(x < low) | (x > high)]

        c1, c2 = st.columns(2)
        with c1:
            st.error(f"⚠️ Uç değer sayısı: {len(df_out)}")
            st.dataframe(df_out, use_container_width=True)

        with c2:
            # points="all" büyük veride ağır; df_viz ile hızlandırıyoruz
            fig = px.box(df_viz, y=num_col, points="outliers", title=f"{num_col} Dağılımı (Örneklenmiş)")
            st.plotly_chart(fig, use_container_width=True)

# ------------------ Grafik Lab ------------------
with t_viz:
    st.subheader("🎨 Grafik Lab")

    chart_type = st.selectbox("Grafik Tipi:", ["Scatter (Trend)", "Bar", "Histogram", "Box"])

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if chart_type == "Scatter (Trend)":
        c1, c2, c3 = st.columns(3)
        x_ax = c1.selectbox("X Ekseni:", all_cols, key="x")
        y_ax = c2.selectbox("Y Ekseni (sayısal):", num_cols, key="y") if num_cols else None
        color_ax = c3.selectbox("Renk (opsiyonel):", [None] + all_cols, key="c")

        if not y_ax:
            st.warning("Scatter için sayısal Y sütunu yok.")
        else:
            fig = px.scatter(
                df_viz, x=x_ax, y=y_ax, color=color_ax,
                trendline="ols", title="İlişki & Trend (Örneklenmiş)"
            )
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar":
        cat_col = st.selectbox("Kategori (X):", all_cols)
        if not num_cols:
            st.warning("Bar grafik için sayısal sütun yok.")
        else:
            val_col = st.selectbox("Değer (Y):", num_cols)
            agg = st.selectbox("Toplama şekli:", ["sum", "mean", "count"])
            g = df.groupby(cat_col)[val_col].agg(agg).reset_index()
            fig = px.bar(g, x=cat_col, y=val_col, title=f"{cat_col} - {agg}({val_col})")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram":
        if not num_cols:
            st.warning("Histogram için sayısal sütun yok.")
        else:
            col = st.selectbox("Sütun:", num_cols)
            fig = px.histogram(df_viz, x=col, title=f"{col} Histogram (Örneklenmiş)")
            st.plotly_chart(fig, use_container_width=True)

    else:  # Box
        if not num_cols:
            st.warning("Box için sayısal sütun yok.")
        else:
            y = st.selectbox("Y (sayısal):", num_cols)
            x = st.selectbox("X (opsiyonel kategori):", [None] + all_cols)
            fig = px.box(df_viz, y=y, x=x, points="outliers", title="Box Plot (Örneklenmiş)")
            st.plotly_chart(fig, use_container_width=True)
