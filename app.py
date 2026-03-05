import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# -----------------------------
# 1) SAYFA AYARLARI
# -----------------------------
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

# -----------------------------
# 2) YARDIMCI FONKSİYONLAR
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolon adlarını temizler: baş/son boşluk, tekrar eden boşluklar, unicode vb."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # "Unnamed" ve tamamen boş kolonlar
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df = df.dropna(how="all", axis=1)
    return df


def detect_header_row_csv(file, n=60, min_non_na=3):
    """
    CSV dosyada header satırını bulmaya çalışır.
    - İlk n satırı header=None okuyup, doluluk oranı en iyi satırı seçer.
    """
    df_test = pd.read_csv(file, nrows=n, header=None, dtype=str, engine="python")
    # Her satırda kaç hücre dolu?
    filled = df_test.notna().sum(axis=1)
    # min_non_na altında kalanları ele
    candidates = filled[filled >= min_non_na]
    if candidates.empty:
        return 0
    # En dolu satırın indexi
    return int(candidates.idxmax())


@st.cache_data(show_spinner=False)
def smart_load_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Cache'li loader: Streamlit uploader her seferinde dosyayı yeniden okumasın.
    """
    import io

    file = io.BytesIO(file_bytes)
    file.name = filename  # detect_header_row_csv bunu kullanıyor

    if filename.lower().endswith(".csv"):
        # header satırını tespit et
        file.seek(0)
        header_row = detect_header_row_csv(file)
        file.seek(0)
        df = pd.read_csv(file, skiprows=header_row, engine="python")
    else:
        file.seek(0)
        df = pd.read_excel(file)
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    df = normalize_columns(df)
    return df


def pick_person_col(df: pd.DataFrame) -> str | None:
    """Sicil yoksa olası personel kolonu seçmeye çalış."""
    preferred = ["Sicil", "SİCİL", "TC", "T.C.", "Personel", "Personel No", "Employee", "ID"]
    for c in preferred:
        if c in df.columns:
            return c
    # Alternatif: benzersizliği yüksek olan string kolonu
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols:
        return None
    uniq = {c: df[c].nunique(dropna=True) for c in obj_cols}
    # Çok küçük uniq genelde kategori olur; çok büyük uniq id olabilir. Orta-üst seçelim.
    return max(uniq, key=uniq.get) if uniq else None


def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dinamik filtre:
    - kategorik: multiselect
    - sayısal: slider (min-max)
    - tarih: date_input (min-max)
    """
    st.sidebar.subheader("🎯 Dinamik Filtreler")
    cols = st.sidebar.multiselect("Filtrelemek istediğiniz sütunları seçin:", df.columns)

    out = df.copy()

    for col in cols:
        series = out[col]

        # Tarih denemesi
        if series.dtype == "object":
            # küçük bir örnekle datetime parse deneyelim
            sample = series.dropna().astype(str).head(50)
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            is_date_like = parsed.notna().mean() > 0.6
        else:
            is_date_like = np.issubdtype(series.dtype, np.datetime64)

        if is_date_like:
            dt = pd.to_datetime(out[col], errors="coerce", dayfirst=True)
            min_d, max_d = dt.min(), dt.max()
            if pd.isna(min_d) or pd.isna(max_d):
                st.sidebar.info(f"{col} sütununda tarih filtrelemesi için yeterli veri yok.")
                continue

            start, end = st.sidebar.date_input(
                f"📅 {col} tarih aralığı",
                value=(min_d.date(), max_d.date()),
                min_value=min_d.date(),
                max_value=max_d.date(),
            )
            out = out[(dt.dt.date >= start) & (dt.dt.date <= end)]
            continue

        # Sayısal
        if np.issubdtype(series.dtype, np.number):
            min_v = float(np.nanmin(series.values)) if series.notna().any() else 0.0
            max_v = float(np.nanmax(series.values)) if series.notna().any() else 0.0
            if min_v == max_v:
                st.sidebar.caption(f"{col}: tek değer ({min_v}) olduğu için slider gösterilmedi.")
                continue
            vmin, vmax = st.sidebar.slider(f"🔢 {col} aralığı", min_v, max_v, (min_v, max_v))
            out = out[(out[col] >= vmin) & (out[col] <= vmax)]
            continue

        # Kategorik (çok fazla unique varsa arama kutusu gibi davranır)
        unique_vals = out[col].dropna().unique().tolist()
        unique_vals_sorted = sorted(unique_vals, key=lambda x: str(x))
        default_vals = unique_vals_sorted if len(unique_vals_sorted) <= 200 else unique_vals_sorted[:200]
        selected = st.sidebar.multiselect(f"🏷️ {col} seçimi", unique_vals_sorted, default=default_vals)
        if selected:
            out = out[out[col].isin(selected)]

    return out


# -----------------------------
# 3) SIDEBAR
# -----------------------------
st.sidebar.title("🛠️ Analiz Ayarları")
uploaded_file = st.sidebar.file_uploader("Karne/Ham Data Dosyasını Yükle", type=["csv", "xlsx"])

if not uploaded_file:
    st.header("📂 Başlamak için Karne Dosyasını Yükleyin")
    st.info("Bu program, 'KARNE ÇALIŞMA GÜNCEL' benzeri dosyalardaki karmaşık yapıyı otomatik çözer.")
    st.stop()

# Dosyayı oku (cache'li)
file_bytes = uploaded_file.getvalue()
df_raw = smart_load_cached(file_bytes, uploaded_file.name)

# Filtre uygula
df = build_filters(df_raw)

# -----------------------------
# 4) ANA PANEL
# -----------------------------
st.title("📊 Gelişmiş Operasyonel Analiz")

person_col = pick_person_col(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Toplam Satır", f"{len(df):,}")
m2.metric("Sütun Sayısı", f"{len(df.columns):,}")
m3.metric("Benzersiz Personel", df[person_col].nunique() if person_col else "N/A")
m4.metric("Hata Oranı (%)", round((df.isnull().sum().sum() / max(df.size, 1)) * 100, 2))

t_data, t_pivot, t_stat, t_viz = st.tabs(["📋 Ham Veri", "🧮 Pivot Tablo", "🔬 İstatistiksel Test", "🎨 Grafik Lab"])

# --- Ham Veri
with t_data:
    st.subheader("Temizlenmiş Veri Önizlemesi")
    st.dataframe(df, use_container_width=True)

# --- Pivot
with t_pivot:
    st.subheader("🧮 Dinamik Özetleyici (Pivot)")

    group_col = st.selectbox("Gruplandırma (Satır):", df.columns, index=0)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.warning("Pivot için sayısal sütun bulunamadı. (Örn: skor, adet, süre vb.)")
    else:
        calc_col = st.selectbox("Hesaplanacak Sütun (Sayı):", num_cols)
        agg_choice = st.multiselect("Agregasyonlar:", ["mean", "sum", "count", "median", "min", "max"], default=["mean", "sum", "count"])

        pivot_table = df.groupby(group_col)[calc_col].agg(agg_choice).reset_index()
        st.dataframe(pivot_table, use_container_width=True)

        if "mean" in pivot_table.columns:
            fig_p = px.bar(pivot_table, x=group_col, y="mean", title=f"{group_col} Bazında Ortalama ({calc_col})")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            fig_p = px.bar(pivot_table, x=group_col, y=agg_choice[0], title=f"{group_col} Bazında {agg_choice[0]} ({calc_col})")
            st.plotly_chart(fig_p, use_container_width=True)

# --- İstatistik / Aykırı
with t_stat:
    st.subheader("🔬 Derin İstatistik: Aykırı Değerler")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.warning("İstatistiksel analiz için sayısal sütun bulunamadı.")
    else:
        num_col = st.selectbox("Analiz Sütunu:", num_cols)
        method = st.radio("Yöntem:", ["Z-Score (|z|>3)", "IQR (1.5x)"], horizontal=True)

        s = df[num_col]

        if method.startswith("Z-Score"):
            clean = s.dropna()
            if clean.nunique() <= 1:
                st.info("Bu sütunda varyans yok gibi görünüyor; z-score anlamlı değil.")
            else:
                z = pd.Series(np.abs(stats.zscore(clean)), index=clean.index)
                out_idx = z[z > 3].index
                df_outliers = df.loc[out_idx]
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            df_outliers = df[(s < low) | (s > high)]

        c1, c2 = st.columns(2)
        with c1:
            st.error(f"⚠️ Saptanan Uç Değer Sayısı: {len(df_outliers)}")
            st.dataframe(df_outliers, use_container_width=True)
        with c2:
            fig_box = px.box(df, y=num_col, points="all", title=f"{num_col} Dağılımı ve Sapmalar")
            st.plotly_chart(fig_box, use_container_width=True)

# --- Grafik Lab
with t_viz:
    st.subheader("🎨 Özel Grafik Oluşturucu")

    chart_type = st.selectbox("Grafik Tipi:", ["Scatter (Trend)", "Bar", "Histogram", "Box"])

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if chart_type == "Scatter (Trend)":
        c1, c2, c3 = st.columns(3)
        x_ax = c1.selectbox("X Ekseni:", all_cols, key="x")
        y_ax = c2.selectbox("Y Ekseni (sayısal):", num_cols, key="y")
        color_ax = c3.selectbox("Renk Ayrımı (kategori):", [None] + all_cols, key="c")

        fig = px.scatter(df, x=x_ax, y=y_ax, color=color_ax, trendline="ols", title="Değişkenler Arası İlişki ve Trend")
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
            fig = px.histogram(df, x=col, title=f"{col} Histogram")
            st.plotly_chart(fig, use_container_width=True)

    else:  # Box
        if not num_cols:
            st.warning("Box için sayısal sütun yok.")
        else:
            y = st.selectbox("Y (sayısal):", num_cols)
            x = st.selectbox("X (opsiyonel kategori):", [None] + all_cols)
            fig = px.box(df, y=y, x=x, points="all", title="Box Plot")
            st.plotly_chart(fig, use_container_width=True)
