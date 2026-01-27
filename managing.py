import streamlit as st
import pandas as pd
import os
import re
import duckdb

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants ---
NPI_COLUMNS = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
ALL_STOCK_COLUMNS = [
    'PhysicalStock', 'OveragedTireQty', 'IntransitQty',
    'QualityInspectionQty', 'BlockedStockQty', 'ATPonHand'
]

# --- Parquet Directory ---
PARQUET_DIR = "parquet_cache"
os.makedirs(PARQUET_DIR, exist_ok=True)

def parquet_path(label: str) -> str:
    return os.path.join(PARQUET_DIR, f"{label.replace(' ', '_')}.parquet")

def save_as_parquet(df: pd.DataFrame, label: str):
    try:
        df.to_parquet(parquet_path(label), index=False)
    except Exception as e:
        st.warning(f"Failed to write Parquet for {label}: {e}")

def load_parquet_if_exists(label: str):
    path = parquet_path(label)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None

# --- Week parsing for BDD400 ---
def parse_week_string(week_str):
    try:
        match = re.search(r'W\s*(\d{4})\s*/\s*(\d{1,2})', str(week_str))
        if match:
            year, week = match.groups()
            return pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%G-W%V-%u")
    except:
        return None
    return week_str

# --- Load Excel/CSV Files ---
def load_file_content(file_path_or_buffer, label):
    try:
        # Load raw
        if isinstance(file_path_or_buffer, str):
            df = (
                pd.read_csv(file_path_or_buffer)
                if file_path_or_buffer.endswith(".csv")
                else pd.read_excel(file_path_or_buffer)
            )
        else:
            try:
                df = pd.read_excel(file_path_or_buffer)
            except:
                df = pd.read_csv(file_path_or_buffer)

        # BDD400 week conversion
        if label == "BDD400" and "DateStamp" in df.columns:
            df["DateStamp"] = df["DateStamp"].apply(parse_week_string)
            df = df.dropna(subset=["DateStamp"])

        # Generic DateStamp parsing
        elif "DateStamp" in df.columns:
            df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
            df = df.dropna(subset=["DateStamp"])

        return df

    except Exception as e:
        st.error(f"Error loading {label}: {e}")
        return None

# --- Session State ---
if "data" not in st.session_state:
    st.session_state["data"] = {
        "Stock History": None,
        "BDD400": None,
        "Plant Capacity": None,
        "T&W Forecasts": None
    }

# --- DuckDB NPI Computation (no caching!) ---
def compute_npi_days_duckdb(df: pd.DataFrame, npi_categories: list[str]) -> pd.DataFrame:
    df2 = df.copy()

    # Ensure numeric cols
    for c in npi_categories:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)

    con = duckdb.connect()
    con.register("stock", df2)

    fields = []
    for c in npi_categories:
        fields.append(f"""
            CASE WHEN {c} = 0 THEN DateStamp END AS zero_{c},
            MAX(CASE WHEN {c} = 0 THEN DateStamp END)
                OVER (
                    PARTITION BY SapCode, PlantID
                    ORDER BY DateStamp
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS last_zero_{c},
            DATE_DIFF(
                'day',
                COALESCE(
                    MAX(CASE WHEN {c} = 0 THEN DateStamp END)
                        OVER (
                            PARTITION BY SapCode, PlantID
                            ORDER BY DateStamp
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ),
                    MIN(DateStamp) OVER (PARTITION BY SapCode, PlantID)
                ),
                DateStamp
            ) AS DaysSinceZero_{c}
        """)

    sql = f"""
        SELECT *, {",".join(fields)}
        FROM stock
        ORDER BY SapCode, PlantID, DateStamp
    """

    result = con.execute(sql).df()
    con.close()
    return result

# --- Sidebar ---
st.sidebar.title("App Navigation")
selection = st.sidebar.radio(
    "Go to:",
    [
        "Data load",
        "Non‑Productive Inventory (NPI) Management",
        "Planning Overview",
        "Storage Capacity Management"
    ]
)

# --- DATA LOAD PAGE ---
if selection == "Data load":

    st.header("📂 Data Management")

    filenames = {
        "Stock History": "StockHistory.xlsx",
        "BDD400": "003BDD400.xlsx",
        "Plant Capacity": "PlantCapacity.xlsx",
        "T&W Forecasts": "TWforecasts.xlsx"
    }

    cols = st.columns(2)

    for i, (label, fname) in enumerate(filenames.items()):
        with cols[i % 2]:
            st.subheader(label)

            upload = st.file_uploader(
                f"Upload {fname}",
                type=["xlsx", "csv"],
                key=f"upload_{label}"
            )

            if upload is not None:
                df_loaded = load_file_content(upload, label)
                if df_loaded is not None:
                    st.session_state["data"][label] = df_loaded
                    save_as_parquet(df_loaded, label)
                    st.success(f"⬆️ Loaded new {label}")
                else:
                    st.error("Upload failed.")
            else:
                # Try Parquet (fast path)
                df_parquet = load_parquet_if_exists(label)
                if df_parquet is not None:
                    st.session_state["data"][label] = df_parquet
                else:
                    # Fallback to default local files
                    if os.path.exists(fname):
                        df_loaded = load_file_content(fname, label)
                    else:
                        alt = fname.replace(".xlsx", ".csv")
                        df_loaded = load_file_content(alt, label) if os.path.exists(alt) else None

                    if df_loaded is not None:
                        st.session_state["data"][label] = df_loaded
                        save_as_parquet(df_loaded, label)
                    # else: keep None

            # Status badge
            if st.session_state["data"][label] is not None:
                st.success(f"✅ {label} Active")
            else:
                st.warning(f"⚠️ {label} missing")

# --- NPI MANAGEMENT ---
elif selection == "Non‑Productive Inventory (NPI) Management":

    st.header("📉 Non‑Productive Inventory (NPI) Management")

    df = st.session_state["data"].get("Stock History")

    if df is None:
        st.error("Please upload Stock History first.")
        st.stop()

    # Compute fresh NPI metrics
    df_aug = compute_npi_days_duckdb(df, NPI_COLUMNS)

    latest_date = df_aug["DateStamp"].max()
    df_latest = df_aug[df_aug["DateStamp"] == latest_date]

    # --- TOP SUMMARY ---
    st.subheader(f"📋 Global Plant Summary (As of {latest_date:%Y-%m-%d})")

    summary = (
        df_latest.groupby("PlantID")[ALL_STOCK_COLUMNS]
        .sum()
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "🔍 Material Drilldown",
        "⏳ Accumulation Monitor",
        "📈 Trend Analysis"
    ])

    # --- TAB 1 ---
    with tab1:
        st.subheader("Inventory by Material with Accumulation Days")

        colA, colB = st.columns([2, 1])

        all_plants = sorted(df_latest["PlantID"].unique())

        with colA:
            selected_plants = st.multiselect(
                "Select Plants",
                options=all_plants,
                default=all_plants
            )

        with colB:
            sort_category = st.selectbox("Sort By", NPI_COLUMNS)

        mat = df_latest[df_latest["PlantID"].isin(selected_plants)]

        agg_dict = {
            **{c: "sum" for c in (NPI_COLUMNS + ["PhysicalStock"])},
            **{f"DaysSinceZero_{c}": "max" for c in NPI_COLUMNS}
        }

        mat_display = (
            mat.groupby(["MaterialDescription", "PlantID", "SapCode"])
            .agg(agg_dict)
            .reset_index()
        )

        mat_display[f"Days Since {sort_category} Zero"] = mat_display[
            f"DaysSinceZero_{sort_category}"
        ]

        st.dataframe(
            mat_display.sort_values(by=sort_category, ascending=False),
            use_container_width=True
        )

    # --- TAB 2 ---
    with tab2:
        st.subheader("Time Since Last Zero (All Categories)")
        st.dataframe(mat_display, use_container_width=True)

    # --- TAB 3 ---
    with tab3:
        st.subheader("Evolution of NPI")

        all_plants = sorted(df["PlantID"].unique())

        plant_filt = st.multiselect(
            "Filter by Plant",
            options=all_plants,
            default=all_plants
        )

        trend = (
            df_aug[df_aug["PlantID"].isin(plant_filt)]
            .groupby("DateStamp")[NPI_COLUMNS]
            .sum()
            .reset_index()
            .sort_values("DateStamp")
        )

        if trend.empty:
            st.warning("No trend data for selected plants.")
        else:
            st.line_chart(trend, x="DateStamp", y=NPI_COLUMNS)

# --- OTHER PAGES Placeholder ---
else:
    st.header(selection)
    st.info("Implementation pending.")



