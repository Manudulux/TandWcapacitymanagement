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
    """Save DataFrame to Parquet for fast reload next session."""
    try:
        df.to_parquet(parquet_path(label), index=False)
    except Exception as e:
        st.warning(f"Failed to write Parquet for {label}: {e}")


def load_parquet_if_exists(label: str):
    """Load Parquet file if available."""
    path = parquet_path(label)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


# --- Specialized parsing for BDD400 week strings ---
def parse_week_string(week_str):
    try:
        match = re.search(r'W\s*(\d{4})\s*/\s*(\d{1,2})', str(week_str))
        if match:
            year, week = match.groups()
            return pd.to_datetime(f'{year}-W{int(week):02d}-1', format='%G-W%V-%u')
    except:
        return None
    return week_str


# --- Load Excel/CSV Files ---
def load_file_content(file_path_or_buffer, label):
    try:
        # Load Raw Data
        if isinstance(file_path_or_buffer, str):
            df = (
                pd.read_csv(file_path_or_buffer)
                if file_path_or_buffer.endswith('.csv')
                else pd.read_excel(file_path_or_buffer)
            )
        else:
            try:
                df = pd.read_excel(file_path_or_buffer)
            except:
                df = pd.read_csv(file_path_or_buffer)

        # Format Handling for BDD400 (week strings)
        if label == "BDD400" and 'DateStamp' in df.columns:
            df['DateStamp'] = df['DateStamp'].apply(parse_week_string)
            df = df.dropna(subset=['DateStamp'])

        # Standard Datetime Parsing
        elif 'DateStamp' in df.columns:
            df['DateStamp'] = pd.to_datetime(df['DateStamp'], errors='coerce')
            df = df.dropna(subset=['DateStamp'])

        return df

    except Exception as e:
        st.error(f"Error loading {label}: {e}")
        return None


# --- Session State Init ---
if 'data' not in st.session_state:
    st.session_state['data'] = {
        "Stock History": None,
        "BDD400": None,
        "Plant Capacity": None,
        "T&W Forecasts": None
    }

# Version token to break cache when new data arrives
if 'data_version' not in st.session_state:
    st.session_state['data_version'] = 0


# --- DUCKDB NPI Aging Calculation ---
@st.cache_data(show_spinner=False)
def compute_npi_days_duckdb(df: pd.DataFrame, npi_categories: list[str], version: int) -> pd.DataFrame:
    """
    Uses DuckDB window functions to compute 'days since last zero'
    for all NPI categories in one vectorized SQL pass.
    The 'version' argument is not used inside but ensures cache invalidation.
    """
    df2 = df.copy()

    # Ensure numeric
    for c in npi_categories:
        df2[c] = pd.to_numeric(df2[c], errors='coerce').fillna(0)

    con = duckdb.connect()
    con.register("stock", df2)

    # Build SQL fields for each NPI category
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
selection = st.sidebar.radio("Go to:", [
    "Data load",
    "Non‑Productive Inventory (NPI) Management",
    "Planning Overview",
    "Storage Capacity Management"
])


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
            up = st.file_uploader(
                f"Upload {fname}", type=["xlsx", "csv"],
                key=f"up_{label}"
            )

            if up is not None:
                # USER UPLOAD -> save, clear caches, bump version, rerun
                df_loaded = load_file_content(up, label)
                st.session_state['data'][label] = df_loaded
                if df_loaded is not None:
                    save_as_parquet(df_loaded, label)
                    st.cache_data.clear()
                    st.session_state['data_version'] += 1
                    st.rerun()

            elif st.session_state['data'][label] is None:
                # Prefer Parquet on disk (no rerun here to avoid loops)
                df_parquet = load_parquet_if_exists(label)
                if df_parquet is not None:
                    st.session_state['data'][label] = df_parquet
                else:
                    # Legacy local Excel/CSV fallback (first-time local run)
                    if os.path.exists(fname):
                        df_loaded = load_file_content(fname, label)
                    else:
                        alt = fname.replace(".xlsx", ".csv")
                        df_loaded = load_file_content(alt, label) if os.path.exists(alt) else None

                    if df_loaded is not None:
                        st.session_state['data'][label] = df_loaded
                        save_as_parquet(df_loaded, label)
                        st.cache_data.clear()
                        st.session_state['data_version'] += 1
                        st.rerun()

            # Status indicator
            if st.session_state['data'][label] is not None:
                st.success(f"✅ {label} Active")
            else:
                st.warning(f"⚠️ {label} missing or failed to parse.")


# --- NPI MANAGEMENT PAGE ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")

    df = st.session_state['data'].get("Stock History")

    if df is not None:
        # Compute DuckDB-derived NPI aging (version guarantees recompute)
        df_aug = compute_npi_days_duckdb(df, NPI_COLUMNS, st.session_state['data_version'])

        latest_date = df_aug['DateStamp'].max()
        df_latest = df_aug[df_aug['DateStamp'] == latest_date]

        # --- TOP SUMMARY ---
        st.subheader(f"📋 Global Plant Summary (As of {latest_date:%Y-%m-%d})")
        summary = df_latest.groupby('PlantID')[ALL_STOCK_COLUMNS].sum().reset_index()
        st.dataframe(summary, use_container_width=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs([
            "🔍 Material Drilldown",
            "⏳ Accumulation Monitor",
            "📈 Trend Analysis"
        ])

        # --- TAB 1 ---
        with tab1:
            st.subheader("Inventory by Material with Accumulation Days")

            col_a, col_b = st.columns([2, 1])
            all_plants = sorted(df_latest['PlantID'].unique())

            with col_a:
                selected_plants = st.multiselect(
                    "Select Plants", options=all_plants, default=all_plants
                )

            with col_b:
                sort_category = st.selectbox(
                    "Analyze & Sort By:", options=NPI_COLUMNS
                )

            mat_latest = df_latest[df_latest['PlantID'].isin(selected_plants)]

            agg_dict = {
                **{c: "sum" for c in (NPI_COLUMNS + ["PhysicalStock"])},
                **{f"DaysSinceZero_{c}": "max" for c in NPI_COLUMNS}
            }

            mat_display = (
                mat_latest
                .groupby(['MaterialDescription', 'PlantID', 'SapCode'])
                .agg(agg_dict)
                .reset_index()
            )

            mat_display[f"Days Since {sort_category} Zero"] = \
                mat_display[f"DaysSinceZero_{sort_category}"]

            st.dataframe(
                mat_display.sort_values(by=sort_category, ascending=False),
                use_container_width=True
            )

        # --- TAB 2 ---
        with tab2:
            st.subheader("Time Since Last Zero (Consolidated View)")
            st.dataframe(mat_display, use_container_width=True)

        # --- TAB 3 ---
        with tab3:
            st.subheader("Evolution of NPI")

            all_plants = sorted(df['PlantID'].unique())
            plant_trend_filter = st.multiselect(
                "Filter Trends by Plant",
                options=all_plants,
                default=all_plants,
                key="trend_plant_filt"
            )

            trend_df = (
                df_aug[df_aug["PlantID"].isin(plant_trend_filter)]
                .groupby('DateStamp')[NPI_COLUMNS]
                .sum()
                .reset_index()
                .sort_values('DateStamp')
            )

            if not trend_df.empty:
                st.line_chart(trend_df, x='DateStamp', y=NPI_COLUMNS)
            else:
                st.warning("Trend data is empty for selected plants.")

    else:
        st.error("Please upload 'Stock History' data first.")


# --- PLACEHOLDERS FOR OTHER PAGES ---
else:
    st.header(selection)
    st.info("Implementation pending.")




