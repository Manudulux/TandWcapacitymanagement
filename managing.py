import streamlit as st
import pandas as pd
import os
import re

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants ---
NPI_COLUMNS = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
ALL_STOCK_COLUMNS = ['PhysicalStock', 'OveragedTireQty', 'IntransitQty', 'QualityInspectionQty', 'BlockedStockQty', 'ATPonHand']

# --- 1. Specialized Data Loading Logic ---
def parse_week_string(week_str):
    """Converts 'W 2026 / 04' into a datetime object (Monday of that week)."""
    try:
        match = re.search(r'W\s*(\d{4})\s*/\s*(\d{1,2})', str(week_str))
        if match:
            year, week = match.groups()
            # ISO week format: Year-WWeek-DayNumber (1 for Monday)
            return pd.to_datetime(f'{year}-W{int(week):02d}-1', format='%G-W%V-%u')
    except:
        return None
    return week_str

def load_file_content(file_path_or_buffer, label):
    try:
        # Load Raw Data
        if isinstance(file_path_or_buffer, str):
            df = pd.read_csv(file_path_or_buffer) if file_path_or_buffer.endswith('.csv') else pd.read_excel(file_path_or_buffer)
        else:
            try:
                df = pd.read_excel(file_path_or_buffer)
            except:
                df = pd.read_csv(file_path_or_buffer)
        
        # Format Handling for BDD400 (Week Strings)
        if label == "BDD400" and 'DateStamp' in df.columns:
            df['DateStamp'] = df['DateStamp'].apply(parse_week_string)
            df = df.dropna(subset=['DateStamp']) # Clean up unparseable rows
        
        # Standard Datetime Parsing for others
        elif 'DateStamp' in df.columns:
            df['DateStamp'] = pd.to_datetime(df['DateStamp'], errors='coerce')
            df = df.dropna(subset=['DateStamp'])

        return df
    except Exception as e:
        st.error(f"Error loading {label}: {e}")
        return None

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state['data'] = {"Stock History": None, "BDD400": None, "Plant Capacity": None, "T&W Forecasts": None}

# --- Sidebar Navigation ---
st.sidebar.title("App Navigation")
selection = st.sidebar.radio("Go to:", [
    "Data load", 
    "Non‑Productive Inventory (NPI) Management", 
    "Planning Overview", 
    "Storage Capacity Management"
])

# --- SECTION 1: Data Load ---
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
            up = st.file_uploader(f"Upload {fname}", type=["xlsx", "csv"], key=f"up_{label}")
            if up is not None:
                st.session_state['data'][label] = load_file_content(up, label)
            elif st.session_state['data'][label] is None and os.path.exists(fname):
                # Check for local file presence (handling the .csv naming from the environment)
                local_path = fname if os.path.exists(fname) else fname.replace(".xlsx", ".csv")
                if os.path.exists(local_path):
                    st.session_state['data'][label] = load_file_content(local_path, label)
            
            if st.session_state['data'][label] is not None:
                st.success(f"✅ {label} Active")
            else:
                st.warning(f"⚠️ {label} missing or failed to parse.")

# --- SECTION 2: NPI Management ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    df = st.session_state['data'].get("Stock History")

    if df is not None:
        latest_date = df['DateStamp'].max()
        
        # --- TOP SUMMARY ---
        st.subheader(f"📋 Global Plant Summary (As of {latest_date.strftime('%Y-%m-%d')})")
        df_latest = df[df['DateStamp'] == latest_date]
        summary = df_latest.groupby('PlantID')[ALL_STOCK_COLUMNS].sum().reset_index()
        st.dataframe(summary.style.format(precision=0).highlight_max(axis=0), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["🔍 Material Drilldown", "⏳ Accumulation Monitor", "📈 Trend Analysis"])

        with tab1:
            st.subheader("Inventory by Material with Accumulation Days")
            col_a, col_b = st.columns([2, 1])
            all_plants = sorted(df['PlantID'].unique().tolist())
            with col_a:
                selected_plants = st.multiselect("Select Plants", options=all_plants, default=all_plants)
            with col_b:
                sort_category = st.selectbox("Analyze & Sort By:", options=NPI_COLUMNS, index=0)
            
            mat_latest = df_latest[df_latest['PlantID'].isin(selected_plants)]
            mat_display = mat_latest.groupby(['MaterialDescription', 'PlantID', 'SapCode'])[NPI_COLUMNS + ['PhysicalStock']].sum().reset_index()

            @st.cache_data
            def get_days_since_zero(sap_code, plant_id, category):
                history = df[(df['SapCode'] == sap_code) & (df['PlantID'] == plant_id)].sort_values('DateStamp')
                zeros = history[history[category] == 0]
                last_zero_date = zeros['DateStamp'].max() if not zeros.empty else history['DateStamp'].min()
                return (latest_date - last_zero_date).days

            if not mat_display.empty:
                with st.spinner("Calculating accumulation..."):
                    mat_display[f'Days Since {sort_category} Zero'] = mat_display.apply(
                        lambda x: get_days_since_zero(x['SapCode'], x['PlantID'], sort_category), axis=1
                    )
                st.dataframe(mat_display.sort_values(by=sort_category, ascending=False), use_container_width=True)
            else:
                st.info("No data available for selected criteria.")

        with tab2:
            st.subheader("Time Since Last Zero (Consolidated View)")
            st.write("Displays aging for all NPI categories simultaneously for current stock.")
            st.dataframe(mat_display, use_container_width=True)

        with tab3:
            st.subheader("Evolution of NPI")
            plant_trend_filter = st.multiselect("Filter Trends by Plant", options=all_plants, default=all_plants, key="trend_plant_filt")
            trend_df = df[df['PlantID'].isin(plant_trend_filter)].groupby('DateStamp')[NPI_COLUMNS].sum().reset_index()
            trend_df = trend_df.sort_values('DateStamp')
            
            if not trend_df.empty:
                st.line_chart(trend_df, x='DateStamp', y=NPI_COLUMNS)
            else:
                st.warning("Trend data is empty for selected plants.")

    else:
        st.error("Please upload 'Stock History' data first.")

else:
    st.header(selection)
    st.info("Implementation pending.")
