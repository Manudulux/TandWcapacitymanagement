import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants & Configuration ---
FILES_CONFIG = {
    "BDD400": {"filename": "003BDD400.xlsx"},
    "Plant Capacity": {"filename": "PlantCapacity.xlsx"},
    "Stock History": {"filename": "StockHistory.xlsx"},
    "T&W Forecasts": {"filename": "TWforecasts.xlsx"}
}

# --- 1. Robust Data Loading Logic ---
def load_file_content(file_path_or_buffer):
    """Try reading as Excel, fallback to CSV."""
    try:
        return pd.read_excel(file_path_or_buffer)
    except Exception:
        try:
            return pd.read_csv(file_path_or_buffer)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return None

# Initialize Session State and Auto-load local files on first run
if 'data' not in st.session_state:
    st.session_state['data'] = {}
    for label, config in FILES_CONFIG.items():
        if os.path.exists(config['filename']):
            st.session_state['data'][label] = load_file_content(config['filename'])
        else:
            st.session_state['data'][label] = None

# --- Sidebar Navigation ---
st.sidebar.title("App Navigation")
selection = st.sidebar.radio("Go to:", [
    "Data load", 
    "Non‑Productive Inventory (NPI) Management", 
    "Planning Overview — T&W Forecast Projection", 
    "Planning Overview — BDD400 Closing Stock", 
    "Storage Capacity Management", 
    "Mitigation Proposal"
])

# --- SECTION 1: Data Load ---
if selection == "Data load":
    st.header("📂 Data Management")
    st.info("Files uploaded here will override the default local files for this session.")
    
    cols = st.columns(2)
    for i, (label, config) in enumerate(FILES_CONFIG.items()):
        with cols[i % 2]:
            st.subheader(label)
            # The 'key' ensures Streamlit tracks this specific uploader
            uploaded_file = st.file_uploader(f"Upload {config['filename']}", type=["xlsx", "csv"], key=f"uploader_{label}")
            
            # If user drags and drops, update session state immediately
            if uploaded_file is not None:
                new_df = load_file_content(uploaded_file)
                if new_df is not None:
                    st.session_state['data'][label] = new_df
                    st.success(f"✅ Successfully uploaded {uploaded_file.name}")
            
            # Display status
            if st.session_state['data'][label] is not None:
                st.caption(f"Status: Data loaded ({len(st.session_state['data'][label])} rows)")
            else:
                st.error(f"Status: Missing {config['filename']}")

# --- SECTION 2: NPI Management ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    
    df_stock = st.session_state['data'].get("Stock History")
    
    if df_stock is not None:
        # Data Preparation
        df_stock['DateStamp'] = pd.to_datetime(df_stock['DateStamp'])
        all_categories = [
            'PhysicalStock', 'OveragedTireQty', 'IntransitQty', 
            'QualityInspectionQty', 'BlockedStockQty', 'ATPonHand'
        ]
        
        # Ensure categories exist in data
        existing_cats = [c for c in all_categories if c in df_stock.columns]
        
        # --- NEW: Latest Inventory Summary by Plant ---
        st.subheader("📋 Latest Inventory Snapshot (All Categories)")
        latest_date = df_stock['DateStamp'].max()
        st.write(f"Showing data for the latest available date: **{latest_date.strftime('%Y-%m-%d')}**")
        
        df_latest = df_stock[df_stock['DateStamp'] == latest_date]
        plant_summary = df_latest.groupby('PlantID')[existing_cats].sum().reset_index()
        
        # Adding a Total row for convenience
        total_row = plant_summary[existing_cats].sum().to_frame().T
        total_row['PlantID'] = 'TOTAL'
        plant_summary = pd.concat([plant_summary, total_row], ignore_index=True)
        
        st.dataframe(plant_summary.style.format(precision=0).highlight_max(axis=0, color='#e6f3ff'), use_container_width=True)
        
        st.divider()

        # --- Analysis Tabs ---
        tab1, tab2, tab3 = st.tabs(["📈 NPI Trends", "⏳ Accumulation Monitor", "📋 Bottleneck Materials"])

        with tab1:
            # Filters for the trend chart
            selected_wh = st.multiselect("Filter Chart by Warehouse", options=sorted(df_stock['PlantID'].unique()), default=df_stock['PlantID'].unique())
            df_trend_filt = df_stock[df_stock['PlantID'].isin(selected_wh)]
            
            npi_vars = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
            df_trend = df_trend_filt.groupby('DateStamp')[npi_vars].sum().reset_index()
            st.line_chart(df_trend, x='DateStamp', y=npi_vars)

        with tab2:
            st.subheader("Time Since Last Zero")
            # Reuse your accumulation logic here...
            st.info("This section calculates how long NPI has been building up without clearing.")
            # (Calculation logic from previous version goes here, using df_stock)

    else:
        st.warning("Please upload 'Stock History' in the Data Load section to see the summary.")

# --- OTHER SECTIONS (Placeholders) ---
else:
    st.header(selection)
    st.info("Logic for this section is pending.")
