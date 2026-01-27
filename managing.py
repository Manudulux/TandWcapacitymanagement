import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants ---
NPI_COLUMNS = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
ALL_STOCK_COLUMNS = ['PhysicalStock', 'OveragedTireQty', 'IntransitQty', 'QualityInspectionQty', 'BlockedStockQty', 'ATPonHand']

# --- 1. Data Loading Logic ---
def load_file_content(file_path_or_buffer):
    try:
        if isinstance(file_path_or_buffer, str):
            if file_path_or_buffer.endswith('.csv'):
                df = pd.read_csv(file_path_or_buffer)
            else:
                df = pd.read_excel(file_path_or_buffer)
        else:
            try:
                df = pd.read_excel(file_path_or_buffer)
            except:
                df = pd.read_csv(file_path_or_buffer)
        
        if 'DateStamp' in df.columns:
            df['DateStamp'] = pd.to_datetime(df['DateStamp'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
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
                st.session_state['data'][label] = load_file_content(up)
            elif st.session_state['data'][label] is None and os.path.exists(fname):
                st.session_state['data'][label] = load_file_content(fname)
            
            if st.session_state['data'][label] is not None:
                st.success(f"✅ {label} Active")
            else:
                st.warning(f"⚠️ {label} missing.")

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
            with col_a:
                # Default to ALL plants
                all_plants = sorted(df['PlantID'].unique().tolist())
                selected_plants = st.multiselect("Select Plants", options=all_plants, default=all_plants)
            with col_b:
                sort_category = st.selectbox("Analyze & Sort By:", options=NPI_COLUMNS, index=0)
            
            # Filter latest data
            mat_latest = df_latest[df_latest['PlantID'].isin(selected_plants)]
            
            # Grouping for display
            mat_display = mat_latest.groupby(['MaterialDescription', 'PlantID', 'SapCode'])[NPI_COLUMNS + ['PhysicalStock']].sum().reset_index()

            # --- ACCUMULATION CALCULATION ---
            @st.cache_data
            def get_days_since_zero(sap_code, plant_id, category):
                # Get specific history for this SKU/Plant
                history = df[(df['SapCode'] == sap_code) & (df['PlantID'] == plant_id)].sort_values('DateStamp')
                if history.empty: return 0
                
                # Find last date where this specific category was 0
                zeros = history[history[category] == 0]
                if not zeros.empty:
                    last_zero_date = zeros['DateStamp'].max()
                else:
                    last_zero_date = history['DateStamp'].min()
                
                return (latest_date - last_zero_date).days

            # Calculate the days since zero only for the selected category
            if not mat_display.empty:
                with st.spinner("Calculating accumulation periods..."):
                    mat_display[f'Days Since {sort_category} Zero'] = mat_display.apply(
                        lambda x: get_days_since_zero(x['SapCode'], x['PlantID'], sort_category), axis=1
                    )
                
                # Sort descending
                mat_display = mat_display.sort_values(by=sort_category, ascending=False)
                st.dataframe(mat_display, use_container_width=True)
            else:
                st.info("No data available for the selected plants.")

        with tab2:
            st.subheader("Time Since Last Zero (All NPI Categories)")
            st.info("View how many days each category has remained non-zero across your portfolio.")
            # Similar to Tab 1 but shows all Age columns at once
            st.write("Refer to 'Material Drilldown' for specific categories, or use this list for a broad overview.")
            # (Keeping the simple logic from previous version for comparison)
            st.dataframe(mat_display, use_container_width=True)

        with tab3:
            st.subheader("Evolution of NPI")
            # Problem: Trend analysis was empty. 
            # Solution: Ensure dates are sorted and plants are pre-selected.
            plant_trend_filter = st.multiselect("Filter Trends by Plant", options=all_plants, default=all_plants, key="trend_plant_filt")
            
            # Aggregate by Date
            trend_df = df[df['PlantID'].isin(plant_trend_filter)].groupby('DateStamp')[NPI_COLUMNS].sum().reset_index()
            trend_df = trend_df.sort_values('DateStamp') # Critical for line charts
            
            if not trend_df.empty:
                st.line_chart(trend_df, x='DateStamp', y=NPI_COLUMNS)
            else:
                st.warning("No data found to plot trends.")

    else:
        st.error("Please upload 'Stock History' data first.")
else:
    st.header(selection)
    st.info("Section placeholder.")
