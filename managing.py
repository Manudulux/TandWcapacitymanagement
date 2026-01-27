import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants & Configuration ---
FILES_CONFIG = {
    "BDD400": {
        "filename": "003BDD400.xlsx",
        "help": "Load the BDD400 closing stock and production data."
    },
    "Plant Capacity": {
        "filename": "PlantCapacity.xlsx",
        "help": "Load the maximum storage and plant capacity limits."
    },
    "Stock History": {
        "filename": "StockHistory.xlsx",
        "help": "Load historical stock levels and aging data."
    },
    "T&W Forecasts": {
        "filename": "TWforecasts.xlsx",
        "help": "Load the transport and warehouse forecast projections."
    }
}

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state['data'] = {key: None for key in FILES_CONFIG.keys()}

# --- Helper Functions ---
def load_data(label, filename, uploaded_file):
    """Logic to prioritize uploaded file over local file."""
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file), "Uploaded"
        except Exception as e:
            st.error(f"Error reading uploaded {filename}: {e}")
    elif os.path.exists(filename):
        try:
            return pd.read_excel(filename), "Local (Default)"
        except Exception as e:
            st.error(f"Error reading local {filename}: {e}")
    return None, None

@st.cache_data
def calculate_accumulation(df, npi_vars, reference_date):
    """Calculates the number of days since each NPI category was last at zero."""
    acc_results = []
    # Group by Material and Warehouse to find specific bottlenecks
    groups = df.groupby(['SapCode', 'PlantID', 'MaterialDescription'])
    
    for (sap, plant, desc), group in groups:
        group = group.sort_values('DateStamp')
        res = {'Material': desc, 'Warehouse': plant, 'SapCode': sap}
        
        for var in npi_vars:
            current_val = group[group['DateStamp'] == reference_date][var].sum()
            if current_val > 0:
                # Find the most recent date where the stock was 0
                zeros = group[group[var] == 0]
                if not zeros.empty:
                    last_zero = zeros['DateStamp'].max()
                else:
                    # If it was never zero in history, use the earliest date in history
                    last_zero = group['DateStamp'].min()
                
                days = (reference_date - last_zero).days
                res[f'{var} Age (Days)'] = days
                res[f'{var} Qty'] = current_val
            else:
                res[f'{var} Age (Days)'] = 0
                res[f'{var} Qty'] = 0
        acc_results.append(res)
    
    return pd.DataFrame(acc_results)

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
    st.write("Upload your specific Excel files below. If no file is uploaded, the app looks for defaults in the root folder.")

    cols = st.columns(2)
    for i, (label, config) in enumerate(FILES_CONFIG.items()):
        with cols[i % 2]:
            st.subheader(label)
            uploaded_file = st.file_uploader(
                f"Upload {config['filename']}", 
                type=["xlsx"], 
                key=f"loader_{label}"
            )
            
            df, source = load_data(label, config['filename'], uploaded_file)
            
            if df is not None:
                st.session_state['data'][label] = df
                st.success(f"✅ {label} loaded from {source}")
                with st.expander("Preview Data"):
                    st.dataframe(df.head(5), use_container_width=True)
            else:
                st.error(f"❌ Missing: {config['filename']}")

# --- SECTION 2: NPI Management ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    
    df_stock = st.session_state['data'].get("Stock History")
    
    if df_stock is not None:
        # Pre-processing
        df_stock['DateStamp'] = pd.to_datetime(df_stock['DateStamp'])
        npi_vars = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
        
        # --- Filters ---
        st.sidebar.subheader("Analysis Filters")
        selected_wh = st.sidebar.multiselect("Warehouse", options=sorted(df_stock['PlantID'].unique()), default=df_stock['PlantID'].unique())
        selected_brand = st.sidebar.multiselect("Brand", options=sorted(df_stock['Brand'].unique()), default=df_stock['Brand'].unique())
        selected_season = st.sidebar.multiselect("Season", options=sorted(df_stock['Season'].unique()), default=df_stock['Season'].unique())
        
        min_date, max_date = df_stock['DateStamp'].min().to_pydatetime(), df_stock['DateStamp'].max().to_pydatetime()
        date_range = st.sidebar.slider("Period", min_date, max_date, (min_date, max_date))

        # Filter Logic
        mask = (
            df_stock['PlantID'].isin(selected_wh) & 
            df_stock['Brand'].isin(selected_brand) & 
            df_stock['Season'].isin(selected_season) &
            (df_stock['DateStamp'] >= date_range[0]) &
            (df_stock['DateStamp'] <= date_range[1])
        )
        df_filtered = df_stock[mask].copy()

        # --- Tabs ---
        tab1, tab2, tab3 = st.tabs(["📈 NPI Trends", "⏳ Accumulation Monitor", "📋 Material Summary"])

        with tab1:
            st.subheader("Inventory Quantities Over Time")
            df_trend = df_filtered.groupby('DateStamp')[npi_vars].sum().reset_index()
            st.line_chart(df_trend, x='DateStamp', y=npi_vars)
            
            latest_date = df_filtered['DateStamp'].max()
            current_totals = df_filtered[df_filtered['DateStamp'] == latest_date][npi_vars].sum()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Blocked Stock", f"{current_totals['BlockedStockQty']:,.0f}")
            m2.metric("Quality Inspection", f"{current_totals['QualityInspectionQty']:,.0f}")
            m3.metric("Overaged Tires", f"{current_totals['OveragedTireQty']:,.0f}")

        with tab2:
            st.subheader("Time Since Last Zero (Operational Bottlenecks)")
            st.info("Calculating days since stock was last cleared...")
            
            latest_dt = df_filtered['DateStamp'].max()
            df_acc = calculate_accumulation(df_filtered, npi_vars, latest_dt)
            
            st.dataframe(df_acc, use_container_width=True)

        with tab3:
            st.subheader("Top Critical Materials")
            metric_choice = st.selectbox("Rank bottlenecks by:", [f"{v} Age (Days)" for v in npi_vars])
            
            top_10 = df_acc.sort_values(by=metric_choice, ascending=False).head(10)
            
            if not top_10.empty and top_10[metric_choice].max() > 0:
                st.table(top_10[['Material', 'Warehouse', metric_choice, metric_choice.replace('Age (Days)', 'Qty')]])
            else:
                st.success("No active accumulation found for the selected filters.")
    else:
        st.warning("Please upload 'Stock History' in the Data Load section.")

# --- OTHER SECTIONS (Placeholders) ---
elif selection == "Planning Overview — T&W Forecast Projection":
    st.header("📈 Planning Overview — T&W Forecast Projection")
    st.info("Using data from TWforecasts.xlsx. Forecast logic to be implemented.")

elif selection == "Planning Overview — BDD400 Closing Stock":
    st.header("📦 Planning Overview — BDD400 Closing Stock")
    st.info("Using data from 003BDD400.xlsx. Stock projection logic to be implemented.")

elif selection == "Storage Capacity Management":
    st.header("🏗️ Storage Capacity Management")
    st.info("Comparing Stock History vs. Plant Capacity. Logic to be implemented.")

elif selection == "Mitigation Proposal":
    st.header("🛡️ Mitigation Proposal")
    st.info("Actionable recommendations based on NPI and Capacity data.")
