import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants & Configuration ---
FILES_CONFIG = {
    "BDD400": {"filename": "003BDD400.xlsx"},
    "Plant Capacity": {"filename": "PlantCapacity.xlsx"},
    "Stock History": {"filename": "StockHistory.xlsx"},
    "T&W Forecasts": {"filename": "TWforecasts.xlsx"}
}

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state['data'] = {key: None for key in FILES_CONFIG.keys()}

# --- Helper Functions ---
def load_data(label, filename, uploaded_file):
    """Priority: 1. Uploaded File, 2. Local File."""
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file), "Uploaded"
        except Exception as e:
            st.error(f"Error reading uploaded {filename}: {e}")
    elif os.path.exists(filename):
        try:
            return pd.read_excel(filename), "Local (Default)"
        except Exception as e:
            # Handle cases where CSVs are named .xlsx
            try:
                return pd.read_csv(filename), "Local (CSV fallback)"
            except:
                st.error(f"Error reading local {filename}: {e}")
    return None, None

@st.cache_data
def calculate_accumulation(df, npi_vars, reference_date):
    """Calculates days since each NPI category was last at zero."""
    acc_results = []
    
    # Ensure columns exist in the output even if no data is found
    columns = ['Material', 'Warehouse', 'SapCode']
    for var in npi_vars:
        columns.extend([f'{var} Age (Days)', f'{var} Qty'])
    
    if df.empty:
        return pd.DataFrame(columns=columns)

    groups = df.groupby(['SapCode', 'PlantID', 'MaterialDescription'])
    
    for (sap, plant, desc), group in groups:
        group = group.sort_values('DateStamp')
        res = {'Material': desc, 'Warehouse': plant, 'SapCode': sap}
        
        for var in npi_vars:
            # Get stock value at the latest date
            current_row = group[group['DateStamp'] == reference_date]
            current_val = current_row[var].sum() if not current_row.empty else 0
            
            if current_val > 0:
                zeros = group[group[var] == 0]
                last_zero = zeros['DateStamp'].max() if not zeros.empty else group['DateStamp'].min()
                res[f'{var} Age (Days)'] = (reference_date - last_zero).days
                res[f'{var} Qty'] = current_val
            else:
                res[f'{var} Age (Days)'] = 0
                res[f'{var} Qty'] = 0
        acc_results.append(res)
    
    return pd.DataFrame(acc_results) if acc_results else pd.DataFrame(columns=columns)

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
    cols = st.columns(2)
    for i, (label, config) in enumerate(FILES_CONFIG.items()):
        with cols[i % 2]:
            st.subheader(label)
            uploaded_file = st.file_uploader(f"Upload {config['filename']}", type=["xlsx", "csv"], key=f"ld_{label}")
            df, source = load_data(label, config['filename'], uploaded_file)
            if df is not None:
                st.session_state['data'][label] = df
                st.success(f"✅ Loaded from {source}")
            else:
                st.error(f"❌ Missing: {config['filename']}")

# --- SECTION 2: NPI Management ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    df_stock = st.session_state['data'].get("Stock History")
    
    if df_stock is not None:
        # Data Cleaning
        df_stock['DateStamp'] = pd.to_datetime(df_stock['DateStamp'])
        npi_vars = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
        
        # Sidebar Filters
        st.sidebar.subheader("Analysis Filters")
        wh_list = sorted(df_stock['PlantID'].unique().tolist())
        selected_wh = st.sidebar.multiselect("Warehouse", options=wh_list, default=wh_list)
        
        # Filter Logic
        df_filtered = df_stock[df_stock['PlantID'].isin(selected_wh)].copy()

        if not df_filtered.empty:
            tab1, tab2, tab3 = st.tabs(["📈 NPI Trends", "⏳ Accumulation Monitor", "📋 Material Summary"])

            with tab1:
                df_trend = df_filtered.groupby('DateStamp')[npi_vars].sum().reset_index()
                st.line_chart(df_trend, x='DateStamp', y=npi_vars)

            with tab2:
                latest_dt = df_filtered['DateStamp'].max()
                df_acc = calculate_accumulation(df_filtered, npi_vars, latest_dt)
                st.dataframe(df_acc, use_container_width=True)

            with tab3:
                # SAFE SORTING: Only sort if the column exists and df is not empty
                metric_choices = [f"{v} Age (Days)" for v in npi_vars]
                metric_choice = st.selectbox("Rank bottlenecks by:", metric_choices)
                
                if not df_acc.empty and metric_choice in df_acc.columns:
                    top_10 = df_acc.sort_values(by=metric_choice, ascending=False).head(10)
                    st.table(top_10[['Material', 'Warehouse', metric_choice, metric_choice.replace('Age (Days)', 'Qty')]])
                else:
                    st.info("No materials with current NPI found for the selected criteria.")
        else:
            st.warning("No data matches the selected filters.")
    else:
        st.warning("Please upload 'Stock History' first.")

# --- Placeholder for other sections ---
else:
    st.header(selection)
    st.info("Logic for this section is coming soon.")
