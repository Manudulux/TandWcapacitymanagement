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
        # Try reading as Excel, then CSV
        try:
            df = pd.read_excel(file_path_or_buffer)
        except:
            df = pd.read_csv(file_path_or_buffer)
        
        # Standardize Date Column
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
            
            # Check for upload or local file
            if up is not None:
                st.session_state['data'][label] = load_file_content(up)
            elif st.session_state['data'][label] is None and os.path.exists(fname):
                st.session_state['data'][label] = load_file_content(fname)
            
            if st.session_state['data'][label] is not None:
                st.success(f"✅ {label} Active ({len(st.session_state['data'][label])} rows)")
            else:
                st.warning(f"⚠️ {label} missing. Please upload.")

# --- SECTION 2: NPI Management ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    df = st.session_state['data'].get("Stock History")

    if df is not None:
        # --- TOP SUMMARY ---
        latest_date = df['DateStamp'].max()
        st.subheader(f"📋 Global Plant Summary (As of {latest_date.strftime('%Y-%m-%d')})")
        
        df_latest = df[df['DateStamp'] == latest_date]
        summary = df_latest.groupby('PlantID')[ALL_STOCK_COLUMNS].sum().reset_index()
        st.dataframe(summary.style.format(precision=0).highlight_max(axis=0), use_container_width=True)

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs(["🔍 Material Drilldown", "⏳ Accumulation Monitor", "📈 Trend Analysis"])

        with tab1:
            st.subheader("Inventory by Material")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                selected_plant = st.selectbox("Select Plant", options=sorted(df['PlantID'].unique()))
                sort_by = st.selectbox("Sort By", options=NPI_COLUMNS, index=0)
            
            # Filter latest data for selected plant
            mat_data = df_latest[df_latest['PlantID'] == selected_plant]
            mat_display = mat_data.groupby('MaterialDescription')[NPI_COLUMNS + ['PhysicalStock']].sum().reset_index()
            mat_display = mat_display.sort_values(by=sort_by, ascending=False)
            
            st.write(f"Showing materials in **{selected_plant}** sorted by **{sort_by}**:")
            st.dataframe(mat_display, use_container_width=True)

        with tab2:
            st.subheader("Time Since Last Zero (Accumulation)")
            st.info("Tracking how many days NPI has been non-zero.")
            
            # Optimization: Only calculate for materials that currently have NPI > 0
            current_npi_mask = (df_latest[NPI_COLUMNS].sum(axis=1) > 0)
            target_skus = df_latest[current_npi_mask][['SapCode', 'PlantID', 'MaterialDescription']].drop_duplicates()

            acc_list = []
            if not target_skus.empty:
                for _, row in target_skus.iterrows():
                    # Get history for this specific SKU/Plant
                    hist = df[(df['SapCode'] == row['SapCode']) & (df['PlantID'] == row['PlantID'])].sort_values('DateStamp')
                    res = {'Material': row['MaterialDescription'], 'Plant': row['PlantID']}
                    
                    for col in NPI_COLUMNS:
                        curr_qty = hist[hist['DateStamp'] == latest_date][col].sum()
                        if curr_qty > 0:
                            # Find last zero date
                            zeros = hist[hist[col] == 0]
                            last_zero = zeros['DateStamp'].max() if not zeros.empty else hist['DateStamp'].min()
                            res[f'{col} Age (Days)'] = (latest_date - last_zero).days
                            res[f'{col} Qty'] = curr_qty
                        else:
                            res[f'{col} Age (Days)'] = 0
                            res[f'{col} Qty'] = 0
                    acc_list.append(res)
                
                df_acc = pd.DataFrame(acc_list)
                st.dataframe(df_acc, use_container_width=True)
            else:
                st.write("No materials currently have Non-Productive Inventory.")

        with tab3:
            st.subheader("Evolution of NPI")
            plant_filter = st.multiselect("Filter Trends by Plant", options=sorted(df['PlantID'].unique()), default=df['PlantID'].unique())
            trend_df = df[df['PlantID'].isin(plant_filter)].groupby('DateStamp')[NPI_COLUMNS].sum().reset_index()
            st.line_chart(trend_df, x='DateStamp', y=NPI_COLUMNS)

    else:
        st.error("No Stock History data found. Please go to the 'Data load' section.")

# --- PLACEHOLDER ---
else:
    st.header("Section under construction")
