import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION & UTILS ---
st.set_page_config(page_title="Supply Chain Hub", layout="wide")

def standardize_data(df):
    """Standardizes column names and date formats across modules."""
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    # Standardizing Plant Keys (User Story: Cross-Module consistency)
    if 'plant' in df.columns:
        df['plant_key'] = df['plant'].astype(str).str.upper()
    return df

def get_download_link(df, filename="data.csv"):
    """Generates a download link for CSV."""
    csv = df.to_csv(index=False)
    return st.download_button(label=f"📥 Download {filename}", data=csv, file_name=filename, mime='text/csv')

# --- MODULES ---

def module_npi(df):
    st.header("📦 Non-Productive Inventory (NPI) Management")
    
    # Filters (User Story: Filter by Brand, AB, Hier2, etc.)
    cols = st.columns(4)
    brand = cols[0].multiselect("Brand", options=df['brand'].unique())
    hier = cols[1].multiselect("Hier4", options=df['hier4'].unique())
    
    filtered_df = df.copy()
    if brand: filtered_df = filtered_df[filtered_df['brand'].isin(brand)]
    
    # Calculate Last Zero Date (User Story: Measure accumulation)
    # Logic: Group by material, find max date where quantity was 0
    st.subheader("Accumulation Analysis")
    # (Simplified logic for demonstration)
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    
    fig = px.line(filtered_df, x='date', y='quantity', color='plant_key', title="NPI Quantities Over Time")
    st.plotly_chart(fig, use_container_width=True)

def module_planning_projection(forecast_df, inventory_df):
    st.header("📈 T&W Forecast Projection")
    
    plant = st.selectbox("Select Plant", options=forecast_df['plant_key'].unique())
    starting_stock = st.number_input("Manual Starting Stock Override", value=0)
    
    # Logic: Stock(t) = Stock(t-1) + Load - Unload
    # User Story: Automatically clean and aggregate
    proj = forecast_df[forecast_df['plant_key'] == plant].groupby('week').sum().reset_index()
    proj['projected_stock'] = starting_stock + (proj['load'].cumsum() - proj['unload'].cumsum())
    
    fig = px.bar(proj, x='week', y='projected_stock', title=f"Inventory Evolution: {plant}")
    st.plotly_chart(fig, use_container_width=True)
    get_download_link(proj, f"projection_{plant}.csv")

def module_capacity(capacity_df, stock_df):
    st.header("🏭 Storage Capacity Management")
    
    # Merge on PlantKey (User Story: Standardized PlantKey matching)
    merged = pd.merge(stock_df, capacity_df, on='plant_key', how='inner')
    merged['utilization'] = (merged['current_stock'] / merged['max_capacity']) * 100
    
    # Heatmap (User Story: Risks visually apparent)
    fig = px.density_heatmap(merged, x="week", y="plant_key", z="utilization", 
                             color_continuous_scale="RdYlGn_r", title="Capacity Utilization Heatmap (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Status Indicators
    merged['status'] = merged['utilization'].apply(lambda x: '🔴 Over' if x > 100 else ('🟡 Near' if x > 85 else '🟢 Safe'))
    st.table(merged[['plant_key', 'current_stock', 'max_capacity', 'utilization', 'status']].tail(10))

# --- MAIN APP FLOW ---

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Center", "NPI Management", "Planning Overview", "Capacity Management", "Mitigation Proposal"])

    if "datasets" not in st.session_state:
        st.session_state.datasets = {}

    if page == "Upload Center":
        st.header("📂 Data Ingestion")
        uploaded_files = st.file_uploader("Upload Supply Chain Files (CSV/XLSX)", accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                clean_df = standardize_data(df)
                st.session_state.datasets[file.name] = clean_df
                st.success(f"Successfully processed {file.name}")
            
            st.info(f"Data Integrity Report: {len(st.session_state.datasets)} modules loaded.")

    # Check if data exists before loading modules
    if not st.session_state.datasets:
        st.warning("Please upload datasets in the Upload Center to begin.")
        return

    # Routing
    if page == "NPI Management":
        # Assumes a file with 'npi' in name exists
        npi_data = next((df for name, df in st.session_state.datasets.items() if 'npi' in name.lower()), None)
        if npi_data is not None: module_npi(npi_data)
        
    elif page == "Planning Overview":
        # Logic for T&W Projection
        st.write("Projecting inventory based on uploaded Forecast and BDD files.")
        # (Pass relevant dataframes from session_state here)

    elif page == "Capacity Management":
        cap_data = next((df for name, df in st.session_state.datasets.items() if 'cap' in name.lower()), None)
        stock_data = next((df for name, df in st.session_state.datasets.items() if 'bdd' in name.lower()), None)
        if cap_data is not None and stock_data is not None:
            module_capacity(cap_data, stock_data)

if __name__ == "__main__":
    main()
