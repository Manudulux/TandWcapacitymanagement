import streamlit as st
import pandas as pd
import altair as alt
import re
from io import BytesIO

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Physical Constraints Management",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def read_excel_robust(upload):
    """
    Reads an uploaded Excel file. 
    Assumes the data is in the first sheet.
    """
    try:
        if hasattr(upload, "seek"):
            upload.seek(0)
        return pd.read_excel(upload, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return pd.DataFrame()

def df_to_excel_bytes(df, sheet_name="Sheet1"):
    """
    Converts a DataFrame to Excel bytes for downloading.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    processed_data = output.getvalue()
    return processed_data

def normalize_columns(df):
    """
    Standardizes column names to ensure logic works regardless of 
    input variations (e.g., 'Plant' vs 'Warehouse').
    """
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    
    for c in df.columns:
        clean_c = c.lower().replace(" ", "").replace("_", "")
        
        # Plant / Warehouse Identification
        if clean_c in ["plant", "warehouse", "site", "location", "plantid"]:
            col_map[c] = "Plant"
        
        # Material Identification
        elif clean_c in ["material", "sku", "item", "sapcode", "materialnumber"]:
            col_map[c] = "Material"
        
        # Description
        elif clean_c in ["materialdescription", "description", "itemname"]:
            col_map[c] = "Description"
            
        # Quantity
        elif clean_c in ["quantity", "qty", "stock", "onhand", "totalqty", "closingstock"]:
            col_map[c] = "Quantity"
            
        # Capacity (Specific to Capacity File)
        elif clean_c in ["maxcapacity", "capacity", "limit", "storagecapacity"]:
            col_map[c] = "MaxCapacity"

        # Batch Status (for NPI logic)
        elif clean_c in ["status", "stocktype", "batchstatus"]:
            col_map[c] = "Status"

    return df.rename(columns=col_map)

# ------------------------------------------------------------
# SESSION STATE MANAGEMENT
# ------------------------------------------------------------
# We use session state to keep files in memory while navigating tabs

if 'stock_df' not in st.session_state:
    st.session_state['stock_df'] = pd.DataFrame()
if 'capacity_df' not in st.session_state:
    st.session_state['capacity_df'] = pd.DataFrame()

# ------------------------------------------------------------
# APP MODULES
# ------------------------------------------------------------

def run_home():
    st.title("🏭 Supply Chain Physical Constraints Management")
    st.markdown("### Data Upload Center")
    st.info("Please upload your data in **Excel (.xlsx)** format.")

    c1, c2 = st.columns(2)

    # --- UPLOAD 1: STOCK/INVENTORY DATA ---
    with c1:
        st.subheader("1. Inventory Data")
        st.markdown("*Required Columns: Plant, Material, Quantity*")
        stock_file = st.file_uploader("Upload Current Stock (Excel)", type=["xlsx", "xls"], key="u_stock")
        
        if stock_file:
            df = read_excel_robust(stock_file)
            if not df.empty:
                df = normalize_columns(df)
                st.session_state['stock_df'] = df
                st.success(f"Loaded {len(df):,} rows.")
                st.dataframe(df.head(3), use_container_width=True)
            else:
                st.error("File is empty or could not be read.")

        if not st.session_state['stock_df'].empty:
            st.caption(f"✅ Active Inventory: {len(st.session_state['stock_df']):,} records")

    # --- UPLOAD 2: CAPACITY DATA ---
    with c2:
        st.subheader("2. Plant Capacity Data")
        st.markdown("*Required Columns: Plant, MaxCapacity*")
        cap_file = st.file_uploader("Upload Capacity Limits (Excel)", type=["xlsx", "xls"], key="u_cap")
        
        if cap_file:
            df = read_excel_robust(cap_file)
            if not df.empty:
                df = normalize_columns(df)
                st.session_state['capacity_df'] = df
                st.success(f"Loaded {len(df):,} plants.")
                st.dataframe(df.head(3), use_container_width=True)

        if not st.session_state['capacity_df'].empty:
            st.caption(f"✅ Active Capacity: {len(st.session_state['capacity_df']):,} plants")

    st.markdown("---")
    st.warning("👈 Use the **Sidebar** to navigate to the Analysis Modules once data is uploaded.")


def run_constraint_overview():
    st.title("📊 Constraints Overview")
    
    df_stock = st.session_state['stock_df']
    df_cap = st.session_state['capacity_df']

    if df_stock.empty or df_cap.empty:
        st.error("Please upload both Inventory and Capacity files on the Home page.")
        return

    # 1. Aggregate Stock by Plant
    if 'Plant' not in df_stock.columns or 'Quantity' not in df_stock.columns:
        st.error("Inventory file missing 'Plant' or 'Quantity' columns.")
        return
        
    agg_stock = df_stock.groupby('Plant')['Quantity'].sum().reset_index()

    # 2. Merge with Capacity
    merged = pd.merge(agg_stock, df_cap, on='Plant', how='outer').fillna(0)

    # 3. Calculate Metrics
    merged['Utilization %'] = (merged['Quantity'] / merged['MaxCapacity'] * 100).round(1)
    merged['Gap'] = merged['MaxCapacity'] - merged['Quantity']
    
    # Define Status
    def get_status(row):
        if row['MaxCapacity'] == 0: return "No Capacity Data"
        if row['Utilization %'] > 100: return "CRITICAL OVERFILL"
        if row['Utilization %'] > 90: return "Warning (High)"
        return "OK"

    merged['Status'] = merged.apply(get_status, axis=1)

    # --- DASHBOARD METRICS ---
    col1, col2, col3 = st.columns(3)
    total_stock = merged['Quantity'].sum()
    total_cap = merged['MaxCapacity'].sum()
    overfilled_plants = merged[merged['Utilization %'] > 100]['Plant'].count()

    col1.metric("Total Network Stock", f"{total_stock:,.0f}")
    col2.metric("Total Network Capacity", f"{total_cap:,.0f}")
    col3.metric("🚨 Overfilled Plants", f"{overfilled_plants}")

    # --- VISUALIZATION ---
    st.subheader("Capacity Utilization by Plant")
    
    # Sort for chart
    chart_data = merged.sort_values('Utilization %', ascending=False)
    
    # Bar Chart with Threshold Colors
    bars = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Plant:N', sort='-y'),
        y=alt.Y('Utilization %:Q'),
        color=alt.condition(
            alt.datum['Utilization %'] > 100,
            alt.value('#d62728'),  # Red for overfill
            alt.condition(
                alt.datum['Utilization %'] > 90,
                alt.value('#ff7f0e'),  # Orange for warning
                alt.value('#2ca02c')   # Green for OK
            )
        ),
        tooltip=['Plant', 'Quantity', 'MaxCapacity', 'Utilization %', 'Status']
    ).properties(height=400)

    # Reference Line at 100%
    rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(color='black', strokeDash=[5, 5]).encode(y='y')

    st.altair_chart((bars + rule), use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("Detailed Plant Status")
    
    # Styling the dataframe
    def color_utilization(val):
        if val > 100: return 'background-color: #ffcccc'
        if val > 90: return 'background-color: #ffe6cc'
        return ''

    st.dataframe(
        merged.style.applymap(color_utilization, subset=['Utilization %'])
        .format({'Quantity': '{:,.0f}', 'MaxCapacity': '{:,.0f}', 'Gap': '{:,.0f}', 'Utilization %': '{:.1f}%'}),
        use_container_width=True
    )

    # Download Button
    excel_data = df_to_excel_bytes(merged)
    st.download_button("⬇️ Download Analysis (Excel)", excel_data, "Constraint_Analysis.xlsx")


def run_material_drilldown():
    st.title("🔎 Material Drill-Down")
    st.markdown("Identify which materials are consuming the most space in critical plants.")

    df_stock = st.session_state['stock_df']
    df_cap = st.session_state['capacity_df']

    if df_stock.empty:
        st.error("Inventory data is missing.")
        return

    # Filter selector
    plant_list = sorted(df_stock['Plant'].unique().tolist())
    selected_plant = st.selectbox("Select Plant to Inspect", plant_list)

    # Filter data
    plant_data = df_stock[df_stock['Plant'] == selected_plant].copy()
    
    # Summary for selected plant
    total_plant_stock = plant_data['Quantity'].sum()
    st.metric(f"Total Stock in {selected_plant}", f"{total_plant_stock:,.0f}")

    # Aggregation by Material
    mat_agg = plant_data.groupby(['Material', 'Description'])['Quantity'].sum().reset_index()
    mat_agg['% Contribution'] = (mat_agg['Quantity'] / total_plant_stock * 100)
    mat_agg = mat_agg.sort_values('Quantity', ascending=False).head(20) # Top 20

    # Pareto Chart
    st.subheader(f"Top 20 Materials in {selected_plant}")
    
    base = alt.Chart(mat_agg).encode(x=alt.X('Material:N', sort='-y'))

    bar = base.mark_bar().encode(
        y='Quantity:Q',
        tooltip=['Material', 'Description', 'Quantity', '% Contribution']
    )

    st.altair_chart(bar, use_container_width=True)

    st.dataframe(mat_agg.style.format({'Quantity': '{:,.0f}', '% Contribution': '{:.1f}%'}), use_container_width=True)


# ------------------------------------------------------------
# NAVIGATION & EXECUTION
# ------------------------------------------------------------

def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    
    options = ["Home", "Constraints Overview", "Material Drill-Down"]
    selection = st.sidebar.radio("Go to:", options)

    if selection == "Home":
        run_home()
    elif selection == "Constraints Overview":
        run_constraint_overview()
    elif selection == "Material Drill-Down":
        run_material_drilldown()

    # Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Supply Chain Toolkit v1.0")
    if st.sidebar.button("🧹 Clear All Data"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    main()
