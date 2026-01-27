import streamlit as st
import pandas as pd
import os

# --- Page Configuration ---
st.set_page_config(page_title="Supply Chain Planning Dashboard", layout="wide")

# --- Constants & Configuration ---
# Map labels to the exact filenames expected locally
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

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state['data'] = {key: None for key in FILES_CONFIG.keys()}

# --- Helper Function ---
def process_file(label, filename, uploaded_file):
    """Priority: 1. Uploaded File, 2. Local File, 3. None"""
    df = None
    source = ""
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            source = "Uploaded"
        except Exception as e:
            st.error(f"Error reading uploaded {filename}: {e}")
    elif os.path.exists(filename):
        try:
            df = pd.read_excel(filename)
            source = "Local (Default)"
        except Exception as e:
            st.error(f"Error reading local {filename}: {e}")
            
    return df, source

# --- SECTION 1: Data Load ---
if selection == "Data load":
    st.header("📂 Data Management")
    st.write("Each section below looks for a default file in the script folder. You can override them by uploading a new version.")

    # Create a grid of 2x2 for the loaders
    cols = st.columns(2)
    
    for i, (label, config) in enumerate(FILES_CONFIG.items()):
        with cols[i % 2]:
            st.subheader(f"{label}")
            # Specific loader for this file
            uploaded_file = st.file_uploader(
                f"Upload {config['filename']}", 
                type=["xlsx"], 
                key=f"loader_{label}",
                help=config['help']
            )
            
            # Process the logic
            df, source = process_file(label, config['filename'], uploaded_file)
            
            if df is not None:
                st.session_state['data'][label] = df
                st.success(f"✅ Loaded from {source}")
                with st.expander(f"Preview {label} Data"):
                    st.dataframe(df.head(5), use_container_width=True)
            else:
                st.error(f"❌ Missing: {config['filename']}")

# --- PLACEHOLDER SECTIONS ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    if st.session_state['data']["Stock History"] is not None:
        st.write("Using data from Stock History...")
    else:
        st.warning("Please upload Stock History in the Data Load section.")

elif selection == "Planning Overview — T&W Forecast Projection":
    st.header("📈 Planning Overview — T&W Forecast Projection")
    if st.session_state['data']["T&W Forecasts"] is not None:
        st.write("Using data from T&W Forecasts...")
    else:
        st.warning("Please upload T&W Forecasts in the Data Load section.")

elif selection == "Planning Overview — BDD400 Closing Stock":
    st.header("📦 Planning Overview — BDD400 Closing Stock")
    if st.session_state['data']["BDD400"] is not None:
        st.write("Using data from BDD400...")
    else:
        st.warning("Please upload BDD400 file in the Data Load section.")

elif selection == "Storage Capacity Management":
    st.header("🏗️ Storage Capacity Management")
    if st.session_state['data']["Plant Capacity"] is not None:
        st.write("Using data from Plant Capacity...")
    else:
        st.warning("Please upload Plant Capacity in the Data Load section.")

elif selection == "Mitigation Proposal":
    st.header("🛡️ Mitigation Proposal")
    st.info("Analysis logic to be implemented.")
