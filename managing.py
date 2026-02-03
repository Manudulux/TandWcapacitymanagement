import streamlit as st
import pandas as pd
import os
import re
import duckdb
from datetime import datetime

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
    try:
        df.to_parquet(parquet_path(label), index=False)
    except Exception as e:
        st.warning(f"Failed to write Parquet for {label}: {e}")

def load_parquet_if_exists(label: str):
    path = parquet_path(label)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None

# --- Validation state helpers ---
def _ensure_validation_state():
    if "validation_detail" not in st.session_state:
        st.session_state["validation_detail"] = {}
    return st.session_state["validation_detail"]

def _record_validation(label: str, detail: dict):
    v = _ensure_validation_state()
    v[label] = detail

def _compute_basic_stats(df: pd.DataFrame, date_col: str = "DateStamp") -> dict:
    stats = {"rows": int(len(df)), "date_min": None, "date_max": None}
    if date_col and date_col in df.columns:
        try:
            dmin = pd.to_datetime(df[date_col]).min()
            dmax = pd.to_datetime(df[date_col]).max()
            if pd.notna(dmin): stats["date_min"] = dmin
            if pd.notna(dmax): stats["date_max"] = dmax
        except Exception:
            pass
    return stats

# --- Week parsing for BDD400 ---
def parse_week_string(week_str):
    try:
        match = re.search(r'W\s*(\d{4})\s*/\s*(\d{1,2})', str(week_str))
        if match:
            year, week = match.groups()
            return pd.to_datetime(f"{year}-W{int(week):02d}-1", format="%G-W%V-%u")
    except:
        return None
    return week_str

# --- Column Normalization ---
def normalize_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Normalize column names to app expectations.
    """
    def clean_one(col: str) -> str:
        c = str(col).strip()
        if len(c) >= 2 and c[0] == '[' and c[-1] == ']':
            c = c[1:-1]
        c = c.strip()
        c = re.sub(r'\s+', ' ', c)
        c = c.replace(' ', '_')
        c = re.sub(r'\_+', '_', c)
        return c

    new_cols = [clean_one(c) for c in df.columns]

    alias_map = {
        # dates
        'period': 'DateStamp',
        'datestamp': 'DateStamp',
        # ids / descriptors
        'sapcode': 'SapCode',
        'plantid': 'PlantID',
        'materialdescription': 'MaterialDescription',
        'qualitycode': 'QualityCode',
        'brand': 'Brand',
        'season': 'Season',
        'ab': 'AB',
        # stock metrics (used across datasets)
        'physicalstock': 'PhysicalStock',
        'overagedtireqty': 'OveragedTireQty',
        'intransitqty': 'IntransitQty',
        'qualityinspectionqty': 'QualityInspectionQty',
        'blockedstockqty': 'BlockedStockQty',
        'atponhand': 'ATPonHand',
        # BDD400 planning metrics
        'closinginventory': 'ClosingInventory',
        'inboundconfirmedpr': 'InboundConfirmedPR',
        # optional helpers
        'period_year': 'Period__Year',
        'period__week': 'Period__Week',
        'period_week': 'Period__Week',
    }

    def canonicalize(col: str) -> str:
        key = col.lower()
        if key in alias_map:
            return alias_map[key]
        if key in ('overagedtireqty', 'overaged_tire_qty', 'overagedtire_qty', 'overagedtyreqty', 'overaged_tireqty', 'overagedtireqty'):
            return 'OveragedTireQty'
        return col

    canon_cols = [canonicalize(c) for c in new_cols]

    # De-duplicate
    seen = {}
    final_cols = []
    for c in canon_cols:
        if c not in seen:
            seen[c] = 1
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}__{seen[c]}")
    df.columns = final_cols
    return df

# --- Load Excel/CSV Files ---
def load_file_content(file_path_or_buffer, label):
    try:
        # Load raw
        if isinstance(file_path_or_buffer, str):
            df = (
                pd.read_csv(file_path_or_buffer)
                if file_path_or_buffer.endswith(".csv")
                else pd.read_excel(file_path_or_buffer)
            )
        else:
            try:
                df = pd.read_excel(file_path_or_buffer)
            except:
                df = pd.read_csv(file_path_or_buffer)

        # Normalize columns early
        if label in ("Stock History", "BDD400"):
            df = normalize_columns(df, dataset=label)

        # Initialize validation snapshot
        detail = {
            "rows": int(len(df)),
            "required_ok": True,
            "missing_required": [],
            "missing_expected": [],
            "auto_created": [],
            "date_min": None,
            "date_max": None,
            "notes": []
        }

        # === Dataset-specific handling ===
        if label == "BDD400":
            required = ["DateStamp", "PlantID", "ClosingInventory"]
            missing_req = [c for c in required if c not in df.columns]
            if missing_req:
                detail["required_ok"] = False
                detail["missing_required"] = missing_req

            expected_soft = ["SapCode", "MaterialDescription", "InboundConfirmedPR"]
            missing_soft = [c for c in expected_soft if c not in df.columns]
            detail["missing_expected"] = missing_soft

            if "DateStamp" in df.columns:
                def _maybe_week_to_dt(x):
                    dt = parse_week_string(x)
                    return pd.to_datetime(dt, errors="coerce")
                df["DateStamp"] = df["DateStamp"].apply(_maybe_week_to_dt)
                df = df.dropna(subset=["DateStamp"])

            for c in ["ClosingInventory", "InboundConfirmedPR"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            stats = _compute_basic_stats(df, "DateStamp")
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            _record_validation(label, detail)

        elif label == "Stock History":
            has_ds = "DateStamp" in df.columns
            has_period = "Period" in df.columns
            missing_req = []
            if not (has_ds or has_period):
                missing_req.append("DateStamp_or_Period")
            if "PlantID" not in df.columns:
                missing_req.append("PlantID")
            if "SapCode" not in df.columns:
                missing_req.append("SapCode")

            if missing_req:
                detail["required_ok"] = False
                detail["missing_required"] = missing_req

            missing_expected = [c for c in (NPI_COLUMNS + ALL_STOCK_COLUMNS) if c not in df.columns]
            detail["missing_expected"] = missing_expected.copy()

            if "Period" in df.columns and "DateStamp" not in df.columns:
                df = df.rename(columns={"Period": "DateStamp"})
                has_ds = True
                detail["notes"].append("Renamed 'Period' → 'DateStamp'")

            if has_ds:
                df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
                df = df.dropna(subset=["DateStamp"])

            if "PlantID" not in df.columns:
                df["PlantID"] = "UNKNOWN"
            if "SapCode" not in df.columns:
                df["SapCode"] = "UNKNOWN"

            auto_created = []
            for c in NPI_COLUMNS:
                if c not in df.columns:
                    df[c] = 0
                    auto_created.append(c)
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            for c in ALL_STOCK_COLUMNS:
                if c not in df.columns:
                    df[c] = 0
                    auto_created.append(c)
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            if auto_created:
                detail["auto_created"] = sorted(auto_created)
            if detail["required_ok"] and auto_created:
                detail["notes"].append("Some expected columns were missing and were auto‑created with value 0.")

            stats = _compute_basic_stats(df, "DateStamp")
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            _record_validation(label, detail)

        elif label == "Plant Capacity":
            required = ["PlantID", "MaxCapacity"]
            missing_req = [c for c in required if c not in df.columns]
            if missing_req:
                detail["required_ok"] = False
                detail["missing_required"] = missing_req

            if "DateStamp" in df.columns:
                try:
                    df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
                except Exception:
                    pass

            stats = _compute_basic_stats(df, "DateStamp")
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            _record_validation(label, detail)

        else:
            stats = _compute_basic_stats(df, "DateStamp" if "DateStamp" in df.columns else None)
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            detail["notes"].append("No strict validation rules configured for this dataset yet.")
            _record_validation(label, detail)

        return df
    except Exception as e:
        st.error(f"Error loading {label}: {e}")
        return None

# --- Session State ---
if "data" not in st.session_state:
    st.session_state["data"] = {
        "Stock History": None,
        "BDD400": None,
        "Plant Capacity": None,
        "T&W Forecasts": None
    }

_ensure_validation_state()

# --- DuckDB NPI Computation ---
def compute_npi_days_duckdb(df: pd.DataFrame, npi_categories: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in npi_categories:
        if c not in df2.columns:
            df2[c] = 0
    for c in npi_categories:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)

    if "SapCode" not in df2.columns:
        df2["SapCode"] = "UNKNOWN"
    if "PlantID" not in df2.columns:
        df2["PlantID"] = "UNKNOWN"

    con = duckdb.connect()
    con.register("stock", df2)

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
st.sidebar.title("Constraint Management")
selection = st.sidebar.radio(
    "Go to:",
    [
        "Data load",
        "Non‑Productive Inventory (NPI) Management",
        "Inventory efficiency",
        "Planning Overview",
        "Storage Capacity Management",
        "Mitigation Proposal"
    ]
)

# --- Helper: Future 18 weeks from BDD400 ---
def get_future_weeks_18(df_bdd: pd.DataFrame) -> list[pd.Timestamp]:
    if df_bdd is None or df_bdd.empty or "DateStamp" not in df_bdd.columns:
        return []
    today = pd.Timestamp.today().normalize()
    weeks = sorted(pd.to_datetime(df_bdd["DateStamp"].unique()))
    future_weeks = [w for w in weeks if w >= today]
    return future_weeks[:18]

def ensure_columns_exist(df: pd.DataFrame, cols: list[str], label: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in {label}: {missing}")
        return False
    return True

# --- Validation panel renderer ---
def render_validation_panel():
    st.markdown("### ✅ Data Validation Panel")
    v = _ensure_validation_state()
    datasets = ["Stock History", "BDD400", "Plant Capacity", "T&W Forecasts"]

    npi_ready = v.get("Stock History", {}).get("required_ok", False)
    plan_ready = v.get("BDD400", {}).get("required_ok", False)
    cap_ready = v.get("Plant Capacity", {}).get("required_ok", False)

    overall_cols = st.columns(3)
    with overall_cols[0]:
        st.markdown(f"**NPI Ready**: {'🟢 Yes' if npi_ready else '🔴 No'}")
    with overall_cols[1]:
        st.markdown(f"**Planning Overview Ready** (BDD400): {'🟢 Yes' if plan_ready else '🔴 No'}")
    with overall_cols[2]:
        st.markdown(f"**Storage Capacity Ready** (Capacity): {'🟢 Yes' if cap_ready else '🔴 No'}")

    st.divider()

    for label in datasets:
        detail = v.get(label)
        st.subheader(f"📄 {label}")
        if detail is None:
            st.info("No data loaded yet.")
            continue

        cols_top = st.columns(4)
        with cols_top[0]:
            st.metric("Rows", value=detail.get("rows", 0))
        with cols_top[1]:
            dm = detail.get("date_min")
            st.metric("Min Date", value=(dm.strftime("%Y-%m-%d") if isinstance(dm, pd.Timestamp) else "—"))
        with cols_top[2]:
            dx = detail.get("date_max")
            st.metric("Max Date", value=(dx.strftime("%Y-%m-%d") if isinstance(dx, pd.Timestamp) else "—"))
        with cols_top[3]:
            st.metric("Required OK", value="✅" if detail.get("required_ok", False) else "❌")

        missing_req = detail.get("missing_required", [])
        missing_exp = detail.get("missing_expected", [])
        auto_created = detail.get("auto_created", [])
        notes = detail.get("notes", [])

        if missing_req:
            st.error(f"Missing **required** columns: {missing_req}")
        else:
            st.success("All required columns are present.")
        if missing_exp:
            st.warning(f"Missing **expected** (optional) columns: {missing_exp}")
        if auto_created:
            st.info(f"Auto‑created columns set to 0 (for compatibility): {auto_created}")
        if notes:
            st.caption("Notes: " + " \n".join(notes))

        st.divider()

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
            upload = st.file_uploader(
                f"Upload {fname}",
                type=["xlsx", "csv"],
                key=f"upload_{label}"
            )
            if upload is not None:
                df_loaded = load_file_content(upload, label)
                if df_loaded is not None:
                    st.session_state["data"][label] = df_loaded
                    save_as_parquet(df_loaded, label)
                    st.success(f"⬆️ Loaded new {label}")
                else:
                    st.error("Upload failed.")
            else:
                # Try Parquet (fast path)
                df_parquet = load_parquet_if_exists(label)
                if df_parquet is not None:
                    if label in ("Stock History", "BDD400"):
                        df_parquet = normalize_columns(df_parquet, dataset=label)
                    st.session_state["data"][label] = df_parquet

                    # Refresh validation for cached data
                    df_tmp = df_parquet
                    # (Simplified validation refresh for cache hit)
                    detail = {
                        "rows": int(len(df_tmp)),
                        "required_ok": True,  # Assume true if cached, or re-run logic
                        "missing_required": [],
                        "missing_expected": [],
                        "auto_created": [],
                        "date_min": _compute_basic_stats(df_tmp, "DateStamp")["date_min"] if "DateStamp" in df_tmp.columns else None,
                        "date_max": _compute_basic_stats(df_tmp, "DateStamp")["date_max"] if "DateStamp" in df_tmp.columns else None,
                        "notes": ["Loaded from cache (Parquet)"]
                    }
                    # A quick check for required cols
                    if label == "BDD400" and not all(c in df_tmp.columns for c in ["DateStamp", "PlantID", "ClosingInventory"]):
                        detail["required_ok"] = False
                    if label == "Stock History" and not all(c in df_tmp.columns for c in ["DateStamp", "PlantID", "SapCode"]):
                        detail["required_ok"] = False
                    _record_validation(label, detail)
                else:
                    if os.path.exists(fname):
                        df_loaded = load_file_content(fname, label)
                    else:
                        alt = fname.replace(".xlsx", ".csv")
                        df_loaded = load_file_content(alt, label) if os.path.exists(alt) else None

                    if df_loaded is not None:
                        st.session_state["data"][label] = df_loaded
                        save_as_parquet(df_loaded, label)

        v = st.session_state["validation_detail"].get(label)
        if st.session_state["data"][label] is not None:
            status_ok = v.get("required_ok", False) if v else False
            st.success(f"✅ {label} Active" + ("" if status_ok else " — but has validation issues"))
        else:
            st.warning(f"⚠️ {label} missing")

    st.divider()
    render_validation_panel()

# --- NPI MANAGEMENT ---
elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")

    df = st.session_state["data"].get("Stock History")
    if df is None:
        st.error("Please upload Stock History first.")
        st.stop()

    df = normalize_columns(df, dataset="Stock History")

    if "Period" in df.columns and "DateStamp" not in df.columns:
        df = df.rename(columns={"Period": "DateStamp"})
    if "DateStamp" not in df.columns:
        st.error("Stock History is missing the 'DateStamp' (Period) column.")
        st.stop()

    df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
    df = df.dropna(subset=["DateStamp"])

    if "PlantID" not in df.columns or "SapCode" not in df.columns:
        st.error("Stock History must include 'PlantID' and 'SapCode' columns.")
        st.stop()

    missing_npi, missing_stock = [], []
    for c in NPI_COLUMNS:
        if c not in df.columns:
            missing_npi.append(c)
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ALL_STOCK_COLUMNS:
        if c not in df.columns:
            missing_stock.append(c)
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if missing_npi or missing_stock:
        st.warning(
            f"Some columns were missing and auto‑created as 0 — "
            f"NPI: {missing_npi or 'none'}, Stock: {missing_stock or 'none'}"
        )

    df_aug = compute_npi_days_duckdb(df, NPI_COLUMNS)
    latest_date = df_aug["DateStamp"].max()
    df_latest = df_aug[df_aug["DateStamp"] == latest_date]

    st.subheader(f"📋 Global Plant Summary (As of {latest_date:%Y-%m-%d})")
    available_cols = [c for c in ALL_STOCK_COLUMNS if c in df_latest.columns]
    if not available_cols:
        st.info("No stock columns available to summarize for the latest date.")
    else:
        df_latest[available_cols] = df_latest[available_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)
        summary = (
            df_latest.groupby("PlantID")[available_cols]
            .sum()
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Material Drilldown",
        "⏳ Accumulation Monitor",
        "📈 Trend Analysis",
        "🕵️ Inventory Search"
    ])

    with tab1:
        st.subheader("Inventory by Material with Accumulation Days")
        colA, colB = st.columns([2, 1])
        all_plants = sorted(df_latest["PlantID"].unique())
        with colA:
            selected_plants = st.multiselect(
                "Select Plants",
                options=all_plants,
                default=all_plants
            )
        with colB:
            sort_category = st.selectbox("Sort By", NPI_COLUMNS)

        mat = df_latest[df_latest["PlantID"].isin(selected_plants)]
        agg_dict = {
            **{c: "sum" for c in (NPI_COLUMNS + ["PhysicalStock"])},
            **{f"DaysSinceZero_{c}": "max" for c in NPI_COLUMNS}
        }
        mat_display = (
            mat.groupby(["MaterialDescription", "PlantID", "SapCode"])
            .agg(agg_dict)
            .reset_index()
        )
        mat_display[f"Days Since {sort_category} Zero"] = mat_display[
            f"DaysSinceZero_{sort_category}"
        ]
        st.dataframe(
            mat_display.sort_values(by=sort_category, ascending=False),
            use_container_width=True
        )

    with tab2:
        st.subheader("Time Since Last Zero (All Categories)")
        if 'mat_display' not in locals():
            mat = df_latest
            agg_dict = {
                **{c: "sum" for c in (NPI_COLUMNS + ["PhysicalStock"])},
                **{f"DaysSinceZero_{c}": "max" for c in NPI_COLUMNS}
            }
            mat_display = (
                mat.groupby(["MaterialDescription", "PlantID", "SapCode"])
                .agg(agg_dict)
                .reset_index()
            )
        st.dataframe(mat_display, use_container_width=True)

    with tab3:
        st.subheader("Evolution of NPI")
        all_plants_trend = sorted(df["PlantID"].unique())
        plant_filt = st.multiselect(
            "Filter by Plant",
            options=all_plants_trend,
            default=all_plants_trend
        )
        trend = (
            df_aug[df_aug["PlantID"].isin(plant_filt)]
            .groupby("DateStamp")[NPI_COLUMNS]
            .sum()
            .reset_index()
            .sort_values("DateStamp")
        )
        if trend.empty:
            st.warning("No trend data for selected plants.")
        else:
            st.line_chart(trend, x="DateStamp", y=NPI_COLUMNS, use_container_width=True)

    with tab4:
        st.subheader("Search Inventory Records (by Plant & SAP Code)")
        left, right = st.columns([2, 2])
        with left:
            plants_all = sorted(df["PlantID"].unique())
            plants_sel = st.multiselect(
                "Plant(s)",
                options=plants_all,
                default=plants_all
            )
            min_d = pd.to_datetime(df["DateStamp"].min()).date()
            max_d = pd.to_datetime(df["DateStamp"].max()).date()
            date_range = st.date_input(
                "Date range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                help="Filter records within the selected date range"
            )
            df_plant = df[df["PlantID"].isin(plants_sel)] if plants_sel else df.copy()
        with right:
            saps_all = sorted(df_plant["SapCode"].dropna().astype(str).unique())
            saps_sel = st.multiselect(
                "SAP code(s)",
                options=saps_all,
                placeholder="Type to search SAP codes..."
            )

        mask = pd.Series(True, index=df.index)
        if plants_sel:
            mask &= df["PlantID"].isin(plants_sel)
        if saps_sel:
            mask &= df["SapCode"].astype(str).isin(saps_sel)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= (df["DateStamp"] >= start_dt) & (df["DateStamp"] <= end_dt)

        base_cols = ["DateStamp", "PlantID", "SapCode", "MaterialDescription"]
        combined = base_cols + ALL_STOCK_COLUMNS + NPI_COLUMNS
        seen = set()
        cols_to_show = [c for c in combined if not (c in seen or seen.add(c))]
        cols_existing = [c for c in cols_to_show if c in df.columns]

        out = df.loc[mask, cols_existing].sort_values(["DateStamp", "PlantID", "SapCode"])
        out = out.loc[:, ~out.columns.duplicated(keep="first")]

        st.dataframe(out, use_container_width=True, height=480)
        st.download_button(
            label="⬇️ Download filtered records (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="inventory_search_results.csv",
            mime="text/csv"
        )

# --- INVENTORY EFFICIENCY ---
elif selection == "Inventory efficiency":
    st.header("⚡ Inventory Efficiency")

    # 1. Load Data
    df = st.session_state["data"].get("Stock History")
    if df is None:
        st.error("Please upload Stock History first.")
        st.stop()
    df = normalize_columns(df, dataset="Stock History")
    df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
    df = df.dropna(subset=["DateStamp"])

    # 2. Plant Selection (Single)
    plants = sorted(df["PlantID"].unique())
    selected_plant = st.selectbox("Select Plant", options=plants)

    # 3. Filter for latest snapshot first to show list
    # We need the latest status for each SKU to determine if it has stock but no ATP
    latest_date = df[df["PlantID"] == selected_plant]["DateStamp"].max()
    df_plant_latest = df[
        (df["PlantID"] == selected_plant) &
        (df["DateStamp"] == latest_date)
    ].copy()

    # 4. Filter: PhysicalStock > 0 AND (ATPonHand == 0 OR Nan)
    # Ensure numeric
    cols_to_check = ["PhysicalStock", "ATPonHand"]
    for c in cols_to_check:
        if c not in df_plant_latest.columns:
            df_plant_latest[c] = 0
        df_plant_latest[c] = pd.to_numeric(df_plant_latest[c], errors="coerce").fillna(0)

    mask_efficiency = (
        (df_plant_latest["PhysicalStock"] > 0) &
        (df_plant_latest["ATPonHand"] <= 0)
    )
    df_filtered = df_plant_latest[mask_efficiency].copy()
    df_filtered = df_filtered.sort_values("PhysicalStock", ascending=False)

    st.subheader(f"⚠️ SKUs with Physical Stock but No ATP (Plant: {selected_plant})")
    st.caption("Select a row below to see the full inventory history for that material.")

    # 5. Display List with Selection
    # We show limited columns for clarity
    display_cols = ["SapCode", "MaterialDescription", "PhysicalStock", "ATPonHand", "DateStamp"]
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    # Use on_select to capture user click
    selection_event = st.dataframe(
        df_filtered[display_cols],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )

    # 6. Show History on Selection
    if selection_event.selection.rows:
        idx = selection_event.selection.rows[0]
        # Get the SapCode from the underlying filtered dataframe using the index
        # (Streamlit returns row integer index relative to the displayed dataframe)
        selected_sap = df_filtered.iloc[idx]["SapCode"]
        selected_desc = df_filtered.iloc[idx].get("MaterialDescription", "")
        st.divider()
        st.subheader(f"📜 History: {selected_sap} - {selected_desc}")

        # Query full history from original DF
        df_history = df[
            (df["PlantID"] == selected_plant) &
            (df["SapCode"] == selected_sap)
        ].copy()

        # Sort by Date Descending (Newest First)
        df_history = df_history.sort_values("DateStamp", ascending=False)

        # --- MINIMAL FIX START ---
        # Show all relevant stock columns (deduplicated!)
        hist_cols = ["DateStamp", "SapCode", "PlantID"] + ALL_STOCK_COLUMNS + NPI_COLUMNS
        # 1) Remove duplicates while preserving order
        hist_cols = list(dict.fromkeys(hist_cols))
        # 2) Keep only columns that actually exist
        hist_cols = [c for c in hist_cols if c in df_history.columns]
        # 3) Also make sure the dataframe itself doesn’t carry duplicate names
        df_history = df_history.loc[:, ~df_history.columns.duplicated(keep="first")]

        st.dataframe(
            df_history[hist_cols],
            use_container_width=True,
            hide_index=True
        )
        # --- MINIMAL FIX END ---

    else:
        st.info("👉 Click on a material in the table above to view its history.")

# --- PLANNING OVERVIEW ---
elif selection == "Planning Overview":
    st.header("🧭 Planning Overview")

    df_bdd = st.session_state["data"].get("BDD400")
    if df_bdd is None:
        st.error("Please upload BDD400 first.")
        st.stop()

    # Normalize BDD400 (covers brackets/casing if any)
    df_bdd = normalize_columns(df_bdd, dataset="BDD400")
    if not ensure_columns_exist(df_bdd, ["DateStamp", "PlantID", "ClosingInventory"], "BDD400"):
        st.stop()

    future_weeks = get_future_weeks_18(df_bdd)
    if not future_weeks:
        st.warning("No future weeks found in BDD400. Showing the most recent available weeks.")
        all_weeks = sorted(pd.to_datetime(df_bdd["DateStamp"].unique()))
        future_weeks = all_weeks[-18:]

    df18 = df_bdd[df_bdd["DateStamp"].isin(future_weeks)]
    plants_all = sorted(df18["PlantID"].unique())
    plants_sel = st.multiselect(
        "Filter plants",
        options=plants_all,
        default=plants_all
    )
    df18 = df18[df18["PlantID"].isin(plants_sel)] if plants_sel else df18.copy()

    # Aggregate by week & plant
    closing = (
        df18.groupby(["DateStamp", "PlantID"])["ClosingInventory"]
        .sum()
        .reset_index()
    )

    # rows = PlantID, columns = DateStamp
    pivot_ci = (
        closing.pivot(index="PlantID", columns="DateStamp", values="ClosingInventory")
        .sort_index()
        .fillna(0)
    )
    st.subheader("Closing Inventory by Plant (Next 18 Weeks)")
    st.dataframe(pivot_ci, use_container_width=True)

    # Optional chart (transpose to plot weeks on x-axis)
    with st.expander("📊 Show chart for the table above (optional)"):
        st.bar_chart(pivot_ci.T, use_container_width=True)

# --- STORAGE CAPACITY MANAGEMENT ---
elif selection == "Storage Capacity Management":
    st.header("🏭 Storage Capacity Planning")

    df_bdd = st.session_state["data"].get("BDD400")
    df_cap = st.session_state["data"].get("Plant Capacity")
    if df_bdd is None:
        st.error("Please upload BDD400 first.")
        st.stop()
    if df_cap is None:
        st.error("Please upload Plant Capacity first.")
        st.stop()

    # Normalize BDD400 (covers brackets/casing if any)
    df_bdd = normalize_columns(df_bdd, dataset="BDD400")
    if not ensure_columns_exist(df_bdd, ["DateStamp", "PlantID", "ClosingInventory"], "BDD400"):
        st.stop()
    if not ensure_columns_exist(df_cap, ["PlantID", "MaxCapacity"], "Plant Capacity"):
        st.stop()

    future_weeks = get_future_weeks_18(df_bdd)
    if not future_weeks:
        st.warning("No future weeks found in BDD400. Showing the most recent available weeks.")
        all_weeks = sorted(pd.to_datetime(df_bdd["DateStamp"].unique()))
        future_weeks = all_weeks[-18:]

    # Closing inventory by plant/week
    df_ci = (
        df_bdd[df_bdd["DateStamp"].isin(future_weeks)]
        .groupby(["DateStamp", "PlantID"])["ClosingInventory"]
        .sum()
        .reset_index()
    )

    # Capacity alignment:
    if "DateStamp" in df_cap.columns:
        df_cap_tmp = df_cap.copy()
        df_cap_tmp["DateStamp"] = pd.to_datetime(df_cap_tmp["DateStamp"])
        df_cap_week = (
            df_cap_tmp[df_cap_tmp["DateStamp"].isin(future_weeks)][["DateStamp", "PlantID", "MaxCapacity"]]
            .drop_duplicates(subset=["DateStamp", "PlantID"], keep="last")
        )
    else:
        weeks_df = pd.DataFrame({"DateStamp": pd.to_datetime(future_weeks)})
        cap_unique = df_cap[["PlantID", "MaxCapacity"]].drop_duplicates()
        weeks_df["key"] = 1
        cap_unique["key"] = 1
        df_cap_week = weeks_df.merge(cap_unique, on="key").drop(columns="key")

    # Merge ClosingInventory with capacities
    df_merge = df_ci.merge(df_cap_week, on=["DateStamp", "PlantID"], how="left")
    missing_cap_plants = sorted(df_merge[df_merge["MaxCapacity"].isna()]["PlantID"].unique().tolist())
    if missing_cap_plants:
        st.warning(f"No capacity found for plant(s): {missing_cap_plants}. Treating as 0 capacity for calculation.")
    df_merge["MaxCapacity"] = pd.to_numeric(df_merge["MaxCapacity"], errors="coerce").fillna(0)

    df_merge["Delta"] = df_merge["ClosingInventory"] - df_merge["MaxCapacity"]
    df_merge["Ratio"] = df_merge.apply(
        lambda r: (r["ClosingInventory"] / r["MaxCapacity"]) if r["MaxCapacity"] else pd.NA,
        axis=1
    )

    # Weekly totals (base, all plants)
    weekly_totals = (
        df_merge.groupby("DateStamp")[["ClosingInventory", "MaxCapacity"]]
        .sum()
        .reset_index()
    )
    weekly_totals["Delta"] = weekly_totals["ClosingInventory"] - weekly_totals["MaxCapacity"]
    weekly_totals["Ratio"] = weekly_totals.apply(
        lambda r: (r["ClosingInventory"] / r["MaxCapacity"]) if r["MaxCapacity"] else pd.NA, axis=1
    )

    # Swapped orientation: rows = PlantID, columns = DateStamp
    st.subheader("Δ to Capacity by Plant & Week")
    st.caption(
        "Orientation: **rows = PlantID**, **columns = DateStamp (weeks)**. "
        "Color scale by % of capacity: >105% red, 95–105% yellow, <95% green. "
        "Δ = ClosingInventory − MaxCapacity. The last row **TOTAL** shows week-level aggregate."
    )

    delta_pivot = (
        df_merge.pivot(index="PlantID", columns="DateStamp", values="Delta")
        .sort_index()
        .fillna(0)
    )
    ratio_pivot = (
        df_merge.pivot(index="PlantID", columns="DateStamp", values="Ratio")
        .sort_index()
    )

    # Append TOTAL row using weekly_totals (per week)
    weekly_delta_ser = weekly_totals.set_index("DateStamp")["Delta"]
    weekly_ratio_ser = weekly_totals.set_index("DateStamp")["Ratio"]

    for col in delta_pivot.columns:
        if col not in weekly_delta_ser.index:
            weekly_delta_ser.loc[col] = 0
    for col in ratio_pivot.columns:
        if col not in weekly_ratio_ser.index:
            weekly_ratio_ser.loc[col] = pd.NA

    delta_pivot.loc["TOTAL", :] = weekly_delta_ser.reindex(delta_pivot.columns).values
    ratio_pivot.loc["TOTAL", :] = weekly_ratio_ser.reindex(ratio_pivot.columns).values

    def colorize_by_ratio(ratio_df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=ratio_df.index, columns=ratio_df.columns)
        for idx in ratio_df.index:
            for col in ratio_df.columns:
                r = ratio_df.loc[idx, col]
                if pd.isna(r):
                    styles.loc[idx, col] = ''
                elif r > 1.05:
                    styles.loc[idx, col] = 'background-color: #f8d7da;'
                elif r >= 0.95:
                    styles.loc[idx, col] = 'background-color: #fff3cd;'
                else:
                    styles.loc[idx, col] = 'background-color: #d4edda;'
        return styles

    styled_delta = (
        delta_pivot.style
        .apply(lambda _: colorize_by_ratio(ratio_pivot), axis=None)
        .format("{:,.0f}")
    )
    st.dataframe(styled_delta, use_container_width=True)

    # --- Plant filter for weekly totals ---
    st.subheader("Weekly Total Inventory vs Capacity")
    total_plants_all = sorted(df_merge["PlantID"].unique())
    selected_plants_for_totals = st.multiselect(
        "Select plant(s) to include in totals",
        options=total_plants_all,
        default=total_plants_all,
        help="Totals below will aggregate only the selected plants."
    )

    if selected_plants_for_totals:
        df_merge_sel = df_merge[df_merge["PlantID"].isin(selected_plants_for_totals)]
    else:
        df_merge_sel = df_merge[df_merge["PlantID"].isin([])]

    weekly_totals_sel = (
        df_merge_sel.groupby("DateStamp")[["ClosingInventory", "MaxCapacity"]]
        .sum()
        .reset_index()
    )
    weekly_totals_sel["Delta"] = weekly_totals_sel["ClosingInventory"] - weekly_totals_sel["MaxCapacity"]
    weekly_totals_sel["Ratio"] = weekly_totals_sel.apply(
        lambda r: (r["ClosingInventory"] / r["MaxCapacity"]) if r["MaxCapacity"] else pd.NA, axis=1
    )

    totals_display = weekly_totals_sel.set_index("DateStamp").rename(columns={
        "ClosingInventory": "TotalClosingInventory",
        "MaxCapacity": "TotalCapacity"
    })
    st.dataframe(
        totals_display[["TotalClosingInventory", "TotalCapacity", "Delta", "Ratio"]]
        .style.format({
            "TotalClosingInventory":"{:,.0f}",
            "TotalCapacity":"{:,.0f}",
            "Delta":"{:,.0f}",
            "Ratio":"{:.2f}"
        }),
        use_container_width=True
    )

# --- MITIGATION PROPOSAL (Fair Share Logic + Smart Defaults) ---
elif selection == "Mitigation Proposal":
    st.subheader("🧯 Mitigation Proposal (Fair Share Distribution)")

    df_bdd = st.session_state["data"].get("BDD400")
    df_stock = st.session_state["data"].get("Stock History")

    # --- Basic Checks ---
    if df_bdd is None:
        st.error("Please upload BDD400 first.")
        st.stop()
    if df_stock is None:
        st.error("Please upload Stock History first.")
        st.stop()

    # --- Normalize & Parse Dates ---
    df_bdd = normalize_columns(df_bdd, dataset="BDD400")
    df_stock = normalize_columns(df_stock, dataset="Stock History")

    df_bdd["DateStamp"] = pd.to_datetime(df_bdd["DateStamp"], errors="coerce")
    df_stock["DateStamp"] = pd.to_datetime(df_stock["DateStamp"], errors="coerce")
    df_bdd = df_bdd.dropna(subset=["DateStamp"])
    df_stock = df_stock.dropna(subset=["DateStamp"])

    # String keys cleanup
    for _df in (df_bdd, df_stock):
        _df["PlantID"] = _df["PlantID"].astype(str).str.strip()
        _df["SapCode"] = _df["SapCode"].astype(str).str.strip()

    # --- UI Controls ---
    all_plants = sorted(df_bdd["PlantID"].dropna().astype(str).unique())
    col1, col2 = st.columns([1, 1])
    with col1:
        receiving_plants = st.multiselect("Receiving Plants", options=all_plants, key="recv_plants_v2")
    with col2:
        shipping_candidates = [p for p in all_plants if p not in receiving_plants]
        shipping_plants = st.multiselect("Shipping Plants", options=shipping_candidates, key="ship_plants_v2")

    col3, col4 = st.columns(2)
    with col3:
        transfer_cap_pct = st.slider(
            "Distribute % of ATP",
            min_value=5, max_value=100, value=75, step=5,
            help="We will only distribute this percentage of the Total Available ATP."
        )
    with col4:
        min_qty_threshold = st.slider(
            "Minimum Transfer Quantity",
            min_value=0, max_value=200, value=50, step=10,
            help="If the calculated proportional share is less than this, do not propose a transfer."
        )

    all_periods = sorted(df_bdd["DateStamp"].unique())
    default_periods = all_periods[-4:] if len(all_periods) >= 4 else all_periods
    selected_periods = st.multiselect(
        "Select week(s)",
        options=all_periods,
        default=default_periods,
        format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
        key="periods_v2"
    )

    if not selected_periods or not receiving_plants or not shipping_plants:
        st.info("Please select Periods, Receiving Plants, and Shipping Plants to begin.")
        st.stop()

    # ==============================================================================
    # 1. CALCULATE DEMAND (Receiving Side)
    # ==============================================================================
    mask_recv = (df_bdd["PlantID"].isin(receiving_plants)) & (df_bdd["DateStamp"].isin(selected_periods))
    df_demand_raw = df_bdd.loc[mask_recv].copy()

    # Sum Demand per Plant per Material
    df_demand = (
        df_demand_raw.groupby(["PlantID", "SapCode", "MaterialDescription"], as_index=False)["InboundConfirmedPR"]
        .sum()
        .rename(columns={"PlantID": "ReceivingPlant", "InboundConfirmedPR": "RecvDemand"})
    )

    # Calculate GLOBAL Demand per Material
    df_global_demand = (
        df_demand.groupby("SapCode")["RecvDemand"]
        .sum()
        .reset_index()
        .rename(columns={"RecvDemand": "GlobalDemand"})
    )

    # Merge Global Demand back to calculate Share
    df_demand = df_demand.merge(df_global_demand, on="SapCode", how="left")

    # Calculate Share Ratio
    df_demand["DemandShare"] = df_demand.apply(
        lambda x: x["RecvDemand"] / x["GlobalDemand"] if x["GlobalDemand"] > 0 else 0, axis=1
    )

    # ==============================================================================
    # 2. CALCULATE SUPPLY (Shipping Side)
    # ==============================================================================
    sap_codes_needed = df_demand[["SapCode"]].drop_duplicates()
    atp_data = {}
    for shp in shipping_plants:
        mask_shp = df_stock["PlantID"] == shp
        df_shp = df_stock.loc[mask_shp].copy()
        if not df_shp.empty:
            df_shp["ATPonHand"] = pd.to_numeric(df_shp["ATPonHand"], errors="coerce").fillna(0)
            df_latest = df_shp.sort_values("DateStamp").groupby("SapCode").tail(1)
            for _, row in df_latest.iterrows():
                s_code = str(row["SapCode"])
                qty = float(row["ATPonHand"])
                if s_code not in atp_data: atp_data[s_code] = {}
                atp_data[s_code][shp] = qty

    supply_rows = []
    for s_code in sap_codes_needed["SapCode"].unique():
        s_code = str(s_code)
        row = {"SapCode": s_code, "TotalRawATP": 0.0}
        mat_atps = atp_data.get(s_code, {})
        for shp in shipping_plants:
            qty = mat_atps.get(shp, 0.0)
            row[f"ATP_{shp}"] = qty
            row["TotalRawATP"] += qty
        supply_rows.append(row)
    df_supply = pd.DataFrame(supply_rows)
    df_supply["DistributableSupply"] = df_supply["TotalRawATP"] * (transfer_cap_pct / 100.0)

    # ==============================================================================
    # 3. MERGE AND CALCULATE PROPOSAL
    # ==============================================================================
    df_main = df_demand.merge(df_supply, on="SapCode", how="left")
    df_main["TotalRawATP"] = df_main["TotalRawATP"].fillna(0)
    df_main["DistributableSupply"] = df_main["DistributableSupply"].fillna(0)
    for shp in shipping_plants:
        col = f"ATP_{shp}"
        if col in df_main.columns:
            df_main[col] = df_main[col].fillna(0)
        else:
            df_main[col] = 0.0

    # 1. Theoretical Share
    df_main["TheoreticalShare"] = df_main["DistributableSupply"] * df_main["DemandShare"]
    # 2. Cap at actual Demand
    df_main["ProposedTotal"] = df_main[["TheoreticalShare", "RecvDemand"]].min(axis=1)
    # 3. Apply Minimum Threshold
    df_main["ProposedTotal"] = df_main["ProposedTotal"].apply(
        lambda x: x if x >= min_qty_threshold else 0.0
    )
    df_main["ProposedTotal"] = df_main["ProposedTotal"].round(0)
    # 4. Sourcing Split
    for shp in shipping_plants:
        atp_col = f"ATP_{shp}"
        out_col = f"From_{shp}"
        df_main[out_col] = df_main.apply(
            lambda r: (r[atp_col] / r["TotalRawATP"] * r["ProposedTotal"])
            if r["TotalRawATP"] > 0 else 0,
            axis=1
        )
        df_main[out_col] = df_main[out_col].round(0)

    # ==============================================================================
    # 4. DISPLAY SETUP (Default Selection & Sorting)
    # ==============================================================================
    # --- AUTO-SELECT LOGIC ---
    # Select True if ProposedTotal > 0
    df_main["Selected"] = df_main["ProposedTotal"] > 0

    # --- SORTING LOGIC ---
    # Sort descending by Proposed Quantity, then Receiving Plant
    df_main = df_main.sort_values(["ProposedTotal", "ReceivingPlant"], ascending=[False, True])

    # Columns to show
    cols_to_show = ["Selected", "ReceivingPlant", "SapCode", "MaterialDescription",
                    "RecvDemand", "TotalRawATP", "DistributableSupply", "ProposedTotal"]
    cols_to_show += [f"From_{shp}" for shp in shipping_plants]

    final_view = df_main[cols_to_show].copy()
    col_config = {
        "Selected": st.column_config.CheckboxColumn("Select", default=False),
        "ReceivingPlant": st.column_config.TextColumn("Recv Plant", width="small"),
        "SapCode": st.column_config.TextColumn("SAP Code", width="small"),
        "MaterialDescription": st.column_config.TextColumn("Description", width="medium"),
        "RecvDemand": st.column_config.NumberColumn("Demand", format="%.0f"),
        "TotalRawATP": st.column_config.NumberColumn("Total ATP", format="%.0f", help="Total Raw ATP in selected shipping plants"),
        "DistributableSupply": st.column_config.NumberColumn("Distributable", format="%.0f", help=f"ATP * {transfer_cap_pct}%"),
        "ProposedTotal": st.column_config.NumberColumn("Proposed Qty", format="%.0f", help="Min(Distributable * Share, Demand)"),
    }
    for shp in shipping_plants:
        col_config[f"From_{shp}"] = st.column_config.NumberColumn(f"From {shp}", format="%.0f")

    st.markdown(f"### 📋 Plan Editor ({transfer_cap_pct}% Cap, Threshold: {min_qty_threshold})")
    # Unique key to force refresh on param change
    key_v2 = f"editor_v2_final_{len(receiving_plants)}_{len(shipping_plants)}_{transfer_cap_pct}_{min_qty_threshold}"
    edited_df = st.data_editor(
        final_view,
        column_config=col_config,
        use_container_width=True,
        height=600,
        hide_index=True,
        key=key_v2
    )

    # ==============================================================================
    # 5. SUMMARY
    # ==============================================================================
    st.divider()
    st.subheader("📊 Summary by Receiving Plant")
    df_sel = edited_df[edited_df["Selected"]==True].copy()
    if not df_sel.empty:
        agg_map = {"ProposedTotal": "sum", "SapCode": "count"}
        for shp in shipping_plants:
            agg_map[f"From_{shp}"] = "sum"
        summary = df_sel.groupby("ReceivingPlant").agg(agg_map).reset_index()
        summary = summary.rename(columns={"SapCode": "Selected SKUs", "ProposedTotal": "Total Transfer Qty"})

        gt = pd.DataFrame(summary.sum(numeric_only=True)).T
        gt["ReceivingPlant"] = "GRAND TOTAL"
        summary = pd.concat([summary, gt], ignore_index=True)

        fmt = {"Total Transfer Qty": "{:,.0f}", "Selected SKUs": "{:.0f}"}
        for shp in shipping_plants:
            fmt[f"From_{shp}"] = "{:,.0f}"

        st.dataframe(summary.style.format(fmt), use_container_width=False)
    else:
        st.info("No items selected.")

    # ==============================================================================
    # 6. DOWNLOAD
    # ==============================================================================
    st.download_button(
        label="⬇️ Download Proposal (CSV)",
        data=edited_df.to_csv(index=False).encode("utf-8"),
        file_name=f"mitigation_v2_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.header(selection)
    st.info("Implementation pending.")


