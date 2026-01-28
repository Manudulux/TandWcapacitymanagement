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

# --- Load Excel/CSV Files (with validation capture) ---
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
            # Required columns for planning pages
            required = ["DateStamp", "PlantID", "ClosingInventory"]
            missing_req = [c for c in required if c not in df.columns]
            if missing_req:
                detail["required_ok"] = False
                detail["missing_required"] = missing_req

            # BDD400 week conversion (keeps column name DateStamp)
            if "DateStamp" in df.columns:
                df["DateStamp"] = df["DateStamp"].apply(parse_week_string)
                df = df.dropna(subset=["DateStamp"])
            # Basic stats
            stats = _compute_basic_stats(df, "DateStamp")
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            _record_validation(label, detail)

        elif label == "Stock History":
            # Required (flex): DateStamp/Period + PlantID + SapCode
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

            # Capture expected (soft) before we auto-create
            missing_expected = [c for c in (NPI_COLUMNS + ALL_STOCK_COLUMNS) if c not in df.columns]
            detail["missing_expected"] = missing_expected.copy()

            # Period→DateStamp (rename)
            if has_period and not has_ds:
                df = df.rename(columns={"Period": "DateStamp"})
                has_ds = True
                detail["notes"].append("Renamed 'Period' → 'DateStamp'")

            # Date parsing & drop invalid
            if has_ds:
                df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
                df = df.dropna(subset=["DateStamp"])

            # Ensure soft-id columns exist (if truly missing)
            if "PlantID" not in df.columns:
                df["PlantID"] = "UNKNOWN"
            if "SapCode" not in df.columns:
                df["SapCode"] = "UNKNOWN"

            # Auto-create missing NPI/STOCK as zeros (and remember them)
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

            # Basic stats
            stats = _compute_basic_stats(df, "DateStamp")
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            _record_validation(label, detail)

        elif label == "Plant Capacity":
            # Required for capacity page
            required = ["PlantID", "MaxCapacity"]
            missing_req = [c for c in required if c not in df.columns]
            if missing_req:
                detail["required_ok"] = False
                detail["missing_required"] = missing_req

            # Optional DateStamp for week-specific capacity
            if "DateStamp" in df.columns:
                try:
                    df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
                except Exception:
                    pass
                stats = _compute_basic_stats(df, "DateStamp")
            else:
                stats = {"rows": int(len(df)), "date_min": None, "date_max": None}
            detail.update({k: v for k, v in stats.items() if k in ["rows", "date_min", "date_max"]})
            _record_validation(label, detail)

        else:
            # T&W Forecasts: no strict schema enforced yet
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

# --- DuckDB NPI Computation (no caching!) ---
def compute_npi_days_duckdb(df: pd.DataFrame, npi_categories: list[str]) -> pd.DataFrame:
    df2 = df.copy()

    # Defensive — create any missing NPI columns as zeros
    for c in npi_categories:
        if c not in df2.columns:
            df2[c] = 0

    # Ensure numeric cols
    for c in npi_categories:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)

    # Ensure base IDs exist for window partitions (soft defaults)
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
st.sidebar.title("App Navigation")
selection = st.sidebar.radio(
    "Go to:",
    [
        "Data load",
        "Non‑Productive Inventory (NPI) Management",
        "Planning Overview",
        "Storage Capacity Management"
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

    # Overall readiness badges
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
            st.caption("Notes: " + " | ".join(notes))

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
                    st.session_state["data"][label] = df_parquet
                    # Validation snapshot for cached data
                    df_tmp = df_parquet
                    if label == "BDD400":
                        required = ["DateStamp", "PlantID", "ClosingInventory"]
                        detail = {
                            "rows": int(len(df_tmp)),
                            "required_ok": all(c in df_tmp.columns for c in required),
                            "missing_required": [c for c in required if c not in df_tmp.columns],
                            "missing_expected": [],
                            "auto_created": [],
                            "date_min": _compute_basic_stats(df_tmp, "DateStamp")["date_min"],
                            "date_max": _compute_basic_stats(df_tmp, "DateStamp")["date_max"],
                            "notes": ["Loaded from cache (Parquet)"]
                        }
                        _record_validation(label, detail)
                    elif label == "Stock History":
                        has_ds = "DateStamp" in df_tmp.columns
                        has_period = "Period" in df_tmp.columns
                        missing_req = []
                        if not (has_ds or has_period):
                            missing_req.append("DateStamp_or_Period")
                        if "PlantID" not in df_tmp.columns:
                            missing_req.append("PlantID")
                        if "SapCode" not in df_tmp.columns:
                            missing_req.append("SapCode")
                        detail = {
                            "rows": int(len(df_tmp)),
                            "required_ok": len(missing_req) == 0,
                            "missing_required": missing_req,
                            "missing_expected": [c for c in (NPI_COLUMNS + ALL_STOCK_COLUMNS) if c not in df_tmp.columns],
                            "auto_created": [],
                            "date_min": _compute_basic_stats(df_tmp, "DateStamp")["date_min"] if has_ds else None,
                            "date_max": _compute_basic_stats(df_tmp, "DateStamp")["date_max"] if has_ds else None,
                            "notes": ["Loaded from cache (Parquet)"]
                        }
                        _record_validation(label, detail)
                    elif label == "Plant Capacity":
                        required = ["PlantID", "MaxCapacity"]
                        detail = {
                            "rows": int(len(df_tmp)),
                            "required_ok": all(c in df_tmp.columns for c in required),
                            "missing_required": [c for c in required if c not in df_tmp.columns],
                            "missing_expected": [],
                            "auto_created": [],
                            "date_min": _compute_basic_stats(df_tmp, "DateStamp")["date_min"] if "DateStamp" in df_tmp.columns else None,
                            "date_max": _compute_basic_stats(df_tmp, "DateStamp")["date_max"] if "DateStamp" in df_tmp.columns else None,
                            "notes": ["Loaded from cache (Parquet)"]
                        }
                        _record_validation(label, detail)
                    else:
                        detail = {
                            "rows": int(len(df_tmp)),
                            "required_ok": True,
                            "missing_required": [],
                            "missing_expected": [],
                            "auto_created": [],
                            "date_min": _compute_basic_stats(df_tmp, "DateStamp")["date_min"] if "DateStamp" in df_tmp.columns else None,
                            "date_max": _compute_basic_stats(df_tmp, "DateStamp")["date_max"] if "DateStamp" in df_tmp.columns else None,
                            "notes": ["Loaded from cache (Parquet)", "No strict validation rules configured for this dataset yet."]
                        }
                        _record_validation(label, detail)
                else:
                    # Fallback to default local files
                    if os.path.exists(fname):
                        df_loaded = load_file_content(fname, label)
                    else:
                        alt = fname.replace(".xlsx", ".csv")
                        df_loaded = load_file_content(alt, label) if os.path.exists(alt) else None
                    if df_loaded is not None:
                        st.session_state["data"][label] = df_loaded
                        save_as_parquet(df_loaded, label)

            # Status badge
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

    # Period→DateStamp safety (in case cached parquet predates change)
    if "Period" in df.columns and "DateStamp" not in df.columns:
        df = df.rename(columns={"Period": "DateStamp"})
    if "DateStamp" not in df.columns:
        st.error("Stock History is missing the 'DateStamp' (Period) column.")
        st.stop()

    # Normalize datetime
    df["DateStamp"] = pd.to_datetime(df["DateStamp"], errors="coerce")
    df = df.dropna(subset=["DateStamp"])

    # Ensure base ID columns exist (hard requirement for NPI logic)
    if "PlantID" not in df.columns or "SapCode" not in df.columns:
        st.error("Stock History must include 'PlantID' and 'SapCode' columns.")
        st.stop()

    # Ensure NPI & stock columns exist at runtime (double safety + numeric)
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

    # Compute fresh NPI metrics
    df_aug = compute_npi_days_duckdb(df, NPI_COLUMNS)
    latest_date = df_aug["DateStamp"].max()
    df_latest = df_aug[df_aug["DateStamp"] == latest_date]

    # --- TOP SUMMARY (defensive against missing columns) ---
    st.subheader(f"📋 Global Plant Summary (As of {latest_date:%Y-%m-%d})")
    available_cols = [c for c in ALL_STOCK_COLUMNS if c in df_latest.columns]
    if not available_cols:
        st.info("No stock columns available to summarize for the latest date.")
    else:
        # Ensure numeric for safety
        df_latest[available_cols] = df_latest[available_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)
        summary = (
            df_latest.groupby("PlantID")[available_cols]
            .sum()
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Material Drilldown",
        "⏳ Accumulation Monitor",
        "📈 Trend Analysis",
        "🔎 Inventory Search"
    ])

    # --- TAB 1 ---
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

    # --- TAB 2 ---
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

    # --- TAB 3 ---
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

    # --- TAB 4 (NEW): Inventory Search ---
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

            # date range filter
            min_d = pd.to_datetime(df["DateStamp"].min()).date()
            max_d = pd.to_datetime(df["DateStamp"].max()).date()
            date_range = st.date_input(
                "Date range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                help="Filter records within the selected date range"
            )

        # derive SAP list after plant filter (for convenience)
        df_plant = df[df["PlantID"].isin(plants_sel)] if plants_sel else df.copy()
        with right:
            saps_all = sorted(df_plant["SapCode"].dropna().astype(str).unique())
            saps_sel = st.multiselect(
                "SAP code(s)",
                options=saps_all,
                placeholder="Type to search SAP codes..."
            )

        # Build filter
        mask = pd.Series(True, index=df.index)
        if plants_sel:
            mask &= df["PlantID"].isin(plants_sel)
        if saps_sel:
            mask &= df["SapCode"].astype(str).isin(saps_sel)

        # Date filter
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= (df["DateStamp"] >= start_dt) & (df["DateStamp"] <= end_dt)

        # Columns to show (deduplicated)
        base_cols = ["DateStamp", "PlantID", "SapCode", "MaterialDescription"]
        combined = base_cols + ALL_STOCK_COLUMNS + NPI_COLUMNS
        seen = set()
        cols_to_show = [c for c in combined if not (c in seen or seen.add(c))]

        cols_existing = [c for c in cols_to_show if c in df.columns]
        out = df.loc[mask, cols_existing].sort_values(["DateStamp", "PlantID", "SapCode"])

        # safety net: drop duplicate columns before display
        out = out.loc[:, ~out.columns.duplicated(keep="first")]

        st.dataframe(out, use_container_width=True, height=480)
        st.download_button(
            label="⬇️ Download filtered records (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="inventory_search_results.csv",
            mime="text/csv"
        )

# --- PLANNING OVERVIEW ---
elif selection == "Planning Overview":
    st.header("🧭 Planning Overview")

    df_bdd = st.session_state["data"].get("BDD400")
    if df_bdd is None:
        st.error("Please upload BDD400 first.")
        st.stop()

    if not ensure_columns_exist(df_bdd, ["DateStamp", "PlantID", "ClosingInventory"], "BDD400"):
        st.stop()

    # Determine next 18 weeks available in BDD400
    future_weeks = get_future_weeks_18(df_bdd)
    if not future_weeks:
        st.warning("No future weeks found in BDD400. Showing the most recent available weeks.")
        # Fallback: last 18 available weeks
        all_weeks = sorted(pd.to_datetime(df_bdd["DateStamp"].unique()))
        future_weeks = all_weeks[-18:]

    # Filter to the next 18 weeks only
    df18 = df_bdd[df_bdd["DateStamp"].isin(future_weeks)]
    plants_all = sorted(df18["PlantID"].unique())

    plants_sel = st.multiselect(
        "Filter plants",
        options=plants_all,
        default=plants_all
    )

    df18 = df18[df18["PlantID"].isin(plants_sel)] if plants_sel else df18.copy()

    # Aggregate by week, plant
    closing = (
        df18.groupby(["DateStamp", "PlantID"])["ClosingInventory"]
        .sum()
        .reset_index()
    )

    # Pivot for display
    pivot_ci = (
        closing.pivot(index="DateStamp", columns="PlantID", values="ClosingInventory")
        .sort_index()
        .fillna(0)
    )

    st.subheader("Closing Inventory by Plant (Next 18 Weeks)")
    st.dataframe(pivot_ci, use_container_width=True)

    st.line_chart(pivot_ci, use_container_width=True)

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

    if not ensure_columns_exist(df_bdd, ["DateStamp", "PlantID", "ClosingInventory"], "BDD400"):
        st.stop()
    if not ensure_columns_exist(df_cap, ["PlantID", "MaxCapacity"], "Plant Capacity"):
        st.stop()

    # Weeks: same "next 18 weeks" logic
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
    # If capacity has DateStamp, use week-specific; else cross-join week list to plant capacities
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

    # Weekly totals (for TOTAL per week)
    weekly_totals = (
        df_merge.groupby("DateStamp")[["ClosingInventory", "MaxCapacity"]]
        .sum()
        .reset_index()
    )
    weekly_totals["Delta"] = weekly_totals["ClosingInventory"] - weekly_totals["MaxCapacity"]
    weekly_totals["Ratio"] = weekly_totals.apply(
        lambda r: (r["ClosingInventory"] / r["MaxCapacity"]) if r["MaxCapacity"] else pd.NA, axis=1
    )

    # --- Swapped orientation: rows = PlantID, columns = DateStamp ---
    st.subheader("Δ to Capacity by Plant & Week")
    st.caption(
        "Orientation: **rows = PlantID**, **columns = DateStamp (weeks)**. "
        "Color scale by % of capacity: >105% red, 95–105% yellow, <95% green. "
        "Δ = ClosingInventory − MaxCapacity. The last row **TOTAL** shows week-level aggregate."
    )

    # Rebuild pivots with swapped axes
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

    # Ensure all week columns exist before assignment
    for col in delta_pivot.columns:
        if col not in weekly_delta_ser.index:
            weekly_delta_ser.loc[col] = 0
    for col in ratio_pivot.columns:
        if col not in weekly_ratio_ser.index:
            weekly_ratio_ser.loc[col] = pd.NA

    # Add TOTAL row (aligned by DateStamp columns)
    delta_pivot.loc["TOTAL", :] = weekly_delta_ser.reindex(delta_pivot.columns).values
    ratio_pivot.loc["TOTAL", :] = weekly_ratio_ser.reindex(ratio_pivot.columns).values

    # Coloring helper based on ratio thresholds (expects same shape/index/cols)
    def colorize_by_ratio(ratio_df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=ratio_df.index, columns=ratio_df.columns)
        for idx in ratio_df.index:
            for col in ratio_df.columns:
                r = ratio_df.loc[idx, col]
                if pd.isna(r):
                    styles.loc[idx, col] = ''
                elif r > 1.05:
                    styles.loc[idx, col] = 'background-color: #f8d7da;'  # light red
                elif r >= 0.95:
                    styles.loc[idx, col] = 'background-color: #fff3cd;'  # light yellow
                else:
                    styles.loc[idx, col] = 'background-color: #d4edda;'  # light green
        return styles

    styled_delta = (
        delta_pivot.style
        .apply(lambda _: colorize_by_ratio(ratio_pivot), axis=None)
        .format("{:,.0f}")
    )
    st.dataframe(styled_delta, use_container_width=True)

    # Weekly totals panel (unchanged)
    st.subheader("Weekly Total Inventory vs Capacity")
    totals_display = weekly_totals.set_index("DateStamp").rename(columns={
        "ClosingInventory": "TotalClosingInventory",
        "MaxCapacity": "TotalCapacity"
    })
    st.dataframe(
        totals_display[["TotalClosingInventory", "TotalCapacity", "Delta", "Ratio"]]
        .style.format({"TotalClosingInventory":"{:,.0f}", "TotalCapacity":"{:,.0f}", "Delta":"{:,.0f}", "Ratio":"{:,.2f}"}),
        use_container_width=True
    )

else:
    st.header(selection)
    st.info("Implementation pending.")
