elif selection == "Non‑Productive Inventory (NPI) Management":
    st.header("📉 Non‑Productive Inventory (NPI) Management")
    
    df_stock = st.session_state['data'].get("Stock History")
    
    if df_stock is not None:
        # --- 1. Data Preparation ---
        df_stock['DateStamp'] = pd.to_datetime(df_stock['DateStamp'])
        npi_vars = ['BlockedStockQty', 'QualityInspectionQty', 'OveragedTireQty']
        
        # --- 2. Sidebar Filters (Local to this page) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("NPI Filters")
        
        warehouses = st.sidebar.multiselect("Warehouse", options=df_stock['PlantID'].unique(), default=df_stock['PlantID'].unique())
        brands = st.sidebar.multiselect("Brand", options=df_stock['Brand'].unique(), default=df_stock['Brand'].unique())
        seasons = st.sidebar.multiselect("Season", options=df_stock['Season'].unique(), default=df_stock['Season'].unique())
        
        # Date Filter
        min_date = df_stock['DateStamp'].min().to_pydatetime()
        max_date = df_stock['DateStamp'].max().to_pydatetime()
        date_range = st.sidebar.slider("Period", min_value=min_date, max_value=max_date, value=(min_date, max_date))

        # Apply Filters
        mask = (
            df_stock['PlantID'].isin(warehouses) & 
            df_stock['Brand'].isin(brands) & 
            df_stock['Season'].isin(seasons) &
            (df_stock['DateStamp'] >= date_range[0]) &
            (df_stock['DateStamp'] <= date_range[1])
        )
        df_filtered = df_stock[mask].copy()

        # --- 3. Dashboard Layout ---
        tab1, tab2, tab3 = st.tabs(["📈 NPI Trends", "⏳ Accumulation Analysis", "📋 Material Summary"])

        with tab1:
            st.subheader("Inventory Quantities Over Time")
            # Grouping for trend chart
            df_trend = df_filtered.groupby('DateStamp')[npi_vars].sum().reset_index()
            st.line_chart(df_trend, x='DateStamp', y=npi_vars)
            
            # KPI Row
            total_npi = df_filtered[df_filtered['DateStamp'] == df_filtered['DateStamp'].max()][npi_vars].sum().sum()
            st.metric("Total Current NPI Units (Latest Date)", f"{total_npi:,.0f}")

        with tab2:
            st.subheader("Days Since Last Zero (Accumulation)")
            st.write("This table shows how long stock has been sitting in NPI categories without reaching zero.")
            
            # Calculation: Last Zero Date
            # We calculate this per Material and Warehouse
            latest_date = df_filtered['DateStamp'].max()
            
            @st.cache_data
            def calculate_accumulation(df, vars, reference_date):
                acc_results = []
                for (sap, plant, desc), group in df.groupby(['SapCode', 'PlantID', 'MaterialDescription']):
                    group = group.sort_values('DateStamp')
                    res = {'Material': desc, 'Warehouse': plant}
                    for var in vars:
                        current_val = group[group['DateStamp'] == reference_date][var].sum()
                        if current_val > 0:
                            # Find last date where stock was 0
                            zeros = group[group[var] == 0]
                            last_zero = zeros['DateStamp'].max() if not zeros.empty else group['DateStamp'].min()
                            days = (reference_date - last_zero).days
                            res[f'{var} Age (Days)'] = days
                            res[f'{var} Qty'] = current_val
                        else:
                            res[f'{var} Age (Days)'] = 0
                            res[f'{var} Qty'] = 0
                    acc_results.append(res)
                return pd.DataFrame(acc_results)

            df_acc = calculate_accumulation(df_filtered, npi_vars, latest_date)
            st.dataframe(df_acc, use_container_width=True)

        with tab3:
            st.subheader("Top Operational Bottlenecks")
            # Filter to materials with accumulation > 30 days
            col_to_check = st.selectbox("Identify bottlenecks for:", [f"{v} Age (Days)" for v in npi_vars])
            bottlenecks = df_acc[df_acc[col_to_check] > 0].sort_values(by=col_to_check, ascending=False).head(10)
            
            if not bottlenecks.empty:
                st.write(f"Top 10 materials with longest {col_to_check}:")
                st.table(bottlenecks[['Material', 'Warehouse', col_to_check, col_to_check.replace('Age (Days)', 'Qty')]])
            else:
                st.success("No long-term accumulation detected in the selected filters!")

    else:
        st.warning("Please upload 'Stock History' in the Data Load section to activate this analysis.")
