# --- Replace from here ---
    st.subheader("Δ to Capacity by Plant & Week")

    st.caption(
        "Orientation: rows = PlantID, columns = DateStamp (weeks). "
        "Color scale by % of capacity: >105% red, 95–105% yellow, <95% green. "
        "Δ = ClosingInventory − MaxCapacity. The last row **TOTAL** shows week-level aggregate."
    )

    # Rebuild pivots with swapped axes (rows=plants, cols=weeks)
    delta_pivot = (
        df_merge.pivot(index="PlantID", columns="DateStamp", values="Delta")
        .sort_index()
        .fillna(0)
    )
    ratio_pivot = (
        df_merge.pivot(index="PlantID", columns="DateStamp", values="Ratio")
        .sort_index()
    )

    # Append TOTAL row (per week) using weekly_totals
    weekly_delta_ser = weekly_totals.set_index("DateStamp")["Delta"]
    weekly_ratio_ser = weekly_totals.set_index("DateStamp")["Ratio"]

    # Ensure all columns present before assignment (in case of missing weeks)
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

    # Apply formatting/colors by ratio
    styled_delta = (
        delta_pivot.style
        .apply(lambda _: colorize_by_ratio(ratio_pivot), axis=None)
        .format("{:,.0f}")
    )
    st.dataframe(styled_delta, use_container_width=True)
    # --- Replace until here ---
