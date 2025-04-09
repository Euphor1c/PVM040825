# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io # Required for Excel export

###############################################
# Streamlit Setup
###############################################
# Call set_page_config() as the FIRST Streamlit command
st.set_page_config(layout="wide")

###############################################
# Initialize Session State Variables
###############################################
# Ensure all session state keys are initialized upfront
if 'filter_selections' not in st.session_state:
    st.session_state.filter_selections = {}
if 'revenue_accounts' not in st.session_state:
    st.session_state.revenue_accounts = []
if 'cost_accounts' not in st.session_state:
    st.session_state.cost_accounts = []
if 'aggregation_level' not in st.session_state:
    st.session_state.aggregation_level = 'SKU' # Default aggregation level
if 'original_columns' not in st.session_state:
    st.session_state.original_columns = []

###############################################
# Helper Functions
###############################################

def convert_df_to_csv(df_: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df_.to_csv(index=False).encode("utf-8")

def create_excel_export(dataframes: dict) -> bytes:
    """Creates an Excel file in memory from a dictionary of DataFrames."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, df_ in dataframes.items():
                safe_sheet_name = sheet_name[:31] # Excel sheet name limit
                if isinstance(df_, pd.DataFrame) and not df_.empty:
                    df_.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                # Optionally log skipped sheets
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return b""
    output.seek(0)
    return output.getvalue()

def margin_ratio(rev_in, cos_in):
    """Calculate margin percentage; handle division by zero, potentially negative revenue, and scalar/array inputs."""
    was_scalar = np.isscalar(rev_in)
    rev = np.atleast_1d(rev_in).astype(float)
    cos = np.atleast_1d(cos_in).astype(float)
    if rev.shape != cos.shape:
        try:
            rev, cos = np.broadcast_arrays(rev, cos)
        except ValueError:
            if was_scalar: return np.nan
            out_shape = getattr(rev_in, 'shape', None) or getattr(cos_in, 'shape', None)
            return np.full(out_shape, np.nan) if out_shape else np.nan
    margin = np.zeros_like(rev, dtype=float)
    valid_mask = rev > 1e-9
    out_array = np.zeros_like(rev[valid_mask]) if np.any(valid_mask) else np.array([])
    where_array = rev[valid_mask]!=0 if np.any(valid_mask) else np.array([])
    if np.any(valid_mask):
        margin[valid_mask] = np.divide((rev[valid_mask] - cos[valid_mask]), rev[valid_mask], out=out_array, where=where_array)
    margin[~valid_mask] = 0.0
    return margin.item() if was_scalar else margin

def add_delta_margin_column(df, margin_col="Margin%"):
    """Add a column with differences (delta) in margin% from the previous row."""
    df = df.copy()
    if margin_col not in df.columns or df.empty:
        df["Delta Margin%"] = 0.0
        return df
    df[margin_col] = pd.to_numeric(df[margin_col], errors='coerce')
    df["Delta Margin%"] = df[margin_col].diff()
    df["Delta Margin%"] = df["Delta Margin%"].fillna(0.0)
    return df

def select_all_buttons(key, options, widget_type='multiselect'):
    """Creates Select All/Deselect All buttons for a multiselect widget."""
    button_key_select = f"select_all_{key}"
    button_key_deselect = f"deselect_all_{key}"
    col1, col2 = st.sidebar.columns(2)
    select_pressed = col1.button("Select All", key=button_key_select)
    deselect_pressed = col2.button("Deselect All", key=button_key_deselect)
    # Ensure filter_selections exists before trying to access key
    if 'filter_selections' not in st.session_state:
        st.session_state.filter_selections = {} # Initialize if somehow missing
    if key not in st.session_state.filter_selections:
        st.session_state.filter_selections[key] = options if options else []
    if select_pressed:
        st.session_state.filter_selections[key] = options
        st.rerun()
    if deselect_pressed:
        st.session_state.filter_selections[key] = []
        st.rerun()

###############################################
# Core PVM Calculation Functions
###############################################

# @st.cache_data # Consider re-enabling caching after testing
def pvm_for_account(df_in, y_s, s_s, y_e, s_e, acct, agg_level):
    """
    Compute the Price, Volume, Mix, New_Items, Discontinued_Items, and Ending values
    for a given account by the specified aggregation level. (CORRECTED MULTIINDEX HANDLING V2)
    """
    sub = df_in[df_in["Account"] == acct].copy()
    sub['Amount'] = pd.to_numeric(sub['Amount'], errors='coerce').fillna(0.0)
    sub['Quantity'] = pd.to_numeric(sub['Quantity'], errors='coerce').fillna(0.0)
    sub['Year'] = sub['Year'].astype(str)
    sub['Scenario'] = sub['Scenario'].astype(str)
    y_s, s_s, y_e, s_e = str(y_s), str(s_s), str(y_e), str(s_e)

    default_return = {"Starting": 0, "Price": 0, "Volume": 0, "Mix": 0, "New_Items": 0, "Discontinued_Items": 0, "Ending": 0}

    if agg_level not in sub.columns:
        return default_return
    sub[agg_level] = sub[agg_level].astype(str).fillna('Unknown')
    if sub.empty:
        return default_return

    piv_raw = None
    piv = None

    try:
        piv_raw = sub.pivot_table(index=agg_level, columns=["Year", "Scenario"], values=["Amount", "Quantity"], aggfunc="sum")
        expected_columns_final = pd.MultiIndex.from_tuples(
            [('Amount', y_s, s_s), ('Amount', y_e, s_e), ('Quantity', y_s, s_s), ('Quantity', y_e, s_e)],
            names=['Measure', 'Year', 'Scenario']
        )
        if piv_raw.empty:
             idx = sub[agg_level].unique()
             if not isinstance(idx, pd.Index): idx = pd.Index(idx, name=agg_level)
             if idx.empty and not sub.empty: idx = pd.Index([f"Unknown_{agg_level}"], name=agg_level)
             piv = pd.DataFrame(0.0, index=idx, columns=expected_columns_final)
             piv.index.name = agg_level
        else:
             if not isinstance(piv_raw.columns, pd.MultiIndex): pass # Allow reindex to handle
             piv = piv_raw.reindex(columns=expected_columns_final, fill_value=0.0)
        piv = piv.sort_index(axis=1)
    except Exception as e:
        # Avoid flooding with errors, maybe log differently or sample errors
        # print(f"Error pivoting data for account '{acct}' at level '{agg_level}'. Error: {e}")
        return default_return

    try:
        p_qty = piv[('Quantity', y_s, s_s)]
        p_amt = piv[('Amount', y_s, s_s)]
        c_qty = piv[('Quantity', y_e, s_e)]
        c_amt = piv[('Amount', y_e, s_e)]
    except KeyError as e:
         # print(f"Internal Error: Failed accessing pivot column {e} for '{acct}'. Columns: {piv.columns}")
         return default_return
    except Exception as e:
        # print(f"Unexpected error accessing pivot columns for '{acct}': {e}")
        return default_return

    p_price = np.divide(p_amt.astype(float), p_qty.astype(float), out=np.zeros_like(p_amt, dtype=float), where=p_qty!=0)
    c_price = np.divide(c_amt.astype(float), c_qty.astype(float), out=np.zeros_like(c_amt, dtype=float), where=c_qty!=0)
    tolerance = 1e-9
    exist_mask = (p_qty.abs() > tolerance) & (c_qty.abs() > tolerance)
    new_mask = (p_qty.abs() <= tolerance) & (c_qty.abs() > tolerance)
    disc_mask = (p_qty.abs() > tolerance) & (c_qty.abs() <= tolerance)
    p_qty_exist, c_qty_exist = p_qty[exist_mask], c_qty[exist_mask]
    p_amt_exist, c_amt_exist = p_amt[exist_mask], c_amt[exist_mask]
    p_price_exist, c_price_exist = p_price[exist_mask], c_price[exist_mask]
    price_effect = ((c_price_exist - p_price_exist) * p_qty_exist).sum()
    volume_effect = ((c_qty_exist - p_qty_exist) * p_price_exist).sum()
    total_change_existing = (c_amt_exist - p_amt_exist).sum()
    mix_effect = total_change_existing - price_effect - volume_effect
    new_effect = c_amt[new_mask].sum()
    disc_effect = -p_amt[disc_mask].sum()

    return {"Starting": p_amt.sum(), "Price": price_effect, "Volume": volume_effect, "Mix": mix_effect,
            "New_Items": new_effect, "Discontinued_Items": disc_effect, "Ending": c_amt.sum()}

# @st.cache_data # Consider re-enabling caching after testing
def compute_pvm_all(df_in, y_s, s_s, y_e, s_e, acct_list, agg_level):
    """Compute aggregated PVM values for all selected accounts."""
    row_order = ["Starting", "Price", "Volume", "Mix", "New_Items", "Discontinued_Items", "Ending"]
    results_dict = {str(acct): pvm_for_account(df_in, str(y_s), str(s_s), str(y_e), str(s_e), str(acct), agg_level)
                    for acct in acct_list}
    if not results_dict:
        return pd.DataFrame(columns=["Combined"], index=row_order).fillna(0.0)
    pdf = pd.DataFrame(results_dict)
    pdf = pdf.reindex(columns=[str(a) for a in acct_list], fill_value=0.0)
    pdf = pdf.reindex(row_order).fillna(0.0)
    pdf["Combined"] = pdf.sum(axis=1)
    return pdf

# @st.cache_data # Consider re-enabling caching after testing
def compute_item_pvm(df_in, y_s, s_s, y_e, s_e, acct_list, hier_cols_to_add, agg_level):
    """Compute Item-level PVM table for the chosen aggregation level."""
    if agg_level not in df_in.columns: return pd.DataFrame()
    df_in_copy = df_in.copy()
    df_in_copy[agg_level] = df_in_copy[agg_level].astype(str).fillna('Unknown')
    unique_items = sorted(df_in_copy[agg_level].unique())
    if not unique_items: return pd.DataFrame()

    hier_cols_unique = [h for h in hier_cols_to_add if h in df_in_copy.columns and h != agg_level]
    hierarchy_data_lookup = {}
    if hier_cols_unique:
         try:
             hier_data_df = df_in_copy.astype({h: str for h in hier_cols_unique})
             hierarchy_data_lookup = hier_data_df.drop_duplicates(subset=[agg_level])[([agg_level] + hier_cols_unique)].set_index(agg_level).to_dict('index')
         except Exception: pass # Ignore hierarchy lookup errors

    item_count = len(unique_items)
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_list = []
    acct_list_str = [str(a) for a in acct_list]

    for i, item in enumerate(unique_items):
        if (i % 50 == 0) or (i == item_count - 1):
             status_text.text(f"Processing {agg_level} {i+1} of {item_count}: {item}")
             progress_bar.progress((i + 1) / item_count)
        sub_item = df_in_copy[df_in_copy[agg_level] == item]
        item_hier_data = hierarchy_data_lookup.get(item, {h: "N/A" for h in hier_cols_unique})
        if sub_item.empty: continue
        for acct in acct_list_str:
            sub_acct = sub_item[sub_item["Account"] == acct]
            pvm_dict = pvm_for_account(sub_acct, str(y_s), str(s_s), str(y_e), str(s_e), acct, agg_level)
            row = {agg_level: item, "Account": acct} # Use the actual agg_level column name
            row.update(pvm_dict)
            row.update({h_col: item_hier_data.get(h_col, "N/A") for h_col in hier_cols_unique})
            results_list.append(row)
    status_text.text(f"Processed {item_count} items.")
    progress_bar.empty()
    return pd.DataFrame(results_list) if results_list else pd.DataFrame()

# @st.cache_data # Consider re-enabling caching after testing
def build_item_factor_sums(item_pvm: pd.DataFrame, revenue_accts: list, cost_accts: list, agg_level) -> pd.DataFrame:
    """For each (Item, Factor), sum the revenue and cost amounts."""
    factors = ["Starting", "Price", "Volume", "Mix", "New_Items", "Discontinued_Items", "Ending"]
    if item_pvm.empty or not all(c in item_pvm.columns for c in [agg_level, "Account"]): return pd.DataFrame()
    value_vars_present = [f for f in factors if f in item_pvm.columns]
    if not value_vars_present: return pd.DataFrame()
    id_vars_present = [agg_level, "Account"]
    df_sub = item_pvm[id_vars_present + value_vars_present].copy()
    try:
        df_melt = df_sub.melt(id_vars=id_vars_present, value_vars=value_vars_present, var_name="Factor", value_name="Amount")
    except Exception: return pd.DataFrame()
    revenue_accts_str = {str(a) for a in revenue_accts}
    cost_accts_str = {str(a) for a in cost_accts}
    def rev_or_cost(acct):
        acct_str = str(acct);
        if acct_str in revenue_accts_str: return "Revenue"
        elif acct_str in cost_accts_str: return "Cost"
        else: return "Ignore"
    df_melt["Type"] = df_melt["Account"].astype(str).apply(rev_or_cost)
    df_melt = df_melt[df_melt["Type"] != "Ignore"]
    if df_melt.empty: return pd.DataFrame()
    try:
        df_melt['Amount'] = pd.to_numeric(df_melt['Amount'], errors='coerce').fillna(0.0)
        df_melt[agg_level] = df_melt[agg_level].astype(str)
        df_melt['Factor'] = df_melt['Factor'].astype(str)
        grouped = df_melt.groupby([agg_level, "Factor", "Type"], as_index=False)["Amount"].sum()
    except Exception: return pd.DataFrame()
    try:
        grouped = grouped.drop_duplicates(subset=[agg_level, "Factor", "Type"], keep='first')
        pivoted = grouped.pivot(index=[agg_level, "Factor"], columns="Type", values="Amount").fillna(0)
    except Exception: return pd.DataFrame()
    if "Revenue" not in pivoted.columns: pivoted["Revenue"] = 0.0
    if "Cost" not in pivoted.columns: pivoted["Cost"] = 0.0
    pivoted = pivoted.reset_index()
    factor_order = pd.Categorical(pivoted['Factor'], categories=factors, ordered=True)
    pivoted['Factor'] = factor_order
    pivoted[agg_level] = pivoted[agg_level].astype(str)
    pivoted = pivoted.sort_values(by=[agg_level, 'Factor'])
    pivoted["Net_Impact"] = pivoted["Revenue"] - pivoted["Cost"]
    return pivoted

def aggregator_step_margin_deltas(pvm_agg_df):
    """Compute the aggregated margin% step deltas."""
    row_list = ["Starting", "Price", "Volume", "Mix", "New_Items", "Discontinued_Items", "Ending"]
    valid_rows = [r for r in row_list if r in pvm_agg_df.index]
    if not valid_rows or not all(c in pvm_agg_df.columns for c in ["Revenue_Sum", "Cost_Sum"]):
        st.warning("Cannot calculate margin steps: Aggregated PVM data missing required rows or columns.")
        return {}
    margin_at_step = {}
    rev_accum, cos_accum = 0.0, 0.0

    for f in valid_rows:
        # Get values, coercing errors and checking for NaN immediately
        r_val = pd.to_numeric(pvm_agg_df.loc[f, "Revenue_Sum"], errors='coerce')
        c_val = pd.to_numeric(pvm_agg_df.loc[f, "Cost_Sum"], errors='coerce')

        # Replace NaN with 0 before using the value
        r_val = 0.0 if pd.isna(r_val) else r_val
        c_val = 0.0 if pd.isna(c_val) else c_val

        if f == "Starting":
            rev_accum, cos_accum = r_val, c_val
            margin_at_step[f] = margin_ratio(float(rev_accum), float(cos_accum))
        elif f == "Ending":
             # Calculate final margin based on absolute ending values
             margin_at_step[f] = margin_ratio(float(r_val), float(c_val))
             # No accumulation needed for the 'Ending' step itself in the loop's logic,
             # as its margin is absolute, not based on prior accumulation.
             # The delta calculation later will handle the diff between last factor and Ending.
             continue # Skip to next factor after calculating Ending margin
        else: # Intermediate steps accumulate
             rev_accum += r_val
             cos_accum += c_val
             margin_at_step[f] = margin_ratio(float(rev_accum), float(cos_accum)) # Calculate cumulative margin

    # --- Calculate deltas between steps ---
    margin_deltas = {"Starting": 0.0} # Delta at start is zero
    prev_margin = margin_at_step.get("Starting", 0.0)
    # Ensure prev_margin is a float, default to 0 if it was NaN/None
    prev_margin = 0.0 if pd.isna(prev_margin) else float(prev_margin)

    # Iterate through factors *including* Ending to get the final delta
    for i in range(1, len(valid_rows)):
        current_factor = valid_rows[i]
        current_margin = margin_at_step.get(current_factor, 0.0) # Use .get for safety
        # Ensure margins are numeric before subtracting
        current_margin = 0.0 if pd.isna(current_margin) else float(current_margin)

        margin_deltas[current_factor] = current_margin - prev_margin
        prev_margin = current_margin # Update previous margin for next delta calculation

    return margin_deltas

# @st.cache_data # Consider re-enabling caching after testing
def build_item_step_table(item_factor_summary, agg_margin_deltas, pvm_agg_df, agg_level):
    """Builds table showing item's dollar bridge, cumulative margin, and allocated share of the aggregator's margin step change."""
    row_list = ["Starting", "Price", "Volume", "Mix", "New_Items", "Discontinued_Items", "Ending"]
    # Check inputs
    if item_factor_summary.empty:
        st.warning("Item factor summary is empty. Cannot build item step table.")
        return pd.DataFrame()
    if not agg_margin_deltas:
        st.warning("Aggregator margin deltas are missing. Cannot build item step table.")
        return pd.DataFrame()
    if pvm_agg_df.empty:
        st.warning("Aggregated PVM data is missing. Cannot build item step table.")
        return pd.DataFrame()

    # Ensure Factor column exists and get valid factors
    if 'Factor' not in item_factor_summary.columns or agg_level not in item_factor_summary.columns:
        st.warning("Missing 'Factor' or aggregation level column in item summary. Cannot build item step table.")
        return pd.DataFrame()
    valid_factors = [f for f in row_list if f in item_factor_summary['Factor'].unique()]
    if not valid_factors:
         st.warning("No valid factors found in item summary. Cannot build item step table.")
         return pd.DataFrame()

    # Pre-calculate aggregator dollar deltas for allocation
    agg_dollar_deltas = {}
    for f in valid_factors:
        if f in pvm_agg_df.index and f not in ["Starting", "Ending"] and "Net_Change" in pvm_agg_df.columns:
             # --- FIX: Apply pd.to_numeric and handle NaN correctly ---
             net_change_val = pd.to_numeric(pvm_agg_df.loc[f, "Net_Change"], errors='coerce')
             agg_dollar_deltas[f] = 0.0 if pd.isna(net_change_val) else net_change_val
             # --- END FIX ---
        else:
             agg_dollar_deltas[f] = 0.0

    unique_items_in_summary = sorted(item_factor_summary[agg_level].unique())
    item_count = len(unique_items_in_summary)
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_list = []

    for i, item in enumerate(unique_items_in_summary):
        if (i % 50 == 0) or (i == item_count - 1):
            status_text.text(f"Calculating margin allocation for {agg_level} {i+1} of {item_count}: {item}")
            progress_bar.progress((i + 1) / item_count)
        sub = item_factor_summary[item_factor_summary[agg_level] == item].set_index("Factor").reindex(valid_factors).fillna(0)
        accum_r, accum_c, prev_item_margin = 0.0, 0.0, 0.0
        for f in valid_factors:
            # --- FIX: Get Revenue/Cost/Net for this factor *step* safely ---
            rev_step_raw = sub.loc[f, "Revenue"] if "Revenue" in sub.columns else 0.0
            cos_step_raw = sub.loc[f, "Cost"] if "Cost" in sub.columns else 0.0
            net_impact_raw = sub.loc[f, "Net_Impact"] if "Net_Impact" in sub.columns else None # Get raw value or None

            rev_step = pd.to_numeric(rev_step_raw, errors='coerce')
            cos_step = pd.to_numeric(cos_step_raw, errors='coerce')

            # Handle NaN after coercion
            rev_step = 0.0 if pd.isna(rev_step) else rev_step
            cos_step = 0.0 if pd.isna(cos_step) else cos_step

            # Calculate item_dollar_delta safely
            if net_impact_raw is not None:
                item_dollar_delta = pd.to_numeric(net_impact_raw, errors='coerce')
                item_dollar_delta = 0.0 if pd.isna(item_dollar_delta) else item_dollar_delta
            else:
                item_dollar_delta = rev_step - cos_step
            # --- END FIX ---

            if f == "Starting": accum_r, accum_c = rev_step, cos_step
            else: accum_r += rev_step; accum_c += cos_step
            item_accum_margin = margin_ratio(accum_r, accum_c)
            item_margin_delta_step = item_accum_margin - prev_item_margin if pd.notna(item_accum_margin) and pd.notna(prev_item_margin) else 0.0
            agg_step_margin_delta = agg_margin_deltas.get(f, 0.0)
            agg_step_dollar_delta = agg_dollar_deltas.get(f, 0.0) # Already cleaned above
            item_margin_impact_alloc = 0.0
            if abs(agg_step_dollar_delta) > 1e-9 and f not in ["Starting", "Ending"]:
                fraction = np.divide(float(item_dollar_delta), float(agg_step_dollar_delta), out=np.zeros(1, dtype=float), where=abs(agg_step_dollar_delta) > 1e-12)
                item_margin_impact_alloc = fraction.item() * agg_step_margin_delta
            results_list.append({
                agg_level: item, "Factor": f, "Revenue_Bridge": rev_step, "Cost_Bridge": cos_step, "Net_Bridge": item_dollar_delta,
                "Item_Accum_Revenue": accum_r, "Item_Accum_Cost": accum_c, "Item_Accum_MarginPct": item_accum_margin,
                "Item_Margin_Delta": item_margin_delta_step, "Aggregator_MarginStep": agg_step_margin_delta,
                "Item_MarginStep_Impact_Alloc": item_margin_impact_alloc
            })
            prev_item_margin = item_accum_margin if pd.notna(item_accum_margin) else 0.0
    status_text.text(f"Calculated margin allocations for {item_count} items.")
    progress_bar.empty()
    if not results_list: return pd.DataFrame()
    final_df = pd.DataFrame(results_list)
    factor_order_cat = pd.Categorical(final_df['Factor'], categories=row_list, ordered=True)
    final_df['Factor'] = factor_order_cat
    final_df[agg_level] = final_df[agg_level].astype(str)
    final_df = final_df.sort_values(by=[agg_level, 'Factor'])
    return final_df

###############################################
# Plotting Functions
###############################################

def create_waterfall(df_, value_col, delta_col_plot, title):
    """Generate a waterfall chart for the specified value column using pre-calculated deltas."""
    if df_.empty or value_col not in df_.columns or delta_col_plot not in df_.columns: return go.Figure()
    df_[value_col] = pd.to_numeric(df_[value_col], errors='coerce')
    df_[delta_col_plot] = pd.to_numeric(df_[delta_col_plot], errors='coerce')
    df_ = df_.dropna(subset=[value_col, delta_col_plot])
    if df_.empty: return go.Figure()
    steps_ = df_["Step"].tolist()
    y_deltas_plot = df_[delta_col_plot].tolist()
    y_cumulative = df_[value_col].tolist()
    measure_ = ["absolute"] + ["relative"] * (len(steps_) - 2) + ["total"]
    y_vals_for_plot = [y_cumulative[0]] + y_deltas_plot[1:-1] + [y_cumulative[-1]]
    text_ = [f"${val:,.0f}" for val in y_vals_for_plot]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measure_, x=steps_, y=y_vals_for_plot, text=text_, textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}}, increasing={"marker":{"color":"#1f77b4"}},
        decreasing={"marker":{"color":"#d62728"}}, totals={"marker":{"color":"#9467bd"}}
    ))
    fig.update_layout(title=title, waterfallgap=0.3)
    return fig

def margin_waterfall(df_, title):
    """Generate a margin waterfall chart."""
    if df_.empty or not all(col in df_.columns for col in ["Step", "Margin%", "Delta Margin%"]): return go.Figure()
    df_['Margin%'] = pd.to_numeric(df_['Margin%'], errors='coerce')
    df_['Delta Margin%'] = pd.to_numeric(df_['Delta Margin%'], errors='coerce')
    df_ = df_.dropna(subset=['Margin%', 'Delta Margin%'])
    if df_.empty: return go.Figure()
    steps_ = df_["Step"].tolist()
    margin_ = df_["Margin%"].tolist()
    delta_margin_ = df_["Delta Margin%"].tolist()
    y_vals_for_plot = [margin_[0]] + delta_margin_[1:-1] + [margin_[-1]]
    measure_ = ["absolute"] + ["relative"] * (len(steps_) - 2) + ["total"]
    text_ = [f"{v:.1%}" for v in y_vals_for_plot]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measure_, x=steps_, y=y_vals_for_plot, text=text_, textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}}, increasing={"marker":{"color":"#1f77b4"}},
        decreasing={"marker":{"color":"#d62728"}}, totals={"marker":{"color":"#9467bd"}}
    ))
    fig.update_layout(title=title, yaxis_tickformat=".1%", waterfallgap=0.3)
    return fig

###############################################
# Margin Bridge Table Building Functions
###############################################

def build_step_margin_table(bridge_dol):
    """Build Step-Based Margin% Table"""
    rows = []
    if bridge_dol.empty or not all(c in bridge_dol.columns for c in ["Step", "Revenue", "Cost"]): return pd.DataFrame()
    bridge_dol['Revenue'] = pd.to_numeric(bridge_dol['Revenue'], errors='coerce').fillna(0.0)
    bridge_dol['Cost'] = pd.to_numeric(bridge_dol['Cost'], errors='coerce').fillna(0.0)
    for i in range(len(bridge_dol)):
        step_, rev_, cos_ = bridge_dol.loc[i, "Step"], bridge_dol.loc[i, "Revenue"], bridge_dol.loc[i, "Cost"]
        m_ = margin_ratio(rev_, cos_)
        rows.append((step_, rev_, cos_, m_))
    df_ = pd.DataFrame(rows, columns=["Step", "Revenue", "Cost", "Margin%"])
    return add_delta_margin_column(df_, margin_col="Margin%")

def build_proportional_margin_table(pvm_):
    """Build Proportional Margin% Table"""
    required_factors = ["Starting", "Ending"]; required_cols = ["Revenue_Sum", "Cost_Sum"]
    if pvm_.empty or not all(f in pvm_.index for f in required_factors) or not all(c in pvm_.columns for c in required_cols): return pd.DataFrame()
    def getcell(df, row, col, default=0.0): return df.loc[row, col] if row in df.index and col in df.columns else default
    sRev, sCos = getcell(pvm_, "Starting", "Revenue_Sum"), getcell(pvm_, "Starting", "Cost_Sum")
    eRev, eCos = getcell(pvm_, "Ending", "Revenue_Sum"), getcell(pvm_, "Ending", "Cost_Sum")
    m_start, m_end = margin_ratio(float(sRev), float(sCos)), margin_ratio(float(eRev), float(eCos))
    diff = m_end - m_start if pd.notna(m_start) and pd.notna(m_end) else 0.0
    factors = ["Price", "Volume", "Mix", "New_Items", "Discontinued_Items"]
    partials = []; valid_factors = [f for f in factors if f in pvm_.index]
    for f in valid_factors:
        pm = margin_ratio(float(sRev + getcell(pvm_, f, "Revenue_Sum")), float(sCos + getcell(pvm_, f, "Cost_Sum")))
        partials.append(pm - m_start if pd.notna(pm) and pd.notna(m_start) else 0.0)
    sum_par = sum(partials)
    scaled = [(p / sum_par) * diff for p in partials] if abs(sum_par) > 1e-12 else [0] * len(valid_factors)
    rows = [{"Step": "Starting", "Margin%": m_start, "Delta Margin%": 0.0}]
    current_margin = m_start
    for i, f in enumerate(valid_factors): delta = scaled[i]; current_margin += delta; rows.append({"Step": f, "Margin%": current_margin, "Delta Margin%": delta})
    final_delta = m_end - current_margin if pd.notna(m_end) and pd.notna(current_margin) else 0.0
    rows.append({"Step": "Ending", "Margin%": m_end, "Delta Margin%": final_delta})
    prop_df = pd.DataFrame(rows)
    if len(prop_df) > 1:
        last_margin, prev_margin = prop_df.loc[len(prop_df)-1, 'Margin%'], prop_df.loc[len(prop_df)-2, 'Margin%']
        prop_df.loc[len(prop_df)-1, 'Delta Margin%'] = last_margin - prev_margin if pd.notna(last_margin) and pd.notna(prev_margin) else 0.0
    return prop_df

def build_additive_margin_table(pvm_):
    """Build Additive Margin% Table"""
    required_factors = ["Starting", "Ending"]; required_cols = ["Revenue_Sum", "Cost_Sum"]
    if pvm_.empty or not all(f in pvm_.index for f in required_factors) or not all(c in pvm_.columns for c in required_cols): return pd.DataFrame()
    def getcell(df, row, col, default=0.0): return df.loc[row, col] if row in df.index and col in df.columns else default
    sRev, sCos = getcell(pvm_, "Starting", "Revenue_Sum"), getcell(pvm_, "Starting", "Cost_Sum")
    eRev, eCos = getcell(pvm_, "Ending", "Revenue_Sum"), getcell(pvm_, "Ending", "Cost_Sum")
    m_start, m_end = margin_ratio(float(sRev), float(sCos)), margin_ratio(float(eRev), float(eCos))
    diff = m_end - m_start if pd.notna(m_start) and pd.notna(m_end) else 0.0
    factors = ["Price", "Volume", "Mix", "New_Items", "Discontinued_Items"]
    valid_factors = [f for f in factors if f in pvm_.index]; partials = {}
    for f in valid_factors:
        pm = margin_ratio(float(sRev + getcell(pvm_, f, "Revenue_Sum")), float(sCos + getcell(pvm_, f, "Cost_Sum")))
        partials[f] = pm - m_start if pd.notna(pm) and pd.notna(m_start) else 0.0
    sumf = sum(partials.values()); leftover = diff - sumf
    rows = [{"Step": "Starting", "Margin%": m_start, "Delta Margin%": 0.0}]
    current_margin = m_start
    for f in valid_factors: delta = partials[f]; current_margin += delta; rows.append({"Step": f, "Margin%": current_margin, "Delta Margin%": delta})
    current_margin += leftover; rows.append({"Step": "Residual", "Margin%": current_margin, "Delta Margin%": leftover})
    final_delta = m_end - current_margin if pd.notna(m_end) and pd.notna(current_margin) else 0.0
    rows.append({"Step": "Ending", "Margin%": m_end, "Delta Margin%": final_delta})
    add_df = pd.DataFrame(rows)
    if len(add_df) > 1:
        last_margin, prev_margin = add_df.loc[len(add_df)-1, 'Margin%'], add_df.loc[len(add_df)-2, 'Margin%']
        add_df.loc[len(add_df)-1, 'Delta Margin%'] = last_margin - prev_margin if pd.notna(last_margin) and pd.notna(prev_margin) else 0.0
    return add_df

def build_constant_holding_table(pvm_):
    """Build Constant-Holding Margin% Table"""
    required_factors = ["Starting", "Ending"]; required_cols = ["Revenue_Sum", "Cost_Sum"]
    if pvm_.empty or not all(f in pvm_.index for f in required_factors) or not all(c in pvm_.columns for c in required_cols): return pd.DataFrame()
    def getcell(df, row, col, default=0.0): return df.loc[row, col] if row in df.index and col in df.columns else default
    sRev, sCos = getcell(pvm_, "Starting", "Revenue_Sum"), getcell(pvm_, "Starting", "Cost_Sum")
    eRev, eCos = getcell(pvm_, "Ending", "Revenue_Sum"), getcell(pvm_, "Ending", "Cost_Sum")
    m_start, m_end = margin_ratio(float(sRev), float(sCos)), margin_ratio(float(eRev), float(eCos))
    diff = m_end - m_start if pd.notna(m_start) and pd.notna(m_end) else 0.0
    factors = ["Price", "Volume", "Mix", "New_Items", "Discontinued_Items"]
    valid_factors = [f for f in factors if f in pvm_.index]; impacts = {}
    for f_exclude in valid_factors:
        sum_rev, sum_cos = sRev, sCos
        for f_include in valid_factors:
            if f_include == f_exclude: continue
            sum_rev += getcell(pvm_, f_include, "Revenue_Sum"); sum_cos += getcell(pvm_, f_include, "Cost_Sum")
        m_no_f = margin_ratio(float(sum_rev), float(sum_cos))
        impacts[f_exclude] = m_end - m_no_f if pd.notna(m_end) and pd.notna(m_no_f) else 0.0
    sum_imp = sum(impacts.values()); leftover = diff - sum_imp
    rows = [{"Step": "Starting", "Margin%": m_start, "Delta Margin%": 0.0}]; current_margin = m_start
    for f in valid_factors: delta = impacts[f]; current_margin += delta; rows.append({"Step": f, "Margin%": current_margin, "Delta Margin%": delta})
    current_margin += leftover; rows.append({"Step": "Leftover", "Margin%": current_margin, "Delta Margin%": leftover})
    final_delta = m_end - current_margin if pd.notna(m_end) and pd.notna(current_margin) else 0.0
    rows.append({"Step": "Ending", "Margin%": m_end, "Delta Margin%": final_delta})
    ch_df = pd.DataFrame(rows)
    if len(ch_df) > 1:
        last_margin, prev_margin = ch_df.loc[len(ch_df)-1, 'Margin%'], ch_df.loc[len(ch_df)-2, 'Margin%']
        ch_df.loc[len(ch_df)-1, 'Delta Margin%'] = last_margin - prev_margin if pd.notna(last_margin) and pd.notna(prev_margin) else 0.0
    return ch_df

###############################################
# --- Main App Logic Starts Here ---
###############################################

st.title("Enhanced PVM Analysis & Bridging")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# --- Read, Validate, Preview ---
try:
    try: 
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError: 
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')
except Exception as e: 
    st.error(f"Error reading CSV: {e}")
    st.stop()

st.session_state.original_columns = df.columns.tolist()
required_cols = {"SKU", "Year", "Month", "Scenario", "Account", "Amount", "Quantity"}
missing = required_cols - set(st.session_state.original_columns)
if missing: 
    st.error(f"Missing required columns: {missing}")
    st.stop()

cols_to_check = ['Amount', 'Quantity']
for col in cols_to_check:
    if col in df.columns:
        try: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e: 
            st.error(f"Error converting '{col}' to numeric: {e}")
            st.stop()
    else: 
        st.error(f"Required column '{col}' not found.")
        st.stop()

if df[cols_to_check].isnull().any().any():
    nan_rows = df[df[cols_to_check].isnull().any(axis=1)]
    st.warning(f"Warning: Found missing/non-numeric values in Amount/Quantity. Rows ({len(nan_rows)}):")
    st.dataframe(nan_rows.head())

with st.expander("Preview Uploaded Data"):
    st.dataframe(df.head())
    # ... (rest of preview) ...

# --- Sidebar Setup ---
st.sidebar.header("Analysis Configuration")

# Aggregation Level
hierarchy_cols_detected = ['SKU'] + sorted([c for c in st.session_state.original_columns if c.startswith("LPH") or c.startswith("GPH")])
required_cols_str = {str(c) for c in required_cols}
cols_to_exclude_from_agg = required_cols_str
hierarchy_cols_valid = [c for c in hierarchy_cols_detected if c in df.columns and c not in cols_to_exclude_from_agg and df[c].nunique(dropna=False) > 1 and df[c].nunique(dropna=False) < len(df) * 0.9]
if 'SKU' in df.columns and 'SKU' not in hierarchy_cols_valid: 
    hierarchy_cols_valid.insert(0, 'SKU')
if not hierarchy_cols_valid: 
    st.sidebar.error("No valid aggregation columns found.")
    st.stop()
if 'aggregation_level' not in st.session_state or st.session_state.aggregation_level not in hierarchy_cols_valid:
     st.session_state.aggregation_level = 'SKU' if 'SKU' in hierarchy_cols_valid else hierarchy_cols_valid[0]
try: 
    current_agg_index = hierarchy_cols_valid.index(st.session_state.aggregation_level)
except ValueError: 
    current_agg_index = 0
st.session_state.aggregation_level = st.sidebar.selectbox("Aggregation Level", hierarchy_cols_valid, index=current_agg_index)
AGGREGATION_LEVEL = st.session_state.aggregation_level
if not AGGREGATION_LEVEL: 
    st.sidebar.error("Aggregation level selection failed.")
    st.stop()
st.sidebar.info(f"Aggregating at: **{AGGREGATION_LEVEL}**")

# Year/Scenario Selection
if 'Year' not in df.columns or 'Scenario' not in df.columns: 
    st.sidebar.error("'Year' or 'Scenario' column missing.")
    st.stop()
df['Year'] = df['Year'].astype(str)
df['Scenario'] = df['Scenario'].astype(str)
scenarios_available = sorted(df["Scenario"].unique())
years_available = sorted(df["Year"].unique())
# ... (selectbox setup for year/scenario - simplified for brevity) ...
scenario_start = st.sidebar.selectbox("Starting Scenario", scenarios_available)
year_start = st.sidebar.selectbox("Starting Year", years_available)
scenario_end = st.sidebar.selectbox("Ending Scenario", scenarios_available, index=len(scenarios_available)-1 if scenarios_available else 0)
year_end = st.sidebar.selectbox("Ending Year", years_available, index=len(years_available)-1 if years_available else 0)
if not (scenario_start and year_start and scenario_end and year_end): 
    st.sidebar.error("Scenario/Year selection incomplete.")
    st.stop()
if (year_start, scenario_start) == (year_end, scenario_end): 
    st.sidebar.warning("Start and End periods are the same.")

# Month Filter
if 'Month' not in df.columns: 
    st.sidebar.error("'Month' column missing.")
    st.stop()
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df = df.dropna(subset=['Month']) # Drop rows where Month is invalid *before* filtering
if df.empty: 
    st.error("No valid numeric data in 'Month' column.")
    st.stop()
df['Month'] = df['Month'].astype(int)
month_options = list(range(1, 13))
month_type = st.sidebar.radio("Month Filter Type", ["Monthly", "Quarterly", "Year-to-Date"], horizontal=True)
# ... (month selection logic - simplified) ...
if month_type == "Monthly": 
    chosen_months = st.sidebar.multiselect("Months", month_options, default=month_options)
elif month_type == "Quarterly": 
    quarters = {"Q1": [1,2,3],"Q2": [4,5,6],"Q3": [7,8,9],"Q4": [10,11,12]}
    chosen_q = st.sidebar.selectbox("Quarter", list(quarters.keys()))
    chosen_months = quarters[chosen_q]
else: 
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    chosen_ytd = st.sidebar.selectbox("YTD Month", month_names, index=11)
    chosen_months = list(range(1, month_names.index(chosen_ytd) + 2))
df_filtered = df[df["Month"].isin(chosen_months)].copy()
if df_filtered.empty: 
    st.error(f"No data found for selected months: {chosen_months}.")
    st.stop()

# Hierarchy Filters
standard_cols_for_filter = {"Year", "Month", "Scenario", "Account", "Amount", "Quantity", AGGREGATION_LEVEL}
potential_hier_cols = set(st.session_state.original_columns) - standard_cols_for_filter
hier_cols_to_filter = sorted([c for c in potential_hier_cols if c in df_filtered.columns and df_filtered[c].nunique(dropna=False) > 1])

# --- ADDED: Defensive check for session state initialization ---
if 'filter_selections' not in st.session_state:
    st.session_state.filter_selections = {}
# --- END Defensive check ---

for hcol in hier_cols_to_filter:
    # ... (hierarchy filter logic - simplified) ...
    df_filtered.loc[:, hcol] = df_filtered[hcol].astype(str) # Use .loc
    unique_vals = sorted(df_filtered[hcol].dropna().unique())
    if not unique_vals: continue
    widget_key_hier = f"filter_{hcol}"
    # Initialize session state for this specific filter key if needed
    # This check should now always pass the first part because of the check above
    if widget_key_hier not in st.session_state.filter_selections:
        st.session_state.filter_selections[widget_key_hier] = unique_vals
    st.sidebar.write(f"Filter {hcol}")
    select_all_buttons(widget_key_hier, unique_vals)
    default_selection = st.session_state.filter_selections[widget_key_hier]
    if not isinstance(default_selection, list): default_selection = unique_vals
    valid_default = [v for v in default_selection if v in unique_vals]
    chosen_vals = st.sidebar.multiselect(f"Select {hcol} values", unique_vals, default=valid_default, key=widget_key_hier)
    st.session_state.filter_selections[widget_key_hier] = chosen_vals
    if chosen_vals and set(chosen_vals) != set(unique_vals): 
        df_filtered = df_filtered[df_filtered[hcol].isin(chosen_vals)]
if df_filtered.empty: 
    st.error("No data remaining after hierarchy filters.")
    st.stop()

# Account Selection
if 'Account' not in df_filtered.columns: 
    st.sidebar.error("'Account' column missing.")
    st.stop()
df_filtered['Account'] = df_filtered['Account'].astype(str)
all_accounts = sorted(df_filtered["Account"].unique())
# ... (account selection logic - simplified) ...
revenue_accounts = st.sidebar.multiselect("Revenue Accounts", all_accounts, default=st.session_state.revenue_accounts, key="ms_revenue")
available_cost_accounts = [a for a in all_accounts if a not in revenue_accounts]
cost_accounts = st.sidebar.multiselect("Cost Accounts", available_cost_accounts, default=[c for c in st.session_state.cost_accounts if c in available_cost_accounts], key="ms_cost")
st.session_state.revenue_accounts = revenue_accounts
st.session_state.cost_accounts = cost_accounts
if not revenue_accounts and not cost_accounts: 
    st.sidebar.error("Select at least one Revenue or Cost account.")
    st.stop()

# Margin Bridge Approach
bridge_mode = st.sidebar.radio("Margin% Approach", ["Step-Based", "Proportional (No Residual)", "Additive", "Constant-Holding"])

# --- Main Calculations & Display ---
st.header(f"Aggregated Analysis at '{AGGREGATION_LEVEL}' Level")
with st.spinner(f"Calculating Aggregated PVM..."):
    pvm_df = compute_pvm_all(df_filtered, year_start, scenario_start, year_end, scenario_end, revenue_accounts + cost_accounts, AGGREGATION_LEVEL)

# Summary Stats
st.subheader("Summary Statistics")
start_rev, start_cost, end_rev, end_cost = 0,0,0,0 # Initialize
if not pvm_df.empty and "Starting" in pvm_df.index and "Ending" in pvm_df.index:
    try:
        start_rev = sum(pvm_df.loc["Starting"].get(str(acc), 0) for acc in revenue_accounts)
        start_cost = sum(pvm_df.loc["Starting"].get(str(acc), 0) for acc in cost_accounts)
        end_rev = sum(pvm_df.loc["Ending"].get(str(acc), 0) for acc in revenue_accounts)
        end_cost = sum(pvm_df.loc["Ending"].get(str(acc), 0) for acc in cost_accounts)
        start_margin_pct = margin_ratio(float(start_rev), float(start_cost))
        end_margin_pct = margin_ratio(float(end_rev), float(end_cost))
        net_dollar_change = (end_rev - end_cost) - (start_rev - start_cost)
        margin_point_change = (end_margin_pct - start_margin_pct) if pd.notna(start_margin_pct) and pd.notna(end_margin_pct) else 0.0
        items_analyzed = df_filtered[AGGREGATION_LEVEL].nunique()
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        # ... (metric display) ...
        stat_col1.metric("Starting Net Profit $", f"${(start_rev - start_cost):,.0f}")
        stat_col1.metric("Starting Margin %", f"{start_margin_pct:.1%}" if pd.notna(start_margin_pct) else "N/A")
        stat_col2.metric("Ending Net Profit $", f"${(end_rev - end_cost):,.0f}")
        stat_col2.metric("Ending Margin %", f"{end_margin_pct:.1%}" if pd.notna(end_margin_pct) else "N/A")
        stat_col3.metric("Net Profit Change $", f"${net_dollar_change:,.0f}")
        stat_col3.metric("Margin Change (Points)", f"{margin_point_change:.1%}")
        st.info(f"Analysis based on **{items_analyzed:,}** unique '{AGGREGATION_LEVEL}' items after filtering.")
    except Exception as stat_err: 
        st.error(f"Error calculating summary stats: {stat_err}")
        st.stop()
else: 
    st.error("Aggregated PVM failed, cannot calculate stats.")
    st.stop()

# Aggregated PVM Table
st.subheader("Aggregated PVM Table (Dollars)")
required_indices = ["Starting", "Price", "Volume", "Mix", "New_Items", "Discontinued_Items", "Ending"]
if all(idx in pvm_df.index for idx in required_indices): 
    st.dataframe(pvm_df.style.format("{:,.0f}"), use_container_width=True)
else: 
    st.warning("Agg PVM table missing factors.")
    st.dataframe(pvm_df, use_container_width=True)
pvm_df["Revenue_Sum"] = pvm_df[[str(a) for a in revenue_accounts if str(a) in pvm_df.columns]].sum(axis=1) if revenue_accounts else 0.0
pvm_df["Cost_Sum"] = pvm_df[[str(a) for a in cost_accounts if str(a) in pvm_df.columns]].sum(axis=1) if cost_accounts else 0.0
if "Revenue_Sum" in pvm_df.columns and "Cost_Sum" in pvm_df.columns: 
    pvm_df["Net_Change"] = pvm_df["Revenue_Sum"] - pvm_df["Cost_Sum"]
else: 
    pvm_df["Net_Change"] = 0.0

# Dollar Bridge
st.subheader("Dollar Bridge Decomposition")
bridge_steps_factors = [f for f in required_indices if f in pvm_df.index]
bridge_dollars = pd.DataFrame({"Step": bridge_steps_factors})
bridge_dollars = pd.merge(bridge_dollars, pvm_df[["Revenue_Sum", "Cost_Sum", "Net_Change"]].reset_index().rename(columns={'index':'Step'}), on="Step", how="left").fillna(0.0)
bridge_dollars = bridge_dollars.rename(columns={"Revenue_Sum": "DeltaRevenue", "Cost_Sum": "DeltaCost", "Net_Change": "DeltaNet"})
if "Starting" in bridge_dollars['Step'].tolist():
    start_row = bridge_dollars[bridge_dollars['Step'] == 'Starting']
    start_rev_bridge, start_cost_bridge, start_net_bridge = start_row['DeltaRevenue'].iloc[0], start_row['DeltaCost'].iloc[0], start_row['DeltaNet'].iloc[0]
    bridge_dollars['Revenue'] = bridge_dollars['DeltaRevenue'].cumsum()
    bridge_dollars['Cost'] = bridge_dollars['DeltaCost'].cumsum()
    bridge_dollars['Net'] = bridge_dollars['DeltaNet'].cumsum()
    bridge_dollars.loc[bridge_dollars['Step'] == 'Starting', ['Revenue', 'Cost', 'Net']] = [start_rev_bridge, start_cost_bridge, start_net_bridge]
else: 
    bridge_dollars['Revenue'] = bridge_dollars['DeltaRevenue'].cumsum()
    bridge_dollars['Cost'] = bridge_dollars['DeltaCost'].cumsum()
    bridge_dollars['Net'] = bridge_dollars['DeltaNet'].cumsum()
if "Ending" in bridge_dollars['Step'].tolist():
     intermediate_factors = [s for s in bridge_steps_factors if s not in ["Starting", "Ending"]]
     start_rev_val = bridge_dollars.loc[bridge_dollars['Step'] == 'Starting', 'Revenue'].iloc[0] if "Starting" in bridge_dollars['Step'].tolist() else 0
     start_cost_val = bridge_dollars.loc[bridge_dollars['Step'] == 'Starting', 'Cost'].iloc[0] if "Starting" in bridge_dollars['Step'].tolist() else 0
     end_rev_calc = start_rev_val + bridge_dollars[bridge_dollars['Step'].isin(intermediate_factors)]["DeltaRevenue"].sum()
     end_cost_calc = start_cost_val + bridge_dollars[bridge_dollars['Step'].isin(intermediate_factors)]["DeltaCost"].sum()
     end_net_calc = end_rev_calc - end_cost_calc
     bridge_dollars.loc[bridge_dollars['Step'] == 'Ending', ['Revenue', 'Cost', 'Net']] = [end_rev_calc, end_cost_calc, end_net_calc]
     if not np.allclose([end_rev_calc, end_cost_calc], [end_rev, end_cost]): 
         st.warning("Calculated ending totals from PVM steps do not match initial ending totals.")
if 'Revenue' in bridge_dollars.columns: 
    bridge_dollars['DeltaRevenue_Plot'] = bridge_dollars['Revenue'].diff().fillna(bridge_dollars['Revenue'].iloc[0] if not bridge_dollars.empty else 0)
else: 
    bridge_dollars['DeltaRevenue_Plot'] = 0.0
if 'Cost' in bridge_dollars.columns: 
    bridge_dollars['DeltaCost_Plot'] = bridge_dollars['Cost'].diff().fillna(bridge_dollars['Cost'].iloc[0] if not bridge_dollars.empty else 0)
else: 
    bridge_dollars['DeltaCost_Plot'] = 0.0
if 'Net' in bridge_dollars.columns: 
    bridge_dollars['DeltaNet_Plot'] = bridge_dollars['Net'].diff().fillna(bridge_dollars['Net'].iloc[0] if not bridge_dollars.empty else 0)
else: 
    bridge_dollars['DeltaNet_Plot'] = 0.0

with st.expander("View Dollar Bridge Step Table"):
    numeric_cols_bd = [ 'DeltaRevenue', 'DeltaCost', 'DeltaNet', 'Revenue', 'Cost', 'Net',
                      'DeltaRevenue_Plot', 'DeltaCost_Plot', 'DeltaNet_Plot' ]
    format_dict_bd = { col: "{:,.0f}" for col in numeric_cols_bd if col in bridge_dollars.columns }
    st.dataframe(bridge_dollars.style.format(format_dict_bd, na_rep='-'), use_container_width=True)

wf_col1, wf_col2, wf_col3 = st.columns(3)
with wf_col1: 
    st.plotly_chart(create_waterfall(bridge_dollars, 'Revenue', 'DeltaRevenue_Plot', "Revenue Bridge ($)"), use_container_width=True)
with wf_col2: 
    st.plotly_chart(create_waterfall(bridge_dollars, 'Cost', 'DeltaCost_Plot', "Cost Bridge ($)"), use_container_width=True)
with wf_col3: 
    st.plotly_chart(create_waterfall(bridge_dollars, 'Net', 'DeltaNet_Plot', "Net Profit Bridge ($)"), use_container_width=True)

# Margin Bridge
st.subheader("Aggregator Margin% Bridging")
margin_df = pd.DataFrame(); fig_margin = go.Figure()
try:
    if bridge_mode == "Step-Based": 
        margin_df = build_step_margin_table(bridge_dollars)
        fig_margin = margin_waterfall(margin_df, "Step-Based Margin% Waterfall")
    elif bridge_mode == "Proportional (No Residual)":
        margin_df = build_proportional_margin_table(pvm_df)
        fig_margin = margin_waterfall(margin_df, "Proportional (No Residual) Margin% Bridge")
    elif bridge_mode == "Additive":
        margin_df = build_additive_margin_table(pvm_df)
        fig_margin = margin_waterfall(margin_df, "Additive Margin% Bridge")
    else: 
        margin_df = build_constant_holding_table(pvm_df)
        fig_margin = margin_waterfall(margin_df, "Constant-Holding Margin% Bridge")
    if not margin_df.empty:
        with st.expander(f"View {bridge_mode} Margin% Bridge Table"):
            format_dict_margin = {col: "{:.2%}" for col in ["Margin%", "Delta Margin%"] if col in margin_df.columns}
            st.dataframe(margin_df.style.format(format_dict_margin, na_rep='-'), use_container_width=True)
    st.plotly_chart(fig_margin, use_container_width=True)
except Exception as e:
    st.error(f"An error occurred during the '{bridge_mode}' margin bridge calculation or display: {e}")
    if 'margin_df' in locals() and not margin_df.empty:
        st.warning("Displaying partially calculated margin bridge table due to error:")
        st.dataframe(margin_df, use_container_width=True)

# Item-Level Analysis
st.header(f"Item-Level Analysis ({AGGREGATION_LEVEL})")
item_hier_cols = [c for c in st.session_state.original_columns if c not in required_cols and c != AGGREGATION_LEVEL]
items_analyzed = df_filtered[AGGREGATION_LEVEL].nunique() if AGGREGATION_LEVEL in df_filtered else 0
with st.spinner(f"Calculating Item-Level PVM..."):
     item_pvm_df = compute_item_pvm(df_filtered, year_start, scenario_start, year_end, scenario_end, revenue_accounts + cost_accounts, item_hier_cols, AGGREGATION_LEVEL)
st.subheader(f"{AGGREGATION_LEVEL}-Level PVM Table (Dollars)")
if not item_pvm_df.empty: 
    st.dataframe(item_pvm_df.head(1000).style.format(precision=0, na_rep='-'), use_container_width=True)
    st.caption(f"Showing first 1000 rows.")
else: 
    st.warning("Item-level PVM calculation returned no results.")

# Item Factor Sums
item_factor_df = pd.DataFrame()
if not item_pvm_df.empty:
    with st.spinner("Summarizing item factor impacts..."): 
        item_factor_df = build_item_factor_sums(item_pvm_df, revenue_accounts, cost_accounts, AGGREGATION_LEVEL)
else: 
    item_factor_df = pd.DataFrame()

# Aggregator Margin Deltas
agg_margin_delta_dict = aggregator_step_margin_deltas(pvm_df) if not pvm_df.empty else {}

# Item Step Table
st.subheader(f"{AGGREGATION_LEVEL}-Level Bridging Table (Dollars & Margin% Allocation)")
with st.expander("Understanding the Item-Level Bridging Table"): 
    st.info(f"""...""") # Explanation omitted
item_step_table = pd.DataFrame()
if not item_factor_df.empty and agg_margin_delta_dict and not pvm_df.empty:
    with st.spinner(f"Building final {AGGREGATION_LEVEL}-level step table..."): 
         item_step_table = build_item_step_table(item_factor_summary=item_factor_df, agg_margin_deltas=agg_margin_delta_dict, pvm_agg_df=pvm_df, agg_level=AGGREGATION_LEVEL) # Explicit keywords
else: 
    st.warning("Cannot build item step table due to missing prerequisite data.")
if not item_step_table.empty:
    format_dict_item_step = { col: fmt for col, fmt in {"Revenue_Bridge": "{:,.0f}", "Cost_Bridge": "{:,.0f}", "Net_Bridge": "{:,.0f}", "Item_Accum_Revenue": "{:,.0f}", "Item_Accum_Cost": "{:,.0f}", "Item_Accum_MarginPct": "{:.2%}", "Item_Margin_Delta": "{:.2%}", "Aggregator_MarginStep": "{:.2%}", "Item_MarginStep_Impact_Alloc": "{:.2%}"}.items() if col in item_step_table.columns }
    display_cols_step = [AGGREGATION_LEVEL, 'Factor'] + [c for c in format_dict_item_step if c not in [AGGREGATION_LEVEL, 'Factor']]
    display_cols_step = [c for c in display_cols_step if c in item_step_table.columns]
    st.dataframe(item_step_table[display_cols_step].head(2000).style.format(format_dict_item_step, na_rep='-'), use_container_width=True)
    st.caption(f"Showing first 2000 rows.")
else: 
    st.warning("Item step table is empty or could not be generated.")

# Top/Bottom N Viz
st.subheader(f"Top/Bottom {AGGREGATION_LEVEL} Contributors by Net Factor Impact")
if not item_factor_df.empty and 'Net_Impact' in item_factor_df.columns and 'Factor' in item_factor_df.columns:
    # Define the factors you want to visualize. For example, "Price", "Volume", and "Mix".
    factors_for_viz = ["Price", "Volume", "Mix", "New_Items", "Discontinued_Items"]
    valid_factors_viz = [f for f in factors_for_viz if f in item_factor_df['Factor'].unique()]
    if valid_factors_viz:
        # Let the user select which factor to visualize.
        selected_factor = st.selectbox("Select Factor", valid_factors_viz, key="viz_factor_sel")
        top_n = st.number_input("Number of Top Contributors", min_value=1, value=10, step=1, key="top_n")
        bottom_n = st.number_input("Number of Bottom Contributors", min_value=1, value=10, step=1, key="bottom_n")
        
        # Filter the summary for the selected factor.
        df_selected = item_factor_df[item_factor_df['Factor'] == selected_factor]
        
        # Sort descending for top contributors and ascending for bottom contributors.
        df_top = df_selected.sort_values(by="Net_Impact", ascending=False).head(top_n)
        df_bottom = df_selected.sort_values(by="Net_Impact", ascending=True).head(bottom_n)
        
        st.write(f"### Top {top_n} Contributors for {selected_factor}")
        st.dataframe(df_top)
        st.write(f"### Bottom {bottom_n} Contributors for {selected_factor}")
        st.dataframe(df_bottom)
    else:
        st.warning("No valid factors found for visualization.")
else:
    st.warning("Item factor summary data unavailable for visualization.")


# Downloads
st.header("Downloads")
dfs_to_export = {
    f"Aggregated_PVM_{AGGREGATION_LEVEL}": pvm_df.reset_index() if isinstance(pvm_df, pd.DataFrame) else pd.DataFrame(),
    "Dollar_Bridge_Steps": bridge_dollars if isinstance(bridge_dollars, pd.DataFrame) else pd.DataFrame(),
    f"{bridge_mode}_Margin_Bridge": margin_df if isinstance(margin_df, pd.DataFrame) else pd.DataFrame(),
    f"{AGGREGATION_LEVEL}_PVM_Detail": item_pvm_df if isinstance(item_pvm_df, pd.DataFrame) else pd.DataFrame(),
    f"{AGGREGATION_LEVEL}_Factor_Summary": item_factor_df if isinstance(item_factor_df, pd.DataFrame) else pd.DataFrame(),
    f"{AGGREGATION_LEVEL}_Step_Allocation": item_step_table if isinstance(item_step_table, pd.DataFrame) else pd.DataFrame()
}
dfs_non_empty = {name: df for name, df in dfs_to_export.items() if isinstance(df, pd.DataFrame) and not df.empty}
if dfs_non_empty:
    with st.spinner("Generating Excel file..."):
        excel_buffer = create_excel_export(dfs_non_empty)
    if excel_buffer:
        st.download_button(label=f"📥 Download All Tables as Excel", data=excel_buffer, file_name=f"pvm_analysis_{AGGREGATION_LEVEL}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else: 
        st.warning("Could not generate Excel file.")
else: 
    st.warning("No data tables generated for download.")

st.success("Analysis Complete.")
