# app.py — Logistics Network Optimization (Cost / Service / Cost+Service) + Optional Maps
# requirements.txt:
# gradio>=4.37
# pandas
# openpyxl
# pyomo>=6.7
# highspy>=1.7
# numpy
# plotly
# pgeocode

import io
import os
import tempfile

import pandas as pd
import gradio as gr
import plotly.express as px
import pgeocode

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals,
    Objective, Constraint, value
)

# =========================
# Global business rules
# =========================
OUTBOUND_MULT = 2.4  # outbound multiplier used in OBJECTIVE (must match reporting)


# =========================
# Helpers
# =========================
def to_float(x):
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return float("nan")
        neg = s.startswith("(") and s.endswith(")")
        s = s.replace("$", "").replace("€", "").replace("£", "").replace(",", "")
        if neg:
            s = s[1:-1].strip()
        v = float(s)
        return -v if neg else v
    except Exception:
        return float("nan")


def norm_key(x):
    s = str(x).strip().lower().replace("&", "and").replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def export_fullscreen(fig, prefix="view"):
    path = tempfile.mktemp(prefix=f"{prefix}_", suffix=".html")
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def _find_prefix_cols(df_cols, prefixes):
    out = []
    for c in df_cols:
        cs = str(c)
        for p in prefixes:
            if cs.lower().startswith(p.lower()):
                out.append(cs)
                break
    return out


# =========================
# Loaders / Parsers
# =========================
def load_rents_csv(rent_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(rent_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"Annual Rent per m²": "Annual Rent per m2"})
    for col in ["Warehouse", "Annual Rent per m2"]:
        if col not in df.columns:
            raise KeyError(f"Rent CSV missing required column: {col}")
    df["Annual Rent per m2"] = df["Annual Rent per m2"].map(to_float).fillna(0.0)
    return df


def load_inbound_costs_split(
    inb_bytes: bytes,
    dest_col="Destination",
    sea_col="Inbound Sea Freight Cost",
    dray_col="Inbound Drayage Cost",
    sheet_name=0,
):
    df = pd.read_excel(io.BytesIO(inb_bytes), sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    for col in [dest_col, sea_col, dray_col]:
        if col not in df.columns:
            raise KeyError(f"Inbound file missing required column: {col}")
    df = df.dropna(subset=[dest_col]).copy()
    df[sea_col] = df[sea_col].map(to_float).fillna(0.0)
    df[dray_col] = df[dray_col].map(to_float).fillna(0.0)
    df["__key__"] = df[dest_col].astype(str).map(norm_key)
    sea_map = df.set_index("__key__")[sea_col].to_dict()
    dray_map = df.set_index("__key__")[dray_col].to_dict()
    return sea_map, dray_map


def parse_outbound(
    out_bytes: bytes,
    sheet_name="Sheet1",
    zip_col="Destination Zip",
    sqm_col="Average Sqm Requirement",
    freight_prefix="Freight Charge ",
    service_prefixes=(
        "Days to Serve ",
        "Days to serve ",
        "Service Days ",
        "Transit Days ",
        "Transit Time ",
        "DaysToServe ",
        "Days to deliver ",
    ),
):
    """
    Returns:
      df, customers, hubs, sqm_req_series, freight_df, service_days_df_or_None
    """
    df = pd.read_excel(io.BytesIO(out_bytes), sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    # Customer key used inside model
    if zip_col in df.columns:
        df["__CustomerID__"] = df[zip_col].astype(str).str.strip() + "_" + df.reset_index().index.astype(str)
    else:
        df["__CustomerID__"] = df.reset_index().index.astype(str)

    if sqm_col not in df.columns:
        raise KeyError(f"Outbound file missing required column: {sqm_col}")
    df[sqm_col] = df[sqm_col].map(to_float).fillna(0.0)

    # hubs from freight columns
    hubs = sorted({
        str(c).replace(freight_prefix, "").strip()
        for c in df.columns if str(c).startswith(freight_prefix)
    })
    if not hubs:
        raise ValueError(f"No '{freight_prefix}<Hub>' columns found.")

    # Freight matrix
    freight_data = {}
    for w in hubs:
        col = f"{freight_prefix}{w}"
        s = df[["__CustomerID__", col]].copy()
        s[col] = s[col].map(to_float).fillna(1e12)  # BIG M
        freight_data[w] = s.set_index("__CustomerID__")[col]
    freight = pd.DataFrame(freight_data)

    # Service days matrix (optional)
    service_cols = _find_prefix_cols(df.columns, service_prefixes)
    svc_by_wh = {}
    if service_cols:
        for sc in service_cols:
            matched_prefix = None
            for p in service_prefixes:
                if sc.lower().startswith(p.lower()):
                    matched_prefix = sc[:len(p)]
                    break
            if matched_prefix is None:
                continue
            wh_name = sc[len(matched_prefix):].strip()
            if wh_name in hubs:
                s = df[["__CustomerID__", sc]].copy()
                s[sc] = s[sc].map(to_float).fillna(1e6)  # BIG M days
                svc_by_wh[wh_name] = s.set_index("__CustomerID__")[sc]
    service_days = pd.DataFrame(svc_by_wh) if svc_by_wh else None

    customers = df["__CustomerID__"].tolist()
    sqm_req = df.set_index("__CustomerID__")[sqm_col]
    return df, customers, hubs, sqm_req, freight, service_days


def load_container_vol(out_bytes: bytes, sheet_name="Sheet1", vol_col="Container Vol."):
    """
    Returns dict: customer_id -> container volume
    If missing, returns empty dict and we treat volume as 0 in reporting and 1 in weights.
    """
    df = pd.read_excel(io.BytesIO(out_bytes), sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    if "__CustomerID__" not in df.columns:
        if "Destination Zip" in df.columns:
            df["__CustomerID__"] = df["Destination Zip"].astype(str).str.strip() + "_" + df.reset_index().index.astype(str)
        else:
            df["__CustomerID__"] = df.index.astype(str)

    if vol_col not in df.columns:
        return {}

    df[vol_col] = df[vol_col].map(to_float).fillna(0.0)
    return df.set_index("__CustomerID__")[vol_col].to_dict()


def peek_hubs_from_outbound(outbound_bytes: bytes, sheet_name="Sheet1", freight_prefix="Freight Charge "):
    df = pd.read_excel(io.BytesIO(outbound_bytes), sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    hubs = sorted({
        str(c).replace(freight_prefix, "").strip()
        for c in df.columns if str(c).startswith(freight_prefix)
    })
    return hubs


# =========================
# Optimization Model
# =========================
def build_model(
    warehouses,
    customers,
    rent_per_m2,
    sqm_req,
    freight,
    sea_map,
    dray_map,
    cont_vol,
    K,
    min_wh_sqm=1500.0,
    service_days=None,
    objective_mode="Optimize Cost",
    service_penalty_per_day=0.0,
):
    """
    Min warehouse size constraint:
      sum_c sqm[c] * assign[w,c] >= min_wh_sqm * open[w]
    """
    m = ConcreteModel()
    m.W = Set(initialize=warehouses, ordered=True)
    m.C = Set(initialize=customers, ordered=True)

    m.rent = Param(m.W, initialize=rent_per_m2, within=NonNegativeReals)
    m.sqm = Param(m.C, initialize=sqm_req, within=NonNegativeReals)

    # Freight (raw) param
    fdict = {(w, c): float(freight.loc[c, w]) for c in customers for w in warehouses}
    m.freight = Param(m.W, m.C, initialize=fdict, within=NonNegativeReals)

    # inbound unit costs
    def lookup(mp, w): return float(mp.get(norm_key(w), 0.0))
    m.sea_unit = Param(m.W, initialize={w: lookup(sea_map, w) for w in warehouses}, within=NonNegativeReals)
    m.dray_unit = Param(m.W, initialize={w: lookup(dray_map, w) for w in warehouses}, within=NonNegativeReals)

    # Weight: container volume if >0 else 1 (used for inbound cost and weighted service days)
    cont_default = {}
    for c in customers:
        v = float(cont_vol.get(c, 0.0))
        cont_default[c] = v if v > 0 else 1.0
    m.weight = Param(m.C, initialize=cont_default, within=NonNegativeReals)

    # NEW: actual container volume (0 if missing) for reporting inbound containers
    m.cont_vol = Param(
        m.C,
        initialize={c: float(cont_vol.get(c, 0.0)) for c in customers},
        within=NonNegativeReals
    )

    # inbound costs per assignment
    def sea_cost(m, w, c): return m.sea_unit[w] * m.weight[c]
    def dray_cost(m, w, c): return m.dray_unit[w] * m.weight[c]
    m.sea_cost = Param(m.W, m.C, initialize=sea_cost, within=NonNegativeReals, mutable=False)
    m.dray_cost = Param(m.W, m.C, initialize=dray_cost, within=NonNegativeReals, mutable=False)

    # optional service days
    if service_days is not None:
        sd = service_days.copy()
        for w in warehouses:
            if w not in sd.columns:
                sd[w] = 1e6
        sd = sd.reindex(customers)
        sdict = {(w, c): float(sd.loc[c, w]) for c in customers for w in warehouses}
        m.days = Param(m.W, m.C, initialize=sdict, within=NonNegativeReals)
    else:
        m.days = None

    m.open = Var(m.W, domain=Binary)
    m.assign = Var(m.W, m.C, domain=Binary)

    m.one_wh = Constraint(m.C, rule=lambda m, c: sum(m.assign[w, c] for w in m.W) == 1)
    m.req_open = Constraint(m.W, m.C, rule=lambda m, w, c: m.assign[w, c] <= m.open[w])
    m.k_wh = Constraint(rule=lambda m: sum(m.open[w] for w in m.W) == K)

    # Minimum warehouse sqm constraint
    min_wh_sqm = float(min_wh_sqm)

    def min_size_rule(m, w):
        return sum(m.sqm[c] * m.assign[w, c] for c in m.C) >= min_wh_sqm * m.open[w]

    m.min_wh_size = Constraint(m.W, rule=min_size_rule)

    def rent_expr(m):
        return sum(m.rent[w] * sum(m.sqm[c] * m.assign[w, c] for c in m.C) for w in m.W)

    # IMPORTANT: apply outbound multiplier in objective
    def outbound_expr(m):
        return sum(OUTBOUND_MULT * m.freight[w, c] * m.assign[w, c] for w in m.W for c in m.C)

    def inbound_expr(m):
        return sum((m.sea_cost[w, c] + m.dray_cost[w, c]) * m.assign[w, c] for w in m.W for c in m.C)

    def total_cost_expr(m):
        return rent_expr(m) + outbound_expr(m) + inbound_expr(m)

    def weighted_service_days_expr(m):
        return sum(m.days[w, c] * m.weight[c] * m.assign[w, c] for w in m.W for c in m.C)

    mode = str(objective_mode).strip()

    if mode == "Optimize Service Days":
        if m.days is None:
            raise ValueError("Service Days optimization selected, but no service day columns were found in the outbound file.")
        m.obj = Objective(rule=lambda m: weighted_service_days_expr(m))
    elif mode == "Optimize Cost and Service Days":
        if m.days is None:
            raise ValueError("Cost+Service optimization selected, but no service day columns were found in the outbound file.")
        pen = float(service_penalty_per_day)
        m.obj = Objective(rule=lambda m: total_cost_expr(m) + pen * weighted_service_days_expr(m))
    else:
        m.obj = Objective(rule=lambda m: total_cost_expr(m))

    m._total_cost_expr = total_cost_expr
    m._weighted_service_days_expr = weighted_service_days_expr if m.days is not None else None
    return m


def get_solver():
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs
        return Highs(), "highs"
    except Exception:
        pass
    from pyomo.opt import SolverFactory
    for name in ("cbc", "glpk"):
        try:
            s = SolverFactory(name)
            if s is not None and s.available(False):
                return s, name
        except Exception:
            pass
    raise RuntimeError("No solver found. Install 'highspy' (HiGHS) recommended.")


def solve(rent_bytes, out_bytes, inb_bytes, allowed_hubs, K, objective_mode, service_penalty_per_day, min_wh_sqm):
    rents = load_rents_csv(rent_bytes)
    _, customers, hubs_all, sqm_req, freight, service_days = parse_outbound(out_bytes)
    sea_map, dray_map = load_inbound_costs_split(inb_bytes)
    cont = load_container_vol(out_bytes)

    hubs = [h for h in hubs_all if (not allowed_hubs) or (h in allowed_hubs)]
    if not hubs:
        raise ValueError("No warehouses left after filtering.")
    if K < 1 or K > len(hubs):
        raise ValueError(f"K must be between 1 and {len(hubs)}")

    # rent map
    rent_per_m2 = {}
    for w in hubs:
        row = rents.loc[rents["Warehouse"].astype(str).str.strip() == str(w)]
        if row.empty:
            raise KeyError(f"Warehouse '{w}' not found in rent CSV.")
        rent_per_m2[w] = float(to_float(row["Annual Rent per m2"].iloc[0]))

    sqm_map = {c: float(sqm_req.loc[c]) for c in customers}

    m = build_model(
        hubs, customers, rent_per_m2, sqm_map, freight,
        sea_map, dray_map, cont, int(K),
        min_wh_sqm=float(min_wh_sqm),
        service_days=service_days,
        objective_mode=objective_mode,
        service_penalty_per_day=service_penalty_per_day
    )

    opt, solver_name = get_solver()
    _ = opt.solve(m)

    rows = []
    for c in m.C:
        for w in m.W:
            if float(m.assign[w, c].value) >= 0.5:
                svc = float(m.days[w, c]) if m.days is not None else None
                rows.append([
                    c, w, 1,
                    float(m.sqm[c]),
                    float(m.weight[c]),
                    float(m.cont_vol[c]),  # NEW: actual container volume
                    float(OUTBOUND_MULT * m.freight[w, c]),
                    float(m.sea_cost[w, c]),
                    float(m.dray_cost[w, c]),
                    svc
                ])

    assign_df = pd.DataFrame(rows, columns=[
        "Customer", "Warehouse", "Assigned", "Sqm", "Weight",
        "Container_Vol",
        "Outbound_Cost_x2_4", "Inbound_Sea_Cost", "Inbound_Drayage_Cost", "Service_Days"
    ])

    total_out = float(assign_df["Outbound_Cost_x2_4"].sum())
    total_sea = float(assign_df["Inbound_Sea_Cost"].sum())
    total_dray = float(assign_df["Inbound_Drayage_Cost"].sum())

    wh_rows = []
    for w in hubs:
        dfw = assign_df[assign_df["Warehouse"] == w]
        sqm_sum = float(dfw["Sqm"].sum())
        rent_w = float(rent_per_m2[w] * sqm_sum)

        out_sum = float(dfw["Outbound_Cost_x2_4"].sum())
        sea_sum = float(dfw["Inbound_Sea_Cost"].sum())
        dray_sum = float(dfw["Inbound_Drayage_Cost"].sum())
        inb_sum = sea_sum + dray_sum
        total_cost = rent_w + out_sum + inb_sum

        inbound_containers = float(dfw["Container_Vol"].sum())  # NEW: inbound containers

        if dfw["Service_Days"].notna().any():
            wsd = float((dfw["Service_Days"] * dfw["Weight"]).sum())
            avg_days_w = float(wsd / max(dfw["Weight"].sum(), 1.0))
        else:
            wsd = None
            avg_days_w = None

        wh_rows.append([
            w,
            float(dfw["Weight"].sum()),
            inbound_containers,               # NEW
            sqm_sum,
            rent_w,
            out_sum,
            sea_sum,
            dray_sum,
            inb_sum,
            total_cost,
            wsd,
            avg_days_w
        ])

    wh = pd.DataFrame(wh_rows, columns=[
        "Warehouse",
        "Total_Weight",
        "Inbound_Containers",               # NEW
        "Total_Sqm",
        "Rent_Cost",
        "Outbound_Cost_x2_4",
        "Inbound_Sea_Cost",
        "Inbound_Drayage_Cost",
        "Inbound+Drayage_Cost",
        "Total_Cost",
        "Weighted_Service_Days",
        "Avg_Service_Days_Weighted"
    ])

    total_cost_component = float(value(m._total_cost_expr(m)))
    wsd_total = float(value(m._weighted_service_days_expr(m))) if m._weighted_service_days_expr is not None else None
    avg_days_total = (wsd_total / max(assign_df["Weight"].sum(), 1.0)) if wsd_total is not None else None

    obj = {
        "mode": str(objective_mode),
        "min_wh_sqm": float(min_wh_sqm),
        "service_penalty_per_day": float(service_penalty_per_day),
        "open_warehouses": sorted(assign_df["Warehouse"].unique().tolist()),
        "objective_value": float(value(m.obj)),
        "total_cost_component": total_cost_component,
        "weighted_service_days_component": wsd_total,
        "avg_service_days_weighted": avg_days_total,
        "solver": solver_name,
        "outbound_cost_x2_4_summary": total_out,
        "inbound_sea_summary": total_sea,
        "inbound_dray_summary": total_dray,
    }

    return obj, wh, assign_df


# =========================
# Optional maps & coverage
# =========================
geo_us = pgeocode.Nominatim("us")


def read_geo_file(file_obj):
    if file_obj is None:
        return None
    if file_obj.name.lower().endswith(".csv"):
        df = pd.read_csv(file_obj.name)
    else:
        df = pd.read_excel(file_obj.name)
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["customer key", "DestinationState", "Dest Zip"]:
        if c not in df.columns:
            raise KeyError(f"Geo file missing column: {c}")
    df["Dest Zip"] = df["Dest Zip"].astype(str).str.strip().str.zfill(5)
    df["DestinationState"] = df["DestinationState"].astype(str).str.strip().str.upper()
    df["customer key"] = df["customer key"].astype(str).str.strip()
    return df


def add_lat_lon(df, zip_col="Dest Zip"):
    zips = df[zip_col].dropna().astype(str).unique().tolist()
    lookup = geo_us.query_postal_code(zips)[["postal_code", "latitude", "longitude"]].dropna()
    z2lat = dict(zip(lookup["postal_code"].astype(str), lookup["latitude"]))
    z2lon = dict(zip(lookup["postal_code"].astype(str), lookup["longitude"]))
    df["lat"] = df[zip_col].astype(str).map(z2lat)
    df["lon"] = df[zip_col].astype(str).map(z2lon)
    return df


def bubble_map(assign_df, geo_df, aggregate_zip=True, bubble_multiplier=3.5, size_max=80, join_mode="customer key"):
    if join_mode == "customer key":
        m = assign_df.merge(
            geo_df[["customer key", "DestinationState", "Dest Zip"]],
            left_on="Customer",
            right_on="customer key",
            how="left"
        )
    else:
        tmp = assign_df.copy()
        tmp["Dest Zip"] = tmp["Customer"].astype(str).str.split("_").str[0].str.zfill(5)
        m = tmp.merge(
            geo_df[["Dest Zip", "DestinationState"]].drop_duplicates(),
            on="Dest Zip",
            how="left"
        )

    m = m.dropna(subset=["Dest Zip"]).copy()
    m["Dest Zip"] = m["Dest Zip"].astype(str).str.zfill(5)
    m = add_lat_lon(m, "Dest Zip").dropna(subset=["lat", "lon"])

    if aggregate_zip:
        plot_df = (m.groupby(["Dest Zip", "Warehouse"], as_index=False)
                     .agg(Weight=("Weight", "sum"),
                          Customers=("Customer", "count"),
                          lat=("lat", "first"),
                          lon=("lon", "first")))
        hover_name = "Dest Zip"
        hover_data = {"Customers": True, "Weight": True}
    else:
        plot_df = m.copy()
        plot_df["Customers"] = 1
        hover_name = "Customer"
        hover_data = {"Weight": True, "Dest Zip": True}

    plot_df["BubbleSize"] = plot_df["Weight"].clip(lower=0) * float(bubble_multiplier)

    fig = px.scatter_geo(
        plot_df,
        lat="lat",
        lon="lon",
        scope="usa",
        projection="albers usa",
        color="Warehouse",
        size="BubbleSize",
        size_max=int(size_max),
        hover_name=hover_name,
        hover_data=hover_data,
        title="Customer Bubble Map (size ~ weight, color ~ warehouse)"
    )
    fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0.6)))
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=0))
    return fig


def state_views(assign_df, geo_df, join_mode="customer key"):
    if join_mode == "customer key":
        m = assign_df.merge(
            geo_df[["customer key", "DestinationState", "Dest Zip"]],
            left_on="Customer",
            right_on="customer key",
            how="left"
        ).dropna(subset=["DestinationState"])
    else:
        tmp = assign_df.copy()
        tmp["Dest Zip"] = tmp["Customer"].astype(str).str.split("_").str[0].str.zfill(5)
        m = tmp.merge(
            geo_df[["Dest Zip", "DestinationState"]].drop_duplicates(),
            on="Dest Zip",
            how="left"
        ).dropna(subset=["DestinationState"])

    m["DestinationState"] = m["DestinationState"].astype(str).str.upper().str.strip()

    state_wh = (m.groupby(["DestinationState", "Warehouse"], as_index=False)
                  .agg(Weight=("Weight", "sum")))

    state_tot = state_wh.groupby("DestinationState", as_index=False)["Weight"].sum().rename(columns={"Weight": "StateTotalWeight"})
    state_wh = state_wh.merge(state_tot, on="DestinationState", how="left")
    state_wh["Share"] = state_wh["Weight"] / state_wh["StateTotalWeight"].replace({0: 1})

    dom = (state_wh.sort_values(["DestinationState", "Weight"], ascending=[True, False])
                 .groupby("DestinationState", as_index=False).first())

    fig_dom = px.choropleth(
        dom,
        locations="DestinationState",
        locationmode="USA-states",
        color="Warehouse",
        scope="usa",
        title="Dominant Warehouse by State (largest share by weight)"
    )
    fig_dom.update_layout(margin=dict(l=10, r=10, t=45, b=10))

    top_states = state_tot.sort_values("StateTotalWeight", ascending=False)["DestinationState"].head(25).tolist()
    state_wh2 = state_wh[state_wh["DestinationState"].isin(top_states)].copy()

    fig_mix = px.bar(
        state_wh2,
        x="DestinationState",
        y="Weight",
        color="Warehouse",
        title="State Supply Mix (stacked bars = weighted shares, top 25 states)"
    )
    fig_mix.update_layout(barmode="stack", margin=dict(l=10, r=10, t=45, b=10))
    fig_mix.update_layout(legend=dict(orientation="h", y=1.02, x=0))

    share_tbl = state_wh.copy()
    share_tbl["SharePct"] = (share_tbl["Share"] * 100).round(1)
    share_tbl = share_tbl[["DestinationState", "Warehouse", "Weight", "SharePct"]].sort_values(
        ["DestinationState", "SharePct"], ascending=[True, False]
    )
    return fig_dom, fig_mix, share_tbl


# =========================
# UI glue
# =========================
def ui_load_hubs(outbound_file):
    try:
        with open(outbound_file.name, "rb") as f:
            hubs = peek_hubs_from_outbound(f.read())
        return gr.update(choices=hubs, value=[]), f"Found {len(hubs)} hubs."
    except Exception as e:
        return gr.update(choices=[], value=[]), f"Error: {e}"


def ui_run_optimizer(rent_csv, outbound_xlsx, inbound_xlsx, k, allowed_hubs, objective_mode, service_penalty, min_wh_sqm):
    if not rent_csv or not outbound_xlsx or not inbound_xlsx:
        return "Please upload Rent, Outbound, and Inbound files.", None, None, None, None, None

    try:
        rent_bytes = open(rent_csv.name, "rb").read()
        out_bytes = open(outbound_xlsx.name, "rb").read()
        inb_bytes = open(inbound_xlsx.name, "rb").read()

        obj, wh, assign_df = solve(
            rent_bytes, out_bytes, inb_bytes,
            allowed_hubs or None,
            int(k),
            str(objective_mode),
            float(service_penalty),
            float(min_wh_sqm)
        )

        wh_csv_path = tempfile.mktemp(suffix="_warehouse_summary.csv")
        asn_csv_path = tempfile.mktemp(suffix="_assignments.csv")
        wh.to_csv(wh_csv_path, index=False)
        assign_df.to_csv(asn_csv_path, index=False)

        if obj["weighted_service_days_component"] is not None:
            svc_line = (
                f"\nWeighted Service Days: {obj['weighted_service_days_component']:,}"
                f" | Avg Days (weighted): {obj['avg_service_days_weighted']:.2f}"
            )
        else:
            svc_line = "\nService Days: (not available — no service-day columns detected)"

        summary_text = (
            f"✅ Mode: {obj['mode']} | Solver: {obj['solver']}\n"
            f"Min WH Size (m²): {obj['min_wh_sqm']}\n"
            f"Open Warehouses: {obj['open_warehouses']}\n"
            f"Objective Value: {obj['objective_value']:,}\n"
            f"Total Cost Component: {obj['total_cost_component']:,}\n"
            f"Outbound Cost (x{OUTBOUND_MULT}): {obj['outbound_cost_x2_4_summary']:,}\n"
            f"Inbound Sea: {obj['inbound_sea_summary']:,} | Drayage: {obj['inbound_dray_summary']:,}\n"
            f"Service Penalty ($/weighted day): {obj['service_penalty_per_day']:,}"
            f"{svc_line}"
        )

        return summary_text, wh, assign_df.head(50), wh_csv_path, asn_csv_path, assign_df

    except Exception as e:
        return f"❌ Error: {e}", None, None, None, None, None


def ui_build_maps(assignments_state, geo_file, join_mode, aggregate_zip, bubble_multiplier, size_max,
                  fullscreen_choice, make_fullscreen):
    if assignments_state is None or not isinstance(assignments_state, pd.DataFrame) or assignments_state.empty:
        return "Run the optimizer first (Optimizer tab).", None, None, None, None, None

    if geo_file is None:
        return "Geo file not provided (maps are optional). Upload geo file to build maps.", None, None, None, None, None

    try:
        geo_df = read_geo_file(geo_file)
        fig_bubble = bubble_map(
            assignments_state,
            geo_df,
            aggregate_zip=bool(aggregate_zip),
            bubble_multiplier=float(bubble_multiplier),
            size_max=int(size_max),
            join_mode=str(join_mode)
        )
        fig_dom, fig_mix, share_tbl = state_views(assignments_state, geo_df, join_mode=str(join_mode))

        fs_file = None
        if make_fullscreen:
            if fullscreen_choice == "Bubble Map":
                fs_file = export_fullscreen(fig_bubble, prefix="bubble_map")
            elif fullscreen_choice == "Dominant State Map":
                fs_file = export_fullscreen(fig_dom, prefix="dominant_state")
            else:
                fs_file = export_fullscreen(fig_mix, prefix="state_mix")

        return "Maps generated.", fig_bubble, fig_dom, fig_mix, share_tbl, fs_file

    except Exception as e:
        return f"❌ Map error: {e}", None, None, None, None, None


# =========================
# Gradio Layout
# =========================
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("## Logistics Network Optimization")

    assignments_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Optimizer"):
            gr.Markdown("### Choose your optimization lens, then run")

            with gr.Row():
                outbound = gr.File(label="Outbound (.xlsx)", file_types=[".xlsx"])
                rent = gr.File(label="Rents (.csv)", file_types=[".csv"])
                inbound = gr.File(label="Inbound (.xlsx)", file_types=[".xlsx"])

            with gr.Row():
                hubs = gr.CheckboxGroup(label="Allowed Warehouses (detected):", choices=[], value=[])
                k = gr.Slider(1, 15, value=2, step=1, label="Number of Warehouses (K)")

            with gr.Row():
                objective_mode = gr.Dropdown(
                    choices=["Optimize Cost", "Optimize Service Days", "Optimize Cost and Service Days"],
                    value="Optimize Cost",
                    label="Optimization lens"
                )
                service_penalty = gr.Number(
                    value=2000,
                    precision=0,
                    label="Service penalty ($ per weighted day) [only used for Cost+Service]"
                )
                min_wh_sqm = gr.Number(
                    value=1500,
                    precision=0,
                    label="Minimum warehouse size (m²) for any OPEN warehouse"
                )

            with gr.Row():
                load_btn = gr.Button("🔎 Load hubs from Outbound")
                run_btn = gr.Button("🚀 Optimize")

            summary = gr.Textbox(label="Summary", lines=9, interactive=False)
            wh_df = gr.Dataframe(label="Warehouse Summary", interactive=False, wrap=True)
            asn_preview = gr.Dataframe(label="Assignments preview (first 50)", interactive=False, wrap=True)

            with gr.Row():
                wh_csv = gr.File(label="Download warehouse_summary.csv", interactive=False)
                asn_csv = gr.File(label="Download assignments.csv", interactive=False)

            load_btn.click(ui_load_hubs, inputs=outbound, outputs=[hubs, summary])

            run_btn.click(
                ui_run_optimizer,
                inputs=[rent, outbound, inbound, k, hubs, objective_mode, service_penalty, min_wh_sqm],
                outputs=[summary, wh_df, asn_preview, wh_csv, asn_csv, assignments_state]
            )

        with gr.Tab("Maps & Coverage (Optional)"):
            gr.Markdown(
                "### Optional maps\n"
                "Upload your geo file only if you want mapping.\n\n"
                "**Geo file columns:** `customer key`, `DestinationState`, `Dest Zip`"
            )

            geo = gr.File(label="Geo file (.csv or .xlsx)", file_types=[".csv", ".xlsx"])

            with gr.Row():
                join_mode = gr.Dropdown(
                    choices=["customer key", "zip"],
                    value="customer key",
                    label="Join method (if keys don't match, use ZIP)"
                )
                aggregate_zip = gr.Checkbox(value=True, label="Aggregate by ZIP (recommended)")
                bubble_multiplier = gr.Slider(0.5, 10.0, value=3.5, step=0.5, label="Bubble size multiplier")
                size_max = gr.Slider(10, 140, value=80, step=5, label="Max bubble size")

            with gr.Row():
                make_fullscreen = gr.Checkbox(value=False, label="Create full-screen HTML")
                fullscreen_choice = gr.Dropdown(
                    choices=["Bubble Map", "Dominant State Map", "State Mix Chart"],
                    value="Bubble Map",
                    label="Full-screen export view"
                )

            build_btn = gr.Button("🗺️ Build maps")

            map_status = gr.Markdown("")
            bubble_plot = gr.Plot(label="Bubble map (bigger bubbles)")
            dom_plot = gr.Plot(label="Dominant warehouse by state")
            mix_plot = gr.Plot(label="State supply mix (stacked shares)")
            share_table = gr.Dataframe(label="State → Warehouse share table", interactive=False, wrap=True)
            fullscreen_file = gr.File(label="Download full-screen HTML", interactive=False)

            build_btn.click(
                ui_build_maps,
                inputs=[assignments_state, geo, join_mode, aggregate_zip, bubble_multiplier, size_max,
                        fullscreen_choice, make_fullscreen],
                outputs=[map_status, bubble_plot, dom_plot, mix_plot, share_table, fullscreen_file]
            )

import os

app = demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
