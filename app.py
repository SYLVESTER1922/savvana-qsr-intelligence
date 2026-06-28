import os, time, warnings
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests as _http
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ── Gradio 5.x schema bug patch ───────────────────────────────────────────────
try:
    import gradio_client.utils as _gcu
    _orig = _gcu._json_schema_to_python_type
    def _safe(s, d):
        if not isinstance(s, dict): return "any"
        try: return _orig(s, d)
        except: return "any"
    _gcu._json_schema_to_python_type = _safe
except: pass

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
COMPLEXES    = ['Westgate Mall', 'City Centre', 'Eastpark', 'Northgate']
BRANDS       = ['Flame & Grill', 'Pie Palace', 'Chill Creamery', 'Sizzle Wings']

# Stores Intelligence colour palette (exact match)
C_NAVY    = '#1B2A4E'
C_GOLD    = '#C9A55C'
C_RED     = '#C0392B'
C_ORANGE  = '#E67E22'
C_GREEN   = '#27AE60'
C_TEAL    = '#1abc9c'
C_PURPLE  = '#9b59b6'
C_CREAM   = '#F7F3EC'
C_BG      = '#F7F3EC'   # body bg — warm cream
C_PLOT    = 'white'     # chart bg — white
C_GRID    = '#e5e7eb'   # light gray grid
C_TEXT    = '#1f2937'   # dark text on light bg
BAR_COLS  = [C_NAVY, C_GOLD, C_GREEN, C_RED, C_ORANGE, C_TEAL, C_PURPLE, '#3498db']

_cache = {'df': pd.DataFrame(), 'loaded_at': 0}
CACHE_TTL = 300

# ── Data layer — paginated so we get ALL rows ─────────────────────────────────
def load_data():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Supabase not configured"); return pd.DataFrame()
    try:
        all_rows, offset, batch = [], 0, 1000
        while True:
            r = _http.get(
                f"{SUPABASE_URL}/rest/v1/daily_input?select=*&order=date&limit={batch}&offset={offset}",
                headers={"apikey": SUPABASE_KEY, "Authorization": "Bearer " + SUPABASE_KEY}
            )
            if r.status_code != 200: print("HTTP", r.status_code); break
            rows = r.json()
            if not rows: break
            all_rows.extend(rows)
            if len(rows) < batch: break
            offset += batch
        if not all_rows: return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        nums = ['actual_revenue_usd','budget_usd','customer_count','prior_month_actual',
                'prior_year_actual','variance_vs_budget','variance_pct',
                'avg_spend_per_cust','revenue_per_counter','counters_open',
                'vs_prior_month_pct','vs_prior_year_pct','is_holiday']
        for c in nums:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date','actual_revenue_usd'])
        df = df[df['actual_revenue_usd'] > 0].sort_values('date').reset_index(drop=True)
        print(f"Loaded {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
        return df
    except Exception as e:
        print("Load error:", e); return pd.DataFrame()

def get_df():
    now = time.time()
    if now - _cache['loaded_at'] > CACHE_TTL or _cache['df'].empty:
        fresh = load_data()
        if not fresh.empty:
            _cache['df'] = fresh
            _cache['loaded_at'] = now
    return _cache['df']

def filter_df(date_from, date_to):
    df = get_df()
    if df.empty: return df
    try:
        if date_from and str(date_from).strip():
            df = df[df['date'] >= pd.to_datetime(str(date_from).strip())]
        if date_to and str(date_to).strip():
            df = df[df['date'] <= pd.to_datetime(str(date_to).strip())]
    except: pass
    return df

print("Loading data..."); _ = get_df()

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(v):
    if v >= 1e6: return f"${v/1e6:.2f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"

def dark_layout(title, height=320, margin=None):
    """Layout matching Stores Intelligence — white charts, navy/gold palette."""
    m = margin or dict(l=150, r=30, t=50, b=40)
    return dict(
        title=dict(text=title, font=dict(color=C_NAVY, size=13, family='Inter, Arial'), x=0),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='#374151', family='Inter, Arial', size=11),
        height=height,
        xaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb',
                   tickfont=dict(color='#6b7280'), title_font=dict(color='#374151')),
        yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb',
                   tickfont=dict(color='#6b7280'), title_font=dict(color='#374151')),
        legend=dict(bgcolor='white', bordercolor='#e5e7eb', borderwidth=1,
                    font=dict(color='#374151')),
        margin=m
    )

def horiz_bar(x_vals, y_labels, colors, texts, title, height=280):
    """Horizontal bar with values inside, matching Stores Intelligence style."""
    fig = go.Figure()
    # Pick text color: dark for gold, white for navy/dark
    text_colors = []
    for c in (colors if isinstance(colors, list) else [colors]*len(x_vals)):
        text_colors.append('#0a1628' if c in (C_GOLD, C_TEAL, '#2ecc71') else '#ffffff')
    fig.add_trace(go.Bar(
        x=x_vals, y=y_labels, orientation='h',
        marker=dict(color=colors if isinstance(colors, list) else [colors]*len(x_vals),
                    line=dict(color=C_GRID, width=1)),
        text=texts,
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color=text_colors if len(set(text_colors))>1 else text_colors[0],
                      size=12, family='Arial')
    ))
    fig.update_layout(**dark_layout(title, height=height))
    return fig

# ── KPI header ────────────────────────────────────────────────────────────────
def build_kpi_html(dff=None):
    if dff is None: dff = get_df()
    if dff is not None and not isinstance(dff, pd.DataFrame): dff = get_df()
    if dff is None or dff.empty:
        rev, ach, tc, tb, dr = "--", "--", "--", "--", "No data"
    else:
        rev = fmt(dff['actual_revenue_usd'].sum())
        bud = dff['budget_usd'].sum() if 'budget_usd' in dff.columns else 0
        ach = f"{dff['actual_revenue_usd'].sum()/bud*100:.1f}%" if bud > 0 else "--"
        tc  = dff.groupby('complex')['actual_revenue_usd'].sum().idxmax()
        tb  = dff.groupby('brand')['actual_revenue_usd'].sum().idxmax()
        dr  = f"{dff['date'].min().strftime('%b %d, %Y')} → {dff['date'].max().strftime('%b %d, %Y')}"
    pills = "".join([
        f'<div class="kpi-pill"><div class="kpi-val">{v}</div><div class="kpi-lbl">{l}</div></div>'
        for v,l in [(rev,"Total Revenue"),(ach,"Budget Ach."),(tc,"Top Complex"),(tb,"Top Brand"),(dr,"Date Range")]
    ])
    return f'''<div id="savanna-hero">
      <div>
        <div class="brand-tag">SAVANNA QSR INTELLIGENCE</div>
        <h2>Live Operations Dashboard</h2>
        <p class="tagline">Zimbabwe · Real-time · Powered by Supabase + Netrisyl Insights</p>
        <div class="kpi-row">{pills}</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
        <img src="/file=NI_logo.png" alt="Netrisyl Insights"
             style="height:56px;width:auto;object-fit:contain;"
             onerror="this.style.display='none';document.getElementById('ni-fallback').style.display='block'"/>
        <div id="ni-fallback" class="ni-label" style="display:none;">NETRISYL INSIGHTS</div>
      </div>
    </div>'''

# ── Dashboard ─────────────────────────────────────────────────────────────────
def build_dashboard(date_from, date_to):
    dff = filter_df(date_from, date_to)
    kpi = build_kpi_html(dff)
    if dff.empty:
        e = go.Figure().update_layout(**dark_layout("No data — click Refresh"))
        return kpi, e, e, e, e, e, e

    # 1. Revenue by Complex — gold actual bar + navy budget overlay
    cx = dff.groupby('complex').agg(
        actual=('actual_revenue_usd','sum'), budget=('budget_usd','sum')
    ).reset_index().sort_values('actual', ascending=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name='Budget', x=cx['budget'], y=cx['complex'], orientation='h',
        marker=dict(color='#2a4a8a', line=dict(color=C_GRID, width=1)), opacity=0.55
    ))
    fig1.add_trace(go.Bar(
        name='Actual', x=cx['actual'], y=cx['complex'], orientation='h',
        marker=dict(color=C_GOLD, line=dict(color=C_NAVY, width=1)),
        text=[fmt(v) for v in cx['actual']], textposition='inside',
        insidetextanchor='middle', textfont=dict(color='#1B2A4E', size=11, family='Inter, Arial', weight='bold')
    ))
    fig1.update_layout(**dark_layout("Revenue by Complex — Actual vs Budget", height=300),
                       barmode='overlay',
                       xaxis_tickprefix='$', xaxis_tickformat=',.0f')

    # 2. Revenue by Brand
    br = dff.groupby('brand')['actual_revenue_usd'].sum().reset_index().sort_values('actual_revenue_usd', ascending=True)
    bud_br = dff.groupby('brand')['budget_usd'].sum() if 'budget_usd' in dff.columns else None
    texts = []
    for _, row in br.iterrows():
        bv = bud_br[row['brand']] if bud_br is not None and bud_br.sum() > 0 else 0
        ach = f"  {row['actual_revenue_usd']/bv*100:.0f}%" if bv > 0 else ""
        texts.append(f"{fmt(row['actual_revenue_usd'])}{ach}")
    fig2 = go.Figure(go.Bar(
        x=br['actual_revenue_usd'], y=br['brand'], orientation='h',
        marker=dict(color=BAR_COLS[:len(br)], line=dict(color=C_GRID, width=1)),
        text=texts, textposition='inside', insidetextanchor='middle',
        textfont=dict(color='white', size=11, family='Inter, Arial', weight='bold')
    ))
    fig2.update_layout(**dark_layout("Revenue by Brand", height=280),
                       xaxis_tickprefix='$', xaxis_tickformat=',.0f')

    # 3. Daily Revenue Trend — bar per day so it is clearly NOT cumulative
    daily = dff.groupby('date')['actual_revenue_usd'].sum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=daily['date'], y=daily['actual_revenue_usd'], name='Actual',
        marker=dict(color=C_GOLD, opacity=0.85, line=dict(color=C_NAVY, width=0.3)),
        hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Daily Revenue</extra>'
    ))
    if 'budget_usd' in dff.columns:
        daily_b = dff.groupby('date')['budget_usd'].sum().reset_index()
        fig3.add_trace(go.Scatter(
            x=daily_b['date'], y=daily_b['budget_usd'], name='Daily Budget',
            line=dict(color='#4a6a9e', width=1.5, dash='dot'),
            hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Budget</extra>'
        ))
    fig3.update_layout(**dark_layout("Daily Revenue Trend", height=340, margin=dict(l=80,r=30,t=45,b=50)),
                       hovermode='x unified', bargap=0.1,
                       yaxis_tickprefix='$', yaxis_tickformat=',.0f')

    # 4. Avg Spend Per Customer
    if 'customer_count' in dff.columns and dff['customer_count'].sum() > 0:
        avs = dff[dff['customer_count']>0].groupby('complex').apply(
            lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()
        ).reset_index(name='avg').sort_values('avg', ascending=True)
        fig4 = go.Figure(go.Bar(
            x=avs['avg'], y=avs['complex'], orientation='h',
            marker=dict(color=C_TEAL, line=dict(color=C_NAVY, width=1)),
            text=[f"${v:.2f}" for v in avs['avg']], textposition='inside',
            insidetextanchor='middle', textfont=dict(color='#1B2A4E', size=11, family='Inter, Arial', weight='bold')
        ))
        fig4.update_layout(**dark_layout("Avg Spend Per Customer", height=280),
                           xaxis_tickprefix='$', xaxis_tickformat=',.2f')
    else:
        fig4 = go.Figure().update_layout(**dark_layout("No customer data", height=280))

    # 5. Budget Achievement Heatmap
    if 'budget_usd' in dff.columns and dff['budget_usd'].sum() > 0:
        try:
            heat = dff.groupby(['complex','brand']).apply(
                lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100
                          if x['budget_usd'].sum()>0 else 0
            ).reset_index(name='ach')
            pivot = heat.pivot(index='complex', columns='brand', values='ach').fillna(0)
            fig5 = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0,'#c0392b'],[0.8,'#f39c12'],[1,'#2ecc71']],
                zmid=100, zmin=70, zmax=130,
                text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}", textfont=dict(color='white', size=12),
                colorbar=dict(tickfont=dict(color=C_TEXT))
            ))
            fig5.update_layout(**dark_layout("Budget Achievement % (Complex × Brand)", height=280,
                                             margin=dict(l=130,r=30,t=45,b=80)))
        except:
            fig5 = go.Figure().update_layout(**dark_layout("Budget heatmap unavailable", height=280))
    else:
        fig5 = go.Figure().update_layout(**dark_layout("No budget data", height=280))

    # 6. Actual vs Prior Month vs Prior Year
    has_pm = 'prior_month_actual' in dff.columns and dff['prior_month_actual'].sum() > 0
    has_py = 'prior_year_actual'  in dff.columns and dff['prior_year_actual'].sum()  > 0
    comp = dff.groupby('complex').agg(
        actual=('actual_revenue_usd','sum'),
        **({'pm':('prior_month_actual','sum')} if has_pm else {}),
        **({'py':('prior_year_actual','sum')} if has_py else {})
    ).reset_index()
    fig6 = go.Figure()
    if has_py:
        fig6.add_trace(go.Bar(name='Prior Year', x=comp['complex'], y=comp['py'],
            marker=dict(color=C_NAVY, line=dict(color=C_GRID,width=1)),
            text=[fmt(v) for v in comp['py']], textposition='inside',
            insidetextanchor='middle', textfont=dict(color='#ffffff',size=11)))
    if has_pm:
        fig6.add_trace(go.Bar(name='Prior Month', x=comp['complex'], y=comp['pm'],
            marker=dict(color=C_TEAL, line=dict(color=C_GRID,width=1)),
            text=[fmt(v) for v in comp['pm']], textposition='inside',
            insidetextanchor='middle', textfont=dict(color='#0a1628',size=11)))
    fig6.add_trace(go.Bar(name='This Period', x=comp['complex'], y=comp['actual'],
        marker=dict(color=C_GOLD, line=dict(color=C_NAVY,width=1)),
        text=[fmt(v) for v in comp['actual']], textposition='inside',
        insidetextanchor='middle', textfont=dict(color='#0a1628',size=11)))
    fig6.update_layout(**dark_layout("Actual vs Prior Month vs Prior Year", height=340,
                                     margin=dict(l=60,r=30,t=45,b=60)),
                       barmode='group', hovermode='x unified',
                       yaxis_tickprefix='$', yaxis_tickformat=',.0f')

    return kpi, fig1, fig2, fig3, fig4, fig5, fig6

# ── Forecast ──────────────────────────────────────────────────────────────────
GROWTH = 0.05
DOW    = {0:0.85,1:0.90,2:0.92,3:0.95,4:1.05,5:1.20,6:1.15}

def update_seg_choices(seg_type):
    if seg_type == 'Overall':
        return gr.update(choices=['All Complexes & Brands'], value='All Complexes & Brands')
    elif seg_type == 'By Complex':
        return gr.update(choices=COMPLEXES, value=COMPLEXES[0])
    elif seg_type == 'By Brand':
        return gr.update(choices=BRANDS, value=BRANDS[0])
    else:
        opts = [f"{cx} | {br}" for cx in COMPLEXES for br in BRANDS]
        return gr.update(choices=opts, value=opts[0])

def generate_forecast(seg_type, seg_name, horizon, date_from, date_to):
    try:
        df = filter_df(date_from, date_to)
        if df.empty:
            return go.Figure().update_layout(**dark_layout("No data for selected range")), "No data."
        h = int(horizon)
        if seg_type == 'Overall':
            seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = 'All Complexes & Brands'
        elif seg_type == 'By Complex':
            seg = df[df['complex']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = seg_name
        elif seg_type == 'By Brand':
            seg = df[df['brand']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = seg_name
        else:
            parts = seg_name.split(' | ')
            seg = df[(df['complex']==parts[0])&(df['brand']==parts[1])].groupby('date')['actual_revenue_usd'].sum().reset_index() if len(parts)==2 else df.groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = seg_name
        seg.columns = ['date','revenue']
        seg = seg.sort_values('date').reset_index(drop=True)
        if len(seg) < 7:
            return go.Figure().update_layout(**dark_layout("Need at least 7 days of data for this segment")), "Insufficient data."
        last = seg['date'].max()
        base = float(seg['revenue'].tail(min(h, len(seg))).mean())
        # Limit historical display to last 90 days so forecast is proportionally visible
        seg_display = seg[seg['date'] >= last - timedelta(days=90)].reset_index(drop=True)
        if np.isnan(base) or base <= 0: base = float(seg['revenue'].mean())
        fc_dates = [last + timedelta(days=i+1) for i in range(h)]
        fc_vals  = [round(base * DOW[d.weekday()] * (1+GROWTH), 2) for d in fc_dates]
        upper = [v*1.15 for v in fc_vals]
        lower = [v*0.85 for v in fc_vals]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seg_display['date'], y=seg_display['revenue'], name='Historical (last 90d)',
            line=dict(color=C_NAVY, width=2.5), opacity=0.9,
            hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Historical</extra>'))
        fig.add_trace(go.Scatter(
            x=fc_dates+fc_dates[::-1], y=upper+lower[::-1],
            fill='toself', fillcolor='rgba(201,168,76,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_vals, name=f'{h}-Day Forecast',
            line=dict(color=C_GOLD, width=2.5, dash='dash'), mode='lines+markers',
            marker=dict(size=5, color=C_GOLD),
            hovertemplate='%{x|%b %d}<br>$%{y:,.0f}<extra>Forecast</extra>'))
        fig.add_vline(x=str(last)[:10], line_dash='dot', line_color=C_GOLD, opacity=0.5)
        all_vals = list(seg_display['revenue']) + fc_vals + upper
        y_max = max(all_vals) * 1.15 if all_vals else 1
        y_min = max(0, min(list(seg_display['revenue']) + lower) * 0.85)
        layout = dark_layout(f"{label} — {h}-Day Forecast", height=480,
                             margin=dict(l=80,r=40,t=60,b=50))
        layout['hovermode'] = 'x unified'
        layout['yaxis_tickprefix'] = '$'
        layout['yaxis_tickformat'] = ',.0f'
        layout['yaxis_range'] = [y_min, y_max]
        fig.update_layout(**layout)
        total = sum(fc_vals); avg = total/h
        peak = max(fc_vals); peak_d = fc_dates[fc_vals.index(peak)].strftime('%b %d, %Y')
        summary = f"**{h}-Day Forecast — {label}**\n\n| Metric | Value |\n|---|---|\n| Predicted Total | **{fmt(total)}** |\n| Daily Average | **{fmt(avg)}** |\n| Peak Day | **{fmt(peak)}** on {peak_d} |\n| Period | {fc_dates[0].strftime('%b %d')} → {fc_dates[-1].strftime('%b %d, %Y')} |\n\n> *Day-of-week adjusted moving average with 5% growth factor.*"
        return fig, summary
    except Exception as e:
        return go.Figure().update_layout(**dark_layout(f"Error: {e}", height=460)), f"Error: {e}"

# ── Intelligence Chat ─────────────────────────────────────────────────────────
# ── Tool-calling intelligence layer (mirrors Stores Intelligence architecture) ─
import json as _json

# Tool definitions — GPT picks which ones to call
TOOLS = [
    {"type": "function", "function": {
        "name": "q_revenue_by_date",
        "description": "Revenue for a specific calendar date. Use for questions like 'how much on 10 Feb 2023', 'what was made on March 5', 'revenue on 2023-07-15'.",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format."}
        }, "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "q_revenue_by_period",
        "description": "Revenue for a named period or date range. Use for 'last 7 days', 'last 30 days', 'this month', 'MTD', 'all time', or 'from X to Y'.",
        "parameters": {"type": "object", "properties": {
            "period": {"type": "string", "enum": ["last_7_days","last_30_days","last_90_days","mtd","all_time"]},
            "date_from": {"type": "string", "description": "Optional start date YYYY-MM-DD for custom range."},
            "date_to":   {"type": "string", "description": "Optional end date YYYY-MM-DD for custom range."}
        }}}},
    {"type": "function", "function": {
        "name": "q_revenue_by_month",
        "description": "Revenue for a specific month and optional year. Use for 'how much in February', 'January 2023 revenue', 'what did we make in March'.",
        "parameters": {"type": "object", "properties": {
            "month": {"type": "integer", "description": "Month number 1-12."},
            "year":  {"type": "integer", "description": "4-digit year. Omit to search all years."}
        }, "required": ["month"]}}},
    {"type": "function", "function": {
        "name": "q_revenue_by_complex",
        "description": "Revenue breakdown by complex/branch. Use for 'which complex is best', 'City Centre revenue', 'how is Westgate Mall doing', 'compare all branches'.",
        "parameters": {"type": "object", "properties": {
            "complex_name": {"type": "string", "description": "Specific complex name, or omit for all."}
        }}}},
    {"type": "function", "function": {
        "name": "q_revenue_by_brand",
        "description": "Revenue breakdown by brand. Use for 'which brand is top', 'Flame & Grill revenue', 'how is Pie Palace doing', 'rank all brands'.",
        "parameters": {"type": "object", "properties": {
            "brand_name": {"type": "string", "description": "Specific brand name, or omit for all."}
        }}}},
    {"type": "function", "function": {
        "name": "q_complex_brand",
        "description": "Revenue for a specific complex AND brand combination. Use for 'Flame & Grill at City Centre', 'Westgate Mall Pie Palace performance'.",
        "parameters": {"type": "object", "properties": {
            "complex_name": {"type": "string"},
            "brand_name":   {"type": "string"}
        }, "required": ["complex_name", "brand_name"]}}},
    {"type": "function", "function": {
        "name": "q_budget_achievement",
        "description": "Budget achievement and variance analysis. Use for 'budget performance', 'are we on target', 'which sites are below budget', 'variance report', 'which complex needs attention'.",
        "parameters": {"type": "object", "properties": {
            "threshold_pct": {"type": "number", "description": "Flag sites below this % of budget. Default 90."},
            "complex_name":  {"type": "string", "description": "Filter to one complex, or omit for all."}
        }}}},
    {"type": "function", "function": {
        "name": "q_customer_metrics",
        "description": "Customer count and average spend per customer. Use for 'avg spend', 'customer turnover', 'foot traffic', 'footfall', 'spend per customer'.",
        "parameters": {"type": "object", "properties": {
            "complex_name": {"type": "string", "description": "Filter to one complex, or omit for all."}
        }}}},
    {"type": "function", "function": {
        "name": "q_prior_period_comparison",
        "description": "Compare current revenue against prior month or prior year. Use for 'vs last month', 'prior year comparison', 'SDLM', 'SDLY', 'year on year', 'month on month'.",
        "parameters": {"type": "object", "properties": {
            "comparison": {"type": "string", "enum": ["prior_month", "prior_year"]}
        }, "required": ["comparison"]}}},
    {"type": "function", "function": {
        "name": "q_daily_trend",
        "description": "Daily revenue trend analysis. Use for 'daily trend', 'revenue pattern', 'day of week performance', 'best performing day', 'busiest day'.",
        "parameters": {"type": "object", "properties": {
            "group_by": {"type": "string", "enum": ["day_of_week","daily"], "description": "day_of_week for weekday analysis, daily for time series."}
        }}}},
    {"type": "function", "function": {
        "name": "q_rankings",
        "description": "Rank and compare all complexes or brands. Use for 'rank all sites', 'full overview', 'performance summary', 'compare everything', 'leaderboard'.",
        "parameters": {"type": "object", "properties": {
            "by": {"type": "string", "enum": ["complex","brand","both"], "description": "What to rank. Default both."}
        }}}},
]

# ── Tool implementation functions (pandas on filtered df) ─────────────────────
def _tool_q_revenue_by_date(dff, date):
    import pandas as _pd
    try:
        target = _pd.to_datetime(date).date()
    except:
        return {"error": f"Could not parse date: {date}"}
    day = dff[dff['date'].dt.date == target]
    if day.empty:
        return {"found": False, "date": date, "note": "No data for this date in the selected range."}
    has_bud = 'budget_usd' in day.columns and day['budget_usd'].sum() > 0
    rev = day['actual_revenue_usd'].sum()
    bud = day['budget_usd'].sum() if has_bud else 0
    cx = day.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).to_dict()
    br = day.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False).to_dict()
    return {"found": True, "date": date, "total_revenue_usd": round(rev, 2),
            "budget_usd": round(bud, 2), "budget_achievement_pct": round(rev/bud*100,1) if bud > 0 else None,
            "by_complex": {k: round(v,2) for k,v in cx.items()},
            "by_brand":   {k: round(v,2) for k,v in br.items()}}

def _tool_q_revenue_by_period(dff, period=None, date_from=None, date_to=None):
    import pandas as _pd
    df = dff.copy()
    latest = df['date'].max()
    if period == 'last_7_days':  df = df[df['date'] >= latest - timedelta(days=7)]
    elif period == 'last_30_days': df = df[df['date'] >= latest - timedelta(days=30)]
    elif period == 'last_90_days': df = df[df['date'] >= latest - timedelta(days=90)]
    elif period == 'mtd':
        df = df[(df['date'].dt.month==latest.month)&(df['date'].dt.year==latest.year)]
    elif date_from or date_to:
        if date_from: df = df[df['date'] >= _pd.to_datetime(date_from)]
        if date_to:   df = df[df['date'] <= _pd.to_datetime(date_to)]
    if df.empty: return {"found": False, "note": "No data for this period."}
    has_bud = 'budget_usd' in df.columns and df['budget_usd'].sum() > 0
    rev = df['actual_revenue_usd'].sum()
    bud = df['budget_usd'].sum() if has_bud else 0
    days = df['date'].nunique()
    return {"period": period or f"{date_from} to {date_to}",
            "total_revenue_usd": round(rev,2), "days": days,
            "daily_avg_usd": round(rev/days,2) if days>0 else 0,
            "budget_usd": round(bud,2),
            "budget_achievement_pct": round(rev/bud*100,1) if bud>0 else None,
            "date_range": f"{df['date'].min().strftime('%b %d, %Y')} to {df['date'].max().strftime('%b %d, %Y')}"}

def _tool_q_revenue_by_month(dff, month, year=None):
    df = dff[dff['date'].dt.month == month]
    if year: df = df[df['date'].dt.year == year]
    if df.empty:
        import calendar
        return {"found": False, "month": calendar.month_name[month], "year": year}
    import calendar
    has_bud = 'budget_usd' in df.columns and df['budget_usd'].sum() > 0
    rev = df['actual_revenue_usd'].sum()
    bud = df['budget_usd'].sum() if has_bud else 0
    cx = df.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).to_dict()
    br = df.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False).to_dict()
    # Detect if multiple years
    years = df['date'].dt.year.unique().tolist()
    return {"found": True, "month": calendar.month_name[month], "year": year or years,
            "total_revenue_usd": round(rev,2),
            "budget_usd": round(bud,2), "budget_achievement_pct": round(rev/bud*100,1) if bud>0 else None,
            "by_complex": {k: round(v,2) for k,v in cx.items()},
            "by_brand":   {k: round(v,2) for k,v in br.items()}}

def _tool_q_revenue_by_complex(dff, complex_name=None):
    df = dff[dff['complex']==complex_name] if complex_name else dff
    if df.empty: return {"found": False}
    has_bud = 'budget_usd' in df.columns and df['budget_usd'].sum() > 0
    if complex_name:
        rev = df['actual_revenue_usd'].sum()
        bud = df['budget_usd'].sum() if has_bud else 0
        br  = df.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False).to_dict()
        return {"complex": complex_name, "total_revenue_usd": round(rev,2),
                "budget_usd": round(bud,2), "budget_achievement_pct": round(rev/bud*100,1) if bud>0 else None,
                "by_brand": {k: round(v,2) for k,v in br.items()}}
    cx = df.groupby('complex').agg(rev=('actual_revenue_usd','sum'), bud=('budget_usd','sum') if has_bud else ('actual_revenue_usd','sum')).reset_index().sort_values('rev',ascending=False)
    result = []
    for r in cx.itertuples():
        result.append({"complex": r.complex, "total_revenue_usd": round(r.rev,2),
            "budget_achievement_pct": round(r.rev/r.bud*100,1) if has_bud and r.bud>0 else None})
    return {"all_complexes": result, "total_revenue_usd": round(df['actual_revenue_usd'].sum(),2)}

def _tool_q_revenue_by_brand(dff, brand_name=None):
    df = dff[dff['brand']==brand_name] if brand_name else dff
    if df.empty: return {"found": False}
    has_bud = 'budget_usd' in df.columns and df['budget_usd'].sum() > 0
    if brand_name:
        rev = df['actual_revenue_usd'].sum()
        bud = df['budget_usd'].sum() if has_bud else 0
        cx  = df.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).to_dict()
        return {"brand": brand_name, "total_revenue_usd": round(rev,2),
                "budget_achievement_pct": round(rev/bud*100,1) if bud>0 else None,
                "by_complex": {k: round(v,2) for k,v in cx.items()}}
    br = df.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False)
    return {"all_brands": [{"brand": b, "total_revenue_usd": round(v,2)} for b,v in br.items()],
            "total_revenue_usd": round(df['actual_revenue_usd'].sum(),2)}

def _tool_q_complex_brand(dff, complex_name, brand_name):
    df = dff[(dff['complex']==complex_name)&(dff['brand']==brand_name)]
    if df.empty: return {"found": False, "complex": complex_name, "brand": brand_name}
    has_bud = 'budget_usd' in df.columns and df['budget_usd'].sum() > 0
    rev = df['actual_revenue_usd'].sum()
    bud = df['budget_usd'].sum() if has_bud else 0
    return {"found": True, "complex": complex_name, "brand": brand_name,
            "total_revenue_usd": round(rev,2), "daily_avg_usd": round(df['actual_revenue_usd'].mean(),2),
            "budget_achievement_pct": round(rev/bud*100,1) if bud>0 else None, "days": len(df)}

def _tool_q_budget_achievement(dff, threshold_pct=90, complex_name=None):
    df = dff[dff['complex']==complex_name] if complex_name else dff
    if df.empty or 'budget_usd' not in df.columns or df['budget_usd'].sum()==0:
        return {"error": "No budget data available."}
    overall_rev = df['actual_revenue_usd'].sum()
    overall_bud = df['budget_usd'].sum()
    cx = df.groupby('complex').apply(lambda x: {
        "complex": x.name, "revenue_usd": round(x['actual_revenue_usd'].sum(),2),
        "budget_usd": round(x['budget_usd'].sum(),2),
        "achievement_pct": round(x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100,1) if x['budget_usd'].sum()>0 else 0
    }).tolist()
    cx.sort(key=lambda x: x['achievement_pct'])
    below = [c for c in cx if c['achievement_pct'] < threshold_pct]
    return {"overall_achievement_pct": round(overall_rev/overall_bud*100,1),
            "overall_revenue_usd": round(overall_rev,2), "overall_budget_usd": round(overall_bud,2),
            "threshold_pct": threshold_pct, "sites_below_threshold": below,
            "all_complexes": sorted(cx, key=lambda x: -x['achievement_pct'])}

def _tool_q_customer_metrics(dff, complex_name=None):
    df = dff[dff['complex']==complex_name] if complex_name else dff
    if 'customer_count' not in df.columns or df['customer_count'].sum()==0:
        return {"error": "No customer count data available."}
    df = df[df['customer_count']>0]
    if complex_name:
        avg = df['actual_revenue_usd'].sum()/df['customer_count'].sum()
        return {"complex": complex_name, "total_customers": int(df['customer_count'].sum()),
                "avg_spend_per_customer_usd": round(avg,2)}
    cx = df.groupby('complex').apply(lambda x: round(x['actual_revenue_usd'].sum()/x['customer_count'].sum(),2)).sort_values(ascending=False)
    return {"total_customers": int(df['customer_count'].sum()),
            "avg_spend_by_complex": {k: v for k,v in cx.items()},
            "overall_avg_usd": round(df['actual_revenue_usd'].sum()/df['customer_count'].sum(),2)}

def _tool_q_prior_period_comparison(dff, comparison):
    col = 'prior_month_actual' if comparison=='prior_month' else 'prior_year_actual'
    label = 'Prior Month' if comparison=='prior_month' else 'Prior Year'
    if col not in dff.columns or dff[col].sum()==0:
        return {"error": f"No {label} data available."}
    act = dff['actual_revenue_usd'].sum(); prior = dff[col].sum()
    cx = dff.groupby('complex').agg(actual=('actual_revenue_usd','sum'), prior=(col,'sum')).reset_index()
    result = [{"complex": r.complex, "actual_usd": round(r.actual,2), "prior_usd": round(r.prior,2),
               "change_pct": round((r.actual-r.prior)/r.prior*100,1) if r.prior>0 else None}
              for r in cx.itertuples()]
    return {"comparison": label, "overall_actual_usd": round(act,2),
            "overall_prior_usd": round(prior,2),
            "overall_change_pct": round((act-prior)/prior*100,1) if prior>0 else None,
            "by_complex": result}

def _tool_q_daily_trend(dff, group_by='day_of_week'):
    if group_by == 'day_of_week':
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = dff.groupby(dff['date'].dt.day_name())['actual_revenue_usd'].mean()
        dow = dow.reindex([d for d in order if d in dow.index])
        return {"group_by": "day_of_week",
                "avg_revenue_by_day": {d: round(v,2) for d,v in dow.items()},
                "best_day": dow.idxmax(), "worst_day": dow.idxmin()}
    else:
        daily = dff.groupby('date')['actual_revenue_usd'].sum()
        return {"group_by": "daily", "days": len(daily),
                "avg_daily_usd": round(daily.mean(),2), "max_daily_usd": round(daily.max(),2),
                "min_daily_usd": round(daily.min(),2), "peak_date": str(daily.idxmax().date())}

def _tool_q_rankings(dff, by='both'):
    has_bud = 'budget_usd' in dff.columns and dff['budget_usd'].sum()>0
    result = {}
    if by in ('complex','both'):
        cx = dff.groupby('complex').agg(rev=('actual_revenue_usd','sum'), bud=('budget_usd','sum') if has_bud else ('actual_revenue_usd','count')).reset_index().sort_values('rev',ascending=False)
        result['complex_ranking'] = [{"rank": i+1, "complex": r.complex, "revenue_usd": round(r.rev,2),
            "budget_achievement_pct": round(r.rev/r.bud*100,1) if has_bud and r.bud>0 else None}
            for i,r in enumerate(cx.itertuples())]
    if by in ('brand','both'):
        br = dff.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False)
        result['brand_ranking'] = [{"rank": i+1, "brand": b, "revenue_usd": round(v,2)}
            for i,(b,v) in enumerate(br.items())]
    return result

TOOL_FUNC_MAP = {
    "q_revenue_by_date":          _tool_q_revenue_by_date,
    "q_revenue_by_period":        _tool_q_revenue_by_period,
    "q_revenue_by_month":         _tool_q_revenue_by_month,
    "q_revenue_by_complex":       _tool_q_revenue_by_complex,
    "q_revenue_by_brand":         _tool_q_revenue_by_brand,
    "q_complex_brand":            _tool_q_complex_brand,
    "q_budget_achievement":       _tool_q_budget_achievement,
    "q_customer_metrics":         _tool_q_customer_metrics,
    "q_prior_period_comparison":  _tool_q_prior_period_comparison,
    "q_daily_trend":              _tool_q_daily_trend,
    "q_rankings":                 _tool_q_rankings,
}

from datetime import date as _date_cls
_TODAY = _date_cls.today().strftime("%d %B %Y")

SYSTEM_PROMPT = f"""You are the Savanna QSR Intelligence Assistant, built by Netrisyl Insights.
Today is {_TODAY}. Data covers January 2023 to January 2024.
Complexes: Westgate Mall, City Centre, Eastpark, Northgate.
Brands: Flame & Grill, Pie Palace, Chill Creamery, Sizzle Wings.

Rules:
- ALWAYS use a tool to fetch real figures. NEVER invent numbers.
- Use q_revenue_by_date for specific date questions.
- Use q_revenue_by_month for month-level questions.
- Use q_budget_achievement when asked about underperformance, targets, or which sites need attention.
- Quote exact figures from tool results. Revenue in USD ($). Be concise and professional.
- If data is not available for a query, say so clearly.
- The user may have applied a date filter — your tools already reflect that filtered dataset.
"""

def chat(message, history, date_from, date_to):
    if not message or not message.strip(): return ""
    dff = filter_df(date_from, date_to)
    if dff.empty: return "No data loaded for the selected date range. Please adjust the filter or click Refresh Data."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or []):
        if isinstance(h, (list, tuple)) and len(h) == 2:
            if h[0]: messages.append({"role": "user",      "content": str(h[0])})
            if h[1]: messages.append({"role": "assistant",  "content": str(h[1])})
    messages.append({"role": "user", "content": message})

    try:
        # Step 1: GPT decides which tool(s) to call
        resp = _http.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": messages,
                  "tools": TOOLS, "tool_choice": "auto", "temperature": 0})
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]

        if not msg.get("tool_calls"):
            # GPT answered directly without a tool
            return msg.get("content", "I couldn't find an answer to that.")

        # Step 2: Execute each tool call with the actual dataframe
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": msg["tool_calls"]})
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn = TOOL_FUNC_MAP.get(fn_name)
            try:
                args = _json.loads(tc["function"]["arguments"] or "{}")
                result = fn(dff, **args) if fn else {"error": f"Unknown tool: {fn_name}"}
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": _json.dumps(result)})

        # Step 3: GPT formats the final answer from tool results
        final = _http.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": messages, "temperature": 0})
        final.raise_for_status()
        return final.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ Error: {e}"

def refresh_and_apply(date_from, date_to):
    _cache['loaded_at'] = 0
    _ = get_df()  # force reload
    dff = filter_df(date_from, date_to)
    full = get_df()
    if dff.empty:
        return build_kpi_html(dff), f"⚠️ No data for selected range — {len(full):,} total rows available (Jan 2023 – Jan 2024)"
    return build_kpi_html(dff), f"✅ {len(full):,} rows total · Showing {len(dff):,} rows for selected range"




# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container {
    font-family: 'Inter', 'Helvetica Neue', system-ui, sans-serif !important;
    max-width: 1500px !important;
    margin: 0 auto !important;
}
/* Hero header — exact Stores Intelligence style */
#savanna-hero {
    background: linear-gradient(135deg, #1B2A4E 0%, #2C4170 100%);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 16px;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    box-shadow: 0 8px 24px rgba(27,42,78,0.18);
    position: relative;
    overflow: hidden;
}
#savanna-hero::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #C9A55C 0%, #E4CC8E 50%, #C9A55C 100%);
}
#savanna-hero .brand-tag {
    font-size: 0.75em;
    color: #C9A55C;
    letter-spacing: 3px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 4px;
}
#savanna-hero h2 { font-size: 1.7em !important; font-weight: 700 !important;
    margin: 0 0 2px !important; color: white !important; }
#savanna-hero .tagline { font-size: 0.88em; color: #cbd5e1; margin: 0; }
#savanna-hero .ni-label { font-size: 0.65em; letter-spacing: 2px;
    color: #C9A55C; font-weight: 700; text-transform: uppercase; }
/* KPI pills */
.kpi-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.kpi-pill { background: rgba(255,255,255,0.10); border: 1px solid rgba(201,165,92,0.35);
    border-radius: 8px; padding: 7px 14px; text-align: center; min-width: 90px; }
.kpi-pill .kpi-val { color: #C9A55C; font-size: 15px; font-weight: 700; }
.kpi-pill .kpi-lbl { color: #94a3b8; font-size: 10px; margin-top: 2px; }
/* Filter bar */
.filter-bar { background: white; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 12px; }
/* Tab nav — Stores Intelligence style */
.tab-nav, div[role="tablist"] {
    background: white !important;
    border-bottom: 2px solid #e5e7eb !important;
    padding: 0 8px !important;
}
button[class*="tab-"], div[role="tablist"] button {
    color: #6b7280 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 10px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: Inter, Arial, sans-serif !important;
}
button[class*="tab-"]:hover { color: #1B2A4E !important; }
button[class*="tab-"][class*="selected"],
div[role="tablist"] button[aria-selected="true"] {
    color: #1B2A4E !important;
    border-bottom: 3px solid #C9A55C !important;
    font-weight: 700 !important;
}
/* Primary buttons — navy + white */
button.primary, button[variant="primary"] {
    background: #1B2A4E !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: Inter, Arial, sans-serif !important;
    font-size: 13px !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: #0F1A35 !important;
}
/* Secondary buttons */
button.secondary, button[variant="secondary"] {
    background: white !important;
    color: #1B2A4E !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    background: #1B2A4E !important;
    color: white !important;
    border-color: #1B2A4E !important;
}
/* Inputs */
.gradio-container input, .gradio-container textarea {
    background: white !important;
    color: #1f2937 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-family: Inter, Arial, sans-serif !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: #1B2A4E !important;
    box-shadow: 0 0 0 3px rgba(27,42,78,0.08) !important;
    outline: none !important;
}
/* Labels */
.gradio-container label, .gradio-container .label-wrap span {
    color: #374151 !important;
    font-weight: 500 !important;
}
/* Blocks */
.gradio-container .block, .gradio-container .form {
    background: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
}
/* Dropdowns */
ul[role="listbox"] { background: white !important; border: 1px solid #1B2A4E !important; border-radius: 8px !important; }
ul[role="listbox"] li { color: #1f2937 !important; background: white !important; }
ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
    background: #1B2A4E !important; color: white !important; }
/* Radio buttons */
input[type="radio"] { accent-color: #1B2A4E !important; }
/* Chat */
div[class*="chatbot"], .chatbot { background: #fafafa !important; border-radius: 12px !important; }
/* Markdown */
.gradio-container .prose { color: #1f2937 !important; }
.gradio-container .prose strong { color: #1B2A4E !important; }
.gradio-container .prose h3 { color: #1B2A4E !important; }
.gradio-container .prose th { background: #f8f9fa !important; color: #1B2A4E !important; }
.gradio-container .prose td { border-color: #e5e7eb !important; color: #374151 !important; }
/* Filter message */
#filter-msg { color: #374151 !important; font-size: 12px !important; }
/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #C9A55C; border-radius: 4px; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#1B2A4E",
    button_primary_background_fill_hover="#0F1A35",
    button_primary_text_color="white",
    body_background_fill="#F7F3EC",
    block_background_fill="white",
    block_border_color="#e5e7eb",
)

with gr.Blocks(title="Savanna QSR Intelligence | Netrisyl Insights", theme=theme, css=CSS) as demo:

    kpi_header = gr.HTML(value=build_kpi_html())

    # ── Global date filter ────────────────────────────────────────────────────
    # Set default dates from the loaded dataset
    _init_df = get_df()
    _d_from  = _init_df['date'].min().strftime('%Y-%m-%d') if not _init_df.empty else ""
    _d_to    = _init_df['date'].max().strftime('%Y-%m-%d') if not _init_df.empty else ""

    with gr.Row():
        date_from = gr.Textbox(label="From Date", value=_d_from, placeholder="YYYY-MM-DD", scale=2)
        date_to   = gr.Textbox(label="To Date",   value=_d_to,   placeholder="YYYY-MM-DD", scale=2)
        apply_btn = gr.Button("Apply Filter", variant="primary", scale=1)
        refresh_btn = gr.Button("Refresh Data", variant="secondary", scale=1)
        filter_msg = gr.Markdown()

    with gr.Tabs():

        # ── Dashboard ─────────────────────────────────────────────────────────
        with gr.TabItem("📊 Dashboard"):
            with gr.Row():
                ch1 = gr.Plot(show_label=False)
                ch2 = gr.Plot(show_label=False)
            with gr.Row():
                ch3 = gr.Plot(show_label=False)
                ch4 = gr.Plot(show_label=False)
            with gr.Row():
                ch5 = gr.Plot(show_label=False)
                ch6 = gr.Plot(show_label=False)

        # ── Forecast ──────────────────────────────────────────────────────────
        with gr.TabItem("📈 Forecast"):
            gr.HTML(f"<p style='color:#7fb3d3;font-size:12px;margin:8px 0;'>Day-of-week adjusted moving average · 5% growth · Uses the date range selected above.</p>")
            with gr.Row():
                seg_type = gr.Radio(choices=['Overall','By Complex','By Brand','Complex × Brand'],
                                    value='Overall', label="Segment Type")
                seg_name = gr.Dropdown(choices=['All Complexes & Brands'],
                                       value='All Complexes & Brands', label="Segment", interactive=True)
                horizon  = gr.Radio(choices=["7","14","30","60"], value="30", label="Horizon (days)")
                fc_btn   = gr.Button("⚡ Generate Forecast", variant="primary")
            fc_chart   = gr.Plot(show_label=False)
            fc_summary = gr.Markdown()

        # ── Chat ──────────────────────────────────────────────────────────────
        with gr.TabItem("💬 Intelligence Chat"):
            gr.HTML(f"""<div style="background:#0d1628;border:1px solid {C_GRID};border-radius:8px;
                padding:12px 16px;margin:8px 0 12px;">
                <p style="color:{C_GOLD};font-weight:700;margin:0 0 5px;font-size:13px;">
                    💡 Ask anything about Savanna QSR Group operations</p>
                <p style="color:#7fb3d3;font-size:12px;margin:0;">
                    "Which complex is below budget?" · "Compare prior month revenue" ·
                    "What's the 7-day trend?" · "Avg spend per customer?" ·
                    "Rank all brands" · "Which site needs attention?"</p></div>""")
            chatbot_box = gr.Chatbot(height=440, show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Ask about operations...",
                                        show_label=False, scale=9, container=False)
                send_btn   = gr.Button("Send ➤", variant="primary", scale=1)

    gr.HTML(f'<div style="text-align:center;margin-top:14px;padding:10px;border-top:1px solid {C_GRID};"><p style="color:{C_GOLD};font-size:10px;font-weight:700;letter-spacing:2px;margin:0;">NETRISYL INSIGHTS</p><p style="color:#4a6a9e;font-size:10px;margin:4px 0 0;">Data · Analytics · Intelligence · <a href="https://netrisyl.com" style="color:#7fb3d3;">netrisyl.com</a></p></div>')

    # ── Wiring ────────────────────────────────────────────────────────────────
    dash_outputs = [kpi_header, ch1, ch2, ch3, ch4, ch5, ch6]

    def apply_filter(date_from, date_to):
        dff = filter_df(date_from, date_to)
        full = get_df()
        if dff.empty:
            e = go.Figure().update_layout(**dark_layout("No data for selected range"))
            msg = f"⚠️ No data for selected dates — available range: Jan 2023 – Jan 2024 ({len(full):,} rows)"
            return [build_kpi_html(dff), msg] + [e]*6
        dash = build_dashboard(date_from, date_to)
        msg = f"✅ Showing {len(dff):,} of {len(full):,} rows"
        return [dash[0], msg] + list(dash[1:])

    apply_btn.click(fn=apply_filter,
                    inputs=[date_from, date_to],
                    outputs=[kpi_header, filter_msg, ch1, ch2, ch3, ch4, ch5, ch6])

    refresh_btn.click(fn=refresh_and_apply, inputs=[date_from, date_to],
                      outputs=[kpi_header, filter_msg])

    seg_type.change(fn=update_seg_choices, inputs=[seg_type], outputs=[seg_name])
    fc_btn.click(fn=generate_forecast,
                 inputs=[seg_type, seg_name, horizon, date_from, date_to],
                 outputs=[fc_chart, fc_summary])

    def respond(message, history, df, dt):
        if not message or not message.strip(): return history or [], ""
        reply = chat(message, history or [], df, dt)
        h = list(history or [])
        h.append((message, reply))
        return h, ""

    send_btn.click(fn=respond, inputs=[chat_input, chatbot_box, date_from, date_to],
                   outputs=[chatbot_box, chat_input])
    chat_input.submit(fn=respond, inputs=[chat_input, chatbot_box, date_from, date_to],
                      outputs=[chatbot_box, chat_input])

    # Auto-load dashboard on startup with full dataset
    demo.load(
        fn=lambda: list(build_dashboard("", "")),
        inputs=[],
        outputs=[kpi_header, ch1, ch2, ch3, ch4, ch5, ch6]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
