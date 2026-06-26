import os
import json
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# GOOGLE SHEETS CONNECTION
# ═══════════════════════════════════════════════════════════════
from google.oauth2 import service_account
from googleapiclient.discovery import build

SHEET_ID   = os.environ.get("GOOGLE_SHEET_ID", "1HhGAibq90EMUj3m_GnVKLaWpCLxRC9GD")
CREDS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
GROWTH_RATE = 0.05

def get_sheets_service():
    creds_dict = json.loads(CREDS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )
    return build("sheets", "v4", credentials=creds)

def load_data():
    try:
        service = get_sheets_service()
        result  = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range="Daily_Input!A:U"
        ).execute()
        rows = result.get("values", [])
        if not rows or len(rows) < 2:
            return pd.DataFrame()
        headers = rows[0]
        data    = rows[1:]
        df = pd.DataFrame(data, columns=headers[:len(data[0])] if data else headers)

        # Clean & type-cast
        numeric_cols = ['budget_usd','actual_revenue_usd','prior_month_actual',
                        'prior_year_actual','customer_count','counters_open',
                        'variance_vs_budget','avg_spend_per_cust','revenue_per_counter',
                        'vs_prior_month_var','vs_prior_year_var','is_holiday']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$','').str.replace(',',''), errors='coerce').fillna(0)

        pct_cols = ['variance_pct','vs_prior_month_pct','vs_prior_year_pct']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('%',''), errors='coerce').fillna(0) / 100

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        df = df.dropna(subset=['date','actual_revenue_usd']).copy()
        df = df[df['actual_revenue_usd'] > 0].copy()
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"Sheet load error: {e}")
        return pd.DataFrame()

# ── Auto-refresh cache (reloads every 5 minutes) ─────────────
import time as _time
_cache = {'df': pd.DataFrame(), 'loaded_at': 0}
CACHE_TTL = 300  # 5 minutes

def get_df():
    now = _time.time()
    if now - _cache['loaded_at'] > CACHE_TTL or _cache['df'].empty:
        fresh = load_data()
        if not fresh.empty:
            _cache['df'] = fresh
            _cache['loaded_at'] = now
            print(f"Data refreshed — {len(fresh)} rows")
    return _cache['df']

# Initial load
print("Loading data from Google Sheets...")
df = get_df()
print(f"Loaded {len(df)} rows" if len(df) else "No data loaded — will retry on first request")

COMPLEXES = ['Westgate Mall','City Centre','Eastpark','Northgate']
BRANDS    = ['Flame & Grill','Pie Palace','Chill Creamery','Sizzle Wings']
REGIONS   = {'Westgate Mall':'Harare','City Centre':'Harare','Eastpark':'Bulawayo','Northgate':'Bulawayo'}

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))

# ── Helper ────────────────────────────────────────────────────
def hex_rgba(h, a=0.15):
    h = h.lstrip('#')
    r,g,b = tuple(int(h[i:i+2],16) for i in (0,2,4))
    return f'rgba({r},{g},{b},{a})'

def fmt(v):
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"

DARK = dict(
    paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
    font=dict(color='#c8d8f0',family='Arial',size=11),
    xaxis=dict(gridcolor='#1a3a6e',linecolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
    yaxis=dict(gridcolor='#1a3a6e',linecolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
    legend=dict(bgcolor='rgba(10,22,40,0.85)',bordercolor='#1a3a6e',
                borderwidth=1,font=dict(color='#c8d8f0')),
)

def dark(title, height=360, **kwargs):
    d = dict(DARK); d.update(kwargs)
    d['title'] = dict(text=title, font=dict(color='#c9a84c',size=14))
    d['height'] = height
    return d

COLORS = ['#c9a84c','#2ecc71','#e74c3c','#9b59b6','#1abc9c','#f39c12','#3498db','#e67e22']

def refresh_data():
    _cache['loaded_at'] = 0  # force reload
    fresh = get_df()
    return f"✅ Refreshed — {len(fresh)} rows loaded"


# ═══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════

def build_dashboard(period):
    df = get_df()
    if df.empty:
        empty = go.Figure().update_layout(**dark("No data available"))
        return [empty]*6

    dff = df.copy()
    latest = dff['date'].max()

    if period == 'Last 7 Days':
        dff = dff[dff['date'] >= latest - timedelta(days=7)]
    elif period == 'Last 30 Days':
        dff = dff[dff['date'] >= latest - timedelta(days=30)]
    elif period == 'Last 90 Days':
        dff = dff[dff['date'] >= latest - timedelta(days=90)]
    elif period == 'MTD':
        dff = dff[(dff['date'].dt.month == latest.month) & (dff['date'].dt.year == latest.year)]

    # ── 1. Revenue by Complex ──────────────────────────────────
    cx_rev = dff.groupby('complex').agg(
        actual=('actual_revenue_usd','sum'),
        budget=('budget_usd','sum'),
        sdlm=('prior_month_actual','sum'),
        sdly=('prior_year_actual','sum')
    ).reset_index().sort_values('actual', ascending=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Budget',x=cx_rev['budget'],y=cx_rev['complex'],
        orientation='h',marker_color='#1a3a6e',opacity=0.7))
    fig1.add_trace(go.Bar(name='Actual',x=cx_rev['actual'],y=cx_rev['complex'],
        orientation='h',marker_color='#c9a84c',
        text=[fmt(v) for v in cx_rev['actual']],textposition='outside',
        textfont=dict(color='#c8d8f0',size=11)))
    fig1.update_layout(**dark("Revenue by Complex — Actual vs Budget",height=300,
        barmode='overlay',margin=dict(l=140,r=80,t=50,b=40)),
        xaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')))

    # ── 2. Revenue by Brand ────────────────────────────────────
    br_rev = dff.groupby('brand').agg(
        actual=('actual_revenue_usd','sum'),
        budget=('budget_usd','sum')
    ).reset_index().sort_values('actual',ascending=True)

    fig2 = go.Figure()
    for i,row in enumerate(br_rev.itertuples()):
        pct = (row.actual-row.budget)/row.budget*100 if row.budget else 0
        color = '#2ecc71' if pct>=0 else '#e74c3c'
        fig2.add_trace(go.Bar(name=row.brand,x=[row.actual],y=[row.brand],
            orientation='h',marker_color=COLORS[i%len(COLORS)],
            text=[f"{fmt(row.actual)} ({pct:+.1f}%)"],textposition='outside',
            textfont=dict(color='#c8d8f0',size=11),showlegend=False))
    fig2.update_layout(**dark("Revenue by Brand — with Budget Achievement",height=280,
        margin=dict(l=140,r=120,t=50,b=40)),
        xaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')))

    # ── 3. Daily Revenue Trend ─────────────────────────────────
    daily = dff.groupby('date').agg(
        actual=('actual_revenue_usd','sum'),
        budget=('budget_usd','sum')
    ).reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=daily['date'],y=daily['budget'],name='Budget',
        line=dict(color='#1a3a6e',width=1.5,dash='dot'),mode='lines'))
    fig3.add_trace(go.Scatter(x=daily['date'],y=daily['actual'],name='Actual',
        line=dict(color='#c9a84c',width=2.5),mode='lines',
        fill='tozeroy',fillcolor='rgba(201,168,76,0.08)'))
    fig3.update_layout(**dark("Daily Revenue Trend — Actual vs Budget",height=360,
        margin=dict(l=70,r=40,t=50,b=50),hovermode='x unified'),
        xaxis=dict(title="Date",gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')))

    # ── 4. Avg Spend Per Customer ──────────────────────────────
    avs = dff[dff['customer_count']>0].groupby('complex').apply(
        lambda x: (x['actual_revenue_usd'].sum()/x['customer_count'].sum())
    ).reset_index(name='avg_spend').sort_values('avg_spend',ascending=True)

    fig4 = go.Figure(go.Bar(
        x=avs['avg_spend'],y=avs['complex'],orientation='h',
        marker_color='#2ecc71',
        text=[f"${v:.2f}" for v in avs['avg_spend']],
        textposition='outside',textfont=dict(color='#c8d8f0',size=11)
    ))
    fig4.update_layout(**dark("Avg Spend Per Customer by Complex",height=280,
        margin=dict(l=140,r=80,t=50,b=40)),
        xaxis=dict(title="Avg Spend (USD)",tickprefix="$",tickformat=",.2f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')))

    # ── 5. Budget Achievement Heatmap ─────────────────────────
    heat = dff.groupby(['complex','brand']).apply(
        lambda x: (x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100)
        if x['budget_usd'].sum()>0 else 0
    ).reset_index(name='achievement')
    heat_pivot = heat.pivot(index='complex',columns='brand',values='achievement')

    fig5 = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale=[[0,'#c0392b'],[0.8,'#f39c12'],[1,'#2ecc71']],
        zmid=100, zmin=70, zmax=120,
        text=[[f"{v:.1f}%" for v in row] for row in heat_pivot.values],
        texttemplate="%{text}",
        textfont=dict(color='white',size=12),
        colorbar=dict(tickfont=dict(color='#c8d8f0'),title=dict(text='%',font=dict(color='#c8d8f0')))
    ))
    fig5.update_layout(**dark("Budget Achievement % — Complex × Brand",height=280,
        margin=dict(l=130,r=60,t=50,b=80)),
        xaxis=dict(tickfont=dict(color='#c8d8f0'),tickangle=15),
        yaxis=dict(tickfont=dict(color='#c8d8f0')))

    # ── 6. SDLM vs SDLY vs Actual ─────────────────────────────
    comp = dff.groupby('complex').agg(
        actual=('actual_revenue_usd','sum'),
        sdlm=('prior_month_actual','sum'),
        sdly=('prior_year_actual','sum')
    ).reset_index()

    fig6 = go.Figure()
    for col,name,color in [
        ('sdly','Prior Year','#4a6a9e'),
        ('sdlm','Prior Month','#7fb3d3'),
        ('actual','This Period','#c9a84c'),
    ]:
        fig6.add_trace(go.Bar(
            name=name, x=comp['complex'], y=comp[col],
            marker_color=color,
            text=[fmt(v) for v in comp[col]],
            textposition='outside',textfont=dict(color='#c8d8f0',size=10)
        ))
    fig6.update_layout(**dark("Period Comparison — Actual vs Prior Month vs Prior Year",height=340,
        margin=dict(l=60,r=40,t=50,b=60),barmode='group',hovermode='x unified'),
        xaxis=dict(tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')))

    return fig1, fig2, fig3, fig4, fig5, fig6


# ═══════════════════════════════════════════════════════════════
# TAB 2 — FORECASTING
# ═══════════════════════════════════════════════════════════════

def generate_forecast(segment_type, segment_name, horizon):
    df = get_df()
    if df.empty:
        return go.Figure().update_layout(**dark("No data available")), ""

    horizon = int(horizon)

    # Filter segment
    if segment_type == 'Overall':
        seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
    elif segment_type == 'By Complex':
        seg = df[df['complex']==segment_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
    elif segment_type == 'By Brand':
        seg = df[df['brand']==segment_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
    else:  # Complex × Brand
        parts = segment_name.split(' | ')
        if len(parts)==2:
            seg = df[(df['complex']==parts[0])&(df['brand']==parts[1])].groupby('date')['actual_revenue_usd'].sum().reset_index()
        else:
            seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()

    seg.columns = ['date','revenue']
    seg = seg.sort_values('date').reset_index(drop=True)

    if len(seg) < 7:
        return go.Figure().update_layout(**dark("Insufficient data for forecast (need 7+ days)")), ""

    # ── Moving average forecast ────────────────────────────────
    last_date = seg['date'].max()
    ma7  = seg['revenue'].tail(7).mean()
    ma14 = seg['revenue'].tail(14).mean() if len(seg)>=14 else ma7
    ma30 = seg['revenue'].tail(30).mean() if len(seg)>=30 else ma7

    # Choose MA based on horizon
    if horizon <= 7:   base_avg = ma7
    elif horizon <= 14: base_avg = ma14
    else:               base_avg = ma30

    # Apply growth rate and DOW seasonality
    forecast_dates  = [last_date + timedelta(days=i+1) for i in range(horizon)]
    DOW_MULT = {0:0.85,1:0.90,2:0.92,3:0.95,4:1.05,5:1.20,6:1.15}
    forecast_values = [round(base_avg * DOW_MULT[d.weekday()] * (1+GROWTH_RATE), 2)
                       for d in forecast_dates]

    # Confidence band ±15%
    upper = [v*1.15 for v in forecast_values]
    lower = [v*0.85 for v in forecast_values]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=seg['date'], y=seg['revenue'],
        name='Historical Revenue',
        line=dict(color='#4a7aae',width=1.2),
        mode='lines', opacity=0.85
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=forecast_dates+forecast_dates[::-1],
        y=upper+lower[::-1],
        fill='toself', fillcolor='rgba(201,168,76,0.18)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band', hoverinfo='skip'
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        name=f'{horizon}-Day Forecast',
        line=dict(color='#c9a84c',width=2.5,dash='dash'),
        mode='lines+markers',
        marker=dict(size=4,color='#c9a84c')
    ))

    # Forecast start line
    fig.add_vline(x=last_date, line_dash='dot', line_color='#c9a84c', opacity=0.7)
    fig.add_annotation(x=last_date, y=1, yref='paper',
        text='▶ Forecast', showarrow=False,
        font=dict(color='#c9a84c',size=11), xanchor='left',
        bgcolor='rgba(10,22,40,0.75)', borderpad=3)

    layout = dark(f"{segment_name} — {horizon}-Day Revenue Forecast", height=480)
    layout['xaxis'] = dict(title='Date',gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0'))
    layout['yaxis'] = dict(title='Revenue (USD)',tickprefix='$',tickformat=',.0f',
                           gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0'))
    layout['hovermode'] = 'x unified'
    layout['legend'] = dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,
                             bgcolor='rgba(10,22,40,0.8)',bordercolor='#1a3a6e',
                             borderwidth=1,font=dict(color='#c8d8f0'))
    layout['margin'] = dict(l=70,r=30,t=70,b=50)
    fig.update_layout(**layout)

    total = sum(forecast_values)
    avg   = total/horizon
    peak  = max(forecast_values)
    peak_d= forecast_dates[forecast_values.index(peak)].strftime('%b %d, %Y')
    low   = min(forecast_values)
    low_d = forecast_dates[forecast_values.index(low)].strftime('%b %d, %Y')

    note = "7-day" if horizon<=7 else ("14-day" if horizon<=14 else "30-day")
    summary = f"""**{horizon}-Day Forecast — {segment_name}**

| Metric | Value |
|---|---|
| Forecast Method | **{note} moving average × {(1+GROWTH_RATE):.0%} growth + DOW adjustment** |
| Predicted Total Revenue | **${total:,.2f}** |
| Average Daily Revenue | **${avg:,.2f}** |
| Peak Day | **${peak:,.2f}** on {peak_d} |
| Lowest Day | **${low:,.2f}** on {low_d} |
| Forecast Period | {forecast_dates[0].strftime('%b %d')} → {forecast_dates[-1].strftime('%b %d, %Y')} |
| Confidence Band | **±15%** around central forecast |

> *Weekend days forecast higher due to day-of-week adjustment. Budget growth rate: {GROWTH_RATE:.0%}.*
"""
    return fig, summary

def update_forecast_segments(seg_type):
    if seg_type == 'Overall':
        return gr.update(choices=['All Complexes & Brands'],value='All Complexes & Brands')
    elif seg_type == 'By Complex':
        return gr.update(choices=COMPLEXES,value=COMPLEXES[0])
    elif seg_type == 'By Brand':
        return gr.update(choices=BRANDS,value=BRANDS[0])
    else:
        opts = [f"{cx} | {br}" for cx in COMPLEXES for br in BRANDS]
        return gr.update(choices=opts,value=opts[0])


# ═══════════════════════════════════════════════════════════════
# TAB 3 — INTELLIGENCE CHAT
# ═══════════════════════════════════════════════════════════════

# ── Deterministic query functions ─────────────────────────────
def q_total_revenue(period_days=None):
    df = get_df()
    d = df.copy()
    if period_days:
        cutoff = d['date'].max() - timedelta(days=period_days)
        d = d[d['date']>=cutoff]
    total = d['actual_revenue_usd'].sum()
    return f"Total revenue: ${total:,.2f}"

def q_top_complex():
    df = get_df()
    top = df.groupby('complex')['actual_revenue_usd'].sum().idxmax()
    val = df.groupby('complex')['actual_revenue_usd'].sum().max()
    return f"Top complex: {top} with ${val:,.2f}"

def q_top_brand():
    df = get_df()
    top = df.groupby('brand')['actual_revenue_usd'].sum().idxmax()
    val = df.groupby('brand')['actual_revenue_usd'].sum().max()
    return f"Top brand: {top} with ${val:,.2f}"

def q_underperforming(threshold=0.80):
    df = get_df()
    latest = df['date'].max()
    recent = df[df['date'] >= latest - timedelta(days=7)]
    perf = recent.groupby(['complex','brand']).apply(
        lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()
        if x['budget_usd'].sum()>0 else 1
    ).reset_index(name='achievement')
    under = perf[perf['achievement']<threshold]
    if under.empty:
        return "No underperforming sites in the last 7 days — all complexes/brands above 80% of budget."
    lines = [f"- {r.complex} / {r.brand}: {r.achievement*100:.1f}% of budget" for r in under.itertuples()]
    return "Underperforming (below 80% budget, last 7 days):\n" + "\n".join(lines)

def q_avg_spend():
    df = get_df()
    avs = df[df['customer_count']>0].groupby('complex').apply(
        lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()
    ).reset_index(name='avg')
    lines = [f"- {r.complex}: ${r.avg:.2f}/customer" for r in avs.sort_values('avg',ascending=False).itertuples()]
    return "Average spend per customer by complex:\n" + "\n".join(lines)

def q_sdlm_comparison():
    df = get_df()
    total_act  = df['actual_revenue_usd'].sum()
    total_sdlm = df['prior_month_actual'].sum()
    if total_sdlm == 0:
        return "Prior month data not yet available — need 30+ days of history."
    diff = total_act - total_sdlm
    pct  = diff/total_sdlm*100
    return f"vs Prior Month (SDLM): ${total_act:,.2f} vs ${total_sdlm:,.2f} — {'+' if diff>=0 else ''}{diff:,.2f} ({pct:+.1f}%)"

def q_sdly_comparison():
    df = get_df()
    total_act  = df['actual_revenue_usd'].sum()
    total_sdly = df['prior_year_actual'].sum()
    if total_sdly == 0:
        return "Prior year data not yet available — need 365+ days of history."
    diff = total_act - total_sdly
    pct  = diff/total_sdly*100
    return f"vs Prior Year (SDLY): ${total_act:,.2f} vs ${total_sdly:,.2f} — {'+' if diff>=0 else ''}{diff:,.2f} ({pct:+.1f}%)"

def q_best_day():
    df = get_df()
    dow_rev = df.groupby(df['date'].dt.day_name())['actual_revenue_usd'].mean()
    best    = dow_rev.idxmax()
    return f"Best revenue day of week: {best} (avg ${dow_rev[best]:,.2f}/day)"

def q_holiday_impact():
    df = get_df()
    if 'is_holiday' not in df.columns:
        return "Holiday data not available."
    hol  = df[df['is_holiday']==1]['actual_revenue_usd'].mean()
    norm = df[df['is_holiday']==0]['actual_revenue_usd'].mean()
    if hol==0: return "No holiday data recorded yet."
    diff = (hol-norm)/norm*100
    return f"Holiday vs normal day: ${hol:,.2f} vs ${norm:,.2f} avg per row ({diff:+.1f}%)"

def q_moving_average(days=7):
    df = get_df()
    latest = df['date'].max()
    cutoff = latest - timedelta(days=days)
    recent = df[df['date']>=cutoff]
    ma = recent.groupby('date')['actual_revenue_usd'].sum().mean()
    return f"{days}-day moving average daily revenue: ${ma:,.2f}"

def q_complex_brand(cx, brd):
    df = get_df()
    sub = df[(df['complex']==cx)&(df['brand']==brd)]
    if sub.empty: return f"No data for {cx} / {brd}"
    total = sub['actual_revenue_usd'].sum()
    avg   = sub['actual_revenue_usd'].mean()
    bgt   = sub['budget_usd'].sum()
    ach   = total/bgt*100 if bgt>0 else 0
    return (f"{cx} / {brd}: ${total:,.2f} total | ${avg:,.2f}/day avg | "
            f"Budget achievement: {ach:.1f}%")

def q_revenue_per_counter():
    df = get_df()
    rpc = df.groupby('complex').apply(
        lambda x: x['actual_revenue_usd'].sum()/x['counters_open'].sum()
        if x['counters_open'].sum()>0 else 0
    ).reset_index(name='rpc')
    lines = [f"- {r.complex}: ${r.rpc:,.2f}/counter" for r in rpc.sort_values('rpc',ascending=False).itertuples()]
    return "Revenue per counter by complex:\n" + "\n".join(lines)

# ── Intent router ──────────────────────────────────────────────
def route_intent(message):
    df = get_df()
    m = message.lower()
    if any(x in m for x in ['underperform','below budget','struggling','worst']):
        return q_underperforming()
    if any(x in m for x in ['top complex','best complex','leading complex','highest complex']):
        return q_top_complex()
    if any(x in m for x in ['top brand','best brand','leading brand','highest brand']):
        return q_top_brand()
    if any(x in m for x in ['avg spend','average spend','spend per customer','per customer']):
        return q_avg_spend()
    if any(x in m for x in ['prior month','sdlm','last month','vs month']):
        return q_sdlm_comparison()
    if any(x in m for x in ['prior year','sdly','last year','vs year','year on year','yoy']):
        return q_sdly_comparison()
    if any(x in m for x in ['best day','day of week','busiest day','peak day']):
        return q_best_day()
    if any(x in m for x in ['holiday','public holiday']):
        return q_holiday_impact()
    if '7-day' in m or '7 day' in m or 'weekly average' in m:
        return q_moving_average(7)
    if '30-day' in m or '30 day' in m or 'monthly average' in m:
        return q_moving_average(30)
    if any(x in m for x in ['per counter','counter performance','revenue per counter']):
        return q_revenue_per_counter()
    if any(x in m for x in ['total revenue','overall revenue','total sales']):
        return q_total_revenue()
    # Complex × brand specific
    for cx in COMPLEXES:
        for brd in BRANDS:
            if cx.lower() in m and brd.lower() in m:
                return q_complex_brand(cx, brd)
    # Complex specific
    for cx in COMPLEXES:
        if cx.lower() in m:
            sub = df[df['complex']==cx]
            return f"{cx}: ${sub['actual_revenue_usd'].sum():,.2f} total | ${sub['actual_revenue_usd'].mean():,.2f}/day avg"
    # Brand specific
    for brd in BRANDS:
        if brd.lower() in m:
            sub = df[df['brand']==brd]
            return f"{brd}: ${sub['actual_revenue_usd'].sum():,.2f} total | ${sub['actual_revenue_usd'].mean():,.2f}/day avg"
    return None

def build_system_prompt():
    df = get_df()
    if df.empty:
        return "You are a QSR intelligence assistant. No data is currently loaded."

    latest  = df['date'].max()
    total   = df['actual_revenue_usd'].sum()
    top_cx  = df.groupby('complex')['actual_revenue_usd'].sum().idxmax()
    top_br  = df.groupby('brand')['actual_revenue_usd'].sum().idxmax()
    date_range = f"{df['date'].min().strftime('%b %d, %Y')} to {df['date'].max().strftime('%b %d, %Y')}"
    ma7 = df[df['date']>=latest-timedelta(days=7)].groupby('date')['actual_revenue_usd'].sum().mean()

    return f"""You are an intelligence assistant for Savanna QSR Group, a multi-site QSR operator in Zimbabwe.
You have access to daily revenue data from Google Sheets for the period {date_range}.

KEY FACTS:
- Total revenue on record: ${total:,.2f}
- Date range: {date_range}
- Complexes: {', '.join(COMPLEXES)}
- Brands: {', '.join(BRANDS)}
- Top complex: {top_cx}
- Top brand: {top_br}
- 7-day moving avg daily revenue: ${ma7:,.2f}
- Budget growth rate: {GROWTH_RATE:.0%} above rolling average

The app uses a Google Sheet with daily data including: actual revenue, budget (DOW-adjusted rolling avg × 5% growth),
prior month actual (SDLM), prior year actual (SDLY), customer count, counters open.

Answer concisely and professionally. Use $ for all amounts.
If asked about forecasts, explain the moving average + DOW adjustment + 5% growth methodology.
If asked about data limitations, be honest — SDLM needs 30+ days, SDLY needs 365+ days of history."""

SYSTEM_PROMPT = build_system_prompt()

def chat(message, history):
    # Try deterministic router first
    data_answer = route_intent(message)

    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    if history:
        for h in history:
            if isinstance(h,dict):
                messages.append({"role":h["role"],"content":h["content"]})
            elif isinstance(h,(list,tuple)) and len(h)==2:
                if h[0]: messages.append({"role":"user","content":str(h[0])})
                if h[1]: messages.append({"role":"assistant","content":str(h[1])})

    if data_answer:
        user_msg = f"{message}\n\n[DATA FROM SYSTEM]: {data_answer}"
    else:
        user_msg = message

    messages.append({"role":"user","content":user_msg})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",messages=messages,temperature=0.2,max_tokens=600
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

css = """
body,.gradio-container{background:#050d1a!important;font-family:Arial,sans-serif!important;}
.tab-nav{background:#0a1628!important;border-bottom:2px solid #1a3a6e!important;}
button[class*="tab-"]{color:#7fb3d3!important;background:transparent!important;
  border:none!important;border-bottom:3px solid transparent!important;
  padding:10px 18px!important;font-size:13px!important;font-weight:500!important;}
button[class*="tab-"]:hover{color:#ffffff!important;}
button[class*="tab-"][class*="selected"],button[class*="tab-"].selected,
div[role="tablist"] button[aria-selected="true"]{
  color:#c9a84c!important;border-bottom:3px solid #c9a84c!important;font-weight:700!important;}
.gradio-container *{color:#c8d8f0;}
.gradio-container input,.gradio-container textarea,.gradio-container select{
  background:#0a1628!important;color:#c8d8f0!important;border:1px solid #1a3a6e!important;border-radius:6px!important;}
ul[role="listbox"]{background:#0d1b2a!important;border:1px solid #c9a84c!important;border-radius:6px!important;}
ul[role="listbox"] li{color:#fff!important;background:#0d1b2a!important;}
ul[role="listbox"] li:hover,ul[role="listbox"] li[aria-selected="true"]{background:#c9a84c!important;color:#0d1b2a!important;}
button.primary,button[variant="primary"]{background:#c9a84c!important;color:#0a1628!important;
  font-weight:700!important;border:none!important;border-radius:6px!important;}
button.primary:hover{background:#e0be6a!important;}
.gradio-container .block,.gradio-container .form,.gradio-container .panel{
  background:#0a1628!important;border:1px solid #1a3a6e!important;border-radius:8px!important;}
.gradio-container label,.gradio-container .label-wrap span{color:#a8c8f0!important;}
.message.user>div,div[data-testid="user"]>div{
  background:linear-gradient(135deg,#1e2d5e,#162d5a)!important;color:#e8f0ff!important;
  border-radius:18px 18px 4px 18px!important;border:1px solid #2a4a8a!important;}
.message.bot>div,div[data-testid="bot"]>div{
  background:linear-gradient(135deg,#0a1e3d,#112952)!important;color:#fff!important;
  border-radius:18px 18px 18px 4px!important;}
div[class*="chatbot"],.chatbot{background:#040c1a!important;border-radius:12px!important;}
.gradio-container .prose{color:#c8d8f0!important;}
.gradio-container .prose strong{color:#c9a84c!important;}
.gradio-container .prose table{border-color:#1a3a6e!important;}
.gradio-container .prose th{background:#0a1628!important;color:#c9a84c!important;}
.gradio-container .prose td{border-color:#1a3a6e!important;color:#c8d8f0!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{background:#c9a84c;border-radius:4px;}
footer{display:none!important;}
"""

def get_kpis():
    df = get_df()
    if df.empty:
        return "No data loaded", "—", "—", "—", "—"
    latest    = df['date'].max()
    total_rev = df['actual_revenue_usd'].sum()
    total_bud = df['budget_usd'].sum()
    ach       = total_rev/total_bud*100 if total_bud else 0
    top_cx    = df.groupby('complex')['actual_revenue_usd'].sum().idxmax()
    top_br    = df.groupby('brand')['actual_revenue_usd'].sum().idxmax()
    ma7       = df[df['date']>=latest-timedelta(days=7)].groupby('date')['actual_revenue_usd'].sum().mean()
    date_rng  = f"{df['date'].min().strftime('%b %d')} – {latest.strftime('%b %d, %Y')}"
    return date_rng, fmt(total_rev), f"{ach:.1f}%", top_cx, top_br

with gr.Blocks(title="Savanna QSR Intelligence | Netrisyl Insights", css=css) as demo:

    date_rng, total_rev, ach_pct, top_cx, top_br = get_kpis()

    gr.HTML(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
                padding:18px 28px 16px;border-radius:12px;margin-bottom:4px;
                border-left:4px solid #c9a84c;box-shadow:0 4px 20px rgba(0,0,0,0.4);">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
            <div>
                <h1 style="color:#fff;margin:0;font-size:22px;font-weight:700;">
                    🍔 Savanna QSR Intelligence
                </h1>
                <p style="color:#aed6f1;margin:4px 0 0;font-size:13px;">
                    Live Operations Intelligence · Zimbabwe · Powered by Google Sheets
                </p>
            </div>
            <div style="text-align:right;">
                <p style="color:#c9a84c;margin:0;font-size:10px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p>
                <p style="color:#7fb3d3;margin:3px 0 0;font-size:11px;">Data · Analytics · Intelligence</p>
            </div>
        </div>
        <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;">
            <div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;">
                <div style="color:#c9a84c;font-size:18px;font-weight:700;">{total_rev}</div>
                <div style="color:#7fb3d3;font-size:11px;">Total Revenue</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;">
                <div style="color:#c9a84c;font-size:18px;font-weight:700;">{ach_pct}</div>
                <div style="color:#7fb3d3;font-size:11px;">Budget Achievement</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;">
                <div style="color:#c9a84c;font-size:18px;font-weight:700;">{top_cx}</div>
                <div style="color:#7fb3d3;font-size:11px;">Top Complex</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;">
                <div style="color:#c9a84c;font-size:18px;font-weight:700;">{top_br}</div>
                <div style="color:#7fb3d3;font-size:11px;">Top Brand</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;">
                <div style="color:#7fb3d3;font-size:18px;font-weight:700;">{date_rng}</div>
                <div style="color:#7fb3d3;font-size:11px;">Data Range</div>
            </div>
        </div>
    </div>""")

    with gr.Tabs():

        # ── Tab 1: Dashboard ──────────────────────────────────
        with gr.TabItem("📊 Dashboard"):
            with gr.Row():
                period_sel = gr.Radio(
                    choices=['Last 7 Days','Last 30 Days','Last 90 Days','MTD','All Time'],
                    value='All Time', label="Period"
                )
                with gr.Column(scale=0):
                    dash_btn     = gr.Button("📊 Load Dashboard", variant="primary")
                    refresh_btn  = gr.Button("🔄 Refresh Data",   variant="secondary")
                    refresh_msg  = gr.Markdown()

            with gr.Row():
                ch_cx  = gr.Plot(show_label=False)
                ch_br  = gr.Plot(show_label=False)
            with gr.Row():
                ch_trend = gr.Plot(show_label=False)
                ch_avs   = gr.Plot(show_label=False)
            with gr.Row():
                ch_heat  = gr.Plot(show_label=False)
                ch_comp  = gr.Plot(show_label=False)

            dash_btn.click(build_dashboard,[period_sel],
                           [ch_cx,ch_br,ch_trend,ch_avs,ch_heat,ch_comp])
            refresh_btn.click(refresh_data,[],[refresh_msg])

        # ── Tab 2: Forecast ───────────────────────────────────
        with gr.TabItem("📈 Forecast"):
            gr.HTML("<p style='color:#7fb3d3;font-size:12px;margin:10px 0 14px;'>Forecasts use day-of-week adjusted moving averages with 5% growth factor.</p>")
            with gr.Row():
                seg_type = gr.Radio(
                    choices=['Overall','By Complex','By Brand','Complex × Brand'],
                    value='Overall', label="Segment Type"
                )
                seg_name = gr.Dropdown(
                    choices=['All Complexes & Brands'],
                    value='All Complexes & Brands',
                    label="Select Segment", interactive=True
                )
                horizon = gr.Radio(choices=[7,14,30,60], value=30, label="Horizon (days)")
                fc_btn  = gr.Button("⚡ Generate Forecast", variant="primary")

            fc_chart   = gr.Plot(show_label=False)
            fc_summary = gr.Markdown()

            seg_type.change(update_forecast_segments,[seg_type],[seg_name])
            fc_btn.click(generate_forecast,[seg_type,seg_name,horizon],[fc_chart,fc_summary])

        # ── Tab 3: Chat ───────────────────────────────────────
        with gr.TabItem("💬 Intelligence Chat"):
            gr.HTML("""
            <div style="background:#0a1628;border:1px solid #1a3a6e;border-radius:8px;
                        padding:14px 18px;margin:10px 0 14px;">
                <p style="color:#c9a84c;font-weight:700;margin:0 0 6px;font-size:13px;">
                    💡 Ask anything about Savanna QSR Group operations
                </p>
                <p style="color:#7fb3d3;font-size:12px;margin:0;line-height:1.8;">
                    "Which complex is underperforming?" &nbsp;·&nbsp;
                    "What is the avg spend per customer?" &nbsp;·&nbsp;
                    "Compare Westgate Mall vs City Centre" &nbsp;·&nbsp;
                    "How is Flame & Grill trending vs prior month?" &nbsp;·&nbsp;
                    "What is the 7-day moving average?"
                </p>
            </div>""")
            chatbot = gr.ChatInterface(
                fn=chat, title="", type="messages",
                examples=[
                    "Which complex is underperforming this week?",
                    "What is the average spend per customer by complex?",
                    "Which brand generates the most revenue?",
                    "How does this period compare to prior month?",
                    "What is the 7-day moving average daily revenue?",
                    "Which day of the week has the highest revenue?",
                    "Compare Westgate Mall and City Centre performance",
                    "What is the revenue per counter for each complex?",
                    "How did holidays affect revenue?",
                    "Which complex should we prioritize for investment?",
                ]
            )

    gr.HTML("""
    <div style="text-align:center;margin-top:16px;padding:12px;border-top:1px solid #1a3a6e;">
        <p style="color:#c9a84c;font-size:11px;font-weight:700;margin:0;letter-spacing:2px;">NETRISYL INSIGHTS</p>
        <p style="color:#4a6a9e;font-size:11px;margin:5px 0 0;">
            Data · Analytics · Intelligence ·
            <a href="https://netrisyl.com" target="_blank" style="color:#7fb3d3;text-decoration:none;">netrisyl.com</a>
        </p>
        <p style="color:#2a4a6e;font-size:10px;margin:4px 0 0;">
            ⚠️ Data anonymized. All names are fictional. Built for demonstration purposes.
        </p>
    </div>""")

demo.launch(server_name="0.0.0.0", server_port=7860)
