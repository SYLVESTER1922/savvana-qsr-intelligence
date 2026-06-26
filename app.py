import os
import json
import time
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
SHEET_ID    = os.environ.get("GOOGLE_SHEET_ID", "1H7rqtZUKOMi2TgV27kcSkbFHQ7H1EDuMc8J2xHfTWrE")
CREDS_JSON  = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
GROWTH_RATE = 0.05
CACHE_TTL   = 300  # 5 minutes

_cache = {'df': pd.DataFrame(), 'loaded_at': 0}

COMPLEXES = ['Westgate Mall','City Centre','Eastpark','Northgate']
BRANDS    = ['Flame & Grill','Pie Palace','Chill Creamery','Sizzle Wings']
REGIONS   = {'Westgate Mall':'Harare','City Centre':'Harare',
             'Eastpark':'Bulawayo','Northgate':'Bulawayo'}
COLORS    = ['#c9a84c','#2ecc71','#e74c3c','#9b59b6','#1abc9c','#f39c12','#3498db','#e67e22']

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))


# ═══════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════

def get_sheets_service():
    if not CREDS_JSON:
        raise ValueError("GOOGLE_CREDENTIALS_JSON secret not set")
    creds_dict = json.loads(CREDS_JSON)
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
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
        data    = [r for r in rows[1:] if any(c.strip() for c in r)]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=headers[:len(data[0])] if data else headers)
        numeric_cols = ['budget_usd','actual_revenue_usd','prior_month_actual',
                        'prior_year_actual','customer_count','counters_open',
                        'variance_vs_budget','avg_spend_per_cust',
                        'revenue_per_counter','is_holiday']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('[$,]','',regex=True),
                    errors='coerce').fillna(0)
        pct_cols = ['variance_pct','vs_prior_month_pct','vs_prior_year_pct']
        for col in pct_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%','',regex=True),
                    errors='coerce').fillna(0)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date','actual_revenue_usd']).copy()
        df = df[df['actual_revenue_usd'] > 0].copy()
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"Sheet load error: {e}")
        return pd.DataFrame()

def get_df():
    now = time.time()
    if now - _cache['loaded_at'] > CACHE_TTL or _cache['df'].empty:
        fresh = load_data()
        if not fresh.empty:
            _cache['df']        = fresh
            _cache['loaded_at'] = now
            print(f"Data refreshed — {len(fresh)} rows")
    return _cache['df']

def refresh_data():
    _cache['loaded_at'] = 0
    fresh = get_df()
    if fresh.empty:
        return "⚠️ Still no data — check logs for connection errors"
    return f"✅ Refreshed — {len(fresh)} rows loaded ({fresh['date'].min().strftime('%b %d, %Y')} → {fresh['date'].max().strftime('%b %d, %Y')})"

print("Loading data from Google Sheets...")
_ = get_df()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def fmt(v):
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"

def hex_rgba(h, a=0.15):
    h = h.lstrip('#')
    r,g,b = tuple(int(h[i:i+2],16) for i in (0,2,4))
    return f'rgba({r},{g},{b},{a})'

def dark(title, height=360):
    return dict(
        title=dict(text=title, font=dict(color='#c9a84c',size=14)),
        paper_bgcolor='#0a1628', plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0',family='Arial',size=11),
        height=height,
        xaxis=dict(gridcolor='#1a3a6e',linecolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(gridcolor='#1a3a6e',linecolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        legend=dict(bgcolor='rgba(10,22,40,0.85)',bordercolor='#1a3a6e',
                    borderwidth=1,font=dict(color='#c8d8f0')),
    )

def get_kpis():
    df = get_df()
    if df.empty:
        return "No data", "—", "—", "—", "—"
    latest    = df['date'].max()
    total_rev = df['actual_revenue_usd'].sum()
    total_bud = df['budget_usd'].sum() if 'budget_usd' in df.columns else 0
    ach       = f"{total_rev/total_bud*100:.1f}%" if total_bud > 0 else "—"
    top_cx    = df.groupby('complex')['actual_revenue_usd'].sum().idxmax()
    top_br    = df.groupby('brand')['actual_revenue_usd'].sum().idxmax()
    date_rng  = f"{df['date'].min().strftime('%b %d')} – {latest.strftime('%b %d, %Y')}"
    return date_rng, fmt(total_rev), ach, top_cx, top_br


# ═══════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════

def build_dashboard(period):
    df = get_df()
    if df.empty:
        empty = go.Figure().update_layout(**dark("No data — click Refresh Data"))
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
        dff = dff[(dff['date'].dt.month == latest.month) &
                  (dff['date'].dt.year  == latest.year)]

    # 1. Revenue by Complex
    cx = dff.groupby('complex').agg(
        actual=('actual_revenue_usd','sum'),
        budget=('budget_usd','sum') if 'budget_usd' in dff.columns else ('actual_revenue_usd','sum')
    ).reset_index().sort_values('actual',ascending=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Budget',x=cx['budget'],y=cx['complex'],
        orientation='h',marker_color='#1a3a6e',opacity=0.7))
    fig1.add_trace(go.Bar(name='Actual',x=cx['actual'],y=cx['complex'],
        orientation='h',marker_color='#c9a84c',
        text=[fmt(v) for v in cx['actual']],textposition='outside',
        textfont=dict(color='#c8d8f0',size=11)))
    fig1.update_layout(**dark("Revenue by Complex — Actual vs Budget",height=300),
        barmode='overlay',
        xaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')),
        margin=dict(l=140,r=80,t=50,b=40))

    # 2. Revenue by Brand
    br = dff.groupby('brand')['actual_revenue_usd'].sum().reset_index().sort_values('actual_revenue_usd',ascending=True)
    fig2 = go.Figure(go.Bar(
        x=br['actual_revenue_usd'],y=br['brand'],orientation='h',
        marker_color=COLORS[:len(br)],
        text=[fmt(v) for v in br['actual_revenue_usd']],textposition='outside',
        textfont=dict(color='#c8d8f0',size=11)))
    fig2.update_layout(**dark("Revenue by Brand",height=280),
        xaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(tickfont=dict(color='#c8d8f0')),
        margin=dict(l=140,r=80,t=50,b=40))

    # 3. Daily trend
    daily = dff.groupby('date')['actual_revenue_usd'].sum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=daily['date'],y=daily['actual_revenue_usd'],
        name='Actual',line=dict(color='#c9a84c',width=2.5),
        fill='tozeroy',fillcolor='rgba(201,168,76,0.08)'))
    fig3.update_layout(**dark("Daily Revenue Trend",height=360),
        xaxis=dict(title="Date",gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        margin=dict(l=70,r=40,t=50,b=50),hovermode='x unified',showlegend=False)

    # 4. Avg spend per customer
    if 'customer_count' in dff.columns:
        avs = dff[dff['customer_count']>0].groupby('complex').apply(
            lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()
        ).reset_index(name='avg').sort_values('avg',ascending=True)
        fig4 = go.Figure(go.Bar(
            x=avs['avg'],y=avs['complex'],orientation='h',
            marker_color='#2ecc71',
            text=[f"${v:.2f}" for v in avs['avg']],textposition='outside',
            textfont=dict(color='#c8d8f0',size=11)))
        fig4.update_layout(**dark("Avg Spend Per Customer by Complex",height=280),
            xaxis=dict(title="Avg Spend (USD)",tickprefix="$",tickformat=",.2f",
                       gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
            yaxis=dict(tickfont=dict(color='#c8d8f0')),
            margin=dict(l=140,r=80,t=50,b=40))
    else:
        fig4 = go.Figure().update_layout(**dark("Avg Spend (no customer data)",height=280))

    # 5. Budget achievement heatmap
    if 'budget_usd' in dff.columns:
        heat = dff.groupby(['complex','brand']).apply(
            lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100
            if x['budget_usd'].sum()>0 else 0
        ).reset_index(name='ach')
        try:
            pivot = heat.pivot(index='complex',columns='brand',values='ach')
            fig5 = go.Figure(go.Heatmap(
                z=pivot.values,x=pivot.columns.tolist(),y=pivot.index.tolist(),
                colorscale=[[0,'#c0392b'],[0.8,'#f39c12'],[1,'#2ecc71']],
                zmid=100,zmin=70,zmax=120,
                text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
                texttemplate="%{text}",textfont=dict(color='white',size=12),
                colorbar=dict(tickfont=dict(color='#c8d8f0'))))
            fig5.update_layout(**dark("Budget Achievement % — Complex × Brand",height=280),
                xaxis=dict(tickfont=dict(color='#c8d8f0'),tickangle=15),
                yaxis=dict(tickfont=dict(color='#c8d8f0')),
                margin=dict(l=130,r=60,t=50,b=80))
        except:
            fig5 = go.Figure().update_layout(**dark("Budget Heatmap",height=280))
    else:
        fig5 = go.Figure().update_layout(**dark("Budget data not available",height=280))

    # 6. Prior period comparison
    comp_cols = {'actual':'actual_revenue_usd'}
    if 'prior_month_actual' in dff.columns:
        comp_cols['sdlm'] = 'prior_month_actual'
    if 'prior_year_actual' in dff.columns:
        comp_cols['sdly'] = 'prior_year_actual'
    comp = dff.groupby('complex')[list(comp_cols.values())].sum().reset_index()
    fig6 = go.Figure()
    color_map = {'sdly':'#4a6a9e','sdlm':'#7fb3d3','actual':'#c9a84c'}
    label_map = {'sdly':'Prior Year','sdlm':'Prior Month','actual':'This Period'}
    for key, col in comp_cols.items():
        if col in comp.columns:
            fig6.add_trace(go.Bar(name=label_map[key],x=comp['complex'],y=comp[col],
                marker_color=color_map[key],
                text=[fmt(v) for v in comp[col]],textposition='outside',
                textfont=dict(color='#c8d8f0',size=10)))
    fig6.update_layout(**dark("Actual vs Prior Month vs Prior Year",height=340),
        barmode='group',hovermode='x unified',
        xaxis=dict(tickfont=dict(color='#c8d8f0')),
        yaxis=dict(title="Revenue (USD)",tickprefix="$",tickformat=",.0f",
                   gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),
        margin=dict(l=60,r=40,t=50,b=60))

    return fig1, fig2, fig3, fig4, fig5, fig6


# ═══════════════════════════════════════════════════════════════
# TAB 2 — FORECAST
# ═══════════════════════════════════════════════════════════════

def generate_forecast(seg_type, seg_name, horizon):
    df = get_df()
    if df.empty:
        fig = go.Figure().update_layout(**dark("No data available"))
        return fig, "No data loaded."
    horizon = int(horizon)
    if seg_type == 'Overall':
        seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
    elif seg_type == 'By Complex':
        seg = df[df['complex']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
    elif seg_type == 'By Brand':
        seg = df[df['brand']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
    else:
        parts = seg_name.split(' | ')
        if len(parts)==2:
            seg = df[(df['complex']==parts[0])&(df['brand']==parts[1])].groupby('date')['actual_revenue_usd'].sum().reset_index()
        else:
            seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
    seg.columns = ['date','revenue']
    seg = seg.sort_values('date').reset_index(drop=True)
    if len(seg) < 7:
        fig = go.Figure().update_layout(**dark("Insufficient data (need 7+ days)"))
        return fig, "Need at least 7 days of data for forecasting."
    last_date = seg['date'].max()
    if horizon <= 7:   base = seg['revenue'].tail(7).mean()
    elif horizon <= 14: base = seg['revenue'].tail(14).mean() if len(seg)>=14 else seg['revenue'].mean()
    else:               base = seg['revenue'].tail(30).mean() if len(seg)>=30 else seg['revenue'].mean()
    DOW = {0:0.85,1:0.90,2:0.92,3:0.95,4:1.05,5:1.20,6:1.15}
    fc_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    fc_vals  = [round(base * DOW[d.weekday()] * (1+GROWTH_RATE), 2) for d in fc_dates]
    upper = [v*1.15 for v in fc_vals]
    lower = [v*0.85 for v in fc_vals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg['date'],y=seg['revenue'],
        name='Historical',line=dict(color='#4a7aae',width=1.2),opacity=0.85))
    fig.add_trace(go.Scatter(
        x=fc_dates+fc_dates[::-1],y=upper+lower[::-1],
        fill='toself',fillcolor='rgba(201,168,76,0.18)',
        line=dict(color='rgba(0,0,0,0)'),name='Confidence ±15%',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fc_dates,y=fc_vals,
        name=f'{horizon}-Day Forecast',line=dict(color='#c9a84c',width=2.5,dash='dash'),
        mode='lines+markers',marker=dict(size=4,color='#c9a84c')))
    fig.add_vline(x=last_date,line_dash='dot',line_color='#c9a84c',opacity=0.6)
    layout = dark(f"{seg_name} — {horizon}-Day Forecast",height=480)
    layout['xaxis'] = dict(title='Date',gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0'))
    layout['yaxis'] = dict(title='Revenue (USD)',tickprefix='$',tickformat=',.0f',
                           gridcolor='#1a3a6e',tickfont=dict(color='#c8d8f0'))
    layout['hovermode'] = 'x unified'
    layout['legend'] = dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,
                             bgcolor='rgba(10,22,40,0.8)',bordercolor='#1a3a6e',
                             borderwidth=1,font=dict(color='#c8d8f0'))
    layout['margin'] = dict(l=70,r=30,t=70,b=50)
    fig.update_layout(**layout)
    total = sum(fc_vals); avg = total/horizon
    peak  = max(fc_vals); peak_d = fc_dates[fc_vals.index(peak)].strftime('%b %d, %Y')
    note = "7-day" if horizon<=7 else "14-day" if horizon<=14 else "30-day"
    summary = f"""**{horizon}-Day Forecast — {seg_name}**

| Metric | Value |
|---|---|
| Method | **{note} moving average × {1+GROWTH_RATE:.0%} growth + DOW adjustment** |
| Predicted Total | **${total:,.2f}** |
| Average Daily | **${avg:,.2f}** |
| Peak Day | **${peak:,.2f}** on {peak_d} |
| Period | {fc_dates[0].strftime('%b %d')} → {fc_dates[-1].strftime('%b %d, %Y')} |

> *Confidence band: ±15%. Weekend days forecast higher via day-of-week multiplier.*
"""
    return fig, summary

def update_seg_choices(seg_type):
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
# TAB 3 — CHAT (deterministic router + GPT)
# ═══════════════════════════════════════════════════════════════

def q_total():
    df = get_df()
    if df.empty: return "No data loaded."
    return f"Total revenue: ${df['actual_revenue_usd'].sum():,.2f}"

def q_top_complex():
    df = get_df()
    if df.empty: return "No data loaded."
    g = df.groupby('complex')['actual_revenue_usd'].sum()
    return f"Top complex: {g.idxmax()} at {fmt(g.max())}"

def q_top_brand():
    df = get_df()
    if df.empty: return "No data loaded."
    g = df.groupby('brand')['actual_revenue_usd'].sum()
    return f"Top brand: {g.idxmax()} at {fmt(g.max())}"

def q_underperforming():
    df = get_df()
    if df.empty: return "No data loaded."
    if 'budget_usd' not in df.columns: return "Budget data not available."
    latest = df['date'].max()
    recent = df[df['date'] >= latest - timedelta(days=7)]
    perf = recent.groupby(['complex','brand']).apply(
        lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()
        if x['budget_usd'].sum()>0 else 1
    ).reset_index(name='ach')
    under = perf[perf['ach']<0.80]
    if under.empty: return "No sites below 80% budget in last 7 days."
    lines = [f"- {r.complex} / {r.brand}: {r.ach*100:.1f}% of budget" for r in under.itertuples()]
    return "Below 80% budget (last 7 days):\n" + "\n".join(lines)

def q_avg_spend():
    df = get_df()
    if df.empty: return "No data loaded."
    if 'customer_count' not in df.columns: return "Customer data not available."
    avs = df[df['customer_count']>0].groupby('complex').apply(
        lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()
    ).sort_values(ascending=False)
    return "Avg spend per customer:\n" + "\n".join([f"- {cx}: ${v:.2f}" for cx,v in avs.items()])

def q_sdlm():
    df = get_df()
    if df.empty: return "No data loaded."
    if 'prior_month_actual' not in df.columns: return "Prior month data not available."
    act = df['actual_revenue_usd'].sum(); pm = df['prior_month_actual'].sum()
    if pm==0: return "Prior month data is zero — may need 30+ days of history."
    return f"vs Prior Month: ${act:,.2f} vs ${pm:,.2f} — {(act-pm)/pm*100:+.1f}%"

def q_sdly():
    df = get_df()
    if df.empty: return "No data loaded."
    if 'prior_year_actual' not in df.columns: return "Prior year data not available."
    act = df['actual_revenue_usd'].sum(); py = df['prior_year_actual'].sum()
    if py==0: return "Prior year data is zero — needs 365+ days of history."
    return f"vs Prior Year: ${act:,.2f} vs ${py:,.2f} — {(act-py)/py*100:+.1f}%"

def q_best_day():
    df = get_df()
    if df.empty: return "No data loaded."
    dow = df.groupby(df['date'].dt.day_name())['actual_revenue_usd'].mean()
    return f"Best day of week: {dow.idxmax()} (avg ${dow.max():,.2f})"

def q_moving_avg(days=7):
    df = get_df()
    if df.empty: return "No data loaded."
    latest = df['date'].max()
    ma = df[df['date']>=latest-timedelta(days=days)].groupby('date')['actual_revenue_usd'].sum().mean()
    return f"{days}-day moving average daily revenue: ${ma:,.2f}"

def route_intent(message):
    df = get_df()
    if df.empty: return None
    m = message.lower()
    if any(x in m for x in ['underperform','below budget','struggling','worst site']):
        return q_underperforming()
    if any(x in m for x in ['top complex','best complex','leading complex']):
        return q_top_complex()
    if any(x in m for x in ['top brand','best brand','leading brand']):
        return q_top_brand()
    if any(x in m for x in ['avg spend','average spend','spend per customer']):
        return q_avg_spend()
    if any(x in m for x in ['prior month','sdlm','last month','vs month']):
        return q_sdlm()
    if any(x in m for x in ['prior year','sdly','last year','yoy','year on year']):
        return q_sdly()
    if any(x in m for x in ['best day','busiest day','peak day','day of week']):
        return q_best_day()
    if '7-day' in m or '7 day' in m: return q_moving_avg(7)
    if '30-day' in m or '30 day' in m: return q_moving_avg(30)
    if any(x in m for x in ['total revenue','overall revenue']): return q_total()
    for cx in COMPLEXES:
        for br in BRANDS:
            if cx.lower() in m and br.lower() in m:
                sub = df[(df['complex']==cx)&(df['brand']==br)]
                if not sub.empty:
                    return f"{cx} / {br}: ${sub['actual_revenue_usd'].sum():,.2f} total, ${sub['actual_revenue_usd'].mean():,.2f}/day avg"
    for cx in COMPLEXES:
        if cx.lower() in m:
            sub = df[df['complex']==cx]
            return f"{cx}: ${sub['actual_revenue_usd'].sum():,.2f} total"
    for br in BRANDS:
        if br.lower() in m:
            sub = df[df['brand']==br]
            return f"{br}: ${sub['actual_revenue_usd'].sum():,.2f} total"
    return None

def build_system_prompt():
    df = get_df()
    if df.empty:
        return "You are a QSR intelligence assistant. No data is currently loaded from Google Sheets."
    total = df['actual_revenue_usd'].sum()
    latest = df['date'].max()
    return f"""You are an intelligence assistant for Savanna QSR Group, Zimbabwe.
Data loaded: {df['date'].min().strftime('%b %d, %Y')} to {latest.strftime('%b %d, %Y')} ({len(df)} rows).
Total revenue on record: ${total:,.2f}
Complexes: {', '.join(COMPLEXES)}
Brands: {', '.join(BRANDS)}
Answer concisely and professionally. Use $ for amounts."""

def chat(message, history):
    if not message or not message.strip():
        return ""
    data_answer = route_intent(message)
    system = build_system_prompt()
    messages = [{"role":"system","content":system}]
    for h in (history or []):
        if isinstance(h, dict):
            messages.append({"role":h["role"],"content":h["content"]})
        elif isinstance(h,(list,tuple)) and len(h)==2:
            if h[0]: messages.append({"role":"user","content":str(h[0])})
            if h[1]: messages.append({"role":"assistant","content":str(h[1])})
    user_msg = f"{message}\n\n[DATA]: {data_answer}" if data_answer else message
    messages.append({"role":"user","content":user_msg})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",messages=messages,temperature=0.2,max_tokens=600)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ {e}"


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════

css = """
body,.gradio-container{background:#050d1a!important;font-family:Arial,sans-serif!important;}
.tab-nav{background:#0a1628!important;border-bottom:2px solid #1a3a6e!important;}
button[class*="tab-"]{color:#7fb3d3!important;background:transparent!important;
  border:none!important;border-bottom:3px solid transparent!important;
  padding:10px 18px!important;font-size:13px!important;font-weight:500!important;}
button[class*="tab-"]:hover{color:#fff!important;}
button[class*="tab-"][class*="selected"],
div[role="tablist"] button[aria-selected="true"]{
  color:#c9a84c!important;border-bottom:3px solid #c9a84c!important;font-weight:700!important;}
.gradio-container *{color:#c8d8f0;}
.gradio-container input,.gradio-container textarea{
  background:#0a1628!important;color:#c8d8f0!important;
  border:1px solid #1a3a6e!important;border-radius:6px!important;}
ul[role="listbox"]{background:#0d1b2a!important;border:1px solid #c9a84c!important;border-radius:6px!important;}
ul[role="listbox"] li{color:#fff!important;background:#0d1b2a!important;}
ul[role="listbox"] li:hover,ul[role="listbox"] li[aria-selected="true"]{background:#c9a84c!important;color:#0d1b2a!important;}
button.primary,button[variant="primary"]{background:#c9a84c!important;color:#0a1628!important;
  font-weight:700!important;border:none!important;border-radius:6px!important;}
button.primary:hover{background:#e0be6a!important;}
button.secondary,button[variant="secondary"]{background:#1a3a6e!important;
  color:#c8d8f0!important;border:1px solid #c9a84c!important;border-radius:6px!important;}
.gradio-container .block,.gradio-container .form,.gradio-container .panel{
  background:#0a1628!important;border:1px solid #1a3a6e!important;border-radius:8px!important;}
.gradio-container label,.gradio-container .label-wrap span{color:#a8c8f0!important;}
input[type="radio"]{accent-color:#c9a84c!important;}
div[class*="chatbot"],.chatbot{background:#040c1a!important;border-radius:12px!important;}
.gradio-container .prose{color:#c8d8f0!important;}
.gradio-container .prose strong{color:#c9a84c!important;}
.gradio-container .prose th{background:#0a1628!important;color:#c9a84c!important;}
.gradio-container .prose td{border-color:#1a3a6e!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{background:#c9a84c;border-radius:4px;}
footer{display:none!important;}
"""

date_rng, total_rev, ach_pct, top_cx, top_br = get_kpis()

with gr.Blocks(title="Savanna QSR Intelligence | Netrisyl Insights", css=css) as demo:

    # Header
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
                <div style="color:#7fb3d3;font-size:16px;font-weight:700;">{date_rng}</div>
                <div style="color:#7fb3d3;font-size:11px;">Data Range</div>
            </div>
        </div>
    </div>""")

    with gr.Tabs():

        # ── Dashboard ──────────────────────────────────────────
        with gr.TabItem("📊 Dashboard"):
            with gr.Row():
                period_sel  = gr.Radio(
                    choices=['Last 7 Days','Last 30 Days','Last 90 Days','MTD','All Time'],
                    value='All Time', label="Period"
                )
                with gr.Column(scale=0, min_width=200):
                    dash_btn    = gr.Button("📊 Load Dashboard", variant="primary")
                    refresh_btn = gr.Button("🔄 Refresh Data",   variant="secondary")
                    refresh_msg = gr.Markdown()

            with gr.Row():
                ch1 = gr.Plot(show_label=False)
                ch2 = gr.Plot(show_label=False)
            with gr.Row():
                ch3 = gr.Plot(show_label=False)
                ch4 = gr.Plot(show_label=False)
            with gr.Row():
                ch5 = gr.Plot(show_label=False)
                ch6 = gr.Plot(show_label=False)

            dash_btn.click(build_dashboard,[period_sel],[ch1,ch2,ch3,ch4,ch5,ch6])
            refresh_btn.click(refresh_data,[],[refresh_msg])

        # ── Forecast ───────────────────────────────────────────
        with gr.TabItem("📈 Forecast"):
            gr.HTML("<p style='color:#7fb3d3;font-size:12px;margin:10px 0 14px;'>Day-of-week adjusted moving average with 5% growth factor.</p>")
            with gr.Row():
                seg_type = gr.Radio(
                    choices=['Overall','By Complex','By Brand','Complex × Brand'],
                    value='Overall', label="Segment Type"
                )
                seg_name = gr.Dropdown(
                    choices=['All Complexes & Brands'],
                    value='All Complexes & Brands',
                    label="Segment", interactive=True
                )
                horizon  = gr.Radio(choices=[7,14,30,60], value=30, label="Horizon (days)")
                fc_btn   = gr.Button("⚡ Generate", variant="primary")

            fc_chart   = gr.Plot(show_label=False)
            fc_summary = gr.Markdown()

            seg_type.change(update_seg_choices,[seg_type],[seg_name])
            fc_btn.click(generate_forecast,[seg_type,seg_name,horizon],[fc_chart,fc_summary])

        # ── Chat ───────────────────────────────────────────────
        with gr.TabItem("💬 Intelligence Chat"):
            gr.HTML("""
            <div style="background:#0a1628;border:1px solid #1a3a6e;border-radius:8px;
                        padding:14px 18px;margin:10px 0 14px;">
                <p style="color:#c9a84c;font-weight:700;margin:0 0 6px;font-size:13px;">
                    💡 Ask anything about Savanna QSR Group operations
                </p>
                <p style="color:#7fb3d3;font-size:12px;margin:0;">
                    Try: "Which complex is underperforming?" · "Avg spend per customer?" ·
                    "Compare Westgate vs City Centre" · "7-day moving average?"
                </p>
            </div>""")

            chatbot_box = gr.Chatbot(height=420, type="messages", show_label=False)

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question about Savanna QSR operations...",
                    show_label=False, scale=9, container=False
                )
                chat_send = gr.Button("Send ➤", variant="primary", scale=1)

            def respond(message, history):
                if not message or not message.strip():
                    return history or [], ""
                reply = chat(message, history or [])
                h = list(history or [])
                h.append({"role":"user",     "content": message})
                h.append({"role":"assistant","content": reply})
                return h, ""

            chat_send.click(respond,[chat_input,chatbot_box],[chatbot_box,chat_input])
            chat_input.submit(respond,[chat_input,chatbot_box],[chatbot_box,chat_input])

            gr.HTML("""
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">
                <span style="color:#7fb3d3;font-size:11px;padding-top:4px;">Try:</span>
            </div>""")

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

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
