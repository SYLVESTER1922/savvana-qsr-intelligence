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

# Stores Intelligence colour palette
C_GOLD    = '#c9a84c'
C_NAVY    = '#1e2d5e'
C_TEAL    = '#1abc9c'
C_RED     = '#e74c3c'
C_PURPLE  = '#9b59b6'
C_BG      = '#0a1628'
C_PLOT    = '#0d1f38'
C_GRID    = '#1a3a6e'
C_TEXT    = '#c8d8f0'
BAR_COLS  = [C_GOLD, C_NAVY, C_TEAL, C_RED, C_PURPLE, '#2ecc71', '#e67e22', '#3498db']

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
    m = margin or dict(l=150, r=30, t=45, b=40)
    return dict(
        title=dict(text=title, font=dict(color=C_GOLD, size=13), x=0),
        paper_bgcolor=C_BG, plot_bgcolor=C_PLOT,
        font=dict(color=C_TEXT, family='Arial', size=11),
        height=height,
        xaxis=dict(gridcolor=C_GRID, linecolor=C_GRID, tickfont=dict(color=C_TEXT), title_font=dict(color=C_TEXT)),
        yaxis=dict(gridcolor=C_GRID, linecolor=C_GRID, tickfont=dict(color=C_TEXT), title_font=dict(color=C_TEXT)),
        legend=dict(bgcolor='rgba(10,22,40,0.85)', bordercolor=C_GRID, borderwidth=1, font=dict(color=C_TEXT)),
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
    return f'''<div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);
        padding:16px 24px;border-radius:10px;border-left:4px solid {C_GOLD};margin-bottom:6px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <h2 style="color:#fff;margin:0;font-size:19px;font-weight:700;">🍔 Savanna QSR Intelligence</h2>
          <p style="color:#aed6f1;margin:3px 0 0;font-size:12px;">Live Operations Intelligence · Zimbabwe · Powered by Supabase</p>
        </div>
        <p style="color:{C_GOLD};margin:0;font-size:10px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p>
      </div>
      <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
        {''.join([f'<div style="background:rgba(10,22,40,0.75);padding:8px 16px;border-radius:7px;border:1px solid {C_GRID};text-align:center;min-width:80px;"><div style="color:{C_GOLD};font-size:16px;font-weight:700;">{v}</div><div style="color:#7fb3d3;font-size:10px;">{l}</div></div>'
        for v,l in [(rev,"Total Revenue"),(ach,"Budget Ach."),(tc,"Top Complex"),(tb,"Top Brand"),(dr,"Date Range")]])}
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
        insidetextanchor='middle', textfont=dict(color='#0a1628', size=12, family='Arial')
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
        textfont=dict(color='#ffffff', size=12, family='Arial')
    ))
    fig2.update_layout(**dark_layout("Revenue by Brand", height=280),
                       xaxis_tickprefix='$', xaxis_tickformat=',.0f')

    # 3. Daily Revenue Trend
    daily = dff.groupby('date')['actual_revenue_usd'].sum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=daily['date'], y=daily['actual_revenue_usd'], name='Actual',
        line=dict(color=C_GOLD, width=2.5), fill='tozeroy',
        fillcolor='rgba(201,168,76,0.10)'
    ))
    if 'budget_usd' in dff.columns:
        daily_b = dff.groupby('date')['budget_usd'].sum().reset_index()
        fig3.add_trace(go.Scatter(
            x=daily_b['date'], y=daily_b['budget_usd'], name='Budget',
            line=dict(color='#4a6a9e', width=1.5, dash='dot')
        ))
    fig3.update_layout(**dark_layout("Daily Revenue Trend", height=340, margin=dict(l=80,r=30,t=45,b=50)),
                       hovermode='x unified',
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
            insidetextanchor='middle', textfont=dict(color='#0a1628', size=12, family='Arial')
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
        if np.isnan(base) or base <= 0: base = float(seg['revenue'].mean())
        fc_dates = [last + timedelta(days=i+1) for i in range(h)]
        fc_vals  = [round(base * DOW[d.weekday()] * (1+GROWTH), 2) for d in fc_dates]
        upper = [v*1.15 for v in fc_vals]
        lower = [v*0.85 for v in fc_vals]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seg['date'], y=seg['revenue'], name='Historical',
            line=dict(color='#4a7aae', width=1.5), opacity=0.9,
            hovertemplate='%{x|%b %d}<br>$%{y:,.0f}<extra>Historical</extra>'))
        fig.add_trace(go.Scatter(
            x=fc_dates+fc_dates[::-1], y=upper+lower[::-1],
            fill='toself', fillcolor='rgba(201,168,76,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_vals, name=f'{h}-Day Forecast',
            line=dict(color=C_GOLD, width=2.5, dash='dash'), mode='lines+markers',
            marker=dict(size=5, color=C_GOLD),
            hovertemplate='%{x|%b %d}<br>$%{y:,.0f}<extra>Forecast</extra>'))
        fig.add_vline(x=str(last)[:10], line_dash='dot', line_color=C_GOLD, opacity=0.5)
        layout = dark_layout(f"{label} — {h}-Day Forecast", height=460,
                             margin=dict(l=80,r=30,t=60,b=50))
        layout['hovermode'] = 'x unified'
        layout['yaxis_tickprefix'] = '$'
        layout['yaxis_tickformat'] = ',.0f'
        fig.update_layout(**layout)
        total = sum(fc_vals); avg = total/h
        peak = max(fc_vals); peak_d = fc_dates[fc_vals.index(peak)].strftime('%b %d, %Y')
        summary = f"**{h}-Day Forecast — {label}**\n\n| Metric | Value |\n|---|---|\n| Predicted Total | **{fmt(total)}** |\n| Daily Average | **{fmt(avg)}** |\n| Peak Day | **{fmt(peak)}** on {peak_d} |\n| Period | {fc_dates[0].strftime('%b %d')} → {fc_dates[-1].strftime('%b %d, %Y')} |\n\n> *Day-of-week adjusted moving average with 5% growth factor.*"
        return fig, summary
    except Exception as e:
        return go.Figure().update_layout(**dark_layout(f"Error: {e}", height=460)), f"Error: {e}"

# ── Intelligence Chat ─────────────────────────────────────────────────────────
def route_intent(message, dff):
    if dff is None or dff.empty: return None
    m = message.lower()
    latest = dff['date'].max()
    has_bud = 'budget_usd' in dff.columns and dff['budget_usd'].sum() > 0

    def ach_str(actual, budget):
        return f" ({actual/budget*100:.1f}% of budget)" if budget > 0 else ""

    # Underperformance
    if any(x in m for x in ['underperform','below budget','struggling','worst','attention','concern','risk','flag']):
        if not has_bud: return "No budget data available."
        recent = dff[dff['date'] >= latest - timedelta(days=7)]
        perf = recent.groupby(['complex','brand']).apply(
            lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum() if x['budget_usd'].sum()>0 else 1
        ).reset_index(name='ach').sort_values('ach')
        under = perf[perf['ach'] < 0.90]
        if under.empty: return "All sites above 90% budget achievement in the last 7 days. No underperformers."
        return "Sites below 90% budget (last 7 days):\n" + "\n".join(
            [f"- {r.complex} / {r.brand}: {r.ach*100:.1f}%" for r in under.itertuples()])

    # Budget achievement
    if any(x in m for x in ['budget','achievement','target','variance']):
        if not has_bud: return "No budget data."
        overall = dff['actual_revenue_usd'].sum()/dff['budget_usd'].sum()*100
        cx_ach = dff.groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0).sort_values(ascending=False)
        br_ach = dff.groupby('brand').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0).sort_values(ascending=False)
        return f"Overall budget achievement: {overall:.1f}%\nBy complex:\n" + "\n".join([f"  - {cx}: {v:.1f}%" for cx,v in cx_ach.items()]) + "\nBy brand:\n" + "\n".join([f"  - {br}: {v:.1f}%" for br,v in br_ach.items()])

    # Prior month
    if any(x in m for x in ['prior month','last month','sdlm','month on month']):
        if 'prior_month_actual' not in dff.columns or dff['prior_month_actual'].sum()==0: return "No prior month data."
        act = dff['actual_revenue_usd'].sum(); pm = dff['prior_month_actual'].sum()
        cx = dff.groupby('complex').agg(a=('actual_revenue_usd','sum'),p=('prior_month_actual','sum'))
        lines = [f"vs Prior Month — Overall: {fmt(act)} vs {fmt(pm)} ({(act-pm)/pm*100:+.1f}%)"]
        lines += [f"  - {r.Index}: {fmt(r.a)} vs {fmt(r.p)} ({(r.a-r.p)/r.p*100:+.1f}%)" for r in cx.itertuples() if r.p>0]
        return "\n".join(lines)

    # Prior year
    if any(x in m for x in ['prior year','last year','sdly','year on year']):
        if 'prior_year_actual' not in dff.columns or dff['prior_year_actual'].sum()==0: return "No prior year data."
        act = dff['actual_revenue_usd'].sum(); py = dff['prior_year_actual'].sum()
        cx = dff.groupby('complex').agg(a=('actual_revenue_usd','sum'),p=('prior_year_actual','sum'))
        lines = [f"vs Prior Year — Overall: {fmt(act)} vs {fmt(py)} ({(act-py)/py*100:+.1f}%)"]
        lines += [f"  - {r.Index}: {fmt(r.a)} vs {fmt(r.p)} ({(r.a-r.p)/r.p*100:+.1f}%)" for r in cx.itertuples() if r.p>0]
        return "\n".join(lines)

    # Yesterday / latest day
    if any(x in m for x in ['yesterday','today','latest day','last day','daily']):
        day = dff[dff['date']==latest]
        if day.empty: return f"No data for {latest.strftime('%b %d')}."
        rev = day['actual_revenue_usd'].sum()
        bud = day['budget_usd'].sum() if has_bud else 0
        cx_b = "\n".join([f"  - {r.complex}: {fmt(r.actual_revenue_usd)}" for r in day.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
        return f"{latest.strftime('%b %d, %Y')} revenue: {fmt(rev)}{ach_str(rev,bud)}\nBy complex:\n{cx_b}"

    # 7-day / weekly
    if any(x in m for x in ['7-day','7 day','last week','this week','weekly','week']):
        w = dff[dff['date'] >= latest - timedelta(days=7)]
        total = w['actual_revenue_usd'].sum()
        bud = w['budget_usd'].sum() if has_bud else 0
        davg = w.groupby('date')['actual_revenue_usd'].sum().mean()
        cx_b = "\n".join([f"  - {r.complex}: {fmt(r.actual_revenue_usd)}" for r in w.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
        return f"Last 7 days: {fmt(total)} total{ach_str(total,bud)}, avg {fmt(davg)}/day\nBy complex:\n{cx_b}"

    # 30-day / MTD
    if any(x in m for x in ['30-day','30 day','this month','monthly','month to date','mtd']):
        w = dff[dff['date'] >= latest - timedelta(days=30)]
        total = w['actual_revenue_usd'].sum()
        bud = w['budget_usd'].sum() if has_bud else 0
        return f"Last 30 days: {fmt(total)} total{ach_str(total,bud)}"

    # Customer / spend
    if any(x in m for x in ['customer','spend per','avg spend','average spend','foot traffic','footfall']):
        if 'customer_count' not in dff.columns or dff['customer_count'].sum()==0: return "No customer count data."
        avs = dff[dff['customer_count']>0].groupby('complex').apply(
            lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()).sort_values(ascending=False)
        tc = int(dff['customer_count'].sum())
        return f"Total customers: {tc:,}\nAvg spend per customer:\n" + "\n".join([f"  - {cx}: ${v:.2f}" for cx,v in avs.items()])

    # Counter / throughput
    if any(x in m for x in ['counter','till','throughput','revenue per counter']):
        if 'revenue_per_counter' not in dff.columns or dff['revenue_per_counter'].sum()==0: return "No counter data."
        rpc = dff[dff['revenue_per_counter']>0].groupby('complex')['revenue_per_counter'].mean().sort_values(ascending=False)
        return "Revenue per counter:\n" + "\n".join([f"  - {cx}: {fmt(v)}" for cx,v in rpc.items()])

    # Day of week
    if any(x in m for x in ['best day','peak day','busiest','day of week','highest day']):
        dow = dff.groupby(dff['date'].dt.day_name())['actual_revenue_usd'].mean()
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = dow.reindex([d for d in order if d in dow.index])
        return "Avg daily revenue by day of week:\n" + "\n".join([f"  - {d}: {fmt(v)}" for d,v in dow.sort_values(ascending=False).items()])

    # Rankings
    if any(x in m for x in ['rank','ranking','all complex','all brand','compare all','overview','summary']):
        cx = dff.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False)
        br = dff.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False)
        cx_ach = dff.groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if has_bud and x['budget_usd'].sum()>0 else None)
        cx_lines = "\n".join([f"  {i+1}. {n}: {fmt(v)}{f' ({cx_ach[n]:.0f}% bud)' if cx_ach[n] else ''}" for i,(n,v) in enumerate(cx.items())])
        br_lines = "\n".join([f"  {i+1}. {n}: {fmt(v)}" for i,(n,v) in enumerate(br.items())])
        return f"Complex ranking:\n{cx_lines}\n\nBrand ranking:\n{br_lines}"

    # Top / best
    if any(x in m for x in ['top','best','highest','leading','number one']):
        g_cx = dff.groupby('complex')['actual_revenue_usd'].sum()
        g_br = dff.groupby('brand')['actual_revenue_usd'].sum()
        return f"Top complex: {g_cx.idxmax()} ({fmt(g_cx.max())})\nTop brand: {g_br.idxmax()} ({fmt(g_br.max())})"

    # Worst / lowest
    if any(x in m for x in ['worst','lowest','bottom','poor','weakest']):
        g_cx = dff.groupby('complex')['actual_revenue_usd'].sum()
        g_br = dff.groupby('brand')['actual_revenue_usd'].sum()
        return f"Lowest complex: {g_cx.idxmin()} ({fmt(g_cx.min())})\nLowest brand: {g_br.idxmin()} ({fmt(g_br.min())})"

    # Total
    if any(x in m for x in ['total','overall revenue']):
        total = dff['actual_revenue_usd'].sum()
        bud = dff['budget_usd'].sum() if has_bud else 0
        return f"Total revenue: {fmt(total)}{ach_str(total,bud)} across {len(dff):,} records"

    # Complex × Brand
    for cx in COMPLEXES:
        for br in BRANDS:
            if cx.lower() in m and br.lower() in m:
                sub = dff[(dff['complex']==cx)&(dff['brand']==br)]
                if not sub.empty:
                    bud = sub['budget_usd'].sum() if has_bud else 0
                    return f"{cx} / {br}: {fmt(sub['actual_revenue_usd'].sum())} total{ach_str(sub['actual_revenue_usd'].sum(),bud)}, {fmt(sub['actual_revenue_usd'].mean())}/day avg"

    for cx in COMPLEXES:
        if cx.lower() in m:
            sub = dff[dff['complex']==cx]
            bud = sub['budget_usd'].sum() if has_bud else 0
            br_b = "\n".join([f"  - {r.brand}: {fmt(r.actual_revenue_usd)}" for r in sub.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
            return f"{cx}: {fmt(sub['actual_revenue_usd'].sum())} total{ach_str(sub['actual_revenue_usd'].sum(),bud)}\nBy brand:\n{br_b}"

    for br in BRANDS:
        if br.lower() in m:
            sub = dff[dff['brand']==br]
            bud = sub['budget_usd'].sum() if has_bud else 0
            cx_b = "\n".join([f"  - {r.complex}: {fmt(r.actual_revenue_usd)}" for r in sub.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
            return f"{br}: {fmt(sub['actual_revenue_usd'].sum())} total{ach_str(sub['actual_revenue_usd'].sum(),bud)}\nBy complex:\n{cx_b}"

    return None

def build_chat_context(dff):
    if dff is None or dff.empty: return "No data loaded."
    cx_rev = dff.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False)
    br_rev = dff.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False)
    has_bud = 'budget_usd' in dff.columns and dff['budget_usd'].sum() > 0
    total = dff['actual_revenue_usd'].sum()
    bud   = dff['budget_usd'].sum() if has_bud else 0
    dr    = f"{dff['date'].min().strftime('%b %d, %Y')} to {dff['date'].max().strftime('%b %d, %Y')}"
    cx_lines = "\n".join([f"- {cx}: {fmt(v)}{f' ({v/dff[dff.complex==cx].budget_usd.sum()*100:.0f}% bud)' if has_bud and dff[dff.complex==cx].budget_usd.sum()>0 else ''}" for cx,v in cx_rev.items()])
    br_lines = "\n".join([f"- {br}: {fmt(v)}" for br,v in br_rev.items()])
    return f"""You are the Savanna QSR Intelligence assistant for Netrisyl Insights, Zimbabwe.
Data range: {dr} | Records: {len(dff):,} | Total revenue: {fmt(total)}{f" ({total/bud*100:.1f}% of budget)" if has_bud else ""}

REVENUE BY COMPLEX:
{cx_lines}

REVENUE BY BRAND:
{br_lines}

Complexes: {', '.join(COMPLEXES)}
Brands: {', '.join(BRANDS)}

Answer directly and concisely. Always cite figures. Use $ for amounts. Be honest when data is insufficient."""

def chat(message, history, date_from, date_to):
    if not message or not message.strip(): return ""
    dff = filter_df(date_from, date_to)
    data_ctx = route_intent(message, dff)
    system   = build_chat_context(dff)
    msgs = [{"role":"system","content":system}]
    for h in (history or []):
        if isinstance(h,(list,tuple)) and len(h)==2:
            if h[0]: msgs.append({"role":"user","content":str(h[0])})
            if h[1]: msgs.append({"role":"assistant","content":str(h[1])})
    user_msg = f"{message}\n\n[COMPUTED DATA]: {data_ctx}" if data_ctx else message
    msgs.append({"role":"user","content":user_msg})
    try:
        r = _http.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"},
            json={"model":"gpt-4o-mini","messages":msgs,"temperature":0.2,"max_tokens":700})
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e: return f"⚠️ {e}"

def refresh_and_apply(date_from, date_to):
    _cache['loaded_at'] = 0
    dff = filter_df(date_from, date_to)
    if dff.empty: return build_kpi_html(), "⚠️ No data — check Supabase connection"
    full = get_df()
    return build_kpi_html(dff), f"✅ {len(full):,} rows loaded · Showing {len(dff):,} rows for selected range"

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = f"""
body,.gradio-container{{background:{C_BG}!important;font-family:Arial,sans-serif!important;}}
.tab-nav{{background:#0d1628!important;border-bottom:2px solid {C_GRID}!important;padding:0 8px!important;}}
button[class*="tab-"]{{color:#7fb3d3!important;background:transparent!important;border:none!important;
  border-bottom:3px solid transparent!important;padding:10px 18px!important;font-size:13px!important;font-weight:500!important;}}
button[class*="tab-"]:hover{{color:#fff!important;background:rgba(201,168,76,0.06)!important;}}
button[class*="tab-"][class*="selected"],div[role="tablist"] button[aria-selected="true"]{{
  color:{C_GOLD}!important;border-bottom:3px solid {C_GOLD}!important;font-weight:700!important;}}
.gradio-container *{{color:{C_TEXT};}}
.gradio-container input,.gradio-container textarea{{
  background:#0d1628!important;color:{C_TEXT}!important;border:1px solid {C_GRID}!important;border-radius:6px!important;}}
.gradio-container label,.gradio-container .label-wrap span{{color:#a8c8f0!important;}}
input[type="radio"]{{accent-color:{C_GOLD}!important;}}
button.primary,button[variant="primary"]{{background:{C_GOLD}!important;color:#0a1628!important;
  font-weight:700!important;border:none!important;border-radius:6px!important;font-size:13px!important;}}
button.primary:hover{{background:#e0be6a!important;}}
button.secondary,button[variant="secondary"]{{background:{C_NAVY}!important;color:{C_TEXT}!important;
  border:1px solid {C_GOLD}!important;border-radius:6px!important;}}
.gradio-container .block,.gradio-container .form{{background:#0d1628!important;
  border:1px solid {C_GRID}!important;border-radius:8px!important;}}
ul[role="listbox"]{{background:#0d1628!important;border:1px solid {C_GOLD}!important;border-radius:6px!important;}}
ul[role="listbox"] li{{color:#fff!important;background:#0d1628!important;}}
ul[role="listbox"] li:hover,ul[role="listbox"] li[aria-selected="true"]{{background:{C_GOLD}!important;color:#0a1628!important;}}
div[class*="chatbot"],.chatbot{{background:#040c1a!important;border-radius:10px!important;}}
.gradio-container .prose{{color:{C_TEXT}!important;}}
.gradio-container .prose strong{{color:{C_GOLD}!important;}}
.gradio-container .prose th{{background:#0d1628!important;color:{C_GOLD}!important;}}
.gradio-container .prose td{{border-color:{C_GRID}!important;}}
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-thumb{{background:{C_GOLD};border-radius:4px;}}
footer{{display:none!important;}}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Savanna QSR Intelligence | Netrisyl Insights", css=CSS) as demo:

    kpi_header = gr.HTML(value=build_kpi_html())

    # ── Global date filter ────────────────────────────────────────────────────
    with gr.Row():
        date_from = gr.Textbox(label="From Date", placeholder="YYYY-MM-DD", scale=2)
        date_to   = gr.Textbox(label="To Date",   placeholder="YYYY-MM-DD", scale=2)
        apply_btn = gr.Button("Apply Filter", variant="primary", scale=1)
        refresh_btn = gr.Button("Refresh Data", variant="secondary", scale=1)
        filter_msg = gr.Markdown(scale=3)

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

    apply_btn.click(
        fn=lambda df, dt: (refresh_and_apply(df, dt)[0], f"Showing filtered range") + build_dashboard(df, dt)[1:],
        inputs=[date_from, date_to],
        outputs=[kpi_header, filter_msg] + [ch1,ch2,ch3,ch4,ch5,ch6]
    )

    def apply_filter(df, dt):
        kpi, msg = refresh_and_apply(df, dt) if False else (build_kpi_html(filter_df(df,dt)), "")
        dash = build_dashboard(df, dt)
        return [dash[0]] + [msg] + list(dash[1:])

    apply_btn.click(fn=lambda df, dt: [build_kpi_html(filter_df(df,dt)), ""] + list(build_dashboard(df,dt)[1:]),
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

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
