import os, json, time, warnings
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timedelta
import requests as _http
warnings.filterwarnings('ignore')
# ── Gradio 5.9.1 schema bug fix ──────────────────────────────────────────────
try:
    import gradio_client.utils as _gcu
    _orig_j2p = _gcu._json_schema_to_python_type
    def _safe_j2p(schema, defs):
        if not isinstance(schema, dict): return "any"
        try: return _orig_j2p(schema, defs)
        except Exception: return "any"
    _gcu._json_schema_to_python_type = _safe_j2p
except Exception: pass
# ─────────────────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
GROWTH_RATE = 0.05
CACHE_TTL = 300
_cache = {'df': pd.DataFrame(), 'loaded_at': 0}
COMPLEXES = ['Westgate Mall','City Centre','Eastpark','Northgate']
BRANDS = ['Flame & Grill','Pie Palace','Chill Creamery','Sizzle Wings']
COLORS = ['#c9a84c','#2ecc71','#e74c3c','#9b59b6','#1abc9c','#f39c12','#3498db','#e67e22']

def load_data():
    if not SUPABASE_URL or not SUPABASE_KEY: print("SUPABASE not configured"); return pd.DataFrame()
    try:
        r = _http.get(SUPABASE_URL+"/rest/v1/daily_input?select=*&order=date&limit=10000", headers={"apikey":SUPABASE_KEY,"Authorization":"Bearer "+SUPABASE_KEY})
        if r.status_code!=200: print("HTTP "+str(r.status_code)); return pd.DataFrame()
        data=r.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        for c in ['budget_usd','actual_revenue_usd','prior_month_actual','prior_year_actual','customer_count','counters_open','variance_vs_budget','avg_spend_per_cust','revenue_per_counter','is_holiday','variance_pct','vs_prior_month_pct','vs_prior_year_pct']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date','actual_revenue_usd']).copy()
        df = df[df['actual_revenue_usd'] > 0].copy()
        print("Loaded "+str(len(df))+" rows from Supabase")
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e: print("Load error: "+str(e)); return pd.DataFrame()
def get_df():
    now = time.time()
    if now - _cache['loaded_at'] > CACHE_TTL or _cache['df'].empty:
        fresh = load_data()
        if not fresh.empty: _cache['df'] = fresh; _cache['loaded_at'] = now
    return _cache['df']
def refresh_data():
    _cache['loaded_at'] = 0; fresh = get_df()
    if fresh.empty: return build_kpi_html(), "No data"
    return build_kpi_html(), "Refreshed: "+str(len(fresh))+" rows ("+fresh['date'].min().strftime('%b %d, %Y')+" to "+fresh['date'].max().strftime('%b %d, %Y')+")"
print("Loading from Supabase..."); _ = get_df()
def fmt(v):
    if v >= 1e6: return "${:.2f}M".format(v/1e6)
    if v >= 1e3: return "${:.0f}K".format(v/1e3)
    return "${:,.0f}".format(v)
def dark(title, height=360):
    return dict(title=dict(text=title,font=dict(color='#c9a84c',size=14)),paper_bgcolor='#0a1628',plot_bgcolor='#0d1f38',font=dict(color='#c8d8f0',family='Arial',size=11),height=height,xaxis=dict(gridcolor='#1a3a6e',linecolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),yaxis=dict(gridcolor='#1a3a6e',linecolor='#1a3a6e',tickfont=dict(color='#c8d8f0')),legend=dict(bgcolor='rgba(10,22,40,0.85)',bordercolor='#1a3a6e',borderwidth=1,font=dict(color='#c8d8f0')))
def build_kpi_html(dff=None):
    if dff is None:
        dff = get_df()
    if dff.empty:
        tr,ap,tc,tb,dr = "--","--","--","--","No data"
    else:
        tr = fmt(dff['actual_revenue_usd'].sum())
        bud = dff['budget_usd'].sum() if 'budget_usd' in dff.columns else 0
        ap = "{:.1f}%".format(dff['actual_revenue_usd'].sum()/bud*100) if bud > 0 else "--"
        tc = dff.groupby('complex')['actual_revenue_usd'].sum().idxmax()
        tb = dff.groupby('brand')['actual_revenue_usd'].sum().idxmax()
        dr = dff['date'].min().strftime('%b %d')+" - "+dff['date'].max().strftime('%b %d, %Y')
    return '<div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);padding:18px 28px 16px;border-radius:12px;margin-bottom:4px;border-left:4px solid #c9a84c;"><div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;"><div><h1 style="color:#fff;margin:0;font-size:22px;">Savanna QSR Intelligence</h1><p style="color:#aed6f1;margin:4px 0 0;font-size:13px;">Live Operations Intelligence - Zimbabwe</p></div><div style="text-align:right;"><p style="color:#c9a84c;margin:0;font-size:10px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p></div></div><div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;"><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+tr+'</div><div style="color:#7fb3d3;font-size:11px;">Total Revenue</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+ap+'</div><div style="color:#7fb3d3;font-size:11px;">Budget Achievement</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+tc+'</div><div style="color:#7fb3d3;font-size:11px;">Top Complex</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+tb+'</div><div style="color:#7fb3d3;font-size:11px;">Top Brand</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#7fb3d3;font-size:16px;font-weight:700;">'+dr+'</div><div style="color:#7fb3d3;font-size:11px;">Data Range</div></div></div></div>'
def build_dashboard(period):
    df = get_df()
    if df.empty:
        e = go.Figure().update_layout(**dark("No data - click Refresh")); return [build_kpi_html()]+[e]*6
    dff = df.copy(); latest = dff['date'].max()
    if period == 'Last 7 Days': dff = dff[dff['date'] >= latest - timedelta(days=7)]
    elif period == 'Last 30 Days': dff = dff[dff['date'] >= latest - timedelta(days=30)]
    elif period == 'Last 90 Days': dff = dff[dff['date'] >= latest - timedelta(days=90)]
    elif period == 'MTD': dff = dff[(dff['date'].dt.month==latest.month)&(dff['date'].dt.year==latest.year)]
    cx = dff.groupby('complex').agg(actual=('actual_revenue_usd','sum'),budget=('budget_usd','sum')).reset_index().sort_values('actual',ascending=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Budget',x=cx['budget'],y=cx['complex'],orientation='h',marker=dict(color='#2a4a8a',line=dict(color='#c9a84c',width=1)),opacity=0.6))
    fig1.add_trace(go.Bar(name='Actual',x=cx['actual'],y=cx['complex'],orientation='h',marker=dict(color='#c9a84c',line=dict(color='#1e2d5e',width=1)),text=[fmt(v) for v in cx['actual']],textposition='inside',insidetextanchor='middle',textfont=dict(color='#0a1628',size=12,family='Arial',weight='bold')))
    fig1.update_layout(**dark("Revenue by Complex",height=300),barmode='overlay',margin=dict(l=140,r=120,t=50,b=40))
    br = dff.groupby('brand')['actual_revenue_usd'].sum().reset_index().sort_values('actual_revenue_usd',ascending=True)
    fig2 = go.Figure(go.Bar(x=br['actual_revenue_usd'],y=br['brand'],orientation='h',marker=dict(color=['#1e2d5e','#c9a84c','#1abc9c','#e74c3c','#9b59b6','#2ecc71','#e67e22','#3498db'][:len(br)],line=dict(color='#c9a84c',width=1)),text=[fmt(v) for v in br['actual_revenue_usd']],textposition='inside',insidetextanchor='middle',textfont=dict(color='#ffffff',size=12,family='Arial',weight='bold')))
    fig2.update_layout(**dark("Revenue by Brand",height=280),margin=dict(l=140,r=120,t=50,b=40))
    daily = dff.groupby('date')['actual_revenue_usd'].sum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=daily['date'],y=daily['actual_revenue_usd'],name='Actual',line=dict(color='#c9a84c',width=2.5),fill='tozeroy',fillcolor='rgba(201,168,76,0.08)'))
    fig3.update_layout(**dark("Daily Revenue Trend",height=360),hovermode='x unified',showlegend=False,margin=dict(l=90,r=40,t=50,b=50),yaxis_tickprefix="$",yaxis_tickformat=",.0f")
    if 'customer_count' in dff.columns:
        avs = dff[dff['customer_count']>0].groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()).reset_index(name='avg').sort_values('avg',ascending=True)
        fig4 = go.Figure(go.Bar(x=avs['avg'],y=avs['complex'],orientation='h',marker_color='#2ecc71',text=["${:.2f}".format(v) for v in avs['avg']],textposition='outside',textfont=dict(color='#c8d8f0',size=11)))
        fig4.update_layout(**dark("Avg Spend Per Customer",height=280),margin=dict(l=140,r=120,t=50,b=40))
    else: fig4 = go.Figure().update_layout(**dark("No customer data",height=280))
    try:
        heat = dff.groupby(['complex','brand']).apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0).reset_index(name='ach')
        pivot = heat.pivot(index='complex',columns='brand',values='ach')
        fig5 = go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),y=pivot.index.tolist(),colorscale=[[0,'#c0392b'],[0.8,'#f39c12'],[1,'#2ecc71']],zmid=100,zmin=70,zmax=120,text=[["{:.1f}%".format(v) for v in row] for row in pivot.values],texttemplate="%{text}",textfont=dict(color='white',size=12)))
        fig5.update_layout(**dark("Budget Achievement %",height=280),margin=dict(l=130,r=60,t=50,b=80))
    except: fig5 = go.Figure().update_layout(**dark("Heatmap error",height=280))
    comp_cols = {'actual':'actual_revenue_usd'}
    if 'prior_month_actual' in dff.columns: comp_cols['sdlm'] = 'prior_month_actual'
    if 'prior_year_actual' in dff.columns: comp_cols['sdly'] = 'prior_year_actual'
    comp = dff.groupby('complex')[list(comp_cols.values())].sum().reset_index()
    fig6 = go.Figure()
    cmap = {'sdly':'#1e2d5e','sdlm':'#1abc9c','actual':'#c9a84c'}
    lmap = {'sdly':'Prior Year','sdlm':'Prior Month','actual':'This Period'}
    for key, col in comp_cols.items():
        if col in comp.columns: fig6.add_trace(go.Bar(name=lmap[key],x=comp['complex'],y=comp[col],marker_color=cmap[key],text=[fmt(v) for v in comp[col]],textposition='outside',textfont=dict(color='#c8d8f0',size=10)))
    fig6.update_layout(**dark("Actual vs Prior Periods",height=340),barmode='group',hovermode='x unified',margin=dict(l=60,r=40,t=50,b=60))
    return build_kpi_html(dff), fig1, fig2, fig3, fig4, fig5, fig6
def generate_forecast(seg_type, seg_name, horizon):
    try:
        df = get_df()
        if df.empty: return go.Figure().update_layout(**dark("No data")), "No data."
        horizon = int(horizon)
        if seg_type == 'Overall':
            seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = 'All Complexes'
        elif seg_type == 'By Complex':
            seg = df[df['complex']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = seg_name
        elif seg_type == 'By Brand':
            seg = df[df['brand']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = seg_name
        else:
            parts = seg_name.split(' | ')
            if len(parts)==2:
                seg = df[(df['complex']==parts[0])&(df['brand']==parts[1])].groupby('date')['actual_revenue_usd'].sum().reset_index()
            else:
                seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
            label = seg_name
        seg.columns = ['date','revenue']
        seg = seg.sort_values('date').reset_index(drop=True)
        if len(seg) < 7:
            return go.Figure().update_layout(**dark("Need 7+ days of data for this segment")), "Insufficient data for this segment."
        last_date = seg['date'].max()
        base = seg['revenue'].tail(min(horizon, len(seg))).mean()
        if pd.isna(base) or base <= 0: base = seg['revenue'].mean()
        DOW = {0:0.85,1:0.90,2:0.92,3:0.95,4:1.05,5:1.20,6:1.15}
        fc_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        fc_vals = [round(float(base) * DOW[d.weekday()] * (1+GROWTH_RATE), 2) for d in fc_dates]
        upper = [v*1.15 for v in fc_vals]; lower = [v*0.85 for v in fc_vals]
        last_date_str = str(last_date)[:10]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seg['date'],y=seg['revenue'],name='Historical',line=dict(color='#4a7aae',width=1.2),opacity=0.85))
        fig.add_trace(go.Scatter(x=fc_dates+fc_dates[::-1],y=upper+lower[::-1],fill='toself',fillcolor='rgba(201,168,76,0.18)',line=dict(color='rgba(0,0,0,0)'),name='Confidence',hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=fc_dates,y=fc_vals,name=str(horizon)+'-Day Forecast',line=dict(color='#c9a84c',width=2.5,dash='dash'),mode='lines+markers',marker=dict(size=4,color='#c9a84c')))
        fig.add_vline(x=last_date_str,line_dash='dot',line_color='#c9a84c',opacity=0.6)
        fig.update_layout(**dark(label+" - "+str(horizon)+"-Day Forecast",height=480),hovermode='x unified',margin=dict(l=70,r=40,t=70,b=50))
        total=sum(fc_vals); avg=total/horizon; peak=max(fc_vals); peak_d=fc_dates[fc_vals.index(peak)].strftime('%b %d, %Y')
        summary = "**"+str(horizon)+"-Day Forecast — "+label+"**\n\n| Metric | Value |\n|---|---|\n| Total | **${:,.2f}".format(total)+"** |\n| Daily Avg | **${:,.2f}".format(avg)+"** |\n| Peak | **${:,.2f}".format(peak)+" on "+peak_d+"** |"
        return fig, summary
    except Exception as e:
        err_fig = go.Figure().update_layout(**dark("Forecast error: "+str(e),height=480))
        return err_fig, "Error: "+str(e)
def update_seg_choices(seg_type):
    if seg_type == 'Overall': return gr.update(choices=['All'],value='All')
    elif seg_type == 'By Complex': return gr.update(choices=COMPLEXES,value=COMPLEXES[0])
    elif seg_type == 'By Brand': return gr.update(choices=BRANDS,value=BRANDS[0])
    else:
        opts = [cx+" | "+br for cx in COMPLEXES for br in BRANDS]
        return gr.update(choices=opts,value=opts[0])
def route_intent(message):
    df = get_df()
    if df.empty: return None
    m = message.lower()
    latest = df['date'].max()

    # Budget / underperformance
    if any(x in m for x in ['underperform','below budget','struggling','worst','attention','concern','risk']):
        if 'budget_usd' not in df.columns: return "No budget data available."
        recent = df[df['date'] >= latest - timedelta(days=7)]
        perf = recent.groupby(['complex','brand']).apply(
            lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum() if x['budget_usd'].sum()>0 else 1
        ).reset_index(name='ach')
        under = perf[perf['ach']<0.90].sort_values('ach')
        if under.empty: return "All sites above 90% budget achievement in last 7 days."
        return "Below 90% budget (last 7 days):\n" + "\n".join(
            ["- {}/{}: {:.1f}%".format(r.complex,r.brand,r.ach*100) for r in under.itertuples()])

    if any(x in m for x in ['budget','achievement','variance','target']):
        if 'budget_usd' not in df.columns: return "No budget data."
        overall = df['actual_revenue_usd'].sum()/df['budget_usd'].sum()*100 if df['budget_usd'].sum()>0 else 0
        cx_ach = df.groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0).sort_values(ascending=False)
        br_ach = df.groupby('brand').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0).sort_values(ascending=False)
        return "Overall budget achievement: {:.1f}%\nBy complex:\n{}\nBy brand:\n{}".format(
            overall,
            "\n".join(["- {}: {:.1f}%".format(cx,v) for cx,v in cx_ach.items()]),
            "\n".join(["- {}: {:.1f}%".format(br,v) for br,v in br_ach.items()]))

    # Prior period comparisons
    if any(x in m for x in ['prior month','last month','sdlm','vs month','month on month']):
        if 'prior_month_actual' not in df.columns: return "No prior month data."
        act=df['actual_revenue_usd'].sum(); pm=df['prior_month_actual'].sum()
        cx = df.groupby('complex').agg(actual=('actual_revenue_usd','sum'),pm=('prior_month_actual','sum'))
        lines = ["vs Prior Month — Overall: ${:,.0f} vs ${:,.0f} ({:+.1f}%)".format(act,pm,(act-pm)/pm*100 if pm else 0)]
        lines += ["- {}: ${:,.0f} vs ${:,.0f} ({:+.1f}%)".format(r.Index,r.actual,r.pm,(r.actual-r.pm)/r.pm*100 if r.pm else 0) for r in cx.itertuples()]
        return "\n".join(lines)

    if any(x in m for x in ['prior year','last year','sdly','vs year','year on year']):
        if 'prior_year_actual' not in df.columns: return "No prior year data."
        act=df['actual_revenue_usd'].sum(); py=df['prior_year_actual'].sum()
        cx = df.groupby('complex').agg(actual=('actual_revenue_usd','sum'),py=('prior_year_actual','sum'))
        lines = ["vs Prior Year — Overall: ${:,.0f} vs ${:,.0f} ({:+.1f}%)".format(act,py,(act-py)/py*100 if py else 0)]
        lines += ["- {}: ${:,.0f} vs ${:,.0f} ({:+.1f}%)".format(r.Index,r.actual,r.py,(r.actual-r.py)/r.py*100 if r.py else 0) for r in cx.itertuples()]
        return "\n".join(lines)

    # Daily / weekly / period summaries
    if any(x in m for x in ['yesterday','today','latest day','last day']):
        day = df[df['date']==latest]
        if day.empty: return "No data for {}.".format(latest.strftime('%b %d'))
        rev = day['actual_revenue_usd'].sum()
        bud = day['budget_usd'].sum() if 'budget_usd' in day.columns else 0
        ach = " ({:.1f}% of budget)".format(rev/bud*100) if bud>0 else ""
        cx_b = "\n".join(["  - {}: ${:,.0f}".format(r.complex,r.actual_revenue_usd) for r in day.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
        return "{} revenue: ${:,.0f}{}\nBy complex:\n{}".format(latest.strftime('%b %d'),rev,ach,cx_b)

    if any(x in m for x in ['7-day','7 day','last week','this week','weekly','week']):
        w = df[df['date'] >= latest - timedelta(days=7)]
        total = w['actual_revenue_usd'].sum(); davg = w.groupby('date')['actual_revenue_usd'].sum().mean()
        bud = w['budget_usd'].sum() if 'budget_usd' in w.columns else 0
        ach = " ({:.1f}% of budget)".format(total/bud*100) if bud>0 else ""
        cx_b = "\n".join(["  - {}: ${:,.0f}".format(r.complex,r.actual_revenue_usd) for r in w.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
        return "Last 7 days: ${:,.0f} total{}, avg ${:,.0f}/day\nBy complex:\n{}".format(total,ach,davg,cx_b)

    if any(x in m for x in ['30-day','30 day','this month','monthly','month to date','mtd']):
        w = df[df['date'] >= latest - timedelta(days=30)]
        total = w['actual_revenue_usd'].sum()
        bud = w['budget_usd'].sum() if 'budget_usd' in w.columns else 0
        ach = " ({:.1f}% of budget)".format(total/bud*100) if bud>0 else ""
        return "Last 30 days: ${:,.0f} total{}".format(total,ach)

    # Customer metrics
    if any(x in m for x in ['customer','spend per','avg spend','average spend','foot traffic','footfall']):
        if 'customer_count' not in df.columns or df['customer_count'].sum()==0:
            return "No customer count data available."
        avs = df[df['customer_count']>0].groupby('complex').apply(
            lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()).sort_values(ascending=False)
        total_c = int(df['customer_count'].sum())
        return "Total customers: {:,}\nAvg spend per customer:\n{}".format(
            total_c, "\n".join(["- {}: ${:.2f}".format(cx,v) for cx,v in avs.items()]))

    # Counter / throughput
    if any(x in m for x in ['counter','till','throughput','revenue per counter']):
        if 'revenue_per_counter' not in df.columns: return "No counter data."
        rpc = df[df['revenue_per_counter']>0].groupby('complex')['revenue_per_counter'].mean().sort_values(ascending=False)
        return "Revenue per counter:\n" + "\n".join(["- {}: ${:,.0f}".format(cx,v) for cx,v in rpc.items()])

    # Day of week
    if any(x in m for x in ['best day','peak day','busiest','day of week','highest day']):
        dow = df.groupby(df['date'].dt.day_name())['actual_revenue_usd'].mean()
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = dow.reindex([d for d in order if d in dow.index])
        return "Average daily revenue by day:\n" + "\n".join(
            ["- {}: ${:,.0f}".format(d,v) for d,v in dow.sort_values(ascending=False).items()])

    # Rankings
    if any(x in m for x in ['rank','ranking','all complex','all brand','compare all','overview']):
        cx = df.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False)
        br = df.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False)
        bud_cx = df.groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if 'budget_usd' in x.columns and x['budget_usd'].sum()>0 else 0) if 'budget_usd' in df.columns else None
        cx_lines = "\n".join(["{}. {}: ${:,.0f}{}".format(i+1,n,v," ({:.0f}% bud)".format(bud_cx[n]) if bud_cx is not None else "") for i,(n,v) in enumerate(cx.items())])
        br_lines = "\n".join(["{}. {}: ${:,.0f}".format(i+1,n,v) for i,(n,v) in enumerate(br.items())])
        return "Complex ranking:\n{}\n\nBrand ranking:\n{}".format(cx_lines,br_lines)

    # Top / bottom
    if any(x in m for x in ['top','best','highest','leading','number one','#1']):
        g_cx=df.groupby('complex')['actual_revenue_usd'].sum(); g_br=df.groupby('brand')['actual_revenue_usd'].sum()
        return "Top complex: {} (${:,.0f})\nTop brand: {} (${:,.0f})".format(g_cx.idxmax(),g_cx.max(),g_br.idxmax(),g_br.max())

    if any(x in m for x in ['worst','lowest','bottom','poor','weakest']):
        g_cx=df.groupby('complex')['actual_revenue_usd'].sum(); g_br=df.groupby('brand')['actual_revenue_usd'].sum()
        return "Lowest complex: {} (${:,.0f})\nLowest brand: {} (${:,.0f})".format(g_cx.idxmin(),g_cx.min(),g_br.idxmin(),g_br.min())

    if any(x in m for x in ['total revenue','overall revenue','total']):
        bud = df['budget_usd'].sum() if 'budget_usd' in df.columns else 0
        ach = " ({:.1f}% of budget)".format(df['actual_revenue_usd'].sum()/bud*100) if bud>0 else ""
        return "Total revenue: ${:,.0f}{} across {:,} records".format(df['actual_revenue_usd'].sum(),ach,len(df))

    # Complex × Brand cross-query
    for cx in COMPLEXES:
        for br in BRANDS:
            if cx.lower() in m and br.lower() in m:
                sub=df[(df['complex']==cx)&(df['brand']==br)]
                if not sub.empty:
                    bud=sub['budget_usd'].sum() if 'budget_usd' in sub.columns else 0
                    ach=" ({:.1f}% budget)".format(sub['actual_revenue_usd'].sum()/bud*100) if bud>0 else ""
                    return "{}/{}: ${:,.0f} total{}, ${:,.0f}/day avg".format(
                        cx,br,sub['actual_revenue_usd'].sum(),ach,sub['actual_revenue_usd'].mean())

    for cx in COMPLEXES:
        if cx.lower() in m:
            sub=df[df['complex']==cx]
            bud=sub['budget_usd'].sum() if 'budget_usd' in sub.columns else 0
            ach=" ({:.1f}% of budget)".format(sub['actual_revenue_usd'].sum()/bud*100) if bud>0 else ""
            br_b="\n".join(["  - {}: ${:,.0f}".format(r.brand,r.actual_revenue_usd) for r in sub.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False).reset_index().itertuples()])
            return "{}: ${:,.0f} total{}\nBy brand:\n{}".format(cx,sub['actual_revenue_usd'].sum(),ach,br_b)

    for br in BRANDS:
        if br.lower() in m:
            sub=df[df['brand']==br]
            bud=sub['budget_usd'].sum() if 'budget_usd' in sub.columns else 0
            ach=" ({:.1f}% of budget)".format(sub['actual_revenue_usd'].sum()/bud*100) if bud>0 else ""
            return "{}: ${:,.0f} total{}".format(br,sub['actual_revenue_usd'].sum(),ach)

    return None
def build_system_prompt():
    df = get_df()
    if df.empty: return "No data loaded. Tell the user to refresh data."
    cx_rev = df.groupby('complex')['actual_revenue_usd'].sum().sort_values(ascending=False)
    br_rev = df.groupby('brand')['actual_revenue_usd'].sum().sort_values(ascending=False)
    has_bud = 'budget_usd' in df.columns and df['budget_usd'].sum() > 0
    cx_bud = df.groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0) if has_bud else None
    br_bud = df.groupby('brand').apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum()*100 if x['budget_usd'].sum()>0 else 0) if has_bud else None
    date_range = df['date'].min().strftime('%b %d, %Y')+" to "+df['date'].max().strftime('%b %d, %Y')
    cx_lines = "\n".join(["- "+cx+": ${:,.0f}".format(v)+((" ({:.1f}% of budget)".format(cx_bud[cx])) if cx_bud is not None and cx in cx_bud else "") for cx,v in cx_rev.items()])
    br_lines = "\n".join(["- "+br+": ${:,.0f}".format(v)+((" ({:.1f}% of budget)".format(br_bud[br])) if br_bud is not None and br in br_bud else "") for br,v in br_rev.items()])
    top_cx = cx_rev.idxmax(); top_br = br_rev.idxmax()
    total = df['actual_revenue_usd'].sum()
    bud_ach = (total / df['budget_usd'].sum() * 100) if 'budget_usd' in df.columns and df['budget_usd'].sum()>0 else None
    return f"""You are a QSR operations intelligence assistant for Savanna QSR Group, Zimbabwe.
You have access to live operational data from {len(df)} records covering {date_range}.

TOTAL REVENUE: ${total:,.2f}{(" (budget achievement: {:.1f}%)".format(bud_ach)) if bud_ach else ""}
TOP COMPLEX: {top_cx}
TOP BRAND: {top_br}

REVENUE BY COMPLEX:
{cx_lines}

REVENUE BY BRAND:
{br_lines}

COMPLEXES: {', '.join(COMPLEXES)}
BRANDS: {', '.join(BRANDS)}

Answer questions directly and concisely using the data above. Use $ formatting for all revenue figures. If asked which is best/top/highest, use the data provided to give a specific answer."""

def chat(message, history):
    if not message or not message.strip(): return ""
    data_answer = route_intent(message)
    system = build_system_prompt()
    messages = [{"role":"system","content":system}]
    for h in (history or []):
        if isinstance(h, dict): messages.append({"role":h["role"],"content":h["content"]})
        elif isinstance(h, (list,tuple)) and len(h)==2:
            if h[0]: messages.append({"role":"user","content":str(h[0])})
            if h[1]: messages.append({"role":"assistant","content":str(h[1])})
    messages.append({"role":"user","content":message+"\n\n[DATA]: "+str(data_answer) if data_answer else message})
    try:
        r2 = _http.post("https://api.openai.com/v1/chat/completions", headers={"Authorization":"Bearer "+os.environ.get("OPENAI_API_KEY",""),"Content-Type":"application/json"}, json={"model":"gpt-4o-mini","messages":messages,"temperature":0.2,"max_tokens":600}); resp = r2.json()
        return resp["choices"][0]["message"]["content"]
    except Exception as e: return "Error: "+str(e)
css = """
body,.gradio-container{background:#050d1a!important;font-family:Arial,sans-serif!important;}
button[class*="tab-"]{color:#7fb3d3!important;background:transparent!important;border:none!important;border-bottom:3px solid transparent!important;padding:10px 18px!important;font-size:13px!important;}
button[class*="tab-"][class*="selected"],div[role="tablist"] button[aria-selected="true"]{color:#c9a84c!important;border-bottom:3px solid #c9a84c!important;font-weight:700!important;}
.gradio-container *{color:#c8d8f0;}
.gradio-container input,.gradio-container textarea{background:#0a1628!important;color:#c8d8f0!important;border:1px solid #1a3a6e!important;border-radius:6px!important;}
button.primary,button[variant="primary"]{background:#c9a84c!important;color:#0a1628!important;font-weight:700!important;border:none!important;border-radius:6px!important;}
button.secondary,button[variant="secondary"]{background:#1a3a6e!important;color:#c8d8f0!important;border:1px solid #c9a84c!important;border-radius:6px!important;}
.gradio-container .block,.gradio-container .form,.gradio-container .panel{background:#0a1628!important;border:1px solid #1a3a6e!important;border-radius:8px!important;}
div[class*="chatbot"],.chatbot{background:#040c1a!important;border-radius:12px!important;}
.gradio-container .prose{color:#c8d8f0!important;}
.gradio-container .prose strong{color:#c9a84c!important;}
footer{display:none!important;}
"""
with gr.Blocks(title="Savanna QSR Intelligence", css=css) as demo:
    kpi_header = gr.HTML(value=build_kpi_html())
    with gr.Tabs():
        with gr.TabItem("Dashboard"):
            with gr.Row():
                period_sel = gr.Radio(choices=['Last 7 Days','Last 30 Days','Last 90 Days','MTD','All Time'],value='All Time',label="Period")
                with gr.Column(scale=0, min_width=200):
                    dash_btn = gr.Button("Load Dashboard", variant="primary")
                    refresh_btn = gr.Button("Refresh Data", variant="secondary")
                    refresh_msg = gr.Markdown()
            with gr.Row(): ch1=gr.Plot(show_label=False); ch2=gr.Plot(show_label=False)
            with gr.Row(): ch3=gr.Plot(show_label=False); ch4=gr.Plot(show_label=False)
            with gr.Row(): ch5=gr.Plot(show_label=False); ch6=gr.Plot(show_label=False)
            dash_btn.click(build_dashboard,[period_sel],[kpi_header,ch1,ch2,ch3,ch4,ch5,ch6])
            period_sel.change(build_dashboard,[period_sel],[kpi_header,ch1,ch2,ch3,ch4,ch5,ch6])
            refresh_btn.click(refresh_data,[],[kpi_header,refresh_msg])
        with gr.TabItem("Forecast"):
            with gr.Row():
                seg_type = gr.Radio(choices=['Overall','By Complex','By Brand','Complex x Brand'],value='Overall',label="Segment")
                seg_name = gr.Dropdown(choices=['All'],value='All',label="Name",interactive=True)
                horizon = gr.Radio(choices=["7","14","30","60"],value="30",label="Horizon (days)")
                fc_btn = gr.Button("Generate", variant="primary")
            fc_chart = gr.Plot(show_label=False); fc_summary = gr.Markdown()
            seg_type.change(update_seg_choices,[seg_type],[seg_name])
            fc_btn.click(generate_forecast,[seg_type,seg_name,horizon],[fc_chart,fc_summary])
        with gr.TabItem("Intelligence Chat"):
            chatbot_box = gr.Chatbot(height=420,show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Ask about Savanna QSR...",show_label=False,scale=9,container=False)
                chat_send = gr.Button("Send",variant="primary",scale=1)
            def respond(message, history):
                if not message or not message.strip(): return history or [], ""
                reply = chat(message, history or [])
                h = list(history or []); h.append((message, reply))
                return h, ""
            chat_send.click(respond,[chat_input,chatbot_box],[chatbot_box,chat_input])
            chat_input.submit(respond,[chat_input,chatbot_box],[chatbot_box,chat_input])
    gr.HTML('<div style="text-align:center;margin-top:16px;padding:12px;border-top:1px solid #1a3a6e;"><p style="color:#c9a84c;font-size:11px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p><p style="color:#4a6a9e;font-size:11px;">Data - Analytics - Intelligence</p></div>')
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

