import os, json, time, warnings
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timedelta
import requests as _http
warnings.filterwarnings('ignore')
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
        r = _http.get(SUPABASE_URL+"/rest/v1/daily_input?select=*&order=date", headers={"apikey":SUPABASE_KEY,"Authorization":"Bearer "+SUPABASE_KEY})
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
def build_kpi_html():
    df = get_df()
    if df.empty:
        tr,ap,tc,tb,dr = "--","--","--","--","No data"
    else:
        tr = fmt(df['actual_revenue_usd'].sum())
        bud = df['budget_usd'].sum() if 'budget_usd' in df.columns else 0
        ap = "{:.1f}%".format(df['actual_revenue_usd'].sum()/bud*100) if bud > 0 else "--"
        tc = df.groupby('complex')['actual_revenue_usd'].sum().idxmax()
        tb = df.groupby('brand')['actual_revenue_usd'].sum().idxmax()
        dr = df['date'].min().strftime('%b %d')+" - "+df['date'].max().strftime('%b %d, %Y')
    return '<div style="background:linear-gradient(135deg,#0d1b2a,#1a3a5c);padding:18px 28px 16px;border-radius:12px;margin-bottom:4px;border-left:4px solid #c9a84c;"><div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;"><div><h1 style="color:#fff;margin:0;font-size:22px;">Savanna QSR Intelligence</h1><p style="color:#aed6f1;margin:4px 0 0;font-size:13px;">Live Operations Intelligence - Zimbabwe</p></div><div style="text-align:right;"><p style="color:#c9a84c;margin:0;font-size:10px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p></div></div><div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;"><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+tr+'</div><div style="color:#7fb3d3;font-size:11px;">Total Revenue</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+ap+'</div><div style="color:#7fb3d3;font-size:11px;">Budget Achievement</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+tc+'</div><div style="color:#7fb3d3;font-size:11px;">Top Complex</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#c9a84c;font-size:18px;font-weight:700;">'+tb+'</div><div style="color:#7fb3d3;font-size:11px;">Top Brand</div></div><div style="background:rgba(10,22,40,0.7);padding:10px 18px;border-radius:8px;border:1px solid #1a3a6e;text-align:center;"><div style="color:#7fb3d3;font-size:16px;font-weight:700;">'+dr+'</div><div style="color:#7fb3d3;font-size:11px;">Data Range</div></div></div></div>'
def build_dashboard(period):
    df = get_df()
    if df.empty:
        e = go.Figure().update_layout(**dark("No data - click Refresh")); return [e]*6
    dff = df.copy(); latest = dff['date'].max()
    if period == 'Last 7 Days': dff = dff[dff['date'] >= latest - timedelta(days=7)]
    elif period == 'Last 30 Days': dff = dff[dff['date'] >= latest - timedelta(days=30)]
    elif period == 'Last 90 Days': dff = dff[dff['date'] >= latest - timedelta(days=90)]
    elif period == 'MTD': dff = dff[(dff['date'].dt.month==latest.month)&(dff['date'].dt.year==latest.year)]
    cx = dff.groupby('complex').agg(actual=('actual_revenue_usd','sum'),budget=('budget_usd','sum')).reset_index().sort_values('actual',ascending=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Budget',x=cx['budget'],y=cx['complex'],orientation='h',marker_color='#1a3a6e',opacity=0.7))
    fig1.add_trace(go.Bar(name='Actual',x=cx['actual'],y=cx['complex'],orientation='h',marker_color='#c9a84c',text=[fmt(v) for v in cx['actual']],textposition='outside',textfont=dict(color='#c8d8f0',size=11)))
    fig1.update_layout(**dark("Revenue by Complex",height=300),barmode='overlay',margin=dict(l=140,r=80,t=50,b=40))
    br = dff.groupby('brand')['actual_revenue_usd'].sum().reset_index().sort_values('actual_revenue_usd',ascending=True)
    fig2 = go.Figure(go.Bar(x=br['actual_revenue_usd'],y=br['brand'],orientation='h',marker_color=COLORS[:len(br)],text=[fmt(v) for v in br['actual_revenue_usd']],textposition='outside',textfont=dict(color='#c8d8f0',size=11)))
    fig2.update_layout(**dark("Revenue by Brand",height=280),margin=dict(l=140,r=80,t=50,b=40))
    daily = dff.groupby('date')['actual_revenue_usd'].sum().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=daily['date'],y=daily['actual_revenue_usd'],name='Actual',line=dict(color='#c9a84c',width=2.5),fill='tozeroy',fillcolor='rgba(201,168,76,0.08)'))
    fig3.update_layout(**dark("Daily Revenue Trend",height=360),hovermode='x unified',showlegend=False,margin=dict(l=70,r=40,t=50,b=50))
    if 'customer_count' in dff.columns:
        avs = dff[dff['customer_count']>0].groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()).reset_index(name='avg').sort_values('avg',ascending=True)
        fig4 = go.Figure(go.Bar(x=avs['avg'],y=avs['complex'],orientation='h',marker_color='#2ecc71',text=["${:.2f}".format(v) for v in avs['avg']],textposition='outside',textfont=dict(color='#c8d8f0',size=11)))
        fig4.update_layout(**dark("Avg Spend Per Customer",height=280),margin=dict(l=140,r=80,t=50,b=40))
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
    cmap = {'sdly':'#4a6a9e','sdlm':'#7fb3d3','actual':'#c9a84c'}
    lmap = {'sdly':'Prior Year','sdlm':'Prior Month','actual':'This Period'}
    for key, col in comp_cols.items():
        if col in comp.columns: fig6.add_trace(go.Bar(name=lmap[key],x=comp['complex'],y=comp[col],marker_color=cmap[key],text=[fmt(v) for v in comp[col]],textposition='outside',textfont=dict(color='#c8d8f0',size=10)))
    fig6.update_layout(**dark("Actual vs Prior Periods",height=340),barmode='group',hovermode='x unified',margin=dict(l=60,r=40,t=50,b=60))
    return fig1, fig2, fig3, fig4, fig5, fig6
def generate_forecast(seg_type, seg_name, horizon):
    df = get_df()
    if df.empty: return go.Figure().update_layout(**dark("No data")), "No data."
    horizon = int(horizon)
    if seg_type == 'Overall': seg = df.groupby('date')['actual_revenue_usd'].sum().reset_index()
    elif seg_type == 'By Complex': seg = df[df['complex']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
    elif seg_type == 'By Brand': seg = df[df['brand']==seg_name].groupby('date')['actual_revenue_usd'].sum().reset_index()
    else:
        parts = seg_name.split(' | ')
        seg = df[(df['complex']==parts[0])&(df['brand']==parts[1])].groupby('date')['actual_revenue_usd'].sum().reset_index() if len(parts)==2 else df.groupby('date')['actual_revenue_usd'].sum().reset_index()
    seg.columns = ['date','revenue']; seg = seg.sort_values('date').reset_index(drop=True)
    if len(seg) < 7: return go.Figure().update_layout(**dark("Need 7+ days")), "Insufficient data."
    last_date = seg['date'].max(); base = seg['revenue'].tail(min(horizon, len(seg))).mean()
    DOW = {0:0.85,1:0.90,2:0.92,3:0.95,4:1.05,5:1.20,6:1.15}
    fc_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    fc_vals = [round(base * DOW[d.weekday()] * (1+GROWTH_RATE), 2) for d in fc_dates]
    upper = [v*1.15 for v in fc_vals]; lower = [v*0.85 for v in fc_vals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg['date'],y=seg['revenue'],name='Historical',line=dict(color='#4a7aae',width=1.2),opacity=0.85))
    fig.add_trace(go.Scatter(x=fc_dates+fc_dates[::-1],y=upper+lower[::-1],fill='toself',fillcolor='rgba(201,168,76,0.18)',line=dict(color='rgba(0,0,0,0)'),name='Confidence',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fc_dates,y=fc_vals,name=str(horizon)+'-Day Forecast',line=dict(color='#c9a84c',width=2.5,dash='dash'),mode='lines+markers',marker=dict(size=4,color='#c9a84c')))
    fig.add_vline(x=last_date,line_dash='dot',line_color='#c9a84c',opacity=0.6)
    fig.update_layout(**dark(seg_name+" - "+str(horizon)+"-Day Forecast",height=480),hovermode='x unified',legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1,bgcolor='rgba(10,22,40,0.8)',bordercolor='#1a3a6e',borderwidth=1,font=dict(color='#c8d8f0')),margin=dict(l=70,r=30,t=70,b=50))
    total=sum(fc_vals); avg=total/horizon; peak=max(fc_vals); peak_d=fc_dates[fc_vals.index(peak)].strftime('%b %d, %Y')
    summary = "**"+str(horizon)+"-Day Forecast**\n\n| Metric | Value |\n|---|---|\n| Total | **${:,.2f}".format(total)+"** |\n| Daily Avg | **${:,.2f}".format(avg)+"** |\n| Peak | **${:,.2f}".format(peak)+" on "+peak_d+"** |"
    return fig, summary
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
    if any(x in m for x in ['underperform','below budget','struggling']):
        if 'budget_usd' not in df.columns: return "No budget data."
        latest = df['date'].max(); recent = df[df['date'] >= latest - timedelta(days=7)]
        perf = recent.groupby(['complex','brand']).apply(lambda x: x['actual_revenue_usd'].sum()/x['budget_usd'].sum() if x['budget_usd'].sum()>0 else 1).reset_index(name='ach')
        under = perf[perf['ach']<0.80]
        if under.empty: return "No sites below 80% budget (7d)."
        return "Below 80% (7d):\n" + "\n".join(["- "+r.complex+"/"+r.brand+": {:.1f}%".format(r.ach*100) for r in under.itertuples()])
    if any(x in m for x in ['top complex','best complex']): g=df.groupby('complex')['actual_revenue_usd'].sum(); return "Top: "+g.idxmax()+" at "+fmt(g.max())
    if any(x in m for x in ['top brand','best brand']): g=df.groupby('brand')['actual_revenue_usd'].sum(); return "Top: "+g.idxmax()+" at "+fmt(g.max())
    if any(x in m for x in ['avg spend','average spend']):
        if 'customer_count' not in df.columns: return "No customer data."
        avs=df[df['customer_count']>0].groupby('complex').apply(lambda x: x['actual_revenue_usd'].sum()/x['customer_count'].sum()).sort_values(ascending=False)
        return "Avg spend:\n"+"\n".join(["- "+cx+": ${:.2f}".format(v) for cx,v in avs.items()])
    if any(x in m for x in ['total revenue','overall']): return "Total: ${:,.2f}".format(df['actual_revenue_usd'].sum())
    if '7-day' in m or '7 day' in m: ma=df[df['date']>=df['date'].max()-timedelta(days=7)].groupby('date')['actual_revenue_usd'].sum().mean(); return "7-day avg: ${:,.2f}/day".format(ma)
    for cx in COMPLEXES:
        if cx.lower() in m:
            sub=df[df['complex']==cx]; return cx+": ${:,.2f}".format(sub['actual_revenue_usd'].sum())
    for br in BRANDS:
        if br.lower() in m:
            sub=df[df['brand']==br]; return br+": ${:,.2f}".format(sub['actual_revenue_usd'].sum())
    return None
def chat(message, history):
    if not message or not message.strip(): return ""
    data_answer = route_intent(message); df = get_df()
    system = "You are an intelligence assistant for Savanna QSR Group, Zimbabwe. Data: "+str(len(df))+" rows. Revenue: ${:,.2f}".format(df['actual_revenue_usd'].sum())+". Complexes: "+', '.join(COMPLEXES)+". Brands: "+', '.join(BRANDS)+". Be concise." if not df.empty else "No data loaded."
    messages = [{"role":"system","content":system}]
    for h in (history or []):
        if isinstance(h,dict): messages.append({"role":h["role"],"content":h["content"]})
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
            dash_btn.click(build_dashboard,[period_sel],[ch1,ch2,ch3,ch4,ch5,ch6])
            refresh_btn.click(refresh_data,[],[kpi_header,refresh_msg])
        with gr.TabItem("Forecast"):
            with gr.Row():
                seg_type = gr.Radio(choices=['Overall','By Complex','By Brand','Complex x Brand'],value='Overall',label="Segment")
                seg_name = gr.Dropdown(choices=['All'],value='All',label="Name",interactive=True)
                horizon = gr.Radio(choices=[7,14,30,60],value=30,label="Days")
                fc_btn = gr.Button("Generate", variant="primary")
            fc_chart = gr.Plot(show_label=False); fc_summary = gr.Markdown()
            seg_type.change(update_seg_choices,[seg_type],[seg_name])
            fc_btn.click(generate_forecast,[seg_type,seg_name,horizon],[fc_chart,fc_summary])
        with gr.TabItem("Intelligence Chat"):
            chatbot_box = gr.Chatbot(height=420,type="messages",show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Ask about Savanna QSR...",show_label=False,scale=9,container=False)
                chat_send = gr.Button("Send",variant="primary",scale=1)
            def respond(message, history):
                if not message or not message.strip(): return history or [], ""
                reply = chat(message, history or [])
                h = list(history or []); h.append({"role":"user","content":message}); h.append({"role":"assistant","content":reply})
                return h, ""
            chat_send.click(respond,[chat_input,chatbot_box],[chatbot_box,chat_input])
            chat_input.submit(respond,[chat_input,chatbot_box],[chatbot_box,chat_input])
    gr.HTML('<div style="text-align:center;margin-top:16px;padding:12px;border-top:1px solid #1a3a6e;"><p style="color:#c9a84c;font-size:11px;font-weight:700;letter-spacing:2px;">NETRISYL INSIGHTS</p><p style="color:#4a6a9e;font-size:11px;">Data - Analytics - Intelligence</p></div>')
demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

