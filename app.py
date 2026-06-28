import os
import base64
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════════════
DATA_URL = "https://github.com/SYLVESTER1922/QSR/raw/refs/heads/main/simbisa_kenya_master_published.csv"
df = pd.read_csv(DATA_URL)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

forecasts_df = pd.read_csv('qsr_forecasts.csv')
forecasts_df['ds'] = pd.to_datetime(forecasts_df['ds'])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

RELIABLE_SITES = [
    'Crossroads Mall', 'Junction Plaza', 'Piazza Court',
    'Garden Court', 'Metro Market', 'Bon Marche Plaza'
]
BRANDS = df['brand'].unique().tolist()

SEGMENT_COLORS = {
    'All Sites & Brands': '#c9a84c',
    'Crust Co.':          '#c9a84c',
    'Flame & Feather':    '#2ecc71',
    'Cala Grill':         '#e74c3c',
    'Frostbite Creamery': '#9b59b6',
    'Crossroads Mall':    '#c9a84c',
    'Junction Plaza':     '#2ecc71',
    'Piazza Court':       '#e74c3c',
    'Garden Court':       '#1abc9c',
    'Metro Market':       '#9b59b6',
    'Bon Marche Plaza':   '#f39c12',
}

# ── Logo: load as base64 so it works on any deployment ─────────
def load_logo_b64():
    for fname in ['NI_logo.png', 'NI logo.png', 'ni_logo.png']:
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    return None

LOGO_B64 = load_logo_b64()
LOGO_HTML = f'<img src="{LOGO_B64}" style="height:58px;width:auto;object-fit:contain;border-radius:6px;" alt="NI"/>' if LOGO_B64 else '<div style="width:58px;height:58px;background:#c9a84c;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:700;color:#0a1628;font-size:14px;">NI</div>'


def hex_to_rgba(hex_color, alpha=0.15):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

def fmt_val(v):
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"

def dark_layout(title, height=350, margin=None):
    m = margin or dict(l=70, r=60, t=55, b=50)
    return dict(
        title=dict(text=title, font=dict(color='#c9a84c', size=14, family='Arial')),
        paper_bgcolor='#0a1628',
        plot_bgcolor='#0d1f38',
        font=dict(color='#c8d8f0', family='Arial', size=12),
        height=height,
        margin=m,
        xaxis=dict(
            gridcolor='#1a3a6e', linecolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0', size=11),
            title_font=dict(color='#a8c8f0', size=12)
        ),
        yaxis=dict(
            gridcolor='#1a3a6e', linecolor='#1a3a6e',
            tickfont=dict(color='#c8d8f0', size=11),
            title_font=dict(color='#a8c8f0', size=12)
        ),
        legend=dict(
            bgcolor='rgba(10,22,40,0.85)',
            bordercolor='#1a3a6e', borderwidth=1,
            font=dict(color='#c8d8f0', size=11)
        ),
    )


# ═══════════════════════════════════════════════════════════════
# TAB 1 — FORECASTING
# ═══════════════════════════════════════════════════════════════

def generate_forecast(segment_type, segment_name, horizon):
    try:
        seg_map = {'Overall': 'overall', 'By Brand': 'brand', 'By Site': 'site'}
        seg_key = seg_map.get(segment_type, 'overall')

        fc = forecasts_df[
            (forecasts_df['segment_type'] == seg_key) &
            (forecasts_df['segment_name'] == segment_name)
        ].copy().head(int(horizon))

        if fc.empty:
            fig = go.Figure()
            fig.update_layout(**dark_layout("No forecast available for this selection"))
            return fig, "⚠️ No forecast data found. Please select a different segment."

        # Historical daily revenue
        if seg_key == 'overall':
            hist = df.groupby('date')['daily_revenue_usd'].sum().reset_index()
        elif seg_key == 'brand':
            hist = df[df['brand'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index()
        else:
            hist = df[df['site'] == segment_name].groupby('date')['daily_revenue_usd'].sum().reset_index()

        hist = hist.sort_values('date').reset_index(drop=True)
        hist.columns = ['ds', 'y']
        color = SEGMENT_COLORS.get(segment_name, '#c9a84c')

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist['ds'], y=hist['y'],
            name='Historical Revenue',
            line=dict(color='#4a7aae', width=1.2),
            mode='lines', opacity=0.9,
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Revenue: $%{y:,.0f}<extra>Historical</extra>'
        ))

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([fc['ds'], fc['ds'][::-1]]),
            y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor=hex_to_rgba(color, 0.20),
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band',
            hoverinfo='skip'
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc['ds'], y=fc['yhat'],
            name=f'{horizon}-Day Forecast',
            line=dict(color=color, width=2.5, dash='dash'),
            mode='lines',
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra>Forecast</extra>'
        ))

        # Forecast start line
        fig.add_shape(
            type="line",
            x0=fc['ds'].min(), x1=fc['ds'].min(),
            y0=0, y1=1, yref="paper",
            line=dict(color=color, dash="dot", width=1.5)
        )
        fig.add_annotation(
            x=fc['ds'].min(), y=0.97, yref="paper",
            text="▶ Forecast Start", showarrow=False,
            font=dict(color=color, size=11, family='Arial'),
            bgcolor="rgba(10,22,40,0.75)", borderpad=4, xanchor="left"
        )

        if seg_key == 'overall':
            fig.add_annotation(
                x=0.01, y=0.04, xref="paper", yref="paper",
                text="Note: Upward trend reflects additional sites opening throughout 2020–2021",
                showarrow=False, font=dict(color='#7fb3d3', size=10),
                bgcolor="rgba(10,22,40,0.65)", borderpad=3, xanchor="left"
            )

        layout = dark_layout(
            f"{segment_name} — {horizon}-Day Revenue Forecast",
            height=500, margin=dict(l=80, r=30, t=70, b=55)
        )
        layout['xaxis']['title'] = 'Date'
        layout['yaxis']['title'] = 'Daily Revenue (USD)'
        layout['yaxis']['tickprefix'] = '$'
        layout['yaxis']['tickformat'] = ',.0f'
        layout['legend'] = dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            bgcolor='rgba(10,22,40,0.8)', bordercolor='#1a3a6e', borderwidth=1,
            font=dict(color='#c8d8f0', size=11)
        )
        layout['hovermode'] = 'x unified'
        fig.update_layout(**layout)

        total    = fc['yhat'].sum()
        avg      = fc['yhat'].mean()
        peak     = fc['yhat'].max()
        peak_day = fc.loc[fc['yhat'].idxmax(), 'ds'].strftime('%b %d, %Y')
        low      = fc['yhat'].min()
        low_day  = fc.loc[fc['yhat'].idxmin(), 'ds'].strftime('%b %d, %Y')

        summary = f"""**{horizon}-Day Forecast — {segment_name}**

| Metric | Value |
|---|---|
| Predicted Total Revenue | **${total:,.2f}** |
| Average Daily Revenue | **${avg:,.2f}** |
| Peak Day | **${peak:,.2f}** on {peak_day} |
| Lowest Day | **${low:,.2f}** on {low_day} |
| Forecast Period | {fc['ds'].min().strftime('%b %d, %Y')} → {fc['ds'].max().strftime('%b %d, %Y')} |

> *Forecast generated using Facebook Prophet trained on 2020–2021 data. Confidence band shows 80% prediction interval.*
"""
        return fig, summary

    except Exception as ex:
        fig = go.Figure()
        fig.update_layout(**dark_layout(f"Error generating forecast: {ex}"))
        return fig, f"⚠️ Error: {ex}"


def update_segment_choices(segment_type):
    if segment_type == 'Overall':
        return gr.update(choices=['All Sites & Brands'], value='All Sites & Brands')
    elif segment_type == 'By Brand':
        return gr.update(choices=BRANDS, value=BRANDS[0])
    else:
        return gr.update(choices=RELIABLE_SITES, value=RELIABLE_SITES[0])


# ═══════════════════════════════════════════════════════════════
# TAB 2 — INTELLIGENCE CHAT
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are a QSR revenue intelligence assistant for the Continental QSR Group, Nairobi Kenya.
You have access to 2 years of verified daily revenue data (Jan 2020 to Dec 2021) across 8 sites and 4 brands.
Answer questions clearly, concisely and professionally. Use $ formatting for all dollar amounts.
Always be transparent about data limitations when relevant.

KEY FACTS:
- Total 2-year revenue: $15,868,509
- Date range: Jan 2020 to Dec 2021
- Brands: {', '.join(BRANDS)}
- Sites: Crossroads Mall, Junction Plaza, Piazza Court, Garden Court, Metro Market, Nairobi Central, Bon Marche Plaza, Harbor Plaza

REVENUE BY BRAND:
- Crust Co.: $8,990,500 (57% of total — dominant brand, leads at 7 of 8 sites)
- Flame & Feather: $3,750,907 (24% — leads only at Metro Market)
- Cala Grill: $1,939,703 (12%)
- Frostbite Creamery: $1,187,399 (7%)

REVENUE BY SITE:
- Junction Plaza: $4,821,962 (top site)
- Crossroads Mall: $4,702,790 (close second)
- Piazza Court: $2,720,070
- Garden Court: $1,764,552
- Metro Market: $720,592
- Nairobi Central: $614,332
- Bon Marche Plaza: $467,850
- Harbor Plaza: $56,362 (smallest — just opened Dec 2021)

YEAR ON YEAR: 2020 $7,460,819 | 2021 $8,407,690 | Growth +12.7%

DAY OF WEEK AVERAGES: Sunday $2,142 (peak) | Saturday $1,555 | Friday $1,288 | Thursday $1,156 | Wednesday $1,045 | Tuesday $1,472 | Monday $988 (lowest)

90-DAY FORECASTS (Jan to Mar 2022):
- All Sites & Brands: $2,742,612 ($30,473/day avg)
- Crust Co.: $1,688,842 | Flame & Feather: $581,721
- Cala Grill: $313,839 | Frostbite Creamery: $204,011
- Crossroads Mall: $731,583 | Junction Plaza: $722,065
- Piazza Court: $453,255 | Garden Court: $317,608

DATA LIMITATIONS:
- Harbor Plaza: only 3 weeks of data (opened Dec 2021) — forecasts unreliable
- Nairobi Central: data only Jan to Mar 2020 — likely closed, no forecast
- Metro Market: data ends Nov 2021
- Bon Marche Plaza: opened Aug 2021, only 4 months of data

COVID-19: April 2020 revenue dropped 56% vs January 2020 ($339K vs $781K). Full recovery by 2021 (+12.7% YoY)."""


def chat(message, history):
    # Gradio 5 passes history as list of dicts with 'role' and 'content'
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history:
        if isinstance(h, dict):
            messages.append({"role": h["role"], "content": h["content"]})
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            if h[0]: messages.append({"role": "user",      "content": str(h[0])})
            if h[1]: messages.append({"role": "assistant", "content": str(h[1])})
    messages.append({"role": "user", "content": message})
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.2, max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as ex:
        return f"⚠️ Error: {ex}"


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
def build_dashboard():
    BG  = '#0a1628'
    PLT = '#0d1f38'
    GRD = '#1a3a6e'
    TXT = '#c8d8f0'
    GLD = '#c9a84c'

    def base(title, height=340):
        return dict(
            title=dict(text=title, font=dict(color=GLD, size=14)),
            paper_bgcolor=BG, plot_bgcolor=PLT,
            font=dict(color=TXT, family='Arial', size=11),
            height=height,
            xaxis=dict(gridcolor=GRD, linecolor=GRD, tickfont=dict(color=TXT)),
            yaxis=dict(gridcolor=GRD, linecolor=GRD, tickfont=dict(color=TXT)),
            legend=dict(bgcolor='rgba(10,22,40,0.85)', bordercolor=GRD,
                        borderwidth=1, font=dict(color=TXT)),
        )

    # 1. Brand revenue
    brand_rev = df.groupby('brand')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig1 = go.Figure(go.Bar(
        x=brand_rev['daily_revenue_usd'], y=brand_rev['brand'],
        orientation='h',
        marker=dict(color=['#9b59b6','#e74c3c','#2ecc71','#c9a84c']),
        text=[fmt_val(v) for v in brand_rev['daily_revenue_usd']],
        textposition='outside', textfont=dict(color=TXT, size=11)
    ))
    fig1.update_layout(
        title=dict(text="Total Revenue by Brand (2020–2021)", font=dict(color=GLD, size=14)),
        paper_bgcolor=BG, plot_bgcolor=PLT,
        font=dict(color=TXT, family='Arial'),
        height=300,
        xaxis=dict(title="Revenue (USD)", gridcolor=GRD, linecolor=GRD,
                   tickfont=dict(color=TXT), tickformat="$,.0f",
                   range=[0, brand_rev['daily_revenue_usd'].max() * 1.3]),
        yaxis=dict(tickfont=dict(color=TXT)),
        margin=dict(l=150, r=90, t=50, b=40)
    )

    # 2. Site revenue
    site_rev = df.groupby('site')['daily_revenue_usd'].sum().sort_values(ascending=True).reset_index()
    fig2 = go.Figure(go.Bar(
        x=site_rev['daily_revenue_usd'], y=site_rev['site'],
        orientation='h',
        marker=dict(color=['#34495e','#34495e','#1abc9c','#9b59b6',
                           '#e74c3c','#2ecc71','#c9a84c','#c9a84c']),
        text=[fmt_val(v) for v in site_rev['daily_revenue_usd']],
        textposition='outside', textfont=dict(color=TXT, size=11)
    ))
    fig2.update_layout(
        title=dict(text="Total Revenue by Site (2020–2021)", font=dict(color=GLD, size=14)),
        paper_bgcolor=BG, plot_bgcolor=PLT,
        font=dict(color=TXT, family='Arial'),
        height=380,
        xaxis=dict(title="Revenue (USD)", gridcolor=GRD, linecolor=GRD,
                   tickfont=dict(color=TXT), tickformat="$,.0f",
                   range=[0, site_rev['daily_revenue_usd'].max() * 1.32]),
        yaxis=dict(tickfont=dict(color=TXT)),
        margin=dict(l=165, r=90, t=50, b=40)
    )

    # 3. Monthly trend
    monthly = df.groupby(df['date'].dt.to_period('M'))['daily_revenue_usd'].sum().reset_index()
    monthly['date'] = monthly['date'].astype(str)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=monthly['date'], y=monthly['daily_revenue_usd'],
        mode='lines+markers',
        line=dict(color=GLD, width=2.5),
        marker=dict(size=5, color=GLD),
        fill='tozeroy', fillcolor='rgba(201,168,76,0.08)'
    ))
    fig3.add_shape(type="rect", x0="2020-03", x1="2020-06",
                   y0=0, y1=1, yref="paper",
                   fillcolor="red", opacity=0.08, line_width=0)
    fig3.add_annotation(x="2020-04", y=0.88, yref="paper",
                        text="COVID-19", showarrow=False,
                        font=dict(color="#ff6b6b", size=10),
                        bgcolor="rgba(10,22,40,0.65)", borderpad=3)
    fig3.update_layout(
        title=dict(text="Monthly Revenue Trend (2020–2021)", font=dict(color=GLD, size=14)),
        paper_bgcolor=BG, plot_bgcolor=PLT,
        font=dict(color=TXT, family='Arial'),
        height=360, showlegend=False,
        xaxis=dict(title="Month", tickangle=45, gridcolor=GRD,
                   tickfont=dict(color=TXT)),
        yaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor=GRD, tickfont=dict(color=TXT)),
        margin=dict(l=80, r=40, t=50, b=80)
    )

    # 4. Day of week
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = df.groupby(df['date'].dt.day_name())['daily_revenue_usd'].mean().reindex(dow_order).reset_index()
    dow.columns = ['day', 'avg_revenue']
    fig4 = go.Figure(go.Bar(
        x=dow['day'], y=dow['avg_revenue'],
        marker=dict(color=['#c9a84c' if d=='Sunday' else '#1e3a6e' for d in dow['day']]),
        text=[f"${v:,.0f}" for v in dow['avg_revenue']],
        textposition='outside', textfont=dict(color=TXT, size=11)
    ))
    fig4.update_layout(
        title=dict(text="Avg Daily Revenue by Day of Week", font=dict(color=GLD, size=14)),
        paper_bgcolor=BG, plot_bgcolor=PLT,
        font=dict(color=TXT, family='Arial'),
        height=320,
        xaxis=dict(title="Day", tickfont=dict(color=TXT)),
        yaxis=dict(title="Avg Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor=GRD, tickfont=dict(color=TXT),
                   range=[0, dow['avg_revenue'].max() * 1.28]),
        margin=dict(l=80, r=40, t=50, b=40)
    )

    # 5. YoY by brand
    yoy = df.groupby([df['date'].dt.year, 'brand'])['daily_revenue_usd'].sum().reset_index()
    yoy.columns = ['year', 'brand', 'revenue']
    bc = {'Crust Co.':'#c9a84c','Flame & Feather':'#2ecc71',
          'Cala Grill':'#e74c3c','Frostbite Creamery':'#9b59b6'}
    fig5 = go.Figure()
    for brand in BRANDS:
        b = yoy[yoy['brand'] == brand]
        fig5.add_trace(go.Bar(
            x=b['year'].astype(str), y=b['revenue'], name=brand,
            marker_color=bc.get(brand, GLD),
            text=[fmt_val(v) for v in b['revenue']],
            textposition='outside', textfont=dict(color=TXT, size=10)
        ))
    fig5.update_layout(
        title=dict(text="Year-on-Year Revenue by Brand", font=dict(color=GLD, size=14)),
        paper_bgcolor=BG, plot_bgcolor=PLT,
        font=dict(color=TXT, family='Arial'),
        height=340, barmode='group',
        xaxis=dict(title="Year", tickfont=dict(color=TXT)),
        yaxis=dict(title="Revenue (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor=GRD, tickfont=dict(color=TXT)),
        legend=dict(bgcolor='rgba(10,22,40,0.85)', bordercolor=GRD,
                    borderwidth=1, font=dict(color=TXT)),
        margin=dict(l=80, r=40, t=50, b=40)
    )

    # 6. Revenue share pie
    site_rev2 = df.groupby('site')['daily_revenue_usd'].sum().reset_index()
    fig6 = go.Figure(go.Pie(
        labels=site_rev2['site'], values=site_rev2['daily_revenue_usd'],
        hole=0.45,
        marker=dict(colors=['#c9a84c','#1e2d5e','#2ecc71','#e74c3c',
                            '#9b59b6','#1abc9c','#f39c12','#34495e'],
                    line=dict(color='#0a1628', width=2)),
        textfont=dict(color='#ffffff', size=11)
    ))
    fig6.update_layout(
        title=dict(text="Revenue Share by Site", font=dict(color=GLD, size=14)),
        paper_bgcolor=BG, font=dict(color=TXT),
        height=340, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor='rgba(10,22,40,0.85)', bordercolor=GRD,
                    borderwidth=1, font=dict(color=TXT))
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


# ═══════════════════════════════════════════════════════════════
# CSS — Comprehensive dark theme for Gradio 5
# ═══════════════════════════════════════════════════════════════

css = """
/* ── Base ──────────────────────────────────────────────── */
body, .gradio-container, .main {
    background-color: #050d1a !important;
    font-family: Arial, sans-serif !important;
}

/* ── ALL text defaults to light ─────────────────────────── */
.gradio-container, .gradio-container * {
    color: #c8d8f0;
}

/* ── Tab bar — Gradio 5 selectors ───────────────────────── */
.tabs > .tab-nav,
[class*="tabs"] > [class*="tab-nav"],
div.tab-nav {
    background-color: #0a1628 !important;
    border-bottom: 2px solid #1a3a6e !important;
}

/* Every tab button */
.tab-nav button,
[class*="tab-nav"] button,
div[role="tablist"] button {
    color: #7fb3d3 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 10px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

.tab-nav button:hover,
[class*="tab-nav"] button:hover,
div[role="tablist"] button:hover {
    color: #ffffff !important;
    background: rgba(201,168,76,0.07) !important;
}

/* Selected tab */
.tab-nav button.selected,
[class*="tab-nav"] button.selected,
div[role="tablist"] button[aria-selected="true"],
button[data-testid*="tab"][aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom: 3px solid #c9a84c !important;
    font-weight: 700 !important;
    background: transparent !important;
}

/* ── Labels ──────────────────────────────────────────────── */
label, .label-wrap, .label-wrap span,
fieldset legend, .form > label {
    color: #a8c8f0 !important;
    font-size: 13px !important;
}

/* ── Radio & checkbox text ───────────────────────────────── */
.gradio-container input[type="radio"] + span,
.gradio-container input[type="radio"] ~ span,
.gradio-container .wrap label span,
.gradio-container [data-testid="radio-group"] span,
.gradio-container .svelte-1cl284s span {
    color: #c8d8f0 !important;
}
input[type="radio"] { accent-color: #c9a84c !important; }
input[type="checkbox"] { accent-color: #c9a84c !important; }

/* ── Inputs & dropdowns ──────────────────────────────────── */
input, textarea, select,
.gradio-container input,
.gradio-container textarea {
    background-color: #0a1628 !important;
    color: #c8d8f0 !important;
    border: 1px solid #1a3a6e !important;
    border-radius: 6px !important;
}

/* ── Dropdown list ───────────────────────────────────────── */
ul[role="listbox"] {
    background-color: #0d1b2a !important;
    border: 1px solid #c9a84c !important;
    border-radius: 6px !important;
}
ul[role="listbox"] li {
    color: #ffffff !important;
    background-color: #0d1b2a !important;
    padding: 8px 12px !important;
}
ul[role="listbox"] li:hover,
ul[role="listbox"] li[aria-selected="true"] {
    background-color: #c9a84c !important;
    color: #0d1b2a !important;
    font-weight: 600 !important;
}

/* ── Panel / block backgrounds ───────────────────────────── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    background-color: #0a1628 !important;
    border-color: #1a3a6e !important;
    border-radius: 8px !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
button.primary, .gradio-container button.primary,
button[variant="primary"] {
    background-color: #c9a84c !important;
    color: #0a1628 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
}
button.primary:hover { background-color: #e0be6a !important; }

button.secondary, .gradio-container button.secondary {
    background-color: #1e2d5e !important;
    color: #c8d8f0 !important;
    border: 1px solid #c9a84c !important;
    border-radius: 6px !important;
}

/* ── Chat bubbles ────────────────────────────────────────── */
.message.user > div,
div[data-testid="user"] > div,
div[class*="message"][class*="user"] > div:last-child {
    background: linear-gradient(135deg, #1e2d5e, #162d5a) !important;
    color: #e8f0ff !important;
    border-radius: 18px 18px 4px 18px !important;
    border: 1px solid #2a4a8a !important;
    padding: 12px 16px !important;
}
.message.bot > div,
div[data-testid="bot"] > div,
div[class*="message"][class*="bot"] > div:last-child {
    background: linear-gradient(135deg, #0a1e3d, #112952) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 18px 4px !important;
    border: 1px solid rgba(42, 106, 160, 0.33) !important;
    padding: 12px 16px !important;
}
.message.bot > div *, div[data-testid="bot"] > div * { color: #ffffff !important; }
.message.bot > div strong, div[data-testid="bot"] strong { color: #c9a84c !important; }

div[class*="chatbot"], .chatbot {
    background-color: #040c1a !important;
    border-radius: 12px !important;
}

/* ── Markdown ────────────────────────────────────────────── */
.prose, .prose p, .prose li { color: #c8d8f0 !important; }
.prose strong { color: #c9a84c !important; }
.prose table { border-color: #1a3a6e !important; width: auto !important; }
.prose th { background-color: #0a1628 !important; color: #c9a84c !important; padding: 6px 12px !important; }
.prose td { border-color: #1a3a6e !important; color: #c8d8f0 !important; padding: 5px 12px !important; }
.prose tr:nth-child(even) td { background-color: rgba(10,22,40,0.5) !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #050d1a; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #e0be6a; }

footer { display: none !important; }
"""


# ═══════════════════════════════════════════════════════════════
# GRADIO APP
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(title="Continental QSR Intelligence | Netrisyl Insights", css=css) as demo:

    # ── Header ─────────────────────────────────────────────────
    gr.HTML(f"""
    <div style="background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 100%);
                padding:20px 28px 18px;border-radius:12px;margin-bottom:2px;
                border-left:4px solid #c9a84c;
                box-shadow:0 4px 24px rgba(0,0,0,0.45);">

        <div style="display:flex;align-items:center;justify-content:space-between;
                    flex-wrap:wrap;gap:14px;">
            <div style="display:flex;align-items:center;gap:16px;">
                {LOGO_HTML}
                <div>
                    <h1 style="color:#ffffff;margin:0;font-size:22px;font-weight:700;
                               letter-spacing:0.3px;line-height:1.2;">
                        🍔 Continental QSR Intelligence
                    </h1>
                    <p style="color:#aed6f1;margin:5px 0 0;font-size:13px;">
                        Revenue Analytics &amp; Forecasting &nbsp;·&nbsp;
                        Nairobi, Kenya &nbsp;·&nbsp; 2020–2021
                    </p>
                </div>
            </div>
            <div style="text-align:right;">
                <p style="color:#c9a84c;margin:0;font-size:10px;font-weight:700;
                          letter-spacing:2.5px;">NETRISYL INSIGHTS</p>
                <p style="color:#7fb3d3;margin:3px 0 0;font-size:11px;">
                    Data &nbsp;·&nbsp; Analytics &nbsp;·&nbsp; Intelligence
                </p>
            </div>
        </div>

        <div style="display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;">
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;
                        border-radius:8px;border:1px solid #1a3a6e;
                        min-width:85px;text-align:center;">
                <div style="color:#c9a84c;font-size:19px;font-weight:700;">$15.9M</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">Total Revenue</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;
                        border-radius:8px;border:1px solid #1a3a6e;
                        min-width:85px;text-align:center;">
                <div style="color:#2ecc71;font-size:19px;font-weight:700;">+12.7%</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">YoY Growth</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;
                        border-radius:8px;border:1px solid #1a3a6e;
                        min-width:85px;text-align:center;">
                <div style="color:#c9a84c;font-size:19px;font-weight:700;">8 Sites</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">4 Brands</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;
                        border-radius:8px;border:1px solid #1a3a6e;
                        min-width:85px;text-align:center;">
                <div style="color:#c9a84c;font-size:19px;font-weight:700;">$2.74M</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">90-Day Forecast</div>
            </div>
            <div style="background:rgba(10,22,40,0.7);padding:10px 20px;
                        border-radius:8px;border:1px solid #1a3a6e;
                        min-width:85px;text-align:center;">
                <div style="color:#c9a84c;font-size:19px;font-weight:700;">$23.9K</div>
                <div style="color:#7fb3d3;font-size:11px;margin-top:2px;">Avg Daily Rev</div>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Revenue Forecast ───────────────────────────
        with gr.TabItem("📈 Revenue Forecast"):
            gr.HTML("""<p style="color:#7fb3d3;font-size:12px;margin:10px 0 14px;">
                Select segment type, choose a specific segment, pick forecast horizon, then click Generate.</p>""")

            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=160):
                    seg_type = gr.Radio(
                        choices=["Overall", "By Brand", "By Site"],
                        value="Overall",
                        label="Segment Type"
                    )
                with gr.Column(scale=2, min_width=180):
                    seg_name = gr.Dropdown(
                        choices=["All Sites & Brands"],
                        value="All Sites & Brands",
                        label="Select Segment",
                        interactive=True
                    )
                with gr.Column(scale=2, min_width=160):
                    horizon = gr.Radio(
                        choices=[30, 60, 90],
                        value=90,
                        label="Forecast Horizon (days)"
                    )
                with gr.Column(scale=1, min_width=140):
                    forecast_btn = gr.Button("⚡ Generate", variant="primary")

            forecast_chart   = gr.Plot(show_label=False)
            forecast_summary = gr.Markdown()

            seg_type.change(update_segment_choices, [seg_type], [seg_name])
            forecast_btn.click(
                generate_forecast,
                [seg_type, seg_name, horizon],
                [forecast_chart, forecast_summary]
            )

        # ── Tab 2: Intelligence Chat ──────────────────────────
        with gr.TabItem("💬 Intelligence Chat"):
            gr.HTML("""
            <div style="background:#0a1628;border:1px solid #1a3a6e;border-radius:8px;
                        padding:14px 18px;margin:10px 0 16px;">
                <p style="color:#c9a84c;font-weight:700;margin:0 0 6px;font-size:13px;">
                    💡 Ask anything about the Continental QSR Group revenue data
                </p>
                <p style="color:#7fb3d3;font-size:12px;margin:0;line-height:1.8;">
                    <strong style="color:#c8d8f0;">"Which brand is the top performer?"</strong>
                    &nbsp;·&nbsp;
                    <strong style="color:#c8d8f0;">"What was the COVID-19 impact?"</strong>
                    &nbsp;·&nbsp;
                    <strong style="color:#c8d8f0;">"Compare Junction Plaza vs Crossroads Mall"</strong>
                    &nbsp;·&nbsp;
                    <strong style="color:#c8d8f0;">"Which site should we invest in?"</strong>
                </p>
            </div>""")
            chatbot = gr.ChatInterface(
                fn=chat,
                title="",
                type="messages",
                examples=[
                    "Which brand generates the most revenue?",
                    "What was the COVID-19 impact on revenue?",
                    "Which site should we prioritize for expansion?",
                    "What is the 90-day revenue forecast for Crust Co.?",
                    "Compare Junction Plaza and Crossroads Mall",
                    "What day of the week has the highest revenue?",
                    "Which sites have data limitations?",
                    "What is the year-on-year growth trend?",
                ]
            )

        # ── Tab 3: Analytics Dashboard ────────────────────────
        with gr.TabItem("📊 Analytics Dashboard"):
            gr.HTML("""<p style="color:#7fb3d3;font-size:12px;margin:10px 0 14px;">
                Click Load Dashboard to render all six analytics charts.</p>""")
            dash_btn = gr.Button("📊 Load Dashboard", variant="primary")
            with gr.Row():
                chart_brand = gr.Plot(show_label=False)
                chart_site  = gr.Plot(show_label=False)
            with gr.Row():
                chart_monthly = gr.Plot(show_label=False)
                chart_dow     = gr.Plot(show_label=False)
            with gr.Row():
                chart_yoy = gr.Plot(show_label=False)
                chart_pie = gr.Plot(show_label=False)
            dash_btn.click(
                build_dashboard, [],
                [chart_brand, chart_site, chart_monthly, chart_dow, chart_yoy, chart_pie]
            )

    # ── Footer ─────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;margin-top:20px;padding:14px 0;
                border-top:1px solid #1a3a6e;">
        <p style="color:#c9a84c;font-size:11px;font-weight:700;
                  margin:0;letter-spacing:2px;">NETRISYL INSIGHTS</p>
        <p style="color:#4a6a9e;font-size:11px;margin:5px 0 0;">
            Data · Analytics · Intelligence ·
            <a href="https://netrisyl.com" target="_blank"
               style="color:#7fb3d3;text-decoration:none;">netrisyl.com</a>
        </p>
        <p style="color:#2a4a6e;font-size:10px;margin:4px 0 0;">
            ⚠️ Data anonymized. All site and brand names are fictional.
            Built for demonstration purposes.
        </p>
    </div>""")

demo.launch(server_name="0.0.0.0", server_port=7860)
