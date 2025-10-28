# build.py
# Generates a self-contained index.html using Polygon.io data.
# Big 8: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AVGO

import os, math, datetime as dt
from polygon import RESTClient
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



API_KEY = os.environ.get("POLYGON_API_KEY")  # set in GitHub Secrets



TICKERS = ["AAPL"]
YEARS_BACK = 2
TIMESPAN = "day"  # free tier: use daily/minute as allowed; daily is safest for rate limits

def fetch_daily_aggs(client: RESTClient, ticker: str, start: str, end: str):
    # list_aggs returns a generator; iterate and collect to list → DataFrame
    rows = []
    for bar in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan=TIMESPAN,
        from_=start,
        to=end,
        adjusted=True,
        limit=50000,
        sort="asc",
    ):
        rows.append({
            "date": dt.datetime.utcfromtimestamp(bar.timestamp/1000).date(),
            "close": bar.close,
        })
    if not rows:
        return pd.DataFrame(columns=["date","close"])
    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("date").drop_duplicates("date")
    return df

def compute_metrics(px_df: pd.DataFrame):
    """px_df: index=date, columns=tickers (adjusted close)."""
    px_df = px_df.dropna(axis=0, how="any")  # align dates
    # daily log returns
    rets = np.log(px_df / px_df.shift(1)).dropna()
    # cumulative (base=100)
    cum = (rets.cumsum()).apply(np.exp) * 100.0
    # 1Y window (~252 trading days)
    window = min(252, len(rets))
    rets_1y = rets.tail(window)
    # total return 1Y
    total_ret_1y = (px_df.tail(1) / px_df.tail(window).iloc[0] - 1.0).iloc[0]
    # annualized vol (stdev daily × sqrt(252))
    ann_vol = rets_1y.std() * math.sqrt(252)
    return cum, total_ret_1y, ann_vol

def build_dashboard_html(cum: pd.DataFrame, total_ret_1y: pd.Series, ann_vol: pd.Series, last_date: str):
    # Figure 1: cumulative returns
    fig1 = go.Figure()
    for t in cum.columns:
        fig1.add_trace(go.Scatter(x=cum.index, y=cum[t], name=t, mode="lines"))
    fig1.update_layout(
        title="Cumulative Returns (base = 100)",
        xaxis_title="Date", yaxis_title="Index Level",
        legend_orientation="h", margin=dict(l=50,r=20,t=50,b=40), height=520
    )

    # Figure 2: annualized vol
    tickers = list(ann_vol.index)
    vols = [float(ann_vol[t]) for t in tickers]
    fig2 = go.Figure(go.Bar(x=tickers, y=vols, text=[f"{v*100:.2f}%" for v in vols], textposition="auto"))
    fig2.update_layout(
        title="Annualized Volatility (last ~1Y)",
        xaxis_title="Ticker", yaxis_title="Volatility",
        margin=dict(l=50,r=20,t=50,b=60), height=520
    )

    # Build a simple HTML shell with Plotly divs + a summary table
    tbl_rows = ""
    for t in tickers:
        r = float(total_ret_1y[t])
        v = float(ann_vol[t])
        tbl_rows += f"<tr><td>{t}</td><td style='text-align:right'>{r*100:.2f}%</td><td style='text-align:right'>{v*100:.2f}%</td></tr>"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Big 8 Tech — Returns & Volatility (Polygon.io)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#111}}
  .wrap{{max-width:1200px;margin:0 auto}}
  .grid{{display:grid;grid-template-columns:1fr;gap:20px}}
  @media(min-width:1000px){{.grid{{grid-template-columns:2fr 1fr}}}}
  .card{{border:1px solid #eee;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04)}}
  table{{width:100%;border-collapse:collapse}}
  th,td{{padding:8px 10px;border-bottom:1px solid #eee}}
  th{{text-align:left}}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Big 8 Tech — Returns & Volatility <span style="font-size:12px;color:#666">Polygon.io daily data</span></h1>
    <div class="grid">
      <div class="card">{fig1.to_html(include_plotlyjs=False, full_html=False)}</div>
      <div class="card">{fig2.to_html(include_plotlyjs=False, full_html=False)}</div>
    </div>
    <div class="card" style="margin-top:20px">
      <h3>Summary (last ~1Y)</h3>
      <table>
        <thead><tr><th>Ticker</th><th style="text-align:right">1Y Return</th><th style="text-align:right">Ann. Vol</th></tr></thead>
        <tbody>
          {tbl_rows}
        </tbody>
      </table>
      <div style="color:#666;font-size:12px;margin-top:8px">
        Data through {last_date}. Daily log returns; annualized volatility = stdev(daily) × √252. Source: Polygon.io
      </div>
    </div>
  </div>
</body>
</html>"""
    return html

def main():
    today = dt.date.today()
    start = (today - dt.timedelta(days=int(365*YEARS_BACK)+10)).isoformat()
    end = today.isoformat()

    client = RESTClient(api_key=API_KEY)
    # Pull daily adjusted close for each ticker
    frames = []
    for t in TICKERS:
        df = fetch_daily_aggs(client, t, start, end)
        df = df.rename(columns={"close": t}).set_index("date")
        frames.append(df)
    px = pd.concat(frames, axis=1).dropna(how="any")  # align on common dates

    cum, total_ret_1y, ann_vol = compute_metrics(px)
    last_date = px.index[-1].strftime("%Y-%m-%d")

    html = build_dashboard_html(cum, total_ret_1y, ann_vol, last_date)
    with open("index.html","w",encoding="utf-8") as f:
        f.write(html)
    print("Wrote index.html")

if __name__ == "__main__":
    main()
