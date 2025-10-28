# build.py
# Generates a self-contained index.html using yfinance daily data.
# Big 8: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AVGO

import math
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# === Config ===
TICKERS = ["AAPL","MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO"]
YEARS_BACK = 2  # how many years of history to fetch
INTERVAL = "1d"  # daily bars

def fetch_daily_closes(tickers, start_date, end_date):
    """
    Download adjusted daily closes for the given tickers and date range.
    We use auto_adjust=True so the 'Close' column is adjusted for splits/dividends.
    """
    df = yf.download(
        tickers=" ".join(tickers),
        start=start_date.isoformat(),
        end=(end_date + dt.timedelta(days=1)).isoformat(),  # include end day
        interval=INTERVAL,
        auto_adjust=True,   # makes 'Close' adjusted close
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # Normalize into a 2D DataFrame: index=date, columns=tickers, values=Close
    closes = {}
    # yfinance returns a multi-index when multiple tickers requested
    for t in tickers:
        try:
            # df[t]['Close'] if multi-index, else df['Close'][t] for some versions
            closes[t] = df[t]["Close"]
        except Exception:
            # fallback if structure differs
            closes[t] = df["Close"][t]
    px = pd.DataFrame(closes).dropna(how="any")  # keep common trading days only
    px.index = pd.to_datetime(px.index).date  # use date objects (cleaner HTML axis)
    return px

def compute_metrics(px_df: pd.DataFrame):
    """
    Input: px_df with date index and columns=tickers (adjusted close).
    Returns:
      cum_df: cumulative returns (base=100)
      total_return_1y: Series of total return over last ~1Y window
      ann_vol_1y: Series of annualized volatility over last ~1Y window
    """
    # Daily log returns
    rets = np.log(px_df / px_df.shift(1)).dropna()

    # Cumulative (base=100 at first valid day)
    cum_df = (rets.cumsum()).apply(np.exp) * 100.0

    # 1Y window (~252 trading days)
    window = min(252, len(rets))
    rets_1y = rets.tail(window)

    # 1Y total return per ticker (simple)
    total_return_1y = (px_df.tail(1) / px_df.tail(window).iloc[0] - 1.0).iloc[0]

    # Annualized volatility (stdev of daily × sqrt(252))
    ann_vol_1y = rets_1y.std() * math.sqrt(252)

    return cum_df, total_return_1y, ann_vol_1y

def build_dashboard_html(cum: pd.DataFrame, total_ret_1y: pd.Series, ann_vol: pd.Series, last_date: str):
    # Figure 1: cumulative returns
    fig1 = go.Figure()
    for t in cum.columns:
        fig1.add_trace(go.Scatter(x=list(cum.index), y=cum[t], name=t, mode="lines"))
    fig1.update_layout(
        title="Cumulative Returns (base = 100)",
        xaxis_title="Date",
        yaxis_title="Index Level",
        legend_orientation="h",
        margin=dict(l=50, r=20, t=50, b=40),
        height=520,
    )

    # Figure 2: annualized vol (bar)
    tickers = list(ann_vol.index)
    vols = [float(ann_vol[t]) for t in tickers]
    fig2 = go.Figure(go.Bar(
        x=tickers,
        y=vols,
        text=[f"{v*100:.2f}%" for v in vols],
        textposition="auto",
        hovertemplate="Ticker: %{x}<br>Ann. Vol: %{y:.4f}<extra></extra>",
    ))
    fig2.update_layout(
        title="Annualized Volatility (last ~1Y)",
        xaxis_title="Ticker",
        yaxis_title="Volatility",
        margin=dict(l=50, r=20, t=50, b=60),
        height=520,
    )

    # Summary table
    rows_html = ""
    for t in tickers:
        r = float(total_ret_1y[t])
        v = float(ann_vol[t])
        rows_html += f"<tr><td>{t}</td><td style='text-align:right'>{r*100:.2f}%</td><td style='text-align:right'>{v*100:.2f}%</td></tr>"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Big 8 Tech — Returns & Volatility (yfinance)</title>
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
    <h1>Big 8 Tech — Returns & Volatility <span style="font-size:12px;color:#666">yfinance daily adjusted data</span></h1>
    <div class="grid">
      <div class="card">{fig1.to_html(include_plotlyjs=False, full_html=False)}</div>
      <div class="card">{fig2.to_html(include_plotlyjs=False, full_html=False)}</div>
    </div>
    <div class="card" style="margin-top:20px">
      <h3>Summary (last ~1Y)</h3>
      <table>
        <thead><tr><th>Ticker</th><th style="text-align:right">1Y Return</th><th style="text-align:right">Ann. Vol</th></tr></thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
      <div style="color:#666;font-size:12px;margin-top:8px">
        Data through {last_date}. Prices are adjusted close. Daily log returns; annualized volatility = stdev(daily) × √252. Source: Yahoo Finance via yfinance.
      </div>
    </div>
  </div>
</body>
</html>"""
    return html

def main():
    today = dt.date.today()
    start = today - dt.timedelta(days=int(365 * YEARS_BACK) + 10)

    px = fetch_daily_closes(TICKERS, start, today)
    if px.empty:
        raise RuntimeError("No price data downloaded. Check tickers or network.")

    cum, total_ret_1y, ann_vol = compute_metrics(px)
    last_date = max(px.index).strftime("%Y-%m-%d")

    html = build_dashboard_html(cum, total_ret_1y, ann_vol, last_date)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote index.html")

if __name__ == "__main__":
    main()
