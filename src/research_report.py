"""Offline interactive research report, also served by the app's Research Lab."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#71859f", "#64d8e4", "#b9a2ff", "#ffc680", "#7ee3b1", "#f294b8", "#5ca9ff"]


def style(figure, height=380):
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color="#cdd8e9"),
        height=height, margin=dict(l=55, r=25, t=35, b=60),
        legend=dict(orientation="h", y=-.25), hovermode="closest",
    )
    figure.update_xaxes(gridcolor="#233148", zeroline=False)
    figure.update_yaxes(gridcolor="#233148", zeroline=False)
    return figure


def report_figures(report):
    folds, summary = pd.DataFrame(report["folds"]), pd.DataFrame(report["summary"])
    baseline = folds.loc[folds.candidate == "Prior baseline"].set_index("fold").log_loss
    timeline = go.Figure()
    ordered = summary.sort_values("log_loss")
    bars = go.Figure(go.Bar(
        x=ordered.log_loss, y=ordered.candidate, orientation="h", marker_color=[
            "#71859f" if name == "Prior baseline" else "#64d8e4" for name in ordered.candidate
        ], hovertemplate="%{y}<br>Log loss: %{x:.4f}<extra></extra>",
    ))
    bars.update_yaxes(autorange="reversed")
    bars.update_xaxes(title="Pooled log loss · lower is better", rangemode="tozero")
    reliability = make_subplots(rows=1, cols=3, subplot_titles=["Underperform", "Within cost band", "Outperform"])
    for col in range(1, 4):
        reliability.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                        line=dict(color="#71859f", dash="dot"),
                                        showlegend=False, hoverinfo="skip"), row=1, col=col)
    for i, (name, group) in enumerate(folds.groupby("candidate", sort=True)):
        color = COLORS[i % len(COLORS)]
        if name != "Prior baseline":
            gain = group.fold.map(baseline).to_numpy() - group.log_loss.to_numpy()
            timeline.add_trace(go.Scatter(
                x=group.evaluation_end, y=gain, name=name, mode="lines+markers",
                line=dict(color=color, width=2), marker=dict(size=7),
                customdata=group[["evaluation_start", "temperature"]].to_numpy(),
                hovertemplate="From %{customdata[0]} to %{x}<br>Improvement: %{y:.4f}"
                              "<br>Temperature: %{customdata[1]:.2f}<extra>%{fullData.name}</extra>",
            ))
        points = pd.DataFrame([p for p in report["reliability"] if p["candidate"] == name])
        for col, label in enumerate(["P_DOWN", "P_NO_TRADE", "P_UP"], 1):
            part = points.loc[points["class"] == label].sort_values("predicted")
            reliability.add_trace(go.Scatter(
                x=part.predicted, y=part.observed, customdata=part["count"],
                name=name, legendgroup=name, showlegend=col == 1,
                visible=True if "XGBoost" in name or name == "Prior baseline" else "legendonly",
                mode="lines+markers", line=dict(color=color),
                hovertemplate="Predicted: %{x:.0%}<br>Observed: %{y:.0%}<br>Rows: %{customdata}<extra>%{fullData.name}</extra>",
            ), row=1, col=col)
    timeline.add_hline(y=0, line_dash="dot", line_color="#cdd8e9")
    timeline.update_yaxes(title="Baseline log loss − model log loss")
    reliability.update_xaxes(range=[0, 1], tickformat=".0%", title="Predicted probability")
    reliability.update_yaxes(range=[0, 1], tickformat=".0%", title="Observed share")
    return [style(timeline, 460), style(bars, 410), style(reliability, 480)]


def render_report(report):
    figures = report_figures(report)
    charts = [figure.to_html(full_html=False, include_plotlyjs=True if i == 0 else False,
                             config={"displaylogo": False, "responsive": True})
              for i, figure in enumerate(figures)]
    baseline = next(s for s in report["summary"] if s["candidate"] == "Prior baseline")
    best = min(report["summary"], key=lambda s: s["log_loss"])
    improvement = baseline["log_loss"] - best["log_loss"]
    verdict = "Baseline still leads" if best["candidate"] == "Prior baseline" else "A candidate beats baseline"
    folds = len({f["fold"] for f in report["folds"]})
    config = report.get("config", {})
    tickers = escape(", ".join(config.get("tickers", [])))
    rows = "".join(
        f"<tr><td>{escape(s['candidate'])}</td><td>{s['log_loss']:.4f}</td>"
        f"<td>{s['brier']:.4f}</td><td>{s['macro_f1']:.3f}</td></tr>"
        for s in sorted(report["summary"], key=lambda s: s["log_loss"])
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MarketSignal · Research Lab</title>
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#080f1c;color:#e8effa;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 8% 0%,#194b6244,transparent 55%),radial-gradient(ellipse at 90% 70%,#51428022,transparent 50%);z-index:-1}}
main{{max-width:1280px;margin:auto;padding:40px 28px}}a{{color:#64d8e4}}nav{{display:flex;justify-content:space-between;margin-bottom:40px;font-size:.85rem}}nav a{{text-decoration:none}}
.eyebrow{{color:#64d8e4;letter-spacing:.17em;font-size:.75rem;text-transform:uppercase;font-weight:700}}h1{{font-size:clamp(2.3rem,5vw,4rem);letter-spacing:-.05em;line-height:1.1;margin:16px 0}}h2{{font-size:1.3rem;margin:0 0 6px}}p{{color:#a5b6cf;margin:8px 0 16px;max-width:880px}}.lead{{font-size:1.1rem}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0}}.card,section{{border:1px solid #27374e;border-radius:20px;background:linear-gradient(130deg,#142138dd,#0d1729ee);box-shadow:0 20px 60px #0002}}.card{{padding:22px;transition:transform .2s,border-color .2s}}.card:hover{{transform:translateY(-3px);border-color:#64d8e4}}.label{{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:#9fb1cd}}.value{{font-size:1.65rem;font-weight:650;margin:10px 0 5px;line-height:1.25}}.note{{font-size:.82rem;color:#9fb1cd}}
section{{padding:24px;margin:20px 0;animation:arrive .5s ease-out both}}.steps{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.step{{flex:1;min-width:170px;padding:14px 18px;border-left:3px solid #64d8e4;background:#64d8e40b;border-radius:4px 12px 12px 4px}}.step:nth-child(2){{border-color:#b9a2ff}}.step:nth-child(3){{border-color:#ffc680}}.step span{{display:block;font-size:.8rem;color:#9fb1cd}}.chart{{min-width:0}}.wide{{overflow-x:auto}}.reliability{{min-width:740px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #27374e}}th{{color:#9fb1cd;font-size:.8rem}}details summary{{cursor:pointer;color:#64d8e4}}footer{{font-size:.8rem;color:#9fb1cd;margin:30px 0}}a:focus-visible,summary:focus-visible{{outline:2px solid #64d8e4;outline-offset:5px}}
@keyframes arrive{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}@media(max-width:700px){{main{{padding:24px 14px}}.cards{{grid-template-columns:1fr}}section{{padding:16px 8px}}.value{{font-size:1.5rem}}}}
</style></head><body><main>
<nav><a href="/">← MarketSignal</a><span>RESEARCH LAB / SAVED EXPERIMENT</span></nav>
<div class="eyebrow">Saved experiment · 5-session outlook</div>
<h1>Does the signal hold up?</h1><p class="lead">Follow the model through time. Compare its predictions with a simple baseline, then see whether its confidence matches reality.</p>
<div class="cards"><div class="card"><div class="label">Current finding</div><div class="value">{verdict}</div><div class="note">{escape(best['candidate'])} · descriptive result</div></div>
<div class="card"><div class="label">Walk-forward windows</div><div class="value">{folds} periods</div><div class="note">{baseline['rows']:,} scored stock-days per candidate</div></div>
<div class="card"><div class="label">Best log-loss improvement</div><div class="value">{improvement:+.4f}</div><div class="note">Compared with the prior-probability baseline</div></div></div>
<div class="steps"><div class="step">01 / Train<span>Learn from the expanding past.</span></div><div class="step">02 / Calibrate<span>Adjust confidence on a later, separate block.</span></div><div class="step">03 / Evaluate<span>Freeze both, then score the following block.</span></div></div>
<p class="note">Labels crossing a boundary are removed (<b>purging</b>). This is a repeated comparison on development data, not an untouched final test or a trading-return backtest.</p>
<section><h2>Performance through time</h2><p>Above zero means lower <b>log loss</b> than the baseline in that period. Log loss penalizes confident mistakes. Click a legend entry to hide a line; double-click to isolate it.</p><div class="chart">{charts[0]}</div></section>
<section><h2>The whole comparison</h2><p>Scores pool all evaluated stock-days. A small lead alone does not establish a reliable advantage. The baseline predicts historical class frequencies.</p><div class="chart">{charts[1]}</div><details><summary>See exact scores</summary><div class="wide"><table><thead><tr><th>Candidate</th><th>Log loss ↓</th><th>Brier ↓</th><th>Macro F1 ↑</th></tr></thead><tbody>{rows}</tbody></table></div></details></section>
<section><h2>Does 70% confidence mean 70%?</h2><p><b>Calibration</b> compares predicted probability with the observed share in each class. Closer to the dotted diagonal is better. Hover to see bin counts; sparse bins are less reliable. XGBoost and the baseline are shown first; other models can be enabled in the legend.</p><div class="wide"><div class="reliability">{charts[2]}</div></div></section>
<section><h2>How to read this experiment</h2><p><b>Temperature scaling</b> adjusts confidence using only the calibration block. It preserves the top class, so it cannot improve classification accuracy by itself. We show raw and calibrated results even when calibration makes scores worse.</p><p>Classes mean underperform SPY, stay within the cost band, or outperform SPY. They differ from the main dashboard’s UP/DOWN target.</p><p>Universe: {tickers}. Development cutoff: {escape(report['development_end'])}. Feature group: {escape(config.get('feature_set', 'custom'))}. {report['unused_tail_dates']} trailing dates are unused because they do not fill an evaluation window.</p><p>This data has already informed research choices. Stocks and overlapping five-day outcomes are correlated, so row counts are not independent sample sizes. No significance or profitability claim is made. Survivor selection, revised price histories, and limited market coverage remain limitations.</p></section>
<footer>MarketSignal · Reproducible research, visible uncertainty. Generated from a saved snapshot; no live market feed.</footer>
</main></body></html>'''
