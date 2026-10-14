"""Compare fixed industry-portfolio strategies with a once-only later test."""

from __future__ import annotations

import argparse
from html import escape
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


DEVELOPMENT_END = pd.Timestamp("2014-12-31")
HOLDOUT_START = pd.Timestamp("2015-01-01")
HOLDOUT_END = pd.Timestamp("2019-12-31")
EVALUATION_START = pd.Timestamp("1970-07-01")
COSTS_BPS = (15, 30, 50)
BASE_COST_BPS = 30
DRAWDOWN_BUDGET = 0.20
TARGET_VOLATILITIES = (0.05, 0.075, 0.10)
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21
VOLATILITY_LOOKBACK = 20
EXECUTION_LAG = 2


def _zip_text(path: Path) -> str:
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV file in {path.name}.")
        return archive.read(members[0]).decode("utf-8-sig")


def _source_frame(path: Path, marker: str, columns: list[str] | None = None) -> pd.DataFrame:
    text = _zip_text(path)
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        raise ValueError(f"Could not find {marker!r} in {path.name}.")
    header = next((i for i in range(start, len(lines))
                   if lines[i].startswith(",") or lines[i].startswith("Date,")), None)
    if header is None:
        raise ValueError(f"Could not find a CSV header in {path.name}.")
    end = next((i for i in range(header + 1, len(lines)) if not lines[i].strip()), len(lines))
    frame = pd.read_csv(StringIO("\n".join(lines[header:end])), skipinitialspace=True)
    frame = frame.rename(columns={frame.columns[0]: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    frame = frame.loc[frame.Date.notna()].set_index("Date").sort_index()
    if columns is not None:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing expected columns in {path.name}: {sorted(missing)}")
        frame = frame[columns]
    return frame.apply(pd.to_numeric, errors="coerce")


def load_industry_panel(industry_zip: Path, factor_zip: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load the value-weighted industry returns and market/T-bill daily returns."""
    industry_text = _zip_text(industry_zip)
    factor_text = _zip_text(factor_zip)
    if "202608 CRSP database" not in industry_text or "202608 CRSP database" not in factor_text:
        raise ValueError("Both source files must be from the frozen 202608 CRSP release.")
    marker = "Average Value Weighted Returns -- Daily"
    industries = _source_frame(industry_zip, marker)
    industry_names = list(industries.columns)
    if len(industry_names) != 49 or len(set(industry_names)) != 49:
        raise ValueError("Expected the 49 value-weighted industry portfolios.")
    factors = _source_frame(factor_zip, "Mkt-RF,SMB,HML,RF", ["Mkt-RF", "SMB", "HML", "RF"])
    # Fama/French values are percentages. -99.99 and -999 are missing sentinels.
    industries = industries.mask(industries <= -99.99) / 100
    factors = factors.mask(factors <= -99.99) / 100
    common = industries.index.intersection(factors.index).sort_values()
    industries = industries.loc[common]
    factors = factors.loc[common]
    complete = industries.notna().all(axis=1) & factors[["Mkt-RF", "RF"]].notna().all(axis=1)
    first_complete = complete[complete].index.min()
    complete_after_start = complete.loc[first_complete:]
    panel = industries.loc[complete_after_start[complete_after_start].index]
    factors = factors.loc[panel.index]
    if panel.empty:
        raise ValueError("The frozen sources do not have a complete common return panel.")
    metadata = {
        "release": "202608 CRSP",
        "industry_count": len(industry_names),
        "industry_names": industry_names,
        "first_complete_date": str(panel.index.min().date()),
        "last_complete_date": str(panel.index.max().date()),
        "complete_sessions": len(panel),
        "dropped_incomplete_sessions": int((~complete_after_start).sum()),
        "industry_zip_sha256": sha256(industry_zip.read_bytes()).hexdigest(),
        "factor_zip_sha256": sha256(factor_zip.read_bytes()).hexdigest(),
    }
    return panel, factors["RF"].rename("Cash"), {**metadata, "market": (factors["Mkt-RF"] + factors["RF"]).rename("Market")}


def _monthly_decisions(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(dates, index=dates).groupby(dates.to_period("M")).last())


def build_strategy_weights(industries: pd.DataFrame, market: pd.Series,
                           cash: pd.Series) -> dict[str, pd.DataFrame]:
    """Build fixed monthly rules; decisions take effect after a two-session lag."""
    dates = industries.index
    assets = list(industries.columns)
    cash_name = "Cash"
    names = [*assets, "Market", cash_name]
    ew_return = industries.mean(axis=1)
    ew_vol = ew_return.rolling(VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).std(ddof=1) * np.sqrt(252)
    wealth = (1 + industries).cumprod()
    market_wealth = (1 + market).cumprod()
    market_ma = market_wealth.rolling(200, min_periods=200).mean()
    momentum = wealth.shift(MOMENTUM_SKIP) / wealth.shift(MOMENTUM_LOOKBACK) - 1
    decisions = _monthly_decisions(dates)

    def scheduled(targets: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
        result = pd.DataFrame(np.nan, index=dates, columns=names)
        pending: dict[pd.Timestamp, pd.Series] = {}
        date_pos = {date: i for i, date in enumerate(dates)}
        for decision, target in targets.items():
            position = date_pos[decision] + EXECUTION_LAG
            if position < len(dates):
                pending[dates[position]] = target
        for effective_date, target in pending.items():
            result.loc[effective_date, :] = target.reindex(names, fill_value=0.0)
        result = result.ffill()
        return result.fillna({**{asset: 0.0 for asset in names if asset != cash_name}, cash_name: 1.0})

    def allocate_industries(decision: pd.Timestamp, chosen: list[str], exposure: float) -> pd.Series:
        weights = pd.Series(0.0, index=names)
        if chosen and exposure > 0:
            weights.loc[chosen] = exposure / len(chosen)
        weights[cash_name] = 1 - weights.drop(cash_name).sum()
        return weights

    all_industries = {decision: allocate_industries(decision, assets, 1.0) for decision in decisions}
    momentum_top10 = {}
    equal_vol = {target: {} for target in TARGET_VOLATILITIES}
    momentum_vol = {target: {} for target in TARGET_VOLATILITIES}
    market_trend = {}
    for decision in decisions:
        vol = ew_vol.loc[decision]
        scores = momentum.loc[decision].dropna().sort_values(ascending=False, kind="stable")
        selected = list(scores.head(10).index) if len(scores) >= 10 and scores.iloc[0] > 0 else []
        momentum_top10[decision] = allocate_industries(decision, selected, 1.0)
        for target_vol in TARGET_VOLATILITIES:
            exposure = min(1.0, target_vol / vol) if np.isfinite(vol) and vol > 0 else 0.0
            equal_vol[target_vol][decision] = allocate_industries(decision, assets, exposure)
            momentum_vol[target_vol][decision] = allocate_industries(decision, selected, exposure)
        trend = market_wealth.loc[decision] > market_ma.loc[decision]
        rule_weights = pd.Series(0.0, index=names)
        rule_weights["Market"] = 1.0 if bool(trend) and pd.notna(trend) else 0.0
        rule_weights[cash_name] = 1 - rule_weights["Market"]
        market_trend[decision] = rule_weights

    # Non-monthly reference strategies are held continuously from the first date.
    buy_hold = pd.DataFrame(0.0, index=dates, columns=names)
    buy_hold["Market"] = 1.0
    cash_only = pd.DataFrame(0.0, index=dates, columns=names)
    cash_only[cash_name] = 1.0
    strategies = {
        "Cash / T-bills": cash_only,
        "Market buy & hold": buy_hold,
        "Market 200-day trend": scheduled(market_trend),
        "Equal-weight 49 industries": scheduled(all_industries),
        "12-1 momentum, top 10": scheduled(momentum_top10),
    }
    for target_vol in TARGET_VOLATILITIES:
        strategies[f"Equal-weight 49, {target_vol:.1%} volatility target"] = scheduled(equal_vol[target_vol])
        strategies[f"12-1 momentum, top 10, {target_vol:.1%} volatility target"] = scheduled(momentum_vol[target_vol])
    return strategies


def backtest_weights(weights: pd.DataFrame, industries: pd.DataFrame, market: pd.Series,
                     cash: pd.Series, cost_bps: int, start: pd.Timestamp,
                     end: pd.Timestamp) -> tuple[pd.Series, pd.Series, float]:
    """Daily close-to-close returns; costs are charged on entry, turnover and exit."""
    data = industries.join(market).join(cash).loc[start:end]
    targets = weights.reindex(data.index).copy()
    if targets.isna().any().any():
        raise ValueError("Strategy weights do not cover the scored dates.")
    return_cols = list(industries.columns) + ["Market", "Cash"]
    if list(targets.columns) != return_cols:
        raise ValueError("Strategy weights and return assets do not match.")
    if not np.allclose(targets.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Every target allocation must include cash and sum to one.")
    cost_rate = cost_bps / 10000
    return_values = data[return_cols].to_numpy(dtype=float)
    target_values = targets.to_numpy(dtype=float)
    holdings = np.zeros(len(return_cols), dtype=float)
    holdings[-1] = 1.0
    equity = 1.0
    equity_values = np.empty(len(data), dtype=float)
    turnover = 0.0
    for i, asset_returns in enumerate(return_values):
        desired = target_values[i]
        traded = 0.5 * float(np.abs(desired - holdings).sum())
        turnover += traded
        equity *= 1 - traded * cost_rate
        gross = float(np.dot(desired, asset_returns))
        if gross <= -1:
            raise ValueError("Portfolio return cannot reduce equity to zero or below.")
        equity *= 1 + gross
        equity_values[i] = equity
        grown = desired * (1 + asset_returns)
        holdings = grown / (1 + gross)
    cash_target = np.zeros(len(return_cols), dtype=float)
    cash_target[-1] = 1.0
    exit_turnover = 0.5 * float(np.abs(holdings - cash_target).sum())
    turnover += exit_turnover
    equity *= 1 - exit_turnover * cost_rate
    # Reconstruct daily net returns from equity snapshots, including final liquidation.
    if len(equity_values):
        equity_values[-1] = equity
    equity_series = pd.Series(equity_values, index=data.index)
    initial = 1.0
    daily_returns = equity_series.div(equity_series.shift(1).fillna(initial)).sub(1)
    return equity_series, daily_returns, turnover


def summarize(equity: pd.Series, returns: pd.Series, cash: pd.Series,
              turnover: float, cost_bps: int) -> dict:
    excess = returns - cash.reindex(returns.index)
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = float(excess.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) else 0.0
    if abs(sharpe) < 1e-9:
        sharpe = 0.0
    peak = equity.cummax()
    years = len(returns) / 252
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else -1.0
    return {"cost_bps_per_side": cost_bps, "annualized_return": cagr,
            "total_return": float(equity.iloc[-1] - 1),
            "max_drawdown": float((equity / peak - 1).min()),
            "annualized_volatility": vol, "sharpe_vs_cash": sharpe,
            "turnover_one_way": float(turnover), "ending_growth_of_10000": float(equity.iloc[-1] * 10000)}


def code_fingerprint() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def build_chart(equity: pd.DataFrame, selected: str, phase: str) -> str:
    """Create a small self-contained, log-scale SVG chart with strategy toggles."""
    width, height = 1080, 520
    left, right, top, bottom = 82, 24, 28, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    values = equity.clip(lower=1e-8) * 10000
    log_values = np.log10(values.to_numpy(dtype=float))
    low, high = float(np.nanmin(log_values)), float(np.nanmax(log_values))
    if high - low < .25:
        middle = (high + low) / 2
        low, high = middle - .25, middle + .25
    palette = ["#71e0c1", "#78a9ff", "#ffc857", "#ff7e9c", "#c49bff", "#79d4e8", "#f4a261"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Strategy portfolio growth chart">',
             f'<rect width="{width}" height="{height}" rx="18" fill="#101a2d"/>']
    for power in range(int(np.floor(low)), int(np.ceil(high)) + 1):
        y = top + (high - power) / (high - low) * plot_h
        if top <= y <= top + plot_h:
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#26344a"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" fill="#9eacc2" font-size="12">${10**power:,.0f}</text>')
    n = len(values)
    for j, name in enumerate(values.columns):
        coords = []
        for i, log_value in enumerate(log_values[:, j]):
            x = left + (i / max(n - 1, 1)) * plot_w
            y = top + (high - log_value) / (high - low) * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        visible = "" if name in [selected, "Market buy & hold", "Cash / T-bills"] else ' style="display:none"'
        parts.append(f'<polyline id="line{j}"{visible} fill="none" stroke="{palette[j % len(palette)]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" points="{" ".join(coords)}"/>')
    parts.extend([f'<text x="{left}" y="{height-22}" fill="#9eacc2" font-size="12">{equity.index[0].date()}</text>',
                  f'<text x="{width-right}" y="{height-22}" text-anchor="end" fill="#9eacc2" font-size="12">{equity.index[-1].date()}</text>',
                  '</svg>'])
    toggles = []
    for j, name in enumerate(values.columns):
        checked = " checked" if name in [selected, "Market buy & hold", "Cash / T-bills"] else ""
        toggles.append(f'<label><input type="checkbox" data-line="line{j}"{checked}> <i style="--swatch:{palette[j % len(palette)]}"></i>{escape(name)}</label>')
    title = "Industry portfolio strategies · development" if phase == "development" else "Locked test · 2015–2019"
    note = ("Development history through 2014; later years remain reserved." if phase == "development"
            else f"Selected using development data through {DEVELOPMENT_END.date()}; settings were frozen before the 2015–2019 test.")
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MarketSignal research</title>
<style>body{{margin:0;background:#0b1220;color:#edf3fc;font:15px system-ui,sans-serif}}main{{max-width:1180px;margin:32px auto;padding:0 22px}}h1{{font-size:26px;margin:0 0 8px}}p{{color:#aab8ce;margin:0 0 20px}}.card{{background:#101a2d;border:1px solid #26344a;border-radius:20px;padding:14px;box-shadow:0 20px 60px #0005}}svg{{display:block;width:100%;height:auto}}.toggles{{display:flex;flex-wrap:wrap;gap:9px 18px;margin:18px 4px 6px}}label{{display:flex;align-items:center;gap:7px;color:#d7e2f1;cursor:pointer;font-size:13px}}i{{width:10px;height:10px;border-radius:50%;background:var(--swatch);display:inline-block}}input{{accent-color:#71e0c1}}.foot{{font-size:12px;margin:14px 2px}}</style>
<main><h1>{escape(title)}</h1><p>{escape(note)} Values show growth of $10,000 after estimated costs, on a logarithmic scale.</p><section class="card">{''.join(parts)}<div class="toggles">{''.join(toggles)}</div></section><p class="foot">Research comparison only. Fixed transaction costs include entry, rebalancing and final liquidation; actual execution and future returns may differ.</p></main>
<script>document.querySelectorAll('input[data-line]').forEach(function(box){{box.addEventListener('change',function(){{document.getElementById(box.dataset.line).style.display=box.checked?'':'none'}})}});</script></html>'''


def run_phase(phase: str, industry_zip: Path, factor_zip: Path,
              development_report: Path, output: Path) -> None:
    panel, cash, metadata = load_industry_panel(industry_zip, factor_zip)
    market = metadata.pop("market")
    weights = build_strategy_weights(panel, market, cash)
    fingerprint = code_fingerprint()
    source_hash = sha256((metadata["industry_zip_sha256"] + metadata["factor_zip_sha256"]).encode()).hexdigest()
    if phase == "development":
        start, end = EVALUATION_START, DEVELOPMENT_END
        if output.exists() or output.with_suffix(".html").exists() or output.with_suffix(".equity.csv").exists():
            raise ValueError("Development outputs already exist; use a new path to preserve the first run.")
        selected_input = None
    else:
        if not development_report.exists():
            raise ValueError("Run the development phase first and provide its frozen report.")
        frozen = json.loads(development_report.read_text(encoding="utf-8"))
        if (frozen.get("phase") != "development" or frozen.get("source_hash") != source_hash
                or frozen.get("code_fingerprint") != fingerprint
                or frozen.get("development_end") != str(DEVELOPMENT_END.date())):
            raise ValueError("Development report, frozen sources, dates, and code must match exactly.")
        if output.exists() or output.with_suffix(".html").exists() or output.with_suffix(".equity.csv").exists():
            raise ValueError("Locked outputs already exist; this source can be scored only once.")
        start, end = HOLDOUT_START, HOLDOUT_END
        selected_input = frozen.get("selected_strategy")
        if selected_input not in weights:
            raise ValueError("Frozen development strategy is unavailable in this run.")

    selected_weights = weights
    rows, yearly_rows, regime_rows = [], [], []
    equity_frames = []
    benchmark_wealth = (1 + market).cumprod()
    market_ma = benchmark_wealth.rolling(200, min_periods=200).mean()
    regime = pd.Series(np.where(benchmark_wealth > market_ma, "Market above 200-day average",
                                "Market at or below 200-day average"), index=panel.index)
    regime.loc[market_ma.isna()] = "Insufficient history for market regime"
    allowed_names = list(selected_weights) if phase == "development" else list(selected_weights)
    for strategy in allowed_names:
        if phase == "locked" and strategy not in [selected_input, "Cash / T-bills", "Market buy & hold",
                                                   "Market 200-day trend", "Equal-weight 49 industries",
                                                   "12-1 momentum, top 10"]:
            # Report fixed baselines alongside the frozen winner, not every losing variant.
            continue
        for cost in COSTS_BPS:
            equity, daily_returns, turnover = backtest_weights(weights[strategy], panel, market, cash, cost, start, end)
            rows.append({"strategy": strategy, **summarize(equity, daily_returns, cash, turnover, cost),
                         "selected_before_holdout": strategy == selected_input if phase == "locked" else False})
            if cost == BASE_COST_BPS:
                equity_frames.append(equity.rename(strategy))
                year_return = daily_returns.groupby(daily_returns.index.year).apply(lambda x: float((1 + x).prod() - 1))
                for year, value in year_return.items():
                    yearly_rows.append({"strategy": strategy, "year": int(year), "net_return": value})
                for label, dates in regime.loc[start:end].groupby(regime.loc[start:end]).groups.items():
                    values = daily_returns.reindex(dates).dropna()
                    if not values.empty:
                        regime_rows.append({"strategy": strategy, "market_regime": label,
                                            "sessions": len(values), "net_return": float((1 + values).prod() - 1)})
    if phase == "development":
        eligible = [row for row in rows if row["cost_bps_per_side"] == BASE_COST_BPS
                    and row["max_drawdown"] >= -DRAWDOWN_BUDGET]
        if not eligible:
            raise ValueError("No non-cash candidate met the development drawdown screen.")
        selected = sorted(eligible, key=lambda row: (-row["annualized_return"], row["strategy"]))[0]["strategy"]
        status = "Development comparison only; the selected candidate is not approved for live trading."
    else:
        selected = selected_input
        status = "The frozen development candidate and fixed baselines were scored on the reserved 2015–2019 period once. This does not establish future profitability."

    result = {"phase": phase, "source_hash": source_hash,
              "sources": {key: value for key, value in metadata.items() if key != "industry_names"},
              "industry_names": metadata["industry_names"], "code_fingerprint": fingerprint,
              "evaluation_start": str(start.date()), "development_end": str(DEVELOPMENT_END.date()),
              "holdout_start": str(HOLDOUT_START.date()), "holdout_end": str(HOLDOUT_END.date()),
              "scored_start": str(panel.loc[start:].index.min().date()),
              "scored_end": str(panel.loc[:end].index.max().date()),
              "selected_strategy": selected,
              "selection_rule": f"Highest development annualized net return at {BASE_COST_BPS} bps per side among fixed non-cash candidates with max drawdown no worse than {-DRAWDOWN_BUDGET:.0%}; alphabetical tie-break." if phase == "development" else "Candidate selected before holdout using the attached development report.",
              "settings": {"cost_scenarios_bps_per_side": list(COSTS_BPS),
                           "base_cost_bps_per_side": BASE_COST_BPS,
                           "drawdown_budget": DRAWDOWN_BUDGET,
                           "volatility_targets": list(TARGET_VOLATILITIES),
                           "momentum_lookback_sessions": MOMENTUM_LOOKBACK,
                           "momentum_skip_sessions": MOMENTUM_SKIP,
                           "momentum_top_industries": 10,
                           "volatility_lookback_sessions": VOLATILITY_LOOKBACK,
                           "execution_lag_sessions": EXECUTION_LAG,
                           "cost_method": "15/30/50 bps per side times one-way traded notional, including terminal liquidation"},
              "summary": rows, "yearly_returns": yearly_rows, "market_regime_returns": regime_rows,
              "limitations": [
                  "These are reconstructed value-weighted industry portfolios, not individual stock picks or a directly investable 49-fund universe.",
                  "Daily industry returns are complete from 1969-07-01; one session with a missing industry return is omitted.",
                  "Portfolio-level costs are fixed assumptions; internal industry reconstitution costs, bid-ask spreads, market impact, taxes, and investor-specific cash rates are not measured.",
                  "The CRSP history is reconstructed and may be revised; the local ZIP files and their hashes identify this snapshot.",
                  "The selected rule is a historical research result and is not a profitability claim or live-trading recommendation."],
              "status": status}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    equity_frame = pd.concat(equity_frames, axis=1)
    equity_frame.index.name = "Date"
    equity_frame.to_csv(output.with_suffix(".equity.csv"))
    output.with_suffix(".html").write_text(build_chart(equity_frame, selected, phase), encoding="utf-8")
    displayed = pd.DataFrame(rows)
    print(f"{phase.title()} period: {result['scored_start']} to {result['scored_end']}")
    print(f"Selected strategy: {selected}")
    print(displayed.loc[displayed.cost_bps_per_side == BASE_COST_BPS,
                        ["strategy", "annualized_return", "max_drawdown", "sharpe_vs_cash", "turnover_one_way"]].to_string(index=False))
    print(f"Saved report: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "locked"], required=True)
    parser.add_argument("--industry-input", default="data/research_inputs/fama_french_49_industry_daily_202608.zip")
    parser.add_argument("--factor-input", default="data/research_inputs/fama_french_3_factors_daily_202608.zip")
    parser.add_argument("--development-report", default="data/industry_development.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    output = Path(args.output or ("data/industry_development.json" if args.phase == "development"
                                  else "data/industry_locked.json"))
    try:
        run_phase(args.phase, Path(args.industry_input), Path(args.factor_input),
                  Path(args.development_report), output)
    except (ValueError, KeyError, OSError, BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
