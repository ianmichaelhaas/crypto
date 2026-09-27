"""Monte Carlo simulation of XRP market direction.

XRP returns are taken from the Canary spot XRP ETF (XRPC) daily closes, which
track XRP spot (minus a small fee drag). Because the ETF trades ~252 days/year
while XRP trades 24/7, each trading-day return is treated as covering
365/252 calendar days when scaling to calendar-day horizons.

Models:
  1. GBM, zero drift        - pure volatility, no directional view
  2. GBM, historical drift  - extrapolates the sample trend
  3. Bootstrap              - resamples actual daily returns (fat tails, skew)
  4. GARCH(1,1) Student-t   - volatility clustering, starts from current vol

Usage: python xrp_simulation.py [spot_price]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPOT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.51203  # XRP-USD mark, 2026-09-27
HORIZONS = [7, 30, 90]  # calendar days
N_PATHS = 100_000
CAL_PER_TRADING = 365 / 252
rng = np.random.default_rng(42)


def load_returns(path="data/XRPC_daily.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[~df["interpolated"]].set_index("date")
    return df["close"], np.log(df["close"]).diff().dropna().values


def fit_garch(r, grid=40):
    """Crude GARCH(1,1) fit by grid-search MLE (avoids an arch dependency)."""
    var = r.var()
    best = (-np.inf, None)
    for a in np.linspace(0.02, 0.25, grid):
        for b in np.linspace(0.60, 0.97, grid):
            if a + b >= 0.995:
                continue
            w = var * (1 - a - b)
            s2 = np.empty_like(r)
            s2[0] = var
            for t in range(1, len(r)):
                s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
            ll = -0.5 * np.sum(np.log(s2) + r**2 / s2)
            if ll > best[0]:
                best = (ll, (w, a, b, w + a * r[-1] ** 2 + b * s2[-1]))
    return best[1]


def simulate(r, days):
    steps = int(round(days / CAL_PER_TRADING))  # trading-day equivalents
    mu, sd = r.mean(), r.std(ddof=1)
    out = {}
    z = rng.standard_normal((N_PATHS, steps))
    out["GBM zero drift"] = SPOT * np.exp((-0.5 * sd**2 + sd * z).sum(1))
    out["GBM hist drift"] = SPOT * np.exp((mu + sd * z).sum(1))
    out["Bootstrap"] = SPOT * np.exp(rng.choice(r, (N_PATHS, steps)).sum(1))
    w, a, b, s2 = fit_garch(r - r.mean())
    nu = 5
    s2 = np.full(N_PATHS, s2)
    total = np.zeros(N_PATHS)
    for _ in range(steps):
        e = rng.standard_t(nu, N_PATHS) * np.sqrt((nu - 2) / nu) * np.sqrt(s2)
        total += e - 0.5 * s2
        s2 = w + a * e**2 + b * s2
    out["GARCH-t"] = SPOT * np.exp(total)
    return out


def indicators(close):
    last = close.iloc[-1]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    return {
        "ETF close": last,
        "vs 20d MA": last / close.rolling(20).mean().iloc[-1] - 1,
        "vs 50d MA": last / close.rolling(50).mean().iloc[-1] - 1,
        "vs 100d MA": last / close.rolling(100).mean().iloc[-1] - 1,
        "RSI(14)": 100 - 100 / (1 + gain / loss),
        "20d return": last / close.iloc[-21] - 1,
        "60d return": last / close.iloc[-61] - 1,
    }


def main():
    close, r = load_returns()
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    print(f"XRP spot: ${SPOT:.4f}")
    print(f"Sample: {close.index[0].date()} -> {close.index[-1].date()} ({len(r)} returns)")
    print(f"Sample return: {close.iloc[-1] / close.iloc[0] - 1:+.1%}   annualised vol: {ann_vol:.0%}")
    print(f"Recent 20d vol (ann.): {r[-20:].std(ddof=1) * np.sqrt(252):.0%}\n")

    print("Technical snapshot (XRPC):")
    for k, v in indicators(close).items():
        print(f"  {k:<11} {v:>8.2f}" if k in ("ETF close", "RSI(14)") else f"  {k:<11} {v:>+8.1%}")

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15, 4.5))
    rows = []
    for ax, days in zip(axes, HORIZONS):
        sims = simulate(r, days)
        for name, px in sims.items():
            q = np.percentile(px, [5, 25, 50, 75, 95])
            rows.append({
                "horizon": f"{days}d", "model": name,
                "P(up)": (px > SPOT).mean(),
                "P(>+20%)": (px > SPOT * 1.2).mean(),
                "P(<-20%)": (px < SPOT * 0.8).mean(),
                "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
            })
            ax.hist(px, bins=200, range=(SPOT * 0.3, SPOT * 2.2), histtype="step", density=True, label=name)
        ax.axvline(SPOT, color="k", ls="--", lw=1)
        ax.set_title(f"XRP price in {days} days")
        ax.set_xlabel("USD")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("xrp_simulation.png", dpi=120)

    res = pd.DataFrame(rows)
    res.to_csv("xrp_simulation_results.csv", index=False)
    pd.set_option("display.width", 200)
    fmt = {c: "{:.1%}".format for c in ("P(up)", "P(>+20%)", "P(<-20%)")}
    fmt.update({c: "${:.3f}".format for c in ("p5", "p25", "median", "p75", "p95")})
    print("\n" + res.to_string(index=False, formatters=fmt))


if __name__ == "__main__":
    main()
