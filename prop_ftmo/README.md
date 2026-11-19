# Prop-firm portfolio (FTMO 2-Step Swing, CFD) — rotating RSI trend-following with a volatility-target overlay

Long-only rotation over 8 base sleeves (BTC, ETH, SOL, US100, US500, XAUUSD, NVDA, AAPL) + optional AMD (JNJ excluded by default: it fails the grid's own filter).
RSI(60) of buy&hold > 50 AND sleeve on; 3 slots at 1/3 × sleeve size; max 2 crypto; 120-day correlation filter at 0.8; XAUUSD as refuge.
Risk layer: k = min(1, τ / EWMA30 vol) × drawdown brake (→ 0 at −10 %) × daily cushion; τ = 10 % in the evaluation, k = 0.25 when funded.

**Read `CAMBIOS_RICK.md` first**: it lists the 10 review points, what was fixed, how it was checked and what is still open.

## Convention (backtest = live)
Signal with closes ≤ t → fill at t+1 → position earns from t+2. Run `senal_diaria.py` in the **morning** with yesterday's closes; trade the same day near the close.
Today's bar is never used. The program **stops** if any asset has no complete recent bar, and if `--equity` is given without `--ref3` (max of the last 3 closing equities from the broker).

## Layout
| Folder | What |
|---|---|
| `data/csv_sleeves/` | the 10 sleeve CSVs (position/size already executed); `data/ftmo_simbolos.csv` = demo conditions (spread, swap, contract, lot, margin) |
| `notebooks/` | `ftmo_rotativo.py` (engine + evaluator + live signal), `senal_diaria.py` (daily wrapper), `sleeves_motor.py` (sleeve rules computed from prices) |
| `montecarlo/` | `mc_prop.py` (block-bootstrap Monte Carlo of the 2 phases + funded stage), reference series, `results/`, `run_log.jsonl` |
| `results_charts/` | pass vs days by τ; pass / burn / unfinished (reference series, swap 30 %) |
| `tests/` | `test_datos.py` (cache, provisional bar, staleness) |

## Honest status
* Pass rates are a **simplified game**: sleeves and AMD were chosen on the same history (OOS included); the only clean out-of-sample is the forward demo from 2026-11-16.
* Intraday loss is modelled, not measured; no floating P&L, no CE(S)T midnight reset; commissions assumed; crypto swap 30 % (demo) until the Swing demo says otherwise.
* Numbers for the new engine in this folder come from a proxy price rebuild (no Yahoo in the sandbox) — re-run with the commands in `CAMBIOS_RICK.md`.
