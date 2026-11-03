"""python diag2.py --json frozen_TQQQ.json --csv ../csv_sleeves/TQQQ_10bp_daily.csv"""
import argparse, json, numpy as np, pandas as pd
from tqqq_engine import download_ohlc
from tqqq_motor import Engine, position_path, COMBOS_NUEVO, ERA_FLAT, WARMUP_START
ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--csv", required=True); a = ap.parse_args()
ref = pd.read_csv(a.csv, index_col=0, parse_dates=True); D = download_ohlc("TQQQ", WARMUP_START)
slow = json.load(open(a.json, encoding="utf-8")); idx = D.index; N = len(idx); i0 = int(idx.searchsorted(pd.Timestamp(ERA_FLAT)))
E = {"L": Engine(D[["high","low","close"]], slow["macd_role"], int(slow["trail_atr"]), bool(slow["exit_latch"])), "N": Engine(D[["high","low","close"]], "both_confirm", 14, False)}
C = {"L": [tuple(c) for c in slow["combos"]], "N": COMBOS_NUEVO}
P = {}
for k in "LN":
    cols = []
    for cb in C[k]:
        u, d = E[k].raw_signals(cb); p = np.zeros(N); p[i0:] = position_path(u[i0:], d[i0:]); cols.append(p)
    P[k] = np.mean(cols, axis=0)
sh = lambda x, d: np.r_[np.zeros(d), x[:N-d]] if d > 0 else x
y = ref["size"].astype(float)
print("delay_lento delay_nuevo  coincide%  dif_media")
for dL in (0, 1, 2, 3):
    for dN in (0, 1, 2, 3):
        ef = pd.Series(0.5*sh(P["L"], dL) + 0.5*sh(P["N"], dN), index=idx).reindex(ref.index)
        ok = np.isclose(ef.values, y.values, atol=1e-9)
        print(f"{dL:^11} {dN:^12} {ok.mean()*100:9.2f}  {np.abs(ef.values-y.values).mean():.4f}")
# cada grupo por separado contra el CSV: ¿qué parte del tamaño explica cada uno?
