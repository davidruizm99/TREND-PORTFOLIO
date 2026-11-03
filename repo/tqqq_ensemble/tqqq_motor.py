"""TQQQ 50/50: 45 sistemas LENTOS (frozen_TQQQ.json, el de 'top-5 x 9 Trails') + 15 NUEVOS (definidos en el notebook), pesos iguales.
Cada sistema = 3 EMA (cruces) + MACD de confirmación (both_confirm) + trailing ATR opcional. size = fracción de sistemas dentro.
Uso:  python tqqq_motor.py --json frozen_TQQQ.json --csv TQQQ_10bp_daily.csv [--yahoo]
"""
import argparse, json
import numpy as np, pandas as pd
from tqqq_engine import Engine, position_path

WARMUP_START = "2009-06-01"; ERA_FLAT = "2018-01-02"
FROZEN_NUEVO = [((3,20,200),(12,26,9),(22,3.0)),((3,34,200),(12,26,9),None),((3,20,150),(12,26,9),(22,3.0)),((3,34,150),(12,26,9),None),((3,34,100),(12,26,9),None),
    ((3,20,100),(12,26,9),None),((3,75,200),(12,26,9),(50,2.5)),((3,75,150),(12,26,9),(50,2.5)),((3,75,100),(12,26,9),(50,2.5)),((5,34,200),(12,26,9),None),
    ((3,50,200),(12,26,9),(50,2.5)),((5,34,150),(12,26,9),None),((5,75,200),(12,26,9),(50,2.5)),((5,75,150),(12,26,9),(50,2.5)),((3,50,150),(12,26,9),(50,2.5))]
COMBOS_NUEVO = [(e[0], e[1], e[2], *m, None, None, None, None, *(tr if tr else (None, None))) for e, m, tr in FROZEN_NUEVO]

def estados(D, slow_json, flat_from=ERA_FLAT, delay=1):
    """Devuelve (objetivo, efectivo). p[t] = tamaño (50/50) tras los eventos del cierre t, con la era plana desde flat_from.
    efectivo[t] = p[t-delay] (delay=1 reproduce el CSV del cuaderno al 99,95%; delay=2 = calcular tras cierre y operar al cierre siguiente).
    objetivo[t] = p[t] = posicion a mantener segun la informacion del cierre t."""
    slow = json.load(open(slow_json, encoding="utf-8"))
    assert slow["ticker"] == "TQQQ" and slow["macd_role"] == "both_confirm" and not slow["exit_latch"] and int(slow["trail_atr"]) == 14
    idx = D.index; i0 = int(idx.searchsorted(pd.Timestamp(flat_from))); N = len(idx)
    ENG = {"L": Engine(D[["high","low","close"]], slow["macd_role"], int(slow["trail_atr"]), bool(slow["exit_latch"])),
           "N": Engine(D[["high","low","close"]], "both_confirm", 14, False)}
    COMB = {"L": [tuple(c) for c in slow["combos"]], "N": COMBOS_NUEVO}
    P = {}
    for k in "LN":
        cols = []
        for cb in COMB[k]:
            u, d = ENG[k].raw_signals(cb); p = np.zeros(N); p[i0:] = position_path(u[i0:], d[i0:]); cols.append(p)
        P[k] = np.mean(cols, axis=0)
    obj = 0.5 * P["L"] + 0.5 * P["N"]
    efectivo = np.r_[np.zeros(delay), obj[:N - delay]]
    return pd.Series(obj, index=idx), pd.Series(efectivo, index=idx)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--csv", required=True); ap.add_argument("--yahoo", action="store_true")
    a = ap.parse_args()
    ref = pd.read_csv(a.csv, parse_dates=["date"], index_col="date")
    if a.yahoo:
        from tqqq_engine import download_ohlc
        D = download_ohlc("TQQQ", WARMUP_START)
    else:
        print("AVISO: sin OHLC real; se usa el cierre como máximo/mínimo (el trailing ATR sale aproximado)")
        c = (1 + ref["bh_return"].fillna(0)).cumprod() * 100; D = pd.DataFrame({"high": c, "low": c, "close": c})
    for dl in (1, 2):
        _, ef = estados(D, a.json, delay=dl)
        c = ref.index.intersection(ef.index); x, y = ef.loc[c].values, ref.loc[c, "size"].values.astype(float)
        print(f"[delay={dl}] TQQQ 50/50: {len(c)} barras · coinciden {np.isclose(x, y, atol=1e-9).mean()*100:.2f}% · dif media {np.abs(x-y).mean():.4f}")
