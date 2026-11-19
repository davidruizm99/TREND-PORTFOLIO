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

def clusters_por_correlacion(X, umbral=0.90):
    """Agrupa los sistemas (columnas de X = caminos de posicion 0/1) por correlacion: enlace promedio, une mientras la correlacion media entre grupos >= umbral.
    Devuelve etiquetas 0..K-1. Un sistema constante (siempre fuera o siempre dentro) forma su propio grupo."""
    n = X.shape[1]; sd = X.std(axis=0)
    C = np.corrcoef(np.where(sd > 0, X, X + np.random.default_rng(0).normal(0, 1e-9, X.shape)), rowvar=False)
    C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    grupos = [[i] for i in range(n)]
    while len(grupos) > 1:
        best, par = -2.0, None
        for a in range(len(grupos)):
            for b in range(a + 1, len(grupos)):
                c = C[np.ix_(grupos[a], grupos[b])].mean()
                if c > best: best, par = c, (a, b)
        if best < umbral: break
        a, b = par; grupos[a] = grupos[a] + grupos[b]; del grupos[b]
    lab = np.zeros(n, int)
    for g, ids in enumerate(grupos): lab[ids] = g
    return lab

def pesos_clusters(X, umbral=0.90):
    """Peso de cada sistema = 1/(K x tamano de su grupo): cada GRUPO de sistemas casi identicos cuenta como UNA opinion (antes: 45 lentos + 15 nuevos al 50/50,
    asi que cada nuevo pesaba 3 veces mas que un lento y muchos eran el mismo MACD/EMA)."""
    lab = clusters_por_correlacion(X, umbral); K = lab.max() + 1
    w = np.array([1.0 / (K * (lab == lab[i]).sum()) for i in range(X.shape[1])])
    return w, lab

def estados(D, slow_json, flat_from=ERA_FLAT, delay=2, pesos="clusters", umbral=0.90):
    """Devuelve (objetivo, efectivo). p[t] = tamaño (50/50) tras los eventos del cierre t, con la era plana desde flat_from.
    efectivo[t] = p[t-delay]. delay=2 (POR DEFECTO) = senal al cierre t, ejecucion al cierre t+1, rinde desde t+2. delay=1 solo reproduce el CSV del cuaderno (la senal
    del cierre t ganaba ese mismo cierre) y NO es operable. pesos="clusters" (por defecto): cada grupo de sistemas correlacionados (>= umbral) vale una opinion;
    pesos="conteo": el 50/50 lento/nuevo antiguo, solo para reproducir el cuaderno.
    objetivo[t] = p[t] = posicion a mantener segun la informacion del cierre t."""
    slow = json.load(open(slow_json, encoding="utf-8"))
    assert slow["ticker"] == "TQQQ" and slow["macd_role"] == "both_confirm" and not slow["exit_latch"] and int(slow["trail_atr"]) == 14
    idx = D.index; i0 = int(idx.searchsorted(pd.Timestamp(flat_from))); N = len(idx)
    ENG = {"L": Engine(D[["high","low","close"]], slow["macd_role"], int(slow["trail_atr"]), bool(slow["exit_latch"])),
           "N": Engine(D[["high","low","close"]], "both_confirm", 14, False)}
    COMB = {"L": [tuple(c) for c in slow["combos"]], "N": COMBOS_NUEVO}
    P, TODOS = {}, []
    for k in "LN":
        cols = []
        for cb in COMB[k]:
            u, d = ENG[k].raw_signals(cb); p = np.zeros(N); p[i0:] = position_path(u[i0:], d[i0:]); cols.append(p)
        P[k] = np.mean(cols, axis=0); TODOS += cols
    if pesos == "conteo":
        obj = 0.5 * P["L"] + 0.5 * P["N"]
    else:
        X = np.array(TODOS).T; w, lab = pesos_clusters(X[i0:], umbral)
        obj = X @ w
        print(f"[clusters] {X.shape[1]} sistemas -> {lab.max() + 1} grupos independientes (correlacion de posiciones >= {umbral} = misma opinion); cada grupo pesa 1/{lab.max() + 1}")
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
    for dl, pw in ((1, "conteo"), (2, "conteo"), (2, "clusters")):
        _, ef = estados(D, a.json, delay=dl, pesos=pw)
        c = ref.index.intersection(ef.index); x, y = ef.loc[c].values, ref.loc[c, "size"].values.astype(float)
        print(f"[delay={dl}, pesos={pw}] TQQQ: {len(c)} barras · coinciden {np.isclose(x, y, atol=1e-9).mean()*100:.2f}% · dif media {np.abs(x-y).mean():.4f}")
