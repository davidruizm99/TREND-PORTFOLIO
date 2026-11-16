"""Rotación TOP3 + filtro de correlación + refugio GLD (port 1:1 de exp5.run(mode='filtro')).
B: retornos diarios buy&hold por sleeve en calendario de 7 días (0 antes de que exista el activo y en fines de semana de bolsa). existe: DataFrame bool, True desde que el activo cotiza.
P: DataFrame bool, el sleeve está dentro (tamaño>0).
lag: 1 = reproduce el backtest; en live se usa la información hasta el cierre t para operar al cierre t+1 (equivale a lag 2 del backtest)."""
import numpy as np, pandas as pd
RSI_N, RSI_THR, TOP, WIN, THR, REFUGIO = 60, 50, 3, 120, 0.8, "GLD"

def rsi_wilder(x, n):
    d = x.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))

def _corr(Bw, fin, win):
    """correlación de retornos de días laborables en la ventana que termina en el último laborable <= fin."""
    x = Bw.loc[:fin].tail(win)
    if len(x) < win: return None
    ok = ~x.isna().any().values; C = np.full((x.shape[1],) * 2, np.nan)
    if ok.sum() >= 2:
        ii = np.where(ok)[0]; C[np.ix_(ii, ii)] = np.corrcoef(x.values[:, ok], rowvar=False)
    return C

def elegir(rsv, C, top=TOP, thr=THR):
    cand = np.where(rsv > RSI_THR)[0]
    if len(cand) == 0: return []
    order = cand[np.argsort(-rsv[cand], kind="stable")]
    chosen, rest = [order[0]], list(order[1:])
    while len(chosen) < top and rest:
        pick = None
        for j in rest:
            cc = C[j, chosen] if C is not None else np.array([])
            cc = cc[~np.isnan(cc)]
            if len(cc) == 0 or cc.max() <= thr: pick = j; break
        if pick is None: break
        chosen.append(pick); rest.remove(pick)
    return chosen

def pesos(B, P, existe, lag=1, win=WIN, thr=THR, top=TOP, refugio=REFUGIO, desde=None):
    """Pesos 'de selección' diarios (sin tamaño del sleeve), igual que exp5.run(mode='filtro')."""
    cols = list(B.columns); idx = B.index
    rs = rsi_wilder((1 + B.fillna(0)).cumprod(), RSI_N).shift(lag); rsv = rs.values
    Bw = B.where(existe)[B.index.dayofweek < 5]
    w = np.zeros(B.shape); ini = 0 if desde is None else idx.searchsorted(pd.Timestamp(desde))
    for t in range(ini, len(idx)):
        if np.isnan(rsv[t]).all(): continue
        p = Bw.index.searchsorted(idx[t], side="left") - 1          # último laborable estrictamente anterior
        C = _corr(Bw, Bw.index[p], win) if p >= 0 else None
        for j in elegir(np.nan_to_num(rsv[t], nan=-1), C, top, thr): w[t, j] = 1.0 / top if P.iloc[t, j] else 0.0
    w = pd.DataFrame(w, index=idx, columns=cols)
    if refugio:
        sob = 1 - w.sum(axis=1); w[refugio] += sob * ((rs[refugio] > RSI_THR) & P[refugio])
    return w

def pesos_live(B, size, existe, win=WIN, thr=THR, top=TOP, refugio=REFUGIO):
    """Objetivo para operar al cierre siguiente: información hasta la última barra de B. size = tamaño objetivo (0..1) de cada sleeve en esa barra.
    Devuelve (pesos objetivo por sleeve, elegidos, rsi)."""
    cols = list(B.columns); Bf = B.copy()
    rs = rsi_wilder((1 + Bf.fillna(0)).cumprod(), RSI_N).iloc[-1]
    Bw = B.where(existe)[B.index.dayofweek < 5]; C = _corr(Bw, Bw.index[-1], win)
    ch = elegir(np.nan_to_num(rs.values, nan=-1), C, top, thr)
    P = (size > 0); w = pd.Series(0.0, index=cols)
    for j in ch:
        if P[cols[j]]: w[cols[j]] = 1.0 / top
    sob = 1 - w.sum()
    if refugio and rs[refugio] > RSI_THR and P[refugio]: w[refugio] += sob
    return w * size, [cols[j] for j in ch], rs
