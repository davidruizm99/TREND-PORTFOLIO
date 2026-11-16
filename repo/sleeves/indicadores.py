"""Indicadores copiados 1:1 del notebook Grid_research_filtros_inverso.ipynb (celda 3),
pero evaluados para UN solo juego de parámetros y sobre un DataFrame OHLC cualquiera."""
import math, re
import numpy as np, pandas as pd

def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()

def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    return x.rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)

def hma(x, n):
    n2, ns = max(int(n / 2), 1), max(int(round(math.sqrt(n))), 1)
    return wma(2 * wma(x, n2) - wma(x, n), ns)

def rsi_wilder(x, n):
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))

def _atr(close, high, low, n):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

def _b(x): return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0).astype(bool)
def _cruza_arriba(a, b): return _b((a > b) & (a.shift(1) <= b.shift(1)))
def _cruza_abajo(a, b):  return _b((a < b) & (a.shift(1) >= b.shift(1)))

def estado_con_memoria(entrada, salida):
    s = np.zeros(len(entrada), dtype=bool); on = False
    for i in range(len(entrada)):
        if on:
            if salida[i]: on = False
        elif entrada[i]:
            on = True
        s[i] = on
    return s

def estado_entrada_prioritaria(entrada, salida):
    s_ = np.zeros(len(entrada), dtype=bool); on = False
    for i in range(len(entrada)):
        if entrada[i]: on = True
        elif salida[i]: on = False
        s_[i] = on
    return s_

# ---- un indicador = una función (df, **params) -> array bool "quiero estar largo al cierre de t" ----
def EMA3(df, f, m, s):
    c = df["close"]
    return _b((ema(c, f) > ema(c, m)) & (ema(c, m) > ema(c, s)))

def EMA3X(df, f, m, s):
    c = df["close"]; e1, e2, e3 = ema(c, f), ema(c, m), ema(c, s)
    ent = _cruza_arriba(e1, e2) | _cruza_arriba(e1, e3) | _cruza_arriba(e2, e3)
    sal = _cruza_abajo(e1, e2) | _cruza_abajo(e1, e3) | _cruza_abajo(e2, e3)
    return estado_entrada_prioritaria(ent, sal)

def SMA(df, n):
    c = df["close"]; return _b(c > c.rolling(n).mean())

def MACD(df, f, s, sig, modo):
    c = df["close"]; macd = ema(c, f) - ema(c, s)
    return _b(macd > 0) if modo == "cero" else _b(macd > macd.ewm(span=sig, adjust=False).mean())

def HMA(df, n, banda):
    c = df["close"]; return _b(c > hma(c, n) * (1 - banda))

def EMAX(df, f, s):
    c = df["close"]; return _b(ema(c, f) > ema(c, s))

def DONCHIAN(df, n_in, n_out):
    hh = df["high"].shift(1).rolling(n_in).max(); ll = df["low"].shift(1).rolling(n_out).min()
    return estado_con_memoria(_b(df["close"] > hh), _b(df["close"] < ll))

def KELTNER(df, n, m, atr_n):
    c = df["close"]; e = ema(c, n); sup = e + m * _atr(c, df["high"], df["low"], atr_n)
    return estado_con_memoria(_b(c > sup), _b(c < e))

def TSMOM(df, n):
    c = df["close"]; return _b(c / c.shift(n) - 1 > 0)

def RSI(df, n, umbral):
    return _b(rsi_wilder(df["close"], n) > umbral)

REGISTRO = dict(EMA3=EMA3, EMA3X=EMA3X, SMA=SMA, MACD=MACD, HMA=HMA, EMAX=EMAX, DONCHIAN=DONCHIAN, KELTNER=KELTNER, TSMOM=TSMOM, RSI=RSI)

_TERM = re.compile(r"(\w+)\(([^)]*)\)")
def _val(v):
    try: return int(v)
    except ValueError:
        try: return float(v)
        except ValueError: return v

def parsear(spec):
    """'EMA3X(f=16,m=120,s=65) OR MACD(f=20,s=300,sig=9,modo=senal)' -> (logica, [(nombre, params), ...])"""
    logica = "AND" if " AND " in spec else "OR"
    terminos = []
    for nombre, args in _TERM.findall(spec):
        terminos.append((nombre, {k.strip(): _val(v.strip()) for k, v in (a.split("=") for a in args.split(","))}))
    return logica, terminos

def estado(df, spec):
    """Estado booleano del sistema (array por barra): True = el sistema quiere estar largo al cierre de t."""
    logica, terminos = parsear(spec)
    mats = [REGISTRO[n](df, **p) for n, p in terminos]
    s = mats[0].copy()
    for m_ in mats[1:]:
        s = (s | m_) if logica == "OR" else (s & m_)
    return s
