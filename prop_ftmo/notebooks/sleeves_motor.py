"""Motor de sleeves en vivo: calcula la posicion de cada sleeve (sistema del grid) a partir de los precios, sin abrir el notebook.
Indicadores y sizing copiados 1:1 de tu repo (sleeves/indicadores.py, sizing.py, sleeves.py). 
Convencion: la senal al cierre de t se opera al cierre de t+delay-1 y rinde desde t+delay (delay=2 por defecto; QQQ/SPY delay=1).
Para el rotativo, el estado del sleeve usado en la decision del cierre t es  obj[t+1-delay]  (= obj.shift(delay-1))."""
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

EXEC_DELAY = 2   # señal al cierre de t -> se opera al cierre de t+1 -> rinde desde t+2

def tamano_objetivo(df, estado, vol_obj=None, anual=252, vol_n=20, paso=0.25, lev_max=1.0):
    """Tamaño objetivo calculado al cierre de cada barra (antes del retraso de ejecución)."""
    tam = estado.astype(float)
    if vol_obj:
        ret = df["close"].pct_change().fillna(0.0).values
        volr = pd.Series(ret).rolling(vol_n).std().values * math.sqrt(anual)
        tam = tam * np.where(np.isnan(volr), 1.0, np.minimum(lev_max, vol_obj / np.maximum(volr, 1e-6)))
        if paso: tam = np.round(tam / paso) * paso
        tam = np.clip(tam, 0, lev_max)
    return tam

def posicion_efectiva(objetivo, delay=EXEC_DELAY):
    pos = np.zeros(len(objetivo)); pos[delay:] = objetivo[:-delay]; return pos

SLEEVES = {
 "BTC":  dict(ticker="BTC-USD", cal="7d", vol=None, spec="EMA3(f=5,m=40,s=65) AND MACD(f=16,s=21,sig=16,modo=senal) AND HMA(n=250,banda=0.0)"),
 "ETH":  dict(ticker="ETH-USD", cal="7d", vol=0.35, spec="EMA3X(f=16,m=120,s=65) OR MACD(f=20,s=300,sig=9,modo=senal) OR DONCHIAN(n_in=150,n_out=10)"),
 "SOL":  dict(ticker="SOL-USD", cal="7d", vol=None, spec="MACD(f=8,s=80,sig=16,modo=senal) AND DONCHIAN(n_in=150,n_out=40) AND TSMOM(n=252)"),
 "AAPL": dict(ticker="AAPL", cal="LV", vol=None, spec="EMA3X(f=5,m=200,s=160) AND MACD(f=8,s=21,sig=9,modo=senal)"),
 "NVDA": dict(ticker="NVDA", cal="LV", vol=0.35, spec="SMA(n=200) OR MACD(f=8,s=200,sig=16,modo=senal)"),
 "AMD":  dict(ticker="AMD", cal="LV", vol=None, spec="EMA3(f=12,m=20,s=80) AND MACD(f=8,s=200,sig=5,modo=senal) AND KELTNER(n=80,m=1.0,atr_n=14)"),
 "GLD":  dict(ticker="GLD", cal="LV", vol=None, spec="EMA3X(f=16,m=120,s=100) OR KELTNER(n=20,m=2.5,atr_n=14)"),
 "QQQ":  dict(ticker="QQQ", cal="LV", vol=None, spec="EMA3(f=20,m=80,s=100) OR MACD(f=8,s=26,sig=0,modo=cero)"),
 "SPY":  dict(ticker="SPY", cal="LV", vol=None, spec="EMA3(f=20,m=80,s=100) OR MACD(f=8,s=26,sig=0,modo=cero)"),
 "JNJ":  dict(ticker="JNJ", cal="LV", vol=None, spec="EMA3X(f=30,m=80,s=65)"),   # candidato #1 del informe JNJ (casi-aprobado: NO supera los filtros de Rick, Sharpe ALL 1.01)
}

def datos_yahoo(tk, cal, hoy=None):
    """OHLC diario de Yahoo SIN la barra de hoy (UTC): es provisional (mercado abierto / dia sin cerrar). La senal se calcula con el cierre de ayer."""
    import yfinance as yf, datetime as dt
    hoy = pd.Timestamp(dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)).normalize() if hoy is None else pd.Timestamp(hoy).normalize()
    df = yf.download(tk, start="1999-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]; df = df[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df.index < hoy]
    return df if cal == "7d" else df[df.index.dayofweek < 5]

def estado_sleeve(nombre, hoy=None):
    """(tamano objetivo al cierre de cada barra, fecha de la ultima barra). SIN desplazar: el motor aplica el retraso (senal cierre t -> ejecucion t+1 -> rinde desde t+2),
    el mismo para TODOS los sleeves (QQQ/SPY incluidos). Tamano = 0..1 (ETH/NVDA con objetivo de vol 0,35 en pasos de 0,25), no solo on/off."""
    s = SLEEVES[nombre]
    df = datos_yahoo(s["ticker"], s["cal"], hoy)
    est = estado(df, s["spec"]); anual = 365 if s["cal"] == "7d" else 252
    tam = tamano_objetivo(df, est, s["vol"], anual)
    return pd.Series(np.asarray(tam, float), index=df.index), df.index[-1]
