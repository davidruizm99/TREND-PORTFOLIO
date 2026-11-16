"""Compara las señales del motor con las columnas position/size de tus CSV del grid.
Uso A (sin internet, solo sistemas SIN Donchian/Keltner):  python verificar.py --csv-dir RUTA
Uso B (en el VPS, todos los sleeves, con OHLC real):         python verificar.py --csv-dir RUTA --yahoo
"""
import argparse, os, sys
import numpy as np, pandas as pd
from indicadores import estado
from sizing import tamano_objetivo, posicion_efectiva
from sleeves import SLEEVES

CSV = {"BTC":"BTC-USD_c2_daily.csv","ETH":"ETH-USD_c1_vol35_daily.csv","SOL":"SOL-USD_c1_daily.csv","AAPL":"AAPL_c1_daily.csv",
       "NVDA":"NVDA_c1_vol35_exacto_daily.csv","AMD":"AMD_c1_daily.csv","GLD":"GLD_c1_daily.csv","DBC":"DBC_c1_daily.csv","QQQ":"QQQ_10bp_daily.csv","SPY":"SPY_10bp_daily.csv"}
OHLC = ("DONCHIAN", "KELTNER")

def datos_desde_csv(d):
    c = (1 + d["bh_return"].fillna(0)).cumprod() * 100
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c})

def datos_yahoo(tk, cal):
    import yfinance as yf
    df = yf.download(tk, start="1999-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]; df = df[["open","high","low","close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df if cal == "7d" else df[df.index.dayofweek < 5]

ap = argparse.ArgumentParser(); ap.add_argument("--csv-dir", required=True); ap.add_argument("--yahoo", action="store_true")
ap.add_argument("--desde", default="2019-07-01", help="se ignora el calentamiento anterior a esta fecha (solo modo CSV)")
a = ap.parse_args()
print(f"{'sleeve':6s} {'barras':>6s} {'coinciden':>10s}  nota")
for nom, fn in CSV.items():
    s = SLEEVES[nom]; ruta = os.path.join(a.csv_dir, fn)
    if not os.path.exists(ruta): print(f"{nom:6s} falta {fn}"); continue
    ref = pd.read_csv(ruta, parse_dates=["date"], index_col="date")
    usa_ohlc = any(t in s["spec"] for t in OHLC)
    if usa_ohlc and not a.yahoo: print(f"{nom:6s} {'-':>6s} {'-':>10s}  usa Donchian/Keltner: necesita --yahoo (OHLC real)"); continue
    df = datos_yahoo(s["ticker"], s["cal"]) if a.yahoo else datos_desde_csv(ref)
    est = estado(df, s["spec"]); anual = 365 if s["cal"] == "7d" else 252
    tam = tamano_objetivo(df, est, s["vol"], anual); pos = pd.Series(posicion_efectiva(tam, s.get('delay', 2)), index=df.index)
    comun = ref.index.intersection(pos.index)
    comun = comun[comun >= pd.Timestamp(a.desde)] if not a.yahoo else comun
    col = "size" if "size" in ref.columns else "position"
    x, y = pos.loc[comun].values, ref.loc[comun, col].values.astype(float)
    ok = np.isclose(x, y, atol=1e-9)
    print(f"{nom:6s} {len(comun):6d} {ok.mean()*100:9.2f}%  comparado con '{col}'" + ("" if ok.all() else f"; primeros desajustes: {[str(d.date()) for d in comun[~ok][:3]]}"))
