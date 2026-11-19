"""Pruebas de los puntos 1 y 6 de Rick (cache de precios, barra provisional, frescura). Sin internet: yfinance se sustituye por un generador.
Ejecuta desde la raiz del repo:  python tests/test_datos.py"""
import sys, os, types, importlib.util, tempfile, numpy as np, pandas as pd
AQUI = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(AQUI, "..", "notebooks", "ftmo_rotativo.py")
os.chdir(tempfile.mkdtemp())                                   # la cache ftmo_precios.pkl se crea aqui, no en tu repo
sys.path.insert(0, os.path.dirname(SRC))
spec = importlib.util.spec_from_file_location("ftmo", SRC); F = importlib.util.module_from_spec(spec); sys.modules["ftmo"] = F; spec.loader.exec_module(F)
FIN = {"fin": pd.Timestamp("2026-11-11")}; LLAMADAS = []
def serie(t, fin):
    idx = pd.date_range("2016-06-01", fin, freq="D")
    if not t.endswith("-USD"): idx = idx[idx.dayofweek < 5]
    r = np.random.default_rng(sum(map(ord, t))).normal(0.0004, 0.02, len(idx)); return pd.Series(100 * np.exp(np.cumsum(r)), idx)
def download(lista, start, auto_adjust=True, progress=False, group_by="column"):
    lista = [lista] if isinstance(lista, str) else lista; LLAMADAS.append((len(lista), start))
    d = pd.DataFrame({t: serie(t, FIN["fin"])[lambda s: s.index >= pd.Timestamp(start)] for t in lista}); d.columns = pd.MultiIndex.from_product([["Close"], d.columns]); return d
sys.modules["yfinance"] = types.SimpleNamespace(download=download)
px = F.descargar(hoy="2026-11-11"); assert px.index[-1] == pd.Timestamp("2026-11-10"), "la barra provisional de hoy NO debe guardarse"; print("OK 1: descarga inicial sin la barra de hoy")
LLAMADAS.clear(); FIN["fin"] = pd.Timestamp("2026-11-14"); px = F.descargar(hoy="2026-11-14")
assert px.index[-1] == pd.Timestamp("2026-11-13") and all(s != F.START for _, s in LLAMADAS); print("OK 2: la cache NO se congela (baja solo el solape y extiende)")
def download2(lista, start, **k):
    d = download(lista, start)
    if "BTC-USD" in d["Close"].columns: d.loc[pd.Timestamp("2026-11-09"), ("Close", "BTC-USD")] = 1.0
    return d
sys.modules["yfinance"].download = download2; px = F.descargar(hoy="2026-11-15")
assert float(px.loc["2026-11-09", "BTCUSD"]) == 1.0; print("OK 3: el solape se reemplaza (una revision de Yahoo sustituye al valor viejo)")
F.comprobar_frescura(px, ["BTCUSD", "ETHUSD", "NVDA", "US100"], hoy="2026-11-15"); print("OK 4: datos frescos -> sigue")
p4 = px.copy(); p4.loc[p4.index > "2026-11-10", "ETHUSD"] = np.nan
try: F.comprobar_frescura(p4, ["BTCUSD", "ETHUSD"], hoy="2026-11-15"); raise AssertionError("no paro con ETH congelado")
except SystemExit: print("OK 5: un activo desactualizado PARA la ejecucion")
F.comprobar_frescura(px, ["NVDA", "US100", "XAUUSD"], hoy="2026-11-16"); print("OK 6: lunes con barra del viernes -> sigue")
