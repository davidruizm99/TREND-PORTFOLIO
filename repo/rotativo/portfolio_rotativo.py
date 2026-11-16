"""PORTFOLIO ROTATIVO — ejecútalo tú.   python portfolio_rotativo.py
Lee los CSV diarios de cada sleeve (columnas bh_return, strategy_return, position o size), aplica la rotación
(RSI60 > 50, TOP_N, filtro de correlación opcional, refugio GLD) y escribe resultados.
La lógica de la rotación vive SOLO en rotacion.py (verificada: 0 días distintos de 3223 contra el backtest original).
Todo lo que se toca está en CONFIGURACIÓN."""
import os, math, sys
import numpy as np, pandas as pd
import rotacion as R

# =============================== CONFIGURACIÓN ===============================
CARPETA = "../data/csv_sleeves"   # CSV diarios de los sleeves
SLEEVES = {                          # sleeve: archivo. Quita o añade líneas para cambiar el universo
    "BTC": "BTC-USD_c2_daily.csv", "ETH": "ETH-USD_c1_vol35_daily.csv", "SOL": "SOL-USD_c1_daily.csv",
    "QQQ": "QQQ_10bp_daily.csv", "SPY": "SPY_10bp_daily.csv", "GLD": "GLD_c1_daily.csv", "TQQQ": "TQQQ_10bp_daily.csv",
    "AAPL": "AAPL_c1_daily.csv", "NVDA": "NVDA_c1_vol35_exacto_daily.csv", "DBC": "DBC_c1_daily.csv", "AMD": "AMD_c1_daily.csv",
}
TOP_N = 3
FILTRO_CORR = True                   # False = TOP_N puro por RSI
UMBRAL, VENTANA = 0.8, 120           # filtro: correlación máxima admitida y ventana (días laborables)
LAG = 1                              # 1 = como el backtest; 2 = ranking un día más tarde (operar tras el cierre)
REFUGIO = "GLD"                      # None = efectivo
COSTE_ROTACION_BP = 10               # coste por unidad de cambio de peso entre sleeves (el backtest original NO lo cobraba; se muestran ambas filas)
ANUAL = 365                          # calendario de 7 días
VENTANAS = {"Todo": (None, None), "2019-07+": ("2019-07-01", None), "Quant 2020-04..2025-11": ("2020-04-01", "2025-11-30"), "OOS 2024+": ("2024-01-01", None)}
SALIDA = "resultados"                # carpeta de salida
# =============================================================================

def leer(f):
    df = pd.read_csv(os.path.join(CARPETA, f), parse_dates=["date"], index_col="date").sort_index()
    for c in ("bh_return", "strategy_return"): assert c in df.columns, f"{f}: falta {c}"
    if "position" not in df.columns:
        df["position"] = (df["size"] > 0).astype(int) if "size" in df.columns else (df["strategy_return"] != 0).astype(int)
    return df[["bh_return", "strategy_return", "position"]].astype(float)

def cargar():
    d = {k: leer(f) for k, f in SLEEVES.items()}
    ini = {k: x.index.min() for k, x in d.items()}
    idx = None
    for x in d.values(): idx = x.index if idx is None else idx.union(x.index)
    idx = pd.date_range(idx.min(), idx.max(), freq="D")
    S = pd.DataFrame({k: x["strategy_return"].reindex(idx).fillna(0.0) for k, x in d.items()})
    B = pd.DataFrame({k: x["bh_return"].reindex(idx).fillna(0.0) for k, x in d.items()})
    AV = pd.DataFrame({k: idx >= ini[k] for k in d}, index=idx)
    P = pd.DataFrame({k: (x["position"].reindex(idx).ffill().fillna(0) > 0) for k, x in d.items()}) & AV
    return S, B, P, AV

def metricas(r):
    r = r.dropna()
    eq = (1 + r).cumprod(); anios = (r.index[-1] - r.index[0]).days / 365.25; sd = r.std(ddof=1)
    return dict(sharpe=r.mean() / sd * math.sqrt(ANUAL), cagr=eq.iloc[-1] ** (1 / anios) - 1, maxdd=(eq / eq.cummax() - 1).min(), vol=sd * math.sqrt(ANUAL))

def main():
    S, B, P, AV = cargar()
    print("Sleeves:", list(S.columns), "| inicio", S.index.min().date(), "fin", S.index.max().date(), "| filtro", UMBRAL if FILTRO_CORR else "no", "| TOP", TOP_N, "| LAG", LAG, "| refugio", REFUGIO)
    w = R.pesos(B, P, AV, lag=LAG, win=VENTANA, thr=UMBRAL if FILTRO_CORR else 1.01, top=TOP_N, refugio=REFUGIO)
    r = (w * S).sum(axis=1)
    giro = w.diff().abs().sum(axis=1).fillna(0)
    rn = r - giro * COSTE_ROTACION_BP / 1e4
    filas = []
    for nom, x in (("Rotación (como el backtest, sin coste de rotación)", r), (f"Rotación neta ({COSTE_ROTACION_BP} pb por giro)", rn)):
        fila = [nom]
        for a, z in VENTANAS.values():
            m = metricas(x.loc[a:z]); fila.append(f"{m['sharpe']:.2f} / {m['cagr']*100:.1f}% / {m['maxdd']*100:.1f}%")
        filas.append(fila)
    print("\nSharpe / CAGR / MaxDD"); print(pd.DataFrame(filas, columns=["", *VENTANAS.keys()]).to_string(index=False))
    print(f"\nGiro: {giro.sum()/(len(giro)/365):.0f}x al año (suma de |Δpeso|) · exposición media {w.sum(axis=1).mean()*100:.0f}% · GLD medio {w.get('GLD', pd.Series(0)).mean()*100:.1f}%")
    os.makedirs(SALIDA, exist_ok=True)
    pd.DataFrame({"ret": rn}).rename_axis("date").to_csv(os.path.join(SALIDA, "rotativo_daily_neto.csv"))
    pd.DataFrame({"ret": r}).rename_axis("date").to_csv(os.path.join(SALIDA, "rotativo_daily_sin_coste_rotacion.csv"))
    w.rename_axis("date").to_csv(os.path.join(SALIDA, "pesos_diarios.csv"))
    print("Escrito en", os.path.abspath(SALIDA), ": rotativo_daily_neto.csv, rotativo_daily_sin_coste_rotacion.csv, pesos_diarios.csv")
if __name__ == "__main__": main()
