#!/usr/bin/env python3
"""
ROTATIVO FTMO  -  el portfolio rotativo reconstruido sobre TODO el menu de FTMO
==============================================================================
Mismas reglas que el rotativo de la Guia (secciones 3, 4 y 14):
  * Ranking por RSI(60) del buy & hold, con 1 dia de retraso, umbral 50.
  * Doble eligibilidad: entra solo si esta en el ranking Y su sleeve esta largo.
  * Filtro de correlacion duro: 120 dias, umbral 0,8 frente a los ya elegidos.
  * Huecos vacios -> refugio ORO (XAUUSD) si su sleeve esta largo; si no, efectivo.
  * Solo largos (los cortos perdieron en todas las pruebas del long-short).
Lo nuevo:
  * Universo = lo que FTMO deja operar: 14 indices, 5 metales, 4 energias, 7 agricolas,
    30 criptos y 57 acciones.
  * Sleeve SIN ENTRENAR, igual para todos (Guia 16.4-B): ensemble Carver EWMAC
    16/64 + 32/128 + 64/256; largo si la prevision combinada > 0. Nada ajustado por activo.
  * Estructura pensada para prop: opcion de cestas por clase de activo (un hueco por clase)
    y peso por inversa de volatilidad, para que un dia malo de cripto no rompa el 5% diario.
  * El grid se ordena por PROBABILIDAD DE PASAR FTMO (evaluador integrado), no por Sharpe.
    Seleccion con inicios 2018-2022 (IS); 2023+ (OOS) solo se mira.

Uso:   pip install yfinance pandas numpy
       python ftmo_rotativo.py              (descarga ~120 series la 1a vez, luego cache)
       python ftmo_rotativo.py --synthetic  (prueba sin internet; numeros sin valor)

Salidas: ftmo_grid.csv (todo el grid) y ftmo_daily_<config>.csv (date,ret) de las mejores
configuraciones: van directas al simulador v4 y al Monte Carlo.

LIMITACIONES CONOCIDAS (leer antes de fiarse):
  * Sesgo de supervivencia: la lista de acciones y criptos es la de HOY.
  * Materias primas = futuro continuo de Yahoo (=F), SIN ajustar el roll: hay saltos falsos
    en el dia de vencimiento (sobre todo gas natural). Los CFD de FTMO tambien hacen roll.
  * Indices = indice de precio (sin dividendos) en divisa local; la conversion a USD se ignora.
  * Costes de acciones, indices y materias primas = SUPUESTOS. Swap = 7,5%/ano sobre la
    exposicion (centro del 6-9% de Rick). Sustituir por los de la cuenta demo.
"""
import sys, os, pickle, itertools, warnings, time
AQUI = os.path.dirname(os.path.abspath(__file__))

def _hallar(*candidatos):
    """Primer camino que existe entre los candidatos (carpeta de trabajo, luego ../data del repo)."""
    for c in candidatos:
        if os.path.exists(c): return c
    return candidatos[0]
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ============================================================================ CONFIG
START, EVAL_START, OOS_START = "2016-06-01", "2018-01-01", "2023-01-01"
CACHE = "ftmo_precios.pkl"
RSI_N, RSI_THR, CORR_WIN, CORR_THR = 60, 50, 120, 0.80   # reglas de la Guia, FIJAS
MIN_OBS = 300            # un activo entra al universo con >= 300 sesiones de historia
SWAP_ANUAL = 0.075       # swap largo sobre la exposicion (Rick: 6-9%)
ANN = 365
REFUGIO = "XAUUSD"

# FTMO 2-Step Swing
OBJ1, OBJ2, DD_DIA, DD_MAX, DIAS_MIN = 0.10, 0.05, 0.05, 0.10, 4
FACTOR_INTRADIA, REF_DIAS = 1.5, 3       # igual que el simulador v4
H_EVAL, PASO, MIN_FUTURO = 900, 5, 365
K_MAX = 1.0                               # sin apalancar (1:1 en acciones/cripto en Swing)
K_MAX_CARVER = 3.0                        # cartera Carver pura sin cripto/acciones: indices y materias primas admiten apalancamiento (CONFIRMAR en FTMO)
SIZINGS = [("A", 0.35), ("A", 0.50), ("C", 0.10), ("C", 0.12), ("C", 0.15)]

# ============================================================================ UNIVERSO FTMO
# clase -> {simbolo FTMO: ticker Yahoo}
UNIVERSO = {
 "indices": {"US500": "^GSPC", "US100": "^NDX", "US30": "^DJI", "US2000": "^RUT", "GER40": "^GDAXI",
             "UK100": "^FTSE", "FRA40": "^FCHI", "EU50": "^STOXX50E", "SPN35": "^IBEX", "N25": "^AEX",
             "JP225": "^N225", "HK50": "^HSI", "AUS200": "^AXJO", "DXY": "DX-Y.NYB"},
 "metales": {"XAUUSD": "GC=F", "XAGUSD": "SI=F", "XPTUSD": "PL=F", "XPDUSD": "PA=F", "XCUUSD": "HG=F"},
 "energia": {"USOIL": "CL=F", "UKOIL": "BZ=F", "NATGAS": "NG=F", "HEATOIL": "HO=F"},
 "agricolas": {"CORN": "ZC=F", "WHEAT": "ZW=F", "SOYBEAN": "ZS=F", "COFFEE": "KC=F", "COCOA": "CC=F",
               "SUGAR": "SB=F", "COTTON": "CT=F"},
 "cripto": {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD", "XRPUSD": "XRP-USD",
            "BNBUSD": "BNB-USD", "ADAUSD": "ADA-USD", "DOGEUSD": "DOGE-USD", "LTCUSD": "LTC-USD",
            "LNKUSD": "LINK-USD", "AVAUSD": "AVAX-USD", "DOTUSD": "DOT-USD", "BCHUSD": "BCH-USD",
            "ETCUSD": "ETC-USD", "XLMUSD": "XLM-USD", "XMRUSD": "XMR-USD", "DASHUSD": "DASH-USD",
            "UNIUSD": "UNI7083-USD", "AAVUSD": "AAVE-USD", "NERUSD": "NEAR-USD", "ALGUSD": "ALGO-USD",
            "ICPUSD": "ICP-USD", "BARUSD": "HBAR-USD", "GALUSD": "GALA-USD", "GRTUSD": "GRT6719-USD",
            "IMXUSD": "IMX10603-USD", "MANUSD": "MANA-USD", "VECUSD": "VET-USD", "XTZUSD": "XTZ-USD",
            "SANUSD": "SAND-USD"},          # NEOUSD excluido: spread ~5,8%
 "acciones": {**{s: s for s in ("AAPL AMZN BA BABA BAC CSCO CVX DIS META FDX GE GM GOOG IBM INTC JNJ JPM KO "
                                 "MCD MSFT NFLX NVDA PFE QCOM RACE T TSLA V WMT XOM ZM RTX LMT PLTR AMD AVGO "
                                 "SBUX MSTR GME NKE ARM SNOW ASML AZN").split()},
              "BRK.B": "BRK-B", "ADSGn": "ADS.DE", "AIRF": "AF.PA", "ALVG": "ALV.DE", "BAYGn": "BAYN.DE",
              "DBKGn": "DBK.DE", "IBE": "IBE.MC", "LVMH": "MC.PA", "SAN": "SAN.MC", "SIEGn": "SIE.DE",
              "VOWG_p": "VOW3.DE", "TTE": "TTE.PA", "BMW": "BMW.DE", "MBG": "MBG.DE"},
}
CRIPTO_MAYORES = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "BNBUSD", "ADAUSD", "DOGEUSD", "LTCUSD", "LNKUSD", "AVAUSD"]

# coste por LADO (fraccion del nocional). Cripto: comision 0,065% + medio spread de tu captura.
SPREAD_CRIPTO = {"BTCUSD": .0013, "ETHUSD": .028, "LTCUSD": .30, "XRPUSD": .14, "XMRUSD": .05, "ADAUSD": .17,
                 "DOTUSD": .26, "DOGEUSD": .13, "DASHUSD": .021}            # % del precio
COSTE_CLASE = {"indices": .00010, "metales": .00020, "energia": .00040, "agricolas": .00100, "acciones": .00050}

def coste(sym, clase):
    if sym in COND:                                    # real: comision de la clase + medio spread observado
        return COMISION_CLASE[clase] + COND[sym]["spread_pct"] / 200.0
    if clase == "cripto":
        return (0.065 + SPREAD_CRIPTO.get(sym, 0.15) / 2) / 100
    return COSTE_CLASE[clase]

CLASE = {s: c for c, d in UNIVERSO.items() for s in d}

# ---- condiciones REALES de la demo de FTMO (salida de ExportarSimbolos.mq5) ----
COMISION_CLASE = {"cripto": 0.00065, "acciones": 0.0002, "indices": 0.0, "metales": 0.0, "energia": 0.0, "agricolas": 0.0}
SPREAD_MAX_PCT = 1.5      # un activo con spread > 1,5% (p.ej. NEO, SPCX, gas natural) se saca del universo
COND = {}                 # simbolo -> dict(spread_pct, swap_anual)

def _clave(s):
    return s[:-5] if s.endswith(".cash") else (s[:-2] if s.endswith(".c") else s)

def cargar_condiciones(path=None):
    path = path or _hallar("ftmo_simbolos.csv", os.path.join(AQUI, "..", "data", "ftmo_simbolos.csv"), os.path.join(AQUI, "data", "ftmo_simbolos.csv"))
    if not os.path.exists(path):
        print(f"[AVISO] no encuentro {path}: costes y swaps = SUPUESTOS (swap {SWAP_ANUAL:.1%} para todo)"); return
    d = pd.read_csv(path, sep=";", decimal=".")
    for _, r in d.iterrows():
        k = _clave(str(r["symbol"]))
        if k not in CLASE: continue
        try: sp = float(r["spread_pct"])
        except Exception: continue
        try: sw = float(r["swap_long_pct_anual"])
        except Exception: sw = np.nan
        # solo cuenta el COSTE: un swap positivo (credito) en la foto del dia se ignora (suele ser ajuste de dividendo/roll)
        COND[k] = dict(spread_pct=sp, swap_anual=(max(0.0, -sw) / 100.0 if np.isfinite(sw) else SWAP_ANUAL))
    print(f"Condiciones reales cargadas para {len(COND)} simbolos")

SWAP_CRIPTO = 0.09        # swap anual de cripto a usar. La demo ESTANDAR muestra 30%; Rick dijo 6-9% (cuenta Swing, sin verificar).
                          # None = usar el de la demo. Se prueba tambien 20% y 30% al final como sensibilidad.
def swap_de(sym):
    if CLASE.get(sym) == "cripto" and SWAP_CRIPTO is not None:
        return SWAP_CRIPTO
    return COND[sym]["swap_anual"] if sym in COND else SWAP_ANUAL

# conjuntos de universo que se comparan (decididos antes de ver resultados)
def conjunto(nombre):
    base = list(UNIVERSO["indices"]) + list(UNIVERSO["metales"]) + list(UNIVERSO["energia"])
    if nombre == "macro":            return base
    if nombre == "+agricolas_sin_cripto": return base + list(UNIVERSO["agricolas"])
    if nombre == "macro+cripto10":   return base + CRIPTO_MAYORES
    if nombre == "+agricolas":       return base + CRIPTO_MAYORES + list(UNIVERSO["agricolas"])
    if nombre == "+acciones":        return base + CRIPTO_MAYORES + list(UNIVERSO["agricolas"]) + list(UNIVERSO["acciones"])
    if nombre == "todo":             return [s for d in UNIVERSO.values() for s in d]
    if nombre == "replica9":         return ["BTCUSD", "ETHUSD", "SOLUSD", "US100", "US500", "XAUUSD", "NVDA", "AAPL"]
    if nombre == "replica10":        return ["BTCUSD", "ETHUSD", "SOLUSD", "US100", "US100x3", "US500", "XAUUSD", "NVDA", "AAPL"]
    raise ValueError(nombre)

# ============================================================================ DATOS
def descargar():
    import yfinance as yf
    todos = {s: t for d in UNIVERSO.values() for s, t in d.items()}
    px = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else pd.DataFrame()
    falta = [s for s in todos if s not in px.columns]
    if falta:
        print(f"Descargando {len(falta)} series de Yahoo...")
        tick = {todos[s]: s for s in falta}
        lista = list(tick)
        for i in range(0, len(lista), 40):
            d = yf.download(lista[i:i + 40], start=START, auto_adjust=True, progress=False, group_by="column")["Close"]
            if isinstance(d, pd.Series): d = d.to_frame(lista[i])
            d = d.rename(columns=tick)
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
            d = d[~d.index.duplicated()]
            px = d if px.empty else px.join(d, how="outer")
        pickle.dump(px, open(CACHE, "wb"))
    px = px[[c for c in px.columns if px[c].notna().sum() > MIN_OBS]]
    sin = [s for s in todos if s not in px.columns]
    if sin: print("Sin datos o historia corta (se omiten):", sin)
    return px

def anadir_x3(px):
    """US100x3: posicion US100 con 3x de nocional (sustituye a TQQQ, que FTMO no ofrece). Costes y swap x3."""
    if "US100" not in px.columns: return px
    r = px["US100"].pct_change()
    x3 = (1 + 3 * r.fillna(0)).cumprod() * 100
    x3[px["US100"].isna()] = np.nan
    px = px.copy(); px["US100x3"] = x3
    CLASE["US100x3"] = "indices"
    base = COND.get("US100", dict(spread_pct=0.006, swap_anual=0.06))
    COND["US100x3"] = dict(spread_pct=3 * base["spread_pct"], swap_anual=3 * base["swap_anual"])
    return px

def sinteticos(seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, "2026-11-10", freq="D")
    out = {}
    for c, d in UNIVERSO.items():
        vol = {"cripto": .045, "acciones": .02, "indices": .011, "metales": .014, "energia": .022, "agricolas": .018}[c]
        f = rng.normal(0, 1, len(idx)).cumsum() * 0.0006            # deriva lenta con tendencia
        for s in d:
            r = rng.normal(0.0002, vol, len(idx)) + np.diff(np.r_[0, f]) * 0.5 + rng.normal(0, 1) * 0.0
            x = pd.Series(100 * np.exp(np.cumsum(r)), idx)
            if c != "cripto": x[x.index.dayofweek >= 5] = np.nan
            if s in ("SOLUSD", "AVAUSD", "PLTR", "ARM", "SNOW", "ICPUSD"): x[:"2020-06-01"] = np.nan
            out[s] = x
    return pd.DataFrame(out)

# ============================================================================ SENALES POR ACTIVO
EWMAC = [(16, 64, 4.10), (32, 128, 2.79), (64, 256, 1.91)]   # scalars de Carver; FDM 1,13

def _propio(px, fn):
    """fn aplicada a cada serie en SUS dias de cotizacion; luego al calendario de 7 dias."""
    out = {}
    for c in px.columns:
        x = px[c].dropna()
        out[c] = fn(x).reindex(px.index).ffill(limit=5)
    return pd.DataFrame(out)

def rsi_wilder(x, n=RSI_N):
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

CARVER_VOL = 0.25        # vol objetivo por instrumento en la cartera Carver pura (antes del tope de exposicion)

def fc_ewmac(x):
    """Prevision Carver combinada (media ~10 en valor absoluto, tope +-20). NaN en el calentamiento."""
    sig = x.pct_change().ewm(span=35, min_periods=20).std() * x       # vol en unidades de precio
    f = sum(((x.ewm(span=a, adjust=False).mean() - x.ewm(span=b, adjust=False).mean()) / sig * s).clip(-20, 20)
            for a, b, s in EWMAC) / len(EWMAC) * 1.13
    f = f.clip(-20, 20)
    f[np.arange(len(x)) < 256] = np.nan
    return f

def sleeve_ewmac(x):
    return (fc_ewmac(x) > 0).astype(float)

def sleeve_sma200(x):
    p = (x > x.rolling(200).mean()).astype(float); p[np.arange(len(x)) < 200] = 0.0
    return p

def vol_anual(x):
    f = np.sqrt(365) if CLASE.get(x.name) == "cripto" else np.sqrt(252)
    return x.pct_change().ewm(span=60, min_periods=30).std() * f

def mom_score(x, con_3m=True):
    """Momentum 12-1: precio de hace 1 mes / precio de hace 12 meses - 1. Con con_3m, solo cuenta si el momentum de 3 meses
    medido sobre la ventana que queda tras saltar el ultimo mes (x[-21]/x[-84]) tambien es positivo. Cripto: barras de 7 dias."""
    k = 365 / 252 if CLASE.get(x.name) == "cripto" else 1.0
    s, m3, m12 = (int(round(n * k)) for n in (21, 84, 252))
    mom12 = x.shift(s) / x.shift(m12) - 1
    if con_3m: mom12 = mom12.where(x.shift(s) / x.shift(m3) - 1 > 0)
    return mom12.where(mom12 > 0)                 # NaN = no elegible

def garch_vol(P, cesta, refit=126, min_obs=500):
    """Volatilidad PRONOSTICADA a 5 dias (diaria, en %) con GARCH(1,1) normal, sin look-ahead: los parametros se reestiman
    cada `refit` barras con datos hasta ese dia (ventana creciente) y entre reestimaciones se actualiza solo la recursion.
    Antes de la primera estimacion no hay pronostico (NaN). Devuelve DataFrame sobre el calendario."""
    from arch import arch_model
    out = {}
    for c in cesta:
        x = P["pxc"][c].dropna(); r = (x.pct_change().dropna() * 100).values; n = len(r)
        fv = np.full(n, np.nan)
        if n > min_obs + 10:
            s2 = np.var(r[:100]); par = None; last = -10 ** 9
            for t in range(n):
                if t >= min_obs and t - last >= refit:
                    try:
                        res = arch_model(r[:t], mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off", show_warning=False)
                        par = (res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]); last = t
                    except Exception:
                        pass
                if par is None:
                    s2 = 0.94 * s2 + 0.06 * r[t] ** 2                          # EWMA mientras no hay GARCH
                    continue
                om, al, be = par
                s2 = om + al * r[t] ** 2 + be * s2                              # varianza de t+1, conocida en t
                rho = al + be
                V = om / (1 - rho) if rho < 0.9999 else s2
                avg = V + (s2 - V) * (1 - rho ** 5) / (5 * (1 - rho)) if rho < 0.9999 else s2
                fv[t] = np.sqrt(max(avg, 1e-12))
        out[c] = pd.Series(fv, index=x.index[1:]).reindex(P["cal"]).ffill(limit=5)
    return pd.DataFrame(out)

def garch_bloqueo(FV, q=0.90):
    """True = activo bloqueado (no elegible) ese dia: la volatilidad pronosticada esta por encima del percentil q de SU PROPIA
    historia hasta ayer. Decide con informacion hasta el cierre anterior (shift 1)."""
    thr = FV.expanding(250).quantile(q).shift(1)
    return ((FV > thr).fillna(False)).shift(1).fillna(False).astype(bool)

def preparar(px):
    cal = pd.date_range(px.index[0], px.index[-1], freq="D")
    pxc = px.reindex(cal)
    nobs = pxc.notna().cumsum()
    P = dict(
        cal=cal, syms=list(px.columns),
        ret=pxc.ffill(limit=5).pct_change(fill_method=None).fillna(0.0),
        rsi=_propio(pxc, rsi_wilder),
        pos={"ewmac": _propio(pxc, sleeve_ewmac), "sma200": _propio(pxc, sleeve_sma200)},
        fc=_propio(pxc, fc_ewmac),
        vol=_propio(pxc, vol_anual),
        ok=(nobs >= MIN_OBS) & pxc.ffill(limit=5).notna(),
        pxc=pxc,
    )
    wd = cal.dayofweek < 5                                             # para correlaciones: solo L-V
    P["ret_wd"] = P["ret"].values[wd]
    P["wd_pos"] = np.cumsum(wd) - 1                                    # fila L-V correspondiente a cada dia
    return P

# ============================================================================ MOTOR DE ROTACION
def rotar(P, syms, estructura="plana", top=3, peso="eq", sleeve="ewmac", cripto_max=None, bloqueo=None, hist=0, rsi=None, thr=None, mercado=None):
    """Devuelve (serie diaria neta a exposicion completa, pesos). Pesos fijados al cierre de d,
    se cobran el dia d+1. Ranking con RSI de d-1 (lag 1); sleeve con la posicion de d."""
    cols = [P["syms"].index(s) for s in syms if s in P["syms"]]
    names = [P["syms"][i] for i in cols]
    R = P["ret"].values[:, cols]
    RSI = (P["rsi"] if rsi is None else rsi).values[:, cols]
    POS = P["pos"][sleeve].values[:, cols]
    VOL = P["vol"].values[:, cols]
    OK = P["ok"].values[:, cols]
    FC = np.nan_to_num(P["fc"].values[:, cols])
    RWD = P["ret_wd"][:, cols]
    clases = np.array([CLASE[s] for s in names])
    costes = np.array([coste(s, CLASE[s]) for s in names])
    swaps = np.array([swap_de(s) for s in names])
    ref = names.index(REFUGIO) if REFUGIO in names else None
    thr = RSI_THR if thr is None else thr
    MK = None if mercado is None else mercado.reindex(P["cal"]).ffill().fillna(False).values.astype(bool)
    T, A = R.shape
    W = np.zeros((T, A))
    BLK = None
    if bloqueo is not None:      # decisiones de peso bloqueadas (dias previos a un gap de resultados): el activo no es elegible ese dia
        BLK = np.zeros((T, A), bool)
        for j, nm in enumerate(names):
            if nm in bloqueo.columns: BLK[:, j] = bloqueo[nm].reindex(P["cal"]).fillna(False).values
    if estructura == "plana":
        grupos = [("todo", np.ones(A, bool), top)]
    else:   # "cestas": un bloque por clase presente, con `top` huecos cada uno
        grupos = [(c, clases == c, top) for c in dict.fromkeys(clases)]
    slots_tot = sum(g[2] for g in grupos)
    for d in range(1, T):
        if estructura == "carver":   # cartera Carver pura: sin ranking; todo activo con prevision > 0
            ok = OK[d]; f = np.clip(FC[d], 0, 20) / 10.0
            w = np.zeros(A)
            if ok.any():
                v = np.where(np.isfinite(VOL[d]) & (VOL[d] > 0), VOL[d], np.inf)
                presentes = list(dict.fromkeys(clases[ok]))
                for c in presentes:                       # mismo peso por clase, mismo peso dentro de la clase
                    m = ok & (clases == c)
                    w[m] = (1.0 / len(presentes) / m.sum()) * f[m] * (CARVER_VOL / v[m])
                if w.sum() > K_MAX: w *= K_MAX / w.sum()
            W[d] = w
            continue
        rs, ps, ok = RSI[d - 1], POS[d], OK[d]
        eleg = ok & (ps > 0) & (rs > thr) & ~np.isnan(rs)
        if BLK is not None: eleg &= ~BLK[d]
        if MK is not None and not MK[d - 1]: eleg &= False     # filtro de mercado (QQQ sobre su media de 200), con lag 1
        if not eleg.any() and ref is None: continue
        w_end = P["wd_pos"][d]
        win = RWD[max(0, w_end - CORR_WIN):w_end + 1]
        elegidos, vacios = [], 0
        for _, mask, k in grupos:
            cand = np.where(eleg & mask)[0]
            cand = cand[np.argsort(-rs[cand])]
            if hist and len(cand):     # histeresis: el que ya se tenia ayer se mantiene mientras siga dentro de los k+hist primeros
                rk = {c: i for i, c in enumerate(cand)}
                keep = [c for c in cand if W[d - 1][c] > 0 and rk[c] < k + hist]
                cand = np.array(keep + [c for c in cand if c not in keep], dtype=int)
            n = 0
            for c in cand:
                if n == k: break
                if cripto_max is not None and clases[c] == "cripto" and sum(clases[e] == "cripto" for e in elegidos) >= cripto_max:
                    continue
                if elegidos and len(win) > 20:
                    cc = [np.corrcoef(win[:, c], win[:, s])[0, 1] for s in elegidos]
                    if np.nanmax(cc) > CORR_THR: continue
                elegidos.append(c); n += 1
            vacios += k - n
        w = np.zeros(A)
        if elegidos:
            el = np.array(elegidos)
            if peso == "eq":
                w[el] = 1.0 / slots_tot
            elif peso == "carver":   # la rotacion elige; Carver dimensiona dentro del hueco
                iv = 1.0 / np.nan_to_num(VOL[d, el], nan=np.nanmedian(VOL[d, el]) if np.isfinite(VOL[d, el]).any() else 0.2)
                w[el] = iv / iv.sum() * len(el) / slots_tot * np.clip(FC[d, el] / 10.0, 0, 2)
                if w.sum() > 1: w /= w.sum()
            else:   # inversa de volatilidad, misma exposicion total que eq
                iv = 1.0 / np.nan_to_num(VOL[d, el], nan=np.nanmedian(VOL[d, el]) if np.isfinite(VOL[d, el]).any() else 0.2)
                w[el] = iv / iv.sum() * len(el) / slots_tot
        if vacios and ref is not None and ok[ref] and POS[d, ref] > 0 and w[ref] == 0:
            w[ref] = vacios / slots_tot
        W[d] = w
    held = np.vstack([np.zeros(A), W[:-1]])
    bruto = (held * R).sum(1)
    tc = (np.abs(np.diff(np.vstack([np.zeros(A), W]), axis=0)) * costes).sum(1)
    swap = (held * swaps).sum(1) / ANN
    neto = pd.Series(bruto - tc - swap, index=P["cal"])
    Wdf = pd.DataFrame(W, index=P["cal"], columns=names)
    m = neto.index >= EVAL_START
    return neto[m], Wdf[m]

# ============================================================================ METRICAS
def metricas(x):
    eq = (1 + x).cumprod(); yrs = len(x) / ANN
    return dict(sharpe=x.mean() / x.std() * np.sqrt(ANN), cagr=eq.iloc[-1] ** (1 / yrs) - 1,
                maxdd=(eq / eq.cummax() - 1).min(), vol=x.std() * np.sqrt(ANN), peor_dia=x.min())

# ============================================================================ EVALUADOR FTMO
def evaluar(r, expo, metodo, par, starts, kmax=K_MAX):
    """Simula FTMO 2-Step Swing desde cada inicio (vectorizado). Devuelve estado y duracion.
    estado: 1 pasa ambas fases, 2 rompe diario, 3 rompe maximo, 0 sin resolver."""
    T, n = len(r), len(starts)
    vd = np.maximum(pd.Series(r).ewm(span=30, min_periods=10).std().shift(1).bfill().values, 0.002)
    e = np.ones(n); fase = np.ones(n, int); est = np.zeros(n, int); dur = np.full(n, np.nan)
    hist = np.ones((n, REF_DIAS)); dias = np.zeros(n, int)
    for j in range(H_EVAL):
        idx = starts + j
        act = (est == 0) & (idx < T)
        if not act.any(): break
        ii = np.minimum(idx, T - 1)
        rr, v = r[ii], vd[ii]
        ref = hist.max(1)
        if metodo == "A":
            k = np.full(n, par)
        else:
            k = np.minimum(kmax, par / np.sqrt(ANN) / v)
            if metodo == "C":
                k = k * np.clip((e - (1 - DD_MAX)) / DD_MAX, 0.25, 1.0)
                k = np.minimum(k, np.maximum(0.01, DD_DIA - (ref - e)) / (2.5 * v))
        ret = k * rr
        low = e * (1 + FACTOR_INTRADIA * np.minimum(ret, 0))
        f_d = act & (ref - low >= DD_DIA)
        f_m = act & ~f_d & (low <= 1 - DD_MAX)
        e = np.where(act, e * (1 + ret), e)
        dias = dias + (act & (expo[ii] > 0))
        hist = np.where(act[:, None], np.c_[hist[:, 1:], e], hist)
        est[f_d], est[f_m] = 2, 3
        dur[f_d | f_m] = j + 1
        ok = act & ~f_d & ~f_m & (dias >= DIAS_MIN)
        p1 = ok & (fase == 1) & (e >= 1 + OBJ1)
        p2 = ok & (fase == 2) & (e >= 1 + OBJ2)
        est[p2] = 1; dur[p2] = j + 1
        fase[p1] = 2; e[p1] = 1.0; hist[p1] = 1.0; dias[p1] = 0
    return est, dur

def resumen_eval(serie, expo, kmax=K_MAX):
    r = serie.values; T = len(r)
    starts = np.arange(0, T - MIN_FUTURO, PASO)
    is_mask = serie.index[starts] < OOS_START
    out = {}
    for m, p in SIZINGS:
        est, dur = evaluar(r, expo, m, p, starts, kmax)
        for lab, msk in (("IS", is_mask), ("OOS", ~is_mask)):
            e_, d_ = est[msk], dur[msk]
            out[f"{m}{p}_pasa_{lab}"] = (e_ == 1).mean()
            out[f"{m}{p}_diario_{lab}"] = (e_ == 2).mean()
            out[f"{m}{p}_dias_{lab}"] = np.nanmedian(d_[e_ == 1]) if (e_ == 1).any() else np.nan
    return out

# ============================================================================ SLEEVES DEL NOTEBOOK (--sleeves-csv)
ALIAS_SLEEVE = {"GLD": "XAUUSD", "QQQ": "US100", "SPY": "US500", "TQQQ": "US100x3", "BTC": "BTCUSD", "ETH": "ETHUSD", "SOL": "SOLUSD",
                "BZF": "UKOIL", "BZ=F": "UKOIL", "N225": "JP225", "_N225": "JP225", "^N225": "JP225"}

FIN_SLEEVE = {}      # ultima fecha de cada CSV de sleeve (para avisar si estan desactualizados)

def cargar_sleeves(carpeta, P, verbose=True):
    """Lee los *_daily.csv exportados por el notebook del grid. Nombre: <ACTIVO>_<lo-que-sea>.csv (ej. ADAUSD_c2_daily.csv).
    Usa la columna `position` (0/1, la posicion ya ejecutada del dia t) desplazada -1: la posicion de t+1 se decide con datos
    hasta t-1 (EXEC_DELAY>=1 en el notebook), asi que al cierre de t ya es conocida -> sin look-ahead.
    Fines de semana: se arrastra la posicion del viernes. Antes del primer dato del CSV: fuera (0)."""
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(carpeta, "*.csv"))):
        base = os.path.basename(f)
        act = base.split("_")[0]
        act_u = act.upper()
        if act_u.endswith("-USD"): act_u = act_u[:-4] + "USD"      # BTC-USD -> BTCUSD
        sym = ALIAS_SLEEVE.get(act_u, act_u if act_u in P["syms"] else act)
        if sym not in P["syms"]:
            print(f"  [sleeves] {base}: '{act}' no esta en el universo FTMO/descargado -> se ignora"); continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        col = "position" if "position" in d.columns else "signal_next"
        sig = (d[col] > 0).astype(float).shift(-1)
        sig.iloc[-1] = float(d[col].iloc[-1] > 0)               # ultimo dia: mantiene la posicion
        sig = sig.reindex(P["cal"]).ffill().fillna(0.0)         # calendario 7 dias; antes del CSV = 0
        out[sym] = sig; FIN_SLEEVE[sym] = d.index[-1]
        if verbose:
            print(f"  [sleeves] {base} -> {sym}: {d.index[0].date()}..{d.index[-1].date()}, en largo {sig[sig.index >= d.index[0]].mean():.0%} del tiempo"
                  + (f" (alias de {act})" if sym != act else ""))
    return out

def modo_sleeves_csv(P, px, carpeta):
    sl = cargar_sleeves(carpeta, P)
    if not sl: print("No se ha cargado ningun sleeve. Revisa la carpeta y los nombres de fichero."); return
    base = [s for s in conjunto("replica9") if s in P["syms"]]
    nuevos = [s for s in sl if s not in base]
    P["pos"]["csv"] = P["pos"]["ewmac"].copy()
    for sym, sig in sl.items(): P["pos"]["csv"][sym] = sig.values
    print(f"\nCesta base: {base}\nActivos nuevos con sleeve propio: {nuevos}\nSleeve del notebook tambien en la base: {[s for s in sl if s in base]}\n")
    variantes = [("A) base 8, sleeve generico", base, "ewmac"),
                 ("B) base + nuevos, sleeve generico", base + nuevos, "ewmac"),
                 ("C) base + nuevos, sleeve del notebook donde hay CSV", base + nuevos, "csv"),
                 ("D) SOLO base 8, sleeve del notebook", base, "csv")]
    for n in nuevos:
        variantes.append((f"E) base + {n} (notebook)", base + [n], "csv"))
    # --ref AMD,TSLA : candidata actual; cada uno de los demas nuevos se anade SOBRE ella (G)
    if "--ref" in sys.argv:
        ref = [r.strip() for r in sys.argv[sys.argv.index("--ref") + 1].split(",") if r.strip() in sl]
        variantes.append((f"R) REFERENCIA base + {'+'.join(ref)}", base + ref, "csv"))
        for n in [n for n in nuevos if n not in ref]:
            variantes.append((f"G) referencia + {n}", base + ref + [n], "csv"))
    def _arg(nombre, defecto):
        return int(sys.argv[sys.argv.index(nombre) + 1]) if nombre in sys.argv else defecto
    TOP = _arg("--top", 3); CAP = _arg("--cap", 0) or None      # configuracion fijada: 3 posiciones, sin tope de cripto
    print(f"Configuracion fijada: {TOP} posiciones, tope de cripto {CAP or 'sin tope'}\n")
    filas = []
    # F) quitar de uno en uno el sleeve del notebook (ese activo vuelve al generico): que sleeve sostiene el resultado?
    orig = P["pos"]["csv"]
    _fset = base + ([r for r in sys.argv[sys.argv.index("--ref") + 1].split(",")] if "--ref" in sys.argv else nuevos)
    for a in [s_ for s_ in sl if s_ in _fset]:
        variantes.append((f"F) C pero {a} con sleeve generico", base + nuevos, f"csv_sin_{a}"))
        P["pos"][f"csv_sin_{a}"] = orig.copy(); P["pos"][f"csv_sin_{a}"][a] = P["pos"]["ewmac"][a].values
    for top in (TOP,):
        for nom, syms, slv in variantes:
            x, W = rotar(P, syms, "plana", top, "eq", slv, CAP)
            o = resumen_eval(x, W.sum(1).shift(1).fillna(0).values)
            m = metricas(x)
            filas.append(dict(top=top, variante=nom, sharpe=m["sharpe"], cagr=m["cagr"], maxdd=m["maxdd"], peor_dia=m["peor_dia"],
                              sharpe_IS=metricas(x[x.index < OOS_START])["sharpe"], sharpe_OOS=metricas(x[x.index >= OOS_START])["sharpe"],
                              **{f"{k}_{t}": o[f"{k}_pasa_{t}"] for k in ("C0.1", "C0.12", "C0.15") for t in ("IS", "OOS") if f"{k}_pasa_{t}" in o}))
            x.to_csv(f"ftmo_daily_sleevescsv_top{top}_{nom.split(')')[0]}_{nom.split(')')[1].strip()[:25].replace(' ','_').replace('/','')}.csv", header=["ret"])
    d = pd.DataFrame(filas); d.to_csv("ftmo_sleeves_csv.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("=== COMPARATIVA (pasa = % de evaluaciones FTMO que pasan, inicios 2018-22 = IS, 2023+ = OOS) ===")
    print(d.round(3).to_string(index=False))
    print("\nLectura: C mejor que B = el sleeve del notebook aporta. C mejor que A en IS Y OOS = merece la pena anadir el activo.\n"
          "Si C solo gana en IS, sospecha de sobreajuste del ensamble.")


# ============================================================================ MEJORAS (--mejoras)
RES_CSV = "ftmo_resultados.csv"

def cargar_resultados(syms, sint=False):
    """{simbolo: [fechas de presentacion de resultados]} de las ACCIONES de la cesta.
    Orden: CSV (ftmo_resultados.csv: columnas simbolo,fecha) -> yfinance -> (solo --synthetic) fechas falsas."""
    acc = {s: t for s, t in UNIVERSO["acciones"].items() if s in syms}
    res = {}
    if sint:
        base = pd.date_range("2016-08-01", "2026-12-31", freq="91D")
        for i, s in enumerate(acc): res[s] = list(base + pd.Timedelta(days=3 * i))
        print("  [resultados] SINTETICO: fechas falsas, solo para probar el codigo"); return res
    if os.path.exists(RES_CSV):
        d = pd.read_csv(RES_CSV, parse_dates=["fecha"])
        for s, g in d.groupby("simbolo"):
            if s in acc: res[s] = list(pd.to_datetime(g["fecha"]).dt.normalize())
    faltan = [s for s in acc if s not in res]
    if faltan:
        try:
            import yfinance as yf
            for s in faltan:
                df = yf.Ticker(acc[s]).get_earnings_dates(limit=100)
                if df is None or len(df) == 0: continue
                ix = pd.DatetimeIndex(df.index)
                ix = ix.tz_localize(None) if ix.tz is not None else ix
                res[s] = sorted(set(ix.normalize()))
        except Exception as e:
            print(f"  [resultados] yfinance no ha dado las fechas ({type(e).__name__}: {e})")
        if res:
            pd.DataFrame([(s, f) for s, fs in res.items() for f in fs], columns=["simbolo", "fecha"]).to_csv(RES_CSV, index=False)
    for s in acc:
        if s not in res: print(f"  [resultados] SIN fechas para {s}: crea {RES_CSV} (columnas simbolo,fecha; una fila por presentacion)")
        else:
            f = [x for x in res[s] if x >= pd.Timestamp(EVAL_START)]
            aviso = "  <-- HISTORIA INCOMPLETA" if (not f or min(f) > pd.Timestamp("2018-06-30")) else ""
            print(f"  [resultados] {s}: {len(f)} fechas desde {EVAL_START[:4]}, primera {min(f).date() if f else '-'}{aviso}")
    return res

def mascaras_resultados(res, cal):
    """BLK[d, activo]: decision de peso del dia d BLOQUEADA (el peso de d se cobra en d+1).
    GAP[g, activo]: dia g en que cae el hueco de resultados. Como no se sabe si el activo informa antes de la apertura (hueco el
    dia D) o despues del cierre (hueco el siguiente dia habil), se cubren los dos: g in {D, D+1 habil}."""
    BLK = pd.DataFrame(False, index=cal, columns=list(res)); GAP = BLK.copy()
    for s, fs in res.items():
        for D in fs:
            for g in {pd.Timestamp(D), pd.Timestamp(D) + pd.offsets.BDay(1)}:
                if g in GAP.index: GAP.loc[g, s] = True
                d = g - pd.Timedelta(days=1)
                if d in BLK.index: BLK.loc[d, s] = True
    return BLK, GAP

def diagnostico_gaps(x, W, P, GAP):
    R = pd.DataFrame(P["ret"].values, index=P["cal"], columns=P["syms"])
    print("\n--- RIESGO DE GAP POR RESULTADOS (acciones de la cesta, desde 2018) ---")
    for s in GAP.columns:
        r = R[s][R.index >= EVAL_START]; g = GAP[s].reindex(r.index).fillna(False)
        r_g, r_n = r[g & (r != 0)], r[~g & (r != 0)]
        p5 = r <= r[r != 0].quantile(0.05)
        print(f"  {s:6s} mov. medio en dia de resultados {r_g.abs().mean():5.1%} (normal {r_n.abs().mean():4.1%}) | peor dia de resultados {r_g.min():+6.1%} "
              f"| {(g & p5).sum() / max(p5.sum(), 1):4.0%} de sus peores dias (5%) son de resultados")
    Wp = W.shift(1).fillna(0); contrib = Wp * R.loc[W.index, W.columns]
    print("\n  10 peores dias de la cartera (a exposicion completa, sin la capa de riesgo):")
    nres = 0
    for t, v in x.nsmallest(10).items():
        c = contrib.loc[t]; top = c.idxmin()
        es = bool(top in GAP.columns and GAP.loc[t, top]); nres += es
        print(f"    {t.date()}  {v:+6.1%}   mayor perdida: {top:7s} {c[top]:+6.1%} aportado (peso {Wp.loc[t, top]:.2f})" + ("   <-- RESULTADOS" if es else ""))
    print(f"  => {nres} de los 10 peores dias son hueco de resultados")

def estudio_inicio(x, W, P, cesta):
    """Que dice el estado del mercado/estrategia el dia ANTES de empezar la evaluacion sobre el exito de la evaluacion.
    Umbral fijado con inicios IS (2018-22) y comprobado en OOS (2023+). Los inicios se solapan: pocas ventanas independientes."""
    expo = W.sum(1).shift(1).fillna(0).values; r = x.values; T = len(r)
    starts = np.arange(1, T - MIN_FUTURO, 2); fechas = x.index[starts]
    cols = [s for s in cesta if s in P["pos"]["csv"].columns]
    sl, rs = P["pos"]["csv"][cols], P["rsi"][cols]
    f = pd.DataFrame(index=x.index)
    f["sleeves_en_largo"] = (sl > 0).sum(1).reindex(x.index)
    f["activos_elegibles"] = ((sl > 0) & (rs > RSI_THR)).sum(1).reindex(x.index)
    f["momentum_60d_estrategia"] = x.rolling(60).sum()
    f["vol_30d_estrategia"] = x.ewm(span=30).std() * np.sqrt(ANN)
    f["invertido_20d"] = (pd.Series(expo, index=x.index) > 0).rolling(20).mean()
    F = f.shift(1).loc[fechas]
    out = {p: evaluar(r, expo, "C", p, starts) for p in (0.10, 0.12)}
    est12, dur12 = out[0.12]; est10 = out[0.10][0]
    pasa12, pasa10 = (est12 == 1).astype(float), (est10 == 1).astype(float)
    quema12 = np.isin(est12, [2, 3]).astype(float)
    is_m = np.asarray(fechas < OOS_START)
    print(f"\n--- CUANDO EMPEZAR LA EVALUACION ({len(starts)} inicios; ventanas independientes aprox. {len(starts) * 2 // 150}) ---")
    print(f"  Base: pasa C0.12 {pasa12.mean():.0%} (IS {pasa12[is_m].mean():.0%} / OOS {pasa12[~is_m].mean():.0%}), C0.10 {pasa10.mean():.0%}")
    for col in F.columns:
        v = F[col]; ok = v.notna().values
        if ok.sum() < 100 or v[ok].nunique() < 3: continue
        try: bins = pd.qcut(v, 3, duplicates="drop")
        except Exception: continue
        print(f"\n  {col}  (correlacion de rangos con pasar: {pd.Series(pasa12[ok]).corr(v[ok].reset_index(drop=True).rank(), method='spearman'):+.2f})")
        print(f"    {'tramo':24s} {'n':>5s} {'pasa10':>7s} {'pasa12':>7s} {'IS':>5s} {'OOS':>5s} {'quema12':>8s} {'dias12':>7s}")
        for b in bins.cat.categories:
            m = (bins == b).values
            ok12 = m & (est12 == 1)
            print(f"    {str(b):24s} {m.sum():5d} {pasa10[m].mean():7.0%} {pasa12[m].mean():7.0%} {pasa12[m & is_m].mean():5.0%} {pasa12[m & ~is_m].mean():5.0%} "
                  f"{quema12[m].mean():8.0%} {np.nanmedian(dur12[ok12]) if ok12.any() else np.nan:7.0f}")
        med = v[is_m & ok].median()
        hi_is, lo_is = pasa12[is_m & ok & (v >= med).values].mean(), pasa12[is_m & ok & (v < med).values].mean()
        dir_ = ">=" if hi_is >= lo_is else "<"
        perm = ((v >= med) if dir_ == ">=" else (v < med)).values & ok
        print(f"    REGLA (umbral {med:.3f} fijado en IS): empezar solo si {col} {dir_} umbral")
        print(f"      IS : permitido {pasa12[is_m & perm].mean():.0%} vs resto {pasa12[is_m & ok & ~perm].mean():.0%} | dias permitidos {perm[is_m].mean():.0%}")
        print(f"      OOS: permitido {pasa12[~is_m & perm].mean():.0%} vs resto {pasa12[~is_m & ok & ~perm].mean():.0%} | dias permitidos {perm[~is_m].mean():.0%}   <-- la prueba que vale")

def modo_mejoras(P, px, carpeta, top, cap, ref, sint):
    import functools
    sl = cargar_sleeves(carpeta, P)
    base = [s for s in conjunto("replica9") if s in P["syms"]]
    ref = [r for r in ref if r in sl]
    cesta = base + ref
    P["pos"]["csv"] = P["pos"]["ewmac"].copy()
    for sym, sig in sl.items(): P["pos"]["csv"][sym] = sig.values
    print(f"\nCesta: {cesta}\nConfiguracion: {top} posiciones, tope cripto {cap or 'sin tope'}, sleeves del notebook\n")
    res = cargar_resultados(cesta, sint)
    BLK = GAP = None
    if res: BLK, GAP = mascaras_resultados(res, P["cal"])
    filas, series = [], {}
    KW = {}
    def correr(nombre, familia, **kw):
        KW[nombre] = dict(kw)
        peso = kw.pop("peso", "eq")
        x, W = rotar(P, cesta, "plana", top, peso, "csv", cap, **kw)
        o = resumen_eval(x, W.sum(1).shift(1).fillna(0).values); m = metricas(x)
        filas.append(dict(variante=nombre, familia=familia, sharpe=m["sharpe"], cagr=m["cagr"], maxdd=m["maxdd"], peor_dia=m["peor_dia"],
                          rotacion=W.diff().abs().sum(1).sum() / (len(W) / ANN),
                          sh_IS=metricas(x[x.index < OOS_START])["sharpe"], sh_OOS=metricas(x[x.index >= OOS_START])["sharpe"],
                          p10_IS=o["C0.1_pasa_IS"], p10_OOS=o["C0.1_pasa_OOS"], p12_IS=o["C0.12_pasa_IS"], p12_OOS=o["C0.12_pasa_OOS"],
                          d12_IS=o["C0.12_dias_IS"], d12_OOS=o["C0.12_dias_OOS"]))
        series[nombre] = x; return x, W
    t0 = time.time()
    x0, W0 = correr("REF", "ref")
    if GAP is not None: diagnostico_gaps(x0, W0, P, GAP)
    if BLK is not None: correr("sin gap resultados", "resultados", bloqueo=BLK)
    correr("pesos 1/vol", "peso", peso="invvol")
    for h in (1, 2, 3): correr(f"histeresis +{h}", "hist", hist=h)
    for n in (30, 45, 90, 120):
        rsi_n = _propio(P["pxc"], functools.partial(rsi_wilder, n=n)); correr(f"RSI({n})", "rsi", rsi=rsi_n)
    # --- ranking 12-1 (en lugar de RSI(60); sigue exigiendo sleeve en largo) y filtro de mercado
    sc12 = _propio(P["pxc"], lambda x: mom_score(x, con_3m=False)); sc123 = _propio(P["pxc"], lambda x: mom_score(x, con_3m=True))
    correr("12-1", "rsi", rsi=sc12, thr=0.0)
    correr("12-1 + 3m>0", "rsi", rsi=sc123, thr=0.0)
    mq = None
    if "US100" in P["pxc"].columns:      # Nasdaq 100 (US100 en FTMO) por encima de su media de 200
        q_ = P["pxc"]["US100"].dropna(); mq = (q_ > q_.rolling(200).mean())
        correr("RSI(60) + US100>MA200", "mercado", mercado=mq)
        correr("12-1 + 3m>0 + US100>MA200", "rsi_mercado", rsi=sc123, thr=0.0, mercado=mq)
    # --- filtro de volatilidad GARCH(1,1): fuera el activo cuya volatilidad pronosticada a 5 dias esta en el decil/quintil alto de su historia
    try:
        import arch  # noqa: F401
        t1 = time.time(); FV = garch_vol(P, cesta); print(f"GARCH ajustado en {time.time() - t1:.0f}s")
        for q in (0.90, 0.80):
            GB = garch_bloqueo(FV, q); m_ = GB.loc[GB.index >= EVAL_START]
            print(f"  GARCH p{int(q * 100)}: bloquea {m_.values.mean():.1%} de los dias-activo desde {EVAL_START}")
            correr(f"GARCH p{int(q * 100)}", "garch", bloqueo=GB)
    except ImportError:
        print("\n[aviso] falta la libreria 'arch' (pip install arch): se omiten las variantes GARCH")
    g = pd.DataFrame(filas); ref0 = g.iloc[0]
    for c in ("sharpe", "p10_IS", "p10_OOS", "p12_IS", "p12_OOS", "peor_dia"): g["d_" + c] = g[c] - ref0[c]
    def veredicto(r):
        if r.variante == "REF": return "-"
        ds = [r.d_p10_IS, r.d_p10_OOS, r.d_p12_IS, r.d_p12_OOS]
        if min(ds) <= -0.03 or r.d_sharpe < -0.10: return "EMPEORA"
        if max(ds) >= 0.01 and min(ds) >= -0.01 and r.d_sharpe >= -0.05: return "MEJORA"
        return "neutro"
    g["veredicto"] = g.apply(veredicto, axis=1)
    # combinacion: lo mejor de cada familia que MEJORA
    mej = g[g.veredicto == "MEJORA"]
    if len(mej):
        elegidas = mej.sort_values("d_p12_IS", ascending=False).drop_duplicates("familia")
        def _or(a, b):
            cols = list(dict.fromkeys(list(a.columns) + list(b.columns)))
            return a.reindex(index=P["cal"], columns=cols, fill_value=False).fillna(False).astype(bool) | b.reindex(index=P["cal"], columns=cols, fill_value=False).fillna(False).astype(bool)
        kw = {}
        for _, r in elegidas.iterrows():
            for k_, v_ in KW[r.variante].items():
                kw[k_] = _or(kw[k_], v_) if k_ == "bloqueo" and k_ in kw else v_
        correr("COMBINADA (" + ", ".join(elegidas.variante) + ")", "combo", **kw)
        g = pd.DataFrame(filas)
        for c in ("sharpe", "p10_IS", "p10_OOS", "p12_IS", "p12_OOS", "peor_dia"): g["d_" + c] = g[c] - ref0[c]
        g["veredicto"] = g.apply(veredicto, axis=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"\n=========== PALANCAS (cada una por separado frente a REF; {time.time() - t0:.0f}s) ===========")
    cols = ["variante", "sharpe", "cagr", "maxdd", "peor_dia", "rotacion", "sh_IS", "sh_OOS", "p10_IS", "p10_OOS", "p12_IS", "p12_OOS", "d12_IS", "d12_OOS", "veredicto"]
    print(g[cols].round(3).to_string(index=False))
    print("\nLectura: p10/p12 = % de evaluaciones que pasan con C0.10/C0.12, inicios 2018-22 (IS) y 2023+ (OOS); d12 = dias (mediana) para pasar.\n"
          "MEJORA = sube el % que pasa sin bajar en ningun tramo. Elegir la mejor de cada familia sobre estos mismos datos es una seleccion: la prueba final es el Monte Carlo.")
    etq = ("_" + sys.argv[sys.argv.index("--tag") + 1]) if "--tag" in sys.argv else ""     # --tag cap1 -> ftmo_mejoras_cap1.csv
    g.to_csv(f"ftmo_mejoras{etq}.csv", index=False)
    for n_, x_ in series.items():
        tag = "".join(ch if ch.isalnum() else "_" for ch in n_)[:40]; x_.to_csv(f"ftmo_daily_mejoras{etq}_{tag}.csv", header=["ret"])
    print(f"Escritos ftmo_mejoras{etq}.csv y ftmo_daily_mejoras{etq}_<variante>.csv (para el Monte Carlo)")
    estudio_inicio(x0, W0, P, cesta)


SLEEVE_DE = {"BTCUSD": "BTC", "ETHUSD": "ETH", "SOLUSD": "SOL", "US100": "QQQ", "US500": "SPY", "XAUUSD": "GLD",
             "AAPL": "AAPL", "NVDA": "NVDA", "AMD": "AMD", "JNJ": "JNJ"}      # simbolo FTMO -> nombre del sleeve en sleeves_motor.SLEEVES

def sleeves_vivos(P, syms, sl_csv):
    """Calcula los sleeves desde los precios (sleeves_motor.py, tus reglas del grid) en lugar de leer los CSV. Para cada sleeve:
    compara con tu CSV en los dias comunes (debe coincidir ~100 %), y devuelve (senales, ultima_barra). Los que no tienen regla
    en sleeves_motor.SLEEVES (p. ej. JNJ hasta que pegues su linea) se quedan con el CSV."""
    import sleeves_motor as sm
    out, fin = {}, {}
    print("\n--- Sleeves calculados en vivo desde los precios (descarga de Yahoo) ---")
    for sym in syms:
        nom = SLEEVE_DE.get(sym)
        if nom is None or nom not in sm.SLEEVES:
            print(f"  {sym:7s} sin regla en sleeves_motor.py -> se usa el CSV" + (" (puede estar desactualizado)" if sym in sl_csv else " (NO HAY CSV: queda fuera)"))
            continue
        try:
            e, ultima = sm.estado_sleeve(nom)
        except Exception as ex:
            print(f"  {sym:7s} fallo al descargar/calcular ({type(ex).__name__}) -> se usa el CSV"); continue
        sig = e.reindex(P["cal"]).ffill().fillna(0.0)
        out[sym] = sig; fin[sym] = ultima
        msg = ""
        if sym in sl_csv:
            ref = sl_csv[sym]; com = e.index.intersection(ref.index)
            com = com[(com >= pd.Timestamp("2019-07-01")) & (com <= ref.index[-1] - pd.Timedelta(days=3))]
            if len(com): 
                ok = float(np.mean(e.loc[com].values == ref.loc[com].values))
                msg = f" | coincide {ok:.2%} con tu CSV en {len(com)} dias" + ("" if ok >= 0.995 else "  <-- REVISA: no reproduce tu CSV")
        print(f"  {sym:7s} {nom:5s} ultima barra {ultima.date()} | hoy: {'EN LARGO' if sig.iloc[-1] > 0 else 'fuera'}{msg}")
    return out, fin

# ============================================================================ SENAL DIARIA (forward test)
def modo_senal(P, px, carpeta, top, cap, ref, tau, equity, base, ref3):
    """Pesos objetivo para MANANA con la cartera de referencia (base 8 + ref, notebook sleeves, RSI(60), pesos iguales),
    tamano k de la capa de riesgo (objetivo de vol tau + freno por drawdown + colchon diario, igual que el Monte Carlo) y
    nominal por activo. Guarda senal_log.csv y compara con las senales de dias anteriores (reproducibilidad)."""
    sl = cargar_sleeves(carpeta, P, verbose=False)
    basec = [s_ for s_ in conjunto("replica9") if s_ in P["syms"]]
    ref = [r_ for r_ in ref if r_ in sl or r_ in SLEEVE_DE]
    cesta = basec + ref
    FIN_SLEEVE.clear()
    for f_ in __import__("glob").glob(os.path.join(carpeta, "*.csv")):       # ultima fecha de cada CSV (para el aviso de desactualizado)
        a_ = os.path.basename(f_).split("_")[0].upper(); a_ = a_[:-4] + "USD" if a_.endswith("-USD") else a_
        sy_ = ALIAS_SLEEVE.get(a_, a_)
        if sy_ in sl: FIN_SLEEVE[sy_] = pd.read_csv(f_, parse_dates=["date"])["date"].iloc[-1]
    if "--solo-csv" not in sys.argv:
        try:
            vivos, fin_v = sleeves_vivos(P, cesta, sl)
            for sy_, sg_ in vivos.items(): sl[sy_] = sg_; FIN_SLEEVE[sy_] = fin_v[sy_]
        except Exception as ex:
            print(f"\n[AVISO] no pude calcular los sleeves en vivo ({type(ex).__name__}: {ex}); se usan los CSV de {carpeta}/")
    P["pos"]["csv"] = P["pos"]["ewmac"].copy()
    for sym, sig in sl.items(): P["pos"]["csv"][sym] = sig.values
    for sym in cesta:
        if sym not in sl:
            P["pos"]["csv"][sym] = 0.0; print(f"  [AVISO] {sym}: no hay sleeve (ni regla ni CSV) -> queda FUERA de la cartera")
    x, W = rotar(P, cesta, "plana", top, "eq", "csv", cap)
    ti = len(P["cal"]) - 1; T = P["cal"][-1]
    w, wp = W.iloc[-1], W.iloc[-2]
    rs = P["rsi"][cesta].iloc[ti - 1]; ps = P["pos"]["csv"][cesta].iloc[ti]; ok = P["ok"][cesta].iloc[ti]
    eleg = ok & (ps > 0) & (rs > RSI_THR) & rs.notna()
    print(f"\n================ SENAL PARA MANANA  (datos hasta el cierre del {T.date()}) ================")
    print(f"Cartera: {cesta} | {top} posiciones, tope cripto {cap or 'sin tope'} | RSI({RSI_N}) de AYER, umbral {RSI_THR}, pesos iguales 1/{top}\n")
    for sym in cesta:
        fin = FIN_SLEEVE.get(sym)
        if fin is not None and (T - fin).days > 3:
            print(f"  [AVISO] sleeve {sym}: su CSV termina el {fin.date()} ({(T - fin).days} dias antes del ultimo dato). "
                  f"Vuelve a ejecutar el notebook para refrescarlo: se esta usando su ultima posicion conocida.")
    filas = []
    for sym in rs.sort_values(ascending=False, na_position="last").index:
        if w[sym] > 0 and wp[sym] > 0: acc = "MANTENER"
        elif w[sym] > 0: acc = "ABRIR"
        elif wp[sym] > 0: acc = "CERRAR"
        else: acc = ""
        motivo = "" if w[sym] > 0 or not eleg[sym] else "elegible, fuera por ranking/correlacion/tope cripto"
        if not eleg[sym]:
            motivo = "sleeve fuera" if ps[sym] <= 0 else ("RSI<=50" if (rs[sym] <= RSI_THR) else "sin datos")
            if w[sym] == 0 and ps[sym] > 0 and rs[sym] > RSI_THR: motivo = "sin datos suficientes"
        filas.append(dict(activo=sym, sleeve="EN LARGO" if ps[sym] > 0 else "fuera", RSI_ayer=round(float(rs[sym]), 1) if pd.notna(rs[sym]) else np.nan,
                          elegible="si" if eleg[sym] else "no", peso_manana=round(float(w[sym]), 3), peso_ayer=round(float(wp[sym]), 3), accion=acc, nota=motivo))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(filas).to_string(index=False))
    if XAU := "XAUUSD":
        if XAU in w.index and w[XAU] > 0 and not eleg.get(XAU, False): print("\n  (XAUUSD esta como refugio: ocupa un hueco vacio de la rotacion)")
    # ---- tamano k (misma formula que el Monte Carlo) ----
    v = float(x.ewm(span=30, adjust=False).std().iloc[-1] * np.sqrt(ANN))
    k_vol = min(1.0, tau / v) if v > 0 else 0.0
    E = (equity / base) if equity else 1.0
    r3 = (ref3 / base) if ref3 else E
    freno = min(1.0, max(0.25, (E - (1 - 0.10)) / 0.10))
    kk = k_vol * freno
    hueco = E - (r3 - 0.05); peor = 2.5 * kk * v / np.sqrt(ANN) * E
    colchon = max(0.0, (hueco - 0.01) / peor) if (peor > 0 and hueco - 0.01 < peor) else 1.0
    k = kk * colchon
    print(f"\n--- Tamano (objetivo de vol {tau:.0%}) ---")
    print(f"  vol anual de la cartera a exposicion completa (EWMA 30): {v:.1%}  ->  k por volatilidad = min(1, {tau:.2f}/{v:.2f}) = {k_vol:.2f}")
    print(f"  equity {E * base:,.0f} sobre base {base:,.0f}  ->  freno por drawdown x{freno:.2f}   | colchon diario x{colchon:.2f}")
    print(f"  k FINAL = {k:.2f}  (exposicion total = suma de pesos {w.sum():.2f} x k = {w.sum() * k:.0%} de la cuenta)")
    cap_ = (equity or base)
    print("\n  Nominal objetivo por activo (en la moneda de la cuenta):")
    for sym in w.index[w > 0]:
        print(f"    {sym:8s} {w[sym] * k * cap_:>12,.0f}   ({w[sym] * k:.1%} de la cuenta)")
    if equity is None: print("  (no has pasado --equity: nominal calculado sobre la base; con --equity y --ref3 se aplican freno y colchon)")
    # ---- log y reproducibilidad ----
    fila = dict(fecha=str(T.date()), tau=tau, k_vol=round(k_vol, 4), k_final=round(k, 4), equity=equity if equity else "", **{f"w_{s_}": round(float(w[s_]), 4) for s_ in cesta})
    L = pd.read_csv("senal_log.csv") if os.path.exists("senal_log.csv") else pd.DataFrame()
    if len(L):
        pasadas = L[L.fecha != fila["fecha"]]
        difs = []
        for _, r_ in pasadas.iterrows():
            d_ = pd.Timestamp(r_.fecha)
            if d_ in W.index:
                dd = sum(abs(float(r_.get(f"w_{s_}", 0) or 0) - float(W.loc[d_, s_])) for s_ in cesta) / 2   # fraccion de la cartera que habria cambiado
                difs.append(dd)
        if difs:
            difs = np.array(difs)
            print(f"\n--- Reproducibilidad: {len(difs)} senales anteriores recalculadas hoy con los datos actuales ---")
            print(f"  coinciden (diferencia < 1% de cartera): {np.mean(difs < 0.01):.0%} | diferencia media {difs.mean():.1%} | maxima {difs.max():.1%}")
            print("  (si hay diferencias: sleeves refrescados, datos revisados o un error al ejecutar la senal; revisa los dias que no coinciden)")
    L = pd.concat([L[L.fecha != fila["fecha"]] if len(L) else L, pd.DataFrame([fila])], ignore_index=True).sort_values("fecha")
    L.to_csv("senal_log.csv", index=False)
    print(f"\nGuardado senal_log.csv ({len(L)} filas). Ahora: 1) coloca las posiciones en la demo, 2) apunta el equity real al cierre "
          f"(mañana lo pasas con --equity).")

# ============================================================================ ACTIVO A ACTIVO
def una_a_una(P, px):
    """Que activo de FTMO MEJORA la cesta de 8 si se le anade como 9o?  Sleeve generico (Carver, igual para todos).
    Un activo solo se marca 'AYUDA' si mejora Sharpe y % que pasa FTMO tanto en inicios 2018-22 como en 2023+.
    OJO: son ~90 comparaciones; es una lista de CANDIDATOS a los que disenar su ensamble, no una conclusion."""
    base = conjunto("replica9")
    def correr_u(syms):
        x, W = rotar(P, syms, "plana", 3, "eq", "ewmac")
        o = resumen_eval(x, W.sum(1).shift(1).fillna(0).values)
        return dict(sharpe_IS=metricas(x[x.index < OOS_START])["sharpe"], sharpe_OOS=metricas(x[x.index >= OOS_START])["sharpe"],
                    dd=metricas(x)["maxdd"], peor=metricas(x)["peor_dia"],
                    pasa_IS=np.mean([o[f"C{p}_pasa_IS"] for p in (0.1, 0.12)]), pasa_OOS=np.mean([o[f"C{p}_pasa_OOS"] for p in (0.1, 0.12)]))
    b = correr_u([s for s in base if s in P["syms"]])
    print(f"BASE (8 activos): Sharpe IS {b['sharpe_IS']:.2f} OOS {b['sharpe_OOS']:.2f} | pasa (C0.10/0.12 media) IS {b['pasa_IS']:.0%} OOS {b['pasa_OOS']:.0%}")
    filas = []
    cands = [c for c in P["syms"] if c not in base and c not in ("US100x3", "NATGAS")]
    for i, c in enumerate(cands):
        r = correr_u([s for s in base if s in P["syms"]] + [c])
        filas.append(dict(activo=c, clase=CLASE.get(c, "?"), desde=px[c].first_valid_index().date(), swap=f"{swap_de(c):.0%}",
                          spread=f"{COND.get(c, {}).get('spread_pct', np.nan):.2f}%",
                          dSharpe_IS=r["sharpe_IS"] - b["sharpe_IS"], dSharpe_OOS=r["sharpe_OOS"] - b["sharpe_OOS"],
                          dPasa_IS=r["pasa_IS"] - b["pasa_IS"], dPasa_OOS=r["pasa_OOS"] - b["pasa_OOS"], peor_dia=r["peor"]))
        if (i + 1) % 10 == 0: print(f"  ... {i + 1}/{len(cands)}")
    d = pd.DataFrame(filas)
    d["AYUDA"] = (d.dSharpe_IS > 0) & (d.dSharpe_OOS > 0) & (d.dPasa_IS > 0) & (d.dPasa_OOS > 0)
    d["score"] = d.dPasa_IS + d.dPasa_OOS
    d = d.sort_values("score", ascending=False)
    d.to_csv("ftmo_una_a_una.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("\n=== ACTIVOS QUE MEJORAN LA CESTA EN LOS DOS TRAMOS (candidatos a disenarles ensamble) ===")
    print(d[d.AYUDA].round(3).to_string(index=False))
    print("\n=== TOP 20 por mejora media del % que pasa (aunque no mejoren en los dos tramos) ===")
    print(d.head(20).round(3).to_string(index=False))
    print("\n=== Por clase: cuantos ayudan / cuantos probados ===")
    print(d.groupby("clase").AYUDA.agg(["sum", "count"]))

# ============================================================================ MAIN
def main(sint=False):
    t0 = time.time()
    cargar_condiciones()
    px = sinteticos() if sint else descargar()
    px = anadir_x3(px)
    if "--senal" in sys.argv:
        i = sys.argv.index("--senal")
        carpeta = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else _hallar("sleeves", os.path.join(AQUI, "..", "data", "csv_sleeves"), os.path.join(AQUI, "data", "csv_sleeves"))
        def _v(n, d, tipo=float): return tipo(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
        ref = sys.argv[sys.argv.index("--ref") + 1].split(",") if "--ref" in sys.argv else ["AMD", "JNJ"]
        modo_senal(preparar(px), px, carpeta, _v("--top", 3, int), _v("--cap", 2, int) or None, [r.strip() for r in ref],
                   _v("--tau", 0.10), _v("--equity", None), _v("--base", 100000.0), _v("--ref3", None)); return
    if "--sleeves-csv" in sys.argv:
        i = sys.argv.index("--sleeves-csv")
        carpeta = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else _hallar("sleeves", os.path.join(AQUI, "..", "data", "csv_sleeves"), os.path.join(AQUI, "data", "csv_sleeves"))
        if "--mejoras" in sys.argv:
            def _a(n, d): return int(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
            ref = sys.argv[sys.argv.index("--ref") + 1].split(",") if "--ref" in sys.argv else ["AMD", "JNJ"]
            modo_mejoras(preparar(px), px, carpeta, _a("--top", 3), _a("--cap", 0) or None, [r.strip() for r in ref], sint); return
        modo_sleeves_csv(preparar(px), px, carpeta); return
    if "--una-a-una" in sys.argv:
        una_a_una(preparar(px), px); return
    if COND:   # fuera los activos con spread prohibitivo
        malos = [s for s in px.columns if s in COND and COND[s]["spread_pct"] > SPREAD_MAX_PCT]
        if malos: print(f"Fuera por spread > {SPREAD_MAX_PCT}%: {malos}"); px = px.drop(columns=malos)
    print(f"Universo con datos: {px.shape[1]} activos | {px.index[0].date()} -> {px.index[-1].date()}")
    P = preparar(px)
    print(f"Senales listas ({time.time() - t0:.0f}s)")

    rapido = "--rapido" in sys.argv
    configs = [("replica9", "plana", 3, "eq", "ewmac", None)]   # referencia: el rotativo actual con sleeves Carver
    if rapido:   # solo las variantes de la cesta FTMO "de verdad": 8 activos, con/sin US100x3 (TQQQ), tope de cripto
        for u, top, cap in itertools.product(["replica9", "replica10"], [3, 4], [None, 1, 2]):
            if (u, top, cap) != ("replica9", 3, None):
                configs.append((u, "plana", top, "eq", "ewmac", cap))
    for u, (est, top), peso in ([] if rapido else itertools.product(["macro+cripto10", "+agricolas", "+acciones", "todo"],
                                                 [("plana", 3), ("plana", 5), ("plana", 8), ("cestas", 1), ("cestas", 2)],
                                                 ["eq", "invvol"])):
        configs.append((u, est, top, peso, "ewmac", None))
    if not rapido: configs.append(("+agricolas", "cestas", 1, "invvol", "sma200", None))   # control: sleeve SMA200 en vez de Carver
    # Carver como motor de TAMANO (no solo como interruptor): dentro de la rotacion y como cartera pura
    for u in ([] if rapido else ["+agricolas", "+acciones", "todo"]):
        configs.append((u, "cestas", 1, "carver", "ewmac", None))
        configs.append((u, "plana", 3, "carver", "ewmac", None))
    for u in ([] if rapido else ["macro", "+agricolas_sin_cripto", "macro+cripto10", "todo"]):
        configs.append((u, "carver", 0, "-", "ewmac", None))

    filas, series = [], {}
    for i, (u, est, top, peso, sl, cap) in enumerate(configs):
        syms = [s for s in conjunto(u) if s in P["syms"]]
        x, W = rotar(P, syms, est, top, peso, sl, cap)
        nombre = f"{u}|{est}{top}|{peso}|{sl}" + (f"|cap{cap}" if cap else "")
        expo = W.sum(1).shift(1).fillna(0).values
        fila = dict(config=nombre, n_activos=len(syms), **metricas(x),
                    sharpe_IS=metricas(x[x.index < OOS_START])["sharpe"], sharpe_OOS=metricas(x[x.index >= OOS_START])["sharpe"],
                    expo_media=W.sum(1).mean(), oro_medio=W.get(REFUGIO, pd.Series(0, index=W.index)).mean(),
                    **{f"peso_{c}": W[[s for s in W.columns if CLASE[s] == c]].sum(1).mean() for c in UNIVERSO})
        km = K_MAX_CARVER if (est == "carver" and u in ("macro", "+agricolas_sin_cripto")) else K_MAX
        fila["k_max"] = km
        fila.update(resumen_eval(x, expo, km))
        filas.append(fila); series[nombre] = x
        print(f"[{i + 1}/{len(configs)}] {nombre:42s} Sharpe {fila['sharpe']:.2f} DD {fila['maxdd']:.0%} "
              f"peor dia {fila['peor_dia']:.1%} | pasa C0.12 IS {fila['C0.12_pasa_IS']:.0%} OOS {fila['C0.12_pasa_OOS']:.0%} "
              f"({time.time() - t0:.0f}s)")

    g = pd.DataFrame(filas)
    pasa_IS = g[[c for c in g.columns if c.endswith("_pasa_IS")]]
    g["mejor_sizing_IS"] = pasa_IS.idxmax(axis=1).str.replace("_pasa_IS", "")
    g["pasa_IS_max"] = pasa_IS.max(axis=1)
    g["pasa_OOS_mismo"] = [g.loc[i, f"{s}_pasa_OOS"] for i, s in zip(g.index, g["mejor_sizing_IS"])]
    g["dias_IS_mismo"] = [g.loc[i, f"{s}_dias_IS"] for i, s in zip(g.index, g["mejor_sizing_IS"])]
    g["dias_OOS_mismo"] = [g.loc[i, f"{s}_dias_OOS"] for i, s in zip(g.index, g["mejor_sizing_IS"])]
    g["diario_IS_mismo"] = [g.loc[i, f"{s}_diario_IS"] for i, s in zip(g.index, g["mejor_sizing_IS"])]
    g.to_csv("ftmo_grid.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["config", "sharpe", "cagr", "maxdd", "peor_dia", "sharpe_IS", "sharpe_OOS", "mejor_sizing_IS",
            "pasa_IS_max", "diario_IS_mismo", "dias_IS_mismo", "pasa_OOS_mismo", "dias_OOS_mismo"]
    print("\n================ RANKING POR % QUE PASA FTMO (inicios 2018-2022) ================")
    print(g.sort_values(["pasa_IS_max", "dias_IS_mismo"], ascending=[False, True])[cols].head(15).round(2).to_string(index=False))
    print("\n--- Referencia: replica del rotativo actual (8 activos FTMO, sleeves Carver) ---")
    print(g[g.config.str.startswith("replica9")][cols].round(2).to_string(index=False))
    if os.path.exists("rotativo_daily_9sleeves.csv"):   # tu rotativo actual, la vara de medir
        s9 = pd.read_csv("rotativo_daily_9sleeves.csv", parse_dates=["date"]).set_index("date")["ret"]
        s9 = s9 - SWAP_ANUAL / ANN * 0.9                    # aprox. swap (exposicion media ~90%)
        o9 = resumen_eval(s9, np.ones(len(s9))); m9 = metricas(s9)
        print(f"\n--- VARA DE MEDIR: tu rotativo 9 sleeves (CSV) --- Sharpe {m9['sharpe']:.2f} DD {m9['maxdd']:.0%} peor dia {m9['peor_dia']:.1%}")
        print("   " + " | ".join(f"{m}{p}: IS {o9[f'{m}{p}_pasa_IS']:.0%} OOS {o9[f'{m}{p}_pasa_OOS']:.0%}" for m, p in SIZINGS))
    print("\n--- Media por estructura (todas las configs, sizing C 0,12) ---")
    g["estructura"] = g.config.str.split("|").str[1]; g["universo"] = g.config.str.split("|").str[0]
    if not rapido: print(g.pivot_table(index="estructura", columns="universo", values="C0.12_pasa_IS").round(2))
    print("\n--- Peso medio por clase en las 5 mejores ---")
    top5 = g.sort_values(["pasa_IS_max", "dias_IS_mismo"], ascending=[False, True]).head(5)
    print(top5[["config"] + [f"peso_{c}" for c in UNIVERSO] + ["oro_medio", "expo_media"]].round(2).to_string(index=False))

    for k, nombre in enumerate(list(top5.config) + [g.config[0]]):
        tag = nombre.replace("|", "_").replace("+", "")
        s = series[nombre].rename("ret"); s.index.name = "date"
        s.to_csv(f"ftmo_daily_{tag}.csv")
        print(f"Escrito ftmo_daily_{tag}.csv  <- {nombre}")
    print("\nPor ano (top1):", {int(y): f"{(1 + v).prod() - 1:+.0%}" for y, v in series[top5.config.iloc[0]].groupby(series[top5.config.iloc[0]].index.year)})
    # ---- sensibilidad al swap de cripto: el dato mas dudoso ----
    global SWAP_CRIPTO
    base = SWAP_CRIPTO
    print("\n================ SENSIBILIDAD AL SWAP DE CRIPTO (% que pasa FTMO, inicios IS / OOS) ================")
    for nombre in [g.config[0]] + list(top5.config[:3]):
        partes = nombre.split("|"); u_, est_, peso_, sl_ = partes[:4]
        cap_ = int(partes[4][3:]) if len(partes) > 4 else None
        est_nom = "".join(ch for ch in est_ if not ch.isdigit()); top_n = int("".join(ch for ch in est_ if ch.isdigit()) or 0)
        for sc in (0.09, 0.20, 0.30):
            SWAP_CRIPTO = sc
            syms = [x for x in conjunto(u_) if x in P["syms"]]
            x, W = rotar(P, syms, est_nom, top_n, peso_ if peso_ != "-" else "eq", sl_, cap_)
            o = resumen_eval(x, W.sum(1).shift(1).fillna(0).values)
            print(f"{nombre:42s} swap cripto {sc:.0%}: " + " | ".join(
                f"{m}{p}: {o[f'{m}{p}_pasa_IS']:.0%}/{o[f'{m}{p}_pasa_OOS']:.0%}" for m, p in SIZINGS) +
                f"  (CAGR {metricas(x)['cagr']:.0%}, Sharpe {metricas(x)['sharpe']:.2f})")
    SWAP_CRIPTO = base
    print(f"Total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main("--synthetic" in sys.argv)
