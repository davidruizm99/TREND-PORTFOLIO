"""Posición/tamaño de un sleeve, igual que el notebook (celdas 4 y 10)."""
import math
import numpy as np, pandas as pd
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
