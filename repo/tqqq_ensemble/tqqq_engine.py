"""Motor del sleeve TQQQ 50/50 (45 lentos congelados + 15 nuevos), copiado 1:1 del notebook TEMACD-TQQQ-5050-NUEVO-REPRODUCCION.ipynb (celda 2)."""

import json, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd


def ema_np(x, n):
    return pd.Series(x).ewm(span=int(n), adjust=False, min_periods=int(n)).mean().to_numpy()


def cross_up(a, b):
    out = np.zeros(len(a), dtype=bool)
    with np.errstate(invalid='ignore'):
        out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_dn(a, b):
    out = np.zeros(len(a), dtype=bool)
    with np.errstate(invalid='ignore'):
        out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def latch(evt, reset):
    """True mientras haya un evento 'evt' mas reciente que el ultimo 'reset' (salida pendiente de confirmacion del MACD)."""
    ix = np.arange(len(evt))
    return np.maximum.accumulate(np.where(evt, ix, -1)) > np.maximum.accumulate(np.where(reset, ix, -1))


def download_ohlc(tk, start):
    import yfinance as yf
    df = yf.download(tk, start=start, interval='1d', auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise SystemExit(f'No hay datos para {tk}')

    def col(field):
        if isinstance(df.columns, pd.MultiIndex):
            s = df[(field, tk)] if (field, tk) in df.columns else df[field].iloc[:, 0]
        else:
            s = df[field]
        return s.astype(float).squeeze()

    return pd.DataFrame({'high': col('High'), 'low': col('Low'), 'close': col('Close')}).dropna()


class Engine:
    """Mismos calculos que el notebook (solo EMAs + MACD + ATR Trail; ADX/Donchian/reentrada no se usan en el congelado)."""

    def __init__(self, sig_df, macd_role, trail_atr, exit_latch=False):
        self.sig = sig_df['close'].to_numpy(float)
        self.hi = sig_df['high'].to_numpy(float)
        self.lo = sig_df['low'].to_numpy(float)
        self.macd_role, self.trail_atr, self.exit_latch = macd_role, trail_atr, exit_latch
        self._ema, self._trail, self._atr = {}, {}, {}

    def ema(self, n):
        if n not in self._ema:
            self._ema[n] = ema_np(self.sig, n)
        return self._ema[n]

    def macd_parts(self, mf, ms, mg):
        line = self.ema(mf) - self.ema(ms)
        sigl = ema_np(line, mg)
        with np.errstate(invalid='ignore'):
            return cross_up(line, sigl), cross_dn(line, sigl), line > sigl, line < sigl

    def trail_event(self, n, k):
        key = (n, k)
        if key not in self._trail:
            if 'atr' not in self._atr:
                h, l, c = pd.Series(self.hi), pd.Series(self.lo), pd.Series(self.sig)
                tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
                self._atr['atr'] = tr.ewm(alpha=1.0 / self.trail_atr, adjust=False, min_periods=self.trail_atr).mean().to_numpy()
            hh = pd.Series(self.hi).rolling(n, min_periods=n).max().to_numpy()
            line = hh - k * self._atr['atr']
            with np.errstate(invalid='ignore'):
                st = self.sig < line
            ev = np.zeros(len(st), dtype=bool)
            ev[1:] = st[1:] & ~st[:-1]
            self._trail[key] = ev
        return self._trail[key]

    def raw_signals(self, combo):
        e1, e2, e3, mf, ms, mg, an, at, dn_, dm, tn, tk = combo
        if an is not None or dn_ is not None:
            raise SystemExit('Este sistema usa ADX/Donchian: no soportado en este script.')
        a, b, c = self.ema(e1), self.ema(e2), self.ema(e3)
        u = cross_up(a, b) | cross_up(a, c) | cross_up(b, c)
        d = cross_dn(a, b) | cross_dn(a, c) | cross_dn(b, c)
        u_raw = u
        if mf is not None:
            mu, md, bull, bear = self.macd_parts(mf, ms, mg)
            if self.macd_role == 'or_events':
                u, d = u | mu, d | md
            else:
                if self.macd_role in ('entry_confirm', 'both_confirm'):
                    u = u & bull
                if self.macd_role in ('exit_confirm', 'both_confirm'):
                    d = (latch(d, u_raw) & bear) if self.exit_latch else (d & bear)
        if tn is not None:
            d = d | self.trail_event(tn, tk)
        return u, d


def position_path(ent, ex):
    """Misma logica que vectorbt from_signals (solo largos): entrada si esta fuera, salida si esta dentro,
    y si coinciden entrada y salida en la misma barra no se hace nada."""
    T = len(ent)
    pos = np.zeros(T, dtype=bool)
    cur = False
    for t in range(T):
        if ent[t] and ex[t]:
            pass
        elif ex[t]:
            cur = False
        elif ent[t]:
            cur = True
        pos[t] = cur
    return pos



