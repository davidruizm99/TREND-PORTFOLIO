"""Definición de los sleeves del portfolio: sistema (candidato #1 de cada informe del grid), calendario y vol-target."""
SLEEVES = {
 # nombre : dict(ticker de datos, calendario, sistema, vol_obj)
 "BTC":  dict(ticker="BTC-USD", cal="7d", vol=None, spec="EMA3(f=5,m=40,s=65) AND MACD(f=16,s=21,sig=16,modo=senal) AND HMA(n=250,banda=0.0)"),
 "ETH":  dict(ticker="ETH-USD", cal="7d", vol=0.35, spec="EMA3X(f=16,m=120,s=65) OR MACD(f=20,s=300,sig=9,modo=senal) OR DONCHIAN(n_in=150,n_out=10)"),
 "SOL":  dict(ticker="SOL-USD", cal="7d", vol=None, spec="MACD(f=8,s=80,sig=16,modo=senal) AND DONCHIAN(n_in=150,n_out=40) AND TSMOM(n=252)"),
 "AAPL": dict(ticker="AAPL", cal="LV", vol=None, spec="EMA3X(f=5,m=200,s=160) AND MACD(f=8,s=21,sig=9,modo=senal)"),
 "NVDA": dict(ticker="NVDA", cal="LV", vol=0.35, spec="SMA(n=200) OR MACD(f=8,s=200,sig=16,modo=senal)"),
 "AMD":  dict(ticker="AMD", cal="LV", vol=None, spec="EMA3(f=12,m=20,s=80) AND MACD(f=8,s=200,sig=5,modo=senal) AND KELTNER(n=80,m=1.0,atr_n=14)"),
 "GLD":  dict(ticker="GLD", cal="LV", vol=None, spec="EMA3X(f=16,m=120,s=100) OR KELTNER(n=20,m=2.5,atr_n=14)"),
 "DBC":  dict(ticker="DBC", cal="LV", vol=None, spec="EMA3X(f=16,m=80,s=65) AND MACD(f=40,s=300,sig=16,modo=senal)"),
 # QQQ/SPY/TQQQ: sistemas del bootcamp anterior (no son del grid)
 # QQQ y SPY: reproducen al 100 % tus CSV *_10bp (verificado). Usan la convención antigua del portfolio (se opera al cierre de la propia señal: delay=1)
 "QQQ":  dict(ticker="QQQ", cal="LV", vol=None, delay=1, spec="EMA3(f=20,m=80,s=100) OR MACD(f=8,s=26,sig=0,modo=cero)"),
 "SPY":  dict(ticker="SPY", cal="LV", vol=None, delay=1, spec="EMA3(f=20,m=80,s=100) OR MACD(f=8,s=26,sig=0,modo=cero)"),
 # TQQQ: ensemble 50/50 de 60 sistemas (45 lentos de frozen_TQQQ.json + 15 nuevos). Se calcula con tqqq_motor.py (motor copiado del notebook de reproducción), delay=2
 "TQQQ": dict(ticker="TQQQ", cal="LV", vol=None, delay=2, spec="ENSEMBLE60", modulo="tqqq_motor"),
}
