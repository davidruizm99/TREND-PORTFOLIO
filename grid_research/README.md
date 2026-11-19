# Grid research: registro de las ejecuciones que dieron los sleeves

`Grid_research_filtros_inverso.ipynb` es el cuaderno generador (indicadores solos y combinados OR/AND con filtros de Sharpe, win rate y profit factor; exporta `{TICKER}_c<n>_daily.csv`).

La carpeta `ejecuciones/` guarda, por cada sleeve de la cartera FTMO, la ejecucion de la que sale. Cada una contiene: `informe_*.txt` (configuracion y todo lo impreso), `candidatos_*.csv`, `casi_aprobados_*.csv`, `shortlist_*.csv`, `fase4_variantes_*.csv`, `validacion_congelada_*.csv`, `*_contexto.txt` y los `*_daily.csv`.

## Que ejecucion produjo cada CSV de sleeve
Comprobado comparando `strategy_return` dia a dia (2018 en adelante).

| Sleeve (CSV) | Ejecucion | Fichero | Coincidencia |
|---|---|---|---|
| AAPL_c1_daily.csv | AAPL_20261026_2031 | AAPL_c1_daily.csv | exacta (diferencia 0) |
| AMD_c1_daily.csv | AMD_20261028_2003 | AMD_c1_daily.csv | exacta |
| BTC-USD_c2_daily.csv | BTC-USD_20261026_1106 | BTC-USD_c2_daily.csv | exacta |
| GLD_c1_daily.csv | GLD_20261026_1120 | GLD_c1_daily.csv | exacta |
| SOL-USD_c1_daily.csv | SOL-USD_20261026_1053 | SOL-USD_c1_daily.csv | exacta |
| ETH-USD_c1_vol35_daily.csv | ETH-USD_20261026_1053 | ETH-USD_f4_daily.csv | aproximada (89,7% de dias iguales): vol35 es una variante posterior |
| NVDA_c1_vol35_exacto_daily.csv | NVDA_20261026_1957 | NVDA_f4_daily.csv | aproximada (92,7%): vol35 es una variante posterior |

## Pendiente (no esta en este registro)
- Como se genero la variante `vol35` de ETH y NVDA.
- QQQ y SPY (`*_10bp_daily.csv`): vienen de otro cuaderno.
- TQQQ (ensemble de 60 votantes): otro cuaderno.
- Archivo "A".
- Datos en bruto: el cuaderno descarga de Yahoo, el precio exacto de cada dia no se guardo. Los `_daily.csv` incluyen los retornos.
- JNJ queda fuera de la cartera y no se incluye.
- Las demas ejecuciones del grid (otros tickers) no estan aqui porque no forman parte de la cartera.
