"""Senal diaria del portfolio FTMO (forward test). Equivale a: python ftmo_rotativo.py --senal <data/csv_sleeves> --top 3 --cap 2 --ref AMD --tau 0.10

CONVENCION (backtest = vivo): la senal usa SOLO cierres hasta AYER (t); se EJECUTA HOY (t+1), en los ultimos minutos antes del cierre; la posicion gana desde hoy.
=> ejecutalo POR LA MAÑANA (la barra de hoy se descarta siempre: es provisional). Si algun activo de la cesta no tiene barra reciente, el programa PARA y no da senal.

Uso:  python senal_diaria.py                                         (primer dia / sin cuenta: nominal sobre 100.000, sin freno ni colchon)
      python senal_diaria.py --equity 103500 --ref3 104200           (equity actual y MAXIMO de las 3 ultimas equity de cierre del BROKER; --ref3 es obligatorio con --equity)
Otras opciones: --tau 0.12  --base 100000  --top 3  --cap 2  --ref AMD  (cesta base 8 + AMD; sin AMD: --ref "" ; con JNJ: --ref AMD,JNJ)  --solo-csv  --hoy AAAA-MM-DD (pruebas)
Salidas: senal_log.csv, snapshots/senal_<fecha>.csv (foto de los datos de entrada), fills_log.csv (apunta aqui cada ejecucion real), run_log.jsonl (reproducibilidad).
Los sleeves se calculan solos desde los precios (sleeves_motor.py, junto a este archivo) y se comparan con los CSV de data/csv_sleeves ("coincide X %")."""
import sys, runpy, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "--senal"] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftmo_rotativo.py"), run_name="__main__")
