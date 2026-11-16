"""Senal diaria del portfolio FTMO (forward test). Equivale a: python ftmo_rotativo.py --senal sleeves --top 3 --cap 2 --ref AMD,JNJ --tau 0.10
Uso:  python senal_diaria.py                       (primer dia / sin cuenta: nominal sobre 100.000)
      python senal_diaria.py --equity 103500 --ref3 104200      (equity actual y maximo de los ultimos 3 cierres de equity)
Los sleeves se calculan solos desde los precios (sleeves_motor.py, junto a este archivo): ya no hace falta abrir el notebook.
Para JNJ pega su linea en sleeves_motor.py (SLEEVES); hasta entonces usa su CSV de la carpeta sleeves/ y avisa si esta desactualizado.
Otras opciones: --solo-csv (usar solo los CSV)   --tau 0.12   --base 100000   --top 3   --cap 2   --ref AMD,JNJ   (carpeta de sleeves: 'sleeves', o la que pases tras --senal)"""
import sys, runpy, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "--senal"] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftmo_rotativo.py"), run_name="__main__")
