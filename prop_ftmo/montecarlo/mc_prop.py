import numpy as np, pandas as pd, matplotlib, os, json, hashlib, platform, datetime, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
CSV = sys.argv[1] if len(sys.argv) > 1 else os.environ["CSV"]

ANN, CAP, TMAX, BL = 365, 1100, 1100 + 366, 30      # CAP = días máximos para pasar las dos fases (3 años)
COSTE = float(os.environ.get("COSTE","0"))
EXCL = [int(x) for x in os.environ.get("EXCL","").split(",") if x]
NREF = int(os.environ.get("NREF","3"))
OBJ1, OBJ2, DDD, DDM, DMIN = 0.10, 0.05, 0.05, 0.10, 4
FACT, PAGO, CADA, DFOND = 1.5, 0.80, 30, 365
LOW_MODE = os.environ.get("LOW_MODE", "max")           # "max" (DEFECTO) = el PEOR de {puente browniano, 1,5 x cierre rojo}: caida tambien en dias verdes sin perder la cola de dias rojos | "puente" solo puente (SUBESTIMA los dias de cola: sube el aprobado 80 -> 96 %) | "factor" = antiguo
BRAKE_MIN = float(os.environ.get("BRAKE_MIN", "0.0"))  # suelo del freno por drawdown: 0 = llega a 0 en el limite del 10 % (antes 0,25)
SEED = 20261103
CUOTA, CUENTA = 540.0, 100_000.0
NP, NB = 6_000, 3_000                              # trayectorias totales y por lote
r = pd.read_csv(CSV, index_col=0, parse_dates=True)["ret"]
r = r[r.index >= "2018-01-03"]
r = r[~r.index.year.isin(EXCL)].values.astype(float); n = len(r); mu = r.mean()

PLANES = {}
def plan(nombre, t1, t2=None, ramp=0.0, fk=("A", 0.35)):
    PLANES[nombre] = (dict(t1=t1, t2=t1 if t2 is None else t2, ramp=ramp), fk)
for t in (0.10, 0.12, 0.15, 0.18, 0.22): plan(f"eval C{t:.2f}", t, fk=("A", 0.25))
EDGES = {"100": 1.0, "50": 0.5}
PRINCIPAL = "rapido C.15"

def boot(rng, P):
    idx = np.empty((P, TMAX + 1), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n, P)
    jump = rng.random((P, TMAX + 1)) < 1 / BL
    new = rng.integers(0, n, (P, TMAX + 1))
    for t in range(1, TMAX + 1):
        idx[:, t] = np.where(jump[:, t], new[:, t], (idx[:, t - 1] + 1) % n)
    return r[idx]

def volat(R):
    V = pd.DataFrame(R.T).ewm(span=30, adjust=False).std().shift(1).bfill().values.T * np.sqrt(ANN)
    return V

def sim(R, V, ev, fk, XI):
    P = R.shape[0]
    stage = np.zeros(P, int); E = np.ones(P); h = np.ones((P, NREF)); dias = np.zeros(P, int)
    ts = np.zeros(P, int); ult = np.zeros(P, int); cob = np.zeros(P); npay = np.zeros(P, int)
    tend = np.zeros(P, int); t1 = np.full(P, -1); t2 = np.full(P, -1); tfirst = np.full(P, -1); out = np.zeros(P, int)
    base = np.array([1, 3, 6, 0])
    for t in range(1, TMAX + 1):
        act = stage < 3
        if not act.any(): break
        cap = act & (stage < 2) & (t > CAP)
        out[cap] = 5; stage[cap] = 3; tend[cap] = t; act = act & ~cap
        st0 = stage.copy(); fund = act & (st0 == 2)
        v = V[:, t]; ref = h.max(1)
        def kC(tau):
            kk = np.where(v > 0, np.minimum(1.0, tau / np.where(v > 0, v, 1)), 0.0)
            kk = kk * np.minimum(1.0, np.maximum(BRAKE_MIN, (E - (1 - DDM)) / DDM))
            hueco = E - (ref - DDD); peor = 2.5 * kk * v / np.sqrt(ANN) * E
            sc = np.where((peor > 0) & (hueco - 0.01 < peor), np.maximum(0.0, (hueco - 0.01) / np.where(peor > 0, peor, 1)), 1.0)
            return kk * sc
        tau = np.where(stage == 0, ev["t1"], ev["t2"]) * (1 + ev["ramp"] * np.clip((E - 1) / 0.05, 0, 1))
        k_ev = kC(tau)
        k_f = np.full(P, fk[1]) if fk[0] == "A" else kC(np.full(P, fk[1]))
        k = np.where(fund, k_f, k_ev)
        x = k * R[:, t] - k * COSTE / ANN
        if LOW_MODE == "factor":
            peor_dia = E * (1 + np.minimum(0.0, x) * FACT)
        else:                                            # minimo intradia: puente browniano, low = (r - sqrt(r^2 + 2 s^2 xi))/2 <= min(r,0); tambien hay caida los dias verdes
            sg = np.nan_to_num(v) / np.sqrt(ANN); rt = R[:, t]
            adv = k * (rt - np.sqrt(rt * rt + 2 * sg * sg * XI[:, t])) / 2
            if LOW_MODE == "max": adv = np.minimum(adv, FACT * k * np.minimum(rt, 0.0))
            peor_dia = E * (1 + adv - k * COSTE / ANN)
        En = E * (1 + x)
        bd = act & (peor_dia < ref - DDD)
        bm = act & ~bd & (peor_dia < 1 - DDM)
        h = np.where(act[:, None], np.concatenate([h[:, 1:], En[:, None]], 1), h)
        E = np.where(act, En, E); dias = dias + (act & (k > 0))
        br = bd | bm
        out[br] = (base[st0] + bm.astype(int))[br]; stage[br] = 3; tend[br] = t
        alive = act & ~br
        p0 = alive & (st0 == 0) & (E >= 1 + OBJ1) & (dias >= DMIN)
        p1 = alive & (st0 == 1) & (E >= 1 + OBJ2) & (dias >= DMIN)
        for m, nxt in ((p0, 1), (p1, 2)):
            stage[m] = nxt; E[m] = 1.0; h[m] = 1.0; dias[m] = 0; ts[m] = t; ult[m] = 0
        t1[p0] = t; t2[p1] = t
        d = t - ts
        pay = alive & (st0 == 2) & (d - ult >= CADA) & (E > 1)
        cob[pay] += PAGO * (E[pay] - 1); E[pay] = 1.0; h[pay] = 1.0; ult[pay] = d[pay]; npay[pay] += 1
        tfirst[pay & (tfirst < 0)] = d[pay & (tfirst < 0)]
        fin = alive & (st0 == 2) & (d >= DFOND)
        out[fin] = 8; stage[fin] = 3; tend[fin] = t
    return dict(out=out, t1=t1, t2=t2, cob=cob, npay=npay, tfirst=tfirst, tend=tend)

rng = np.random.default_rng(SEED)
RES = {pl: {ed: [] for ed in EDGES} for pl in PLANES}
for b in range(NP // NB):
    R0 = boot(rng, NB); XI = rng.exponential(1.0, R0.shape)
    for ed, s in EDGES.items():
        R = R0 - (1 - s) * mu; V = volat(R)
        for pl, (ev, fk) in PLANES.items():
            RES[pl][ed].append(sim(R, V, ev, fk, XI))
    pass
D = {pl: {ed: {k: np.concatenate([x[k] for x in RES[pl][ed]]) for k in RES[pl][ed][0]} for ed in EDGES} for pl in PLANES}


D = {pl: {ed: {k: np.concatenate([x[k] for x in RES[pl][ed]]) for k in RES[pl][ed][0]} for ed in EDGES} for pl in PLANES}
rows=[]
for pl in PLANES:
    for ed in EDGES:
        d=D[pl][ed]; o=d["out"]
        f2=d["t2"]>=0; pc=(d["npay"]>0).mean()
        ev = CUENTA*d["cob"].mean() - CUOTA + CUOTA*pc; T = d["tend"].mean()
        rows.append(dict(plan=pl, edge=ed, pasa_las_2=100*f2.mean(), quema_eval=100*np.isin(o,[1,2,3,4]).mean(), sin_resolver=100*(o==5).mean(),
            dias_med_las2=np.median(d["t2"][f2]) if f2.any() else np.nan, cobra=100*pc,
            quema_fond_de_los_que_pasan=100*np.isin(o,[6,7]).mean()/max(f2.mean(),1e-9), EV=ev, EV_dia=ev/T))
res = pd.DataFrame(rows)
OUTF = sys.argv[2] if len(sys.argv) > 2 else "mc_" + os.path.basename(CSV)
res.to_csv(OUTF, index=False)
pd.set_option("display.width", 200)
print(f"\n{os.path.basename(CSV)}  (pasa_las_2 y quema_eval en %, dias = mediana para pasar las dos fases)")
print(res[["plan", "edge", "pasa_las_2", "quema_eval", "sin_resolver", "dias_med_las2"]].round(1).to_string(index=False))

with open("run_log.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), modo="montecarlo", argv=sys.argv[1:], csv=os.path.basename(CSV),
        csv_sha256=hashlib.sha256(open(CSV, "rb").read()).hexdigest()[:16], script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, seed=SEED, paths=NP, bloque=BL, cap_dias=CAP, dias_fondeada=DFOND,
        low_mode=LOW_MODE, brake_min=BRAKE_MIN, factor_intradia=FACT if LOW_MODE == "factor" else None, coste_extra=COSTE, excluye_anios=EXCL, filas_serie=int(n),
        desde=str(pd.read_csv(CSV, index_col=0, parse_dates=True).index[0].date()), hasta=str(pd.read_csv(CSV, index_col=0, parse_dates=True).index[-1].date()),
        resultados=OUTF)) + "\n")
