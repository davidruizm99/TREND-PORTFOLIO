import numpy as np, pandas as pd, matplotlib, os, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "rotativo_daily_9sleeves.csv")
OUT = os.path.join(HERE, "salida"); os.makedirs(OUT, exist_ok=True)

ANN, CAP, TMAX, BL = 365, 1100, 1100 + 366, 30      # CAP = días máximos para pasar las dos fases (3 años)
COSTE = 0.04                                         # swap extra anual sobre la exposición
OBJ1, OBJ2, DDD, DDM, DMIN = 0.10, 0.05, 0.05, 0.10, 4
FACT, PAGO, CADA, DFOND = 1.5, 0.80, 30, 365
CUOTA, CUENTA = 540.0, 100_000.0
NP, NB = 20_000, 5_000                              # trayectorias totales y por lote
r = pd.read_csv(CSV, index_col=0, parse_dates=True)["ret"]
r = r[r.index >= "2018-01-03"].values.astype(float); n = len(r); mu = r.mean()

PLANES = {                                           # evaluación (método, parámetro) -> fondeada k fijo
    "Plan rápido: C τ=0,15 → k 0,35": (("C", 0.15), 0.35),
    "Plan prudente: C τ=0,10 → k 0,35": (("C", 0.10), 0.35),
    "Sin control: k=1 → k 0,35": (("A", 1.0), 0.35),
    "Muy lento: k 0,35 → k 0,35": (("A", 0.35), 0.35),
}
EDGES = {"Ventaja histórica (100 %)": 1.0, "Mitad de ventaja (50 %)": 0.5, "Sin ventaja (0 %)": 0.0}
PRINCIPAL = "Plan rápido: C τ=0,15 → k 0,35"

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

def sim(R, V, ev, fk):
    P = R.shape[0]
    stage = np.zeros(P, int); E = np.ones(P); h = np.ones((P, 3)); dias = np.zeros(P, int)
    ts = np.zeros(P, int); ult = np.zeros(P, int); cob = np.zeros(P); npay = np.zeros(P, int)
    t1 = np.full(P, -1); t2 = np.full(P, -1); tfirst = np.full(P, -1); out = np.zeros(P, int)
    base = np.array([1, 3, 6, 0])
    for t in range(1, TMAX + 1):
        act = stage < 3
        if not act.any(): break
        cap = act & (stage < 2) & (t > CAP)
        out[cap] = 5; stage[cap] = 3; act = act & ~cap
        st0 = stage.copy(); fund = act & (st0 == 2)
        v = V[:, t]; ref = h.max(1)
        if ev[0] == "A":
            k_ev = np.full(P, ev[1])
        else:
            k_ev = np.where(v > 0, np.minimum(1.0, ev[1] / np.where(v > 0, v, 1)), 0.0)
            k_ev = k_ev * np.minimum(1.0, np.maximum(0.25, (E - (1 - DDM)) / DDM))
            hueco = E - (ref - DDD); peor = 2.5 * k_ev * v / np.sqrt(ANN) * E
            sc = np.where((peor > 0) & (hueco - 0.01 < peor), np.maximum(0.0, (hueco - 0.01) / np.where(peor > 0, peor, 1)), 1.0)
            k_ev = k_ev * sc
        k = np.where(fund, fk, k_ev)
        x = k * R[:, t] - k * COSTE / ANN
        peor_dia = E * (1 + np.minimum(0.0, x) * FACT)
        En = E * (1 + x)
        bd = act & (peor_dia < ref - DDD)
        bm = act & ~bd & (peor_dia < 1 - DDM)
        h = np.where(act[:, None], np.concatenate([h[:, 1:], En[:, None]], 1), h)
        E = np.where(act, En, E); dias = dias + (act & (k > 0))
        br = bd | bm
        out[br] = (base[st0] + bm.astype(int))[br]; stage[br] = 3
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
        out[fin] = 8; stage[fin] = 3
    return dict(out=out, t1=t1, t2=t2, cob=cob, npay=npay, tfirst=tfirst)

rng = np.random.default_rng(20261103)
RES = {pl: {ed: [] for ed in EDGES} for pl in PLANES}
for b in range(NP // NB):
    R0 = boot(rng, NB)
    for ed, s in EDGES.items():
        R = R0 - (1 - s) * mu; V = volat(R)
        for pl, (ev, fk) in PLANES.items():
            RES[pl][ed].append(sim(R, V, ev, fk))
    print("lote", b + 1, "de", NP // NB, flush=True)
D = {pl: {ed: {k: np.concatenate([x[k] for x in RES[pl][ed]]) for k in RES[pl][ed][0]} for ed in EDGES} for pl in PLANES}

# ---------- métricas ----------
rows = []
for pl in PLANES:
    for ed in EDGES:
        d = D[pl][ed]; o = d["out"]; N = len(o)
        pass1 = (d["t1"] >= 0).mean(); funded = (d["t2"] >= 0).mean()
        f1 = np.isin(o, [1, 2]).mean(); f2 = np.isin(o, [3, 4]).mean(); unres = (o == 5).mean()
        fm = d["t2"] >= 0
        surv = (o == 8).sum() / max(fm.sum(), 1); anypay = (d["npay"][fm] > 0).mean() if fm.any() else 0
        net = -CUOTA + CUOTA * (d["npay"] > 0) + CUENTA * d["cob"]
        tot = d["t2"][fm]
        rows.append(dict(plan=pl, ventaja=ed, pasa_fase1=pass1, pasa_ambas=funded, falla_f1=f1, falla_f2=f2, sin_resolver=unres,
                         fondeada_sobrevive_12m=surv, fondeada_cobra_alguna_vez=anypay, p_funded_y_cobra=(d["npay"] > 0).mean(),
                         dias_p10=np.percentile(tot, 10) if len(tot) else np.nan, dias_mediana=np.median(tot) if len(tot) else np.nan,
                         dias_p90=np.percentile(tot, 90) if len(tot) else np.nan,
                         cobro_medio_pct=100 * d["cob"][fm].mean() if fm.any() else 0,
                         ev_por_intento_eur=net.mean(), p_neto_positivo=(net > 0).mean()))
M = pd.DataFrame(rows); M.to_csv(os.path.join(OUT, "montecarlo_resumen.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
txt = M.round(3).to_string(index=False)
open(os.path.join(OUT, "montecarlo_resumen.txt"), "w", encoding="utf-8").write(txt); print(txt)

# ---------- gráficas ----------
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a8a79f"
BLUES = ["#9ec5f0", "#5a9be0", "#1d5aa8"]   # secuencial de una sola tonalidad: sin ventaja → ventaja histórica
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.family": "DejaVu Sans",
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
def pct(x, d=0): return f"{100*x:.{d}f} %".replace(".", ",")
def grid(ax, axis="y"): ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
def titulo(fig, t, s):
    fig.text(0.04, 0.965, t, fontsize=15, fontweight="bold", va="top", ha="left")
    fig.text(0.04, 0.915, s, fontsize=10, color=INK2, va="top", ha="left")
def fila(pl, ed): return M[(M.plan == pl) & (M.ventaja == ed)].iloc[0]
eds = list(EDGES)

# 1) Embudo del plan principal según la ventaja que conserve
fig, ax = plt.subplots(figsize=(10, 5.8)); fig.subplots_adjust(top=0.84, left=0.28, right=0.95, bottom=0.22)
etapas = [("Pasa la fase 1 (+10 %)", "pasa_fase1"), ("Pasa las dos fases\n(cuenta fondeada)", "pasa_ambas"),
          ("Llega a cobrar alguna vez", "p_funded_y_cobra")]
h = 0.22
for j, ed in enumerate(reversed(eds)):                    # sin ventaja arriba…
    for i, (lab, col) in enumerate(etapas):
        v = fila(PRINCIPAL, ed)[col]; y = i + (j - 1) * h * 1.05
        ax.barh(y, v, height=h, color=BLUES[j], zorder=3)
        ax.text(v + 0.008, y, pct(v), va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(etapas))); ax.set_yticklabels([e[0] for e in etapas]); ax.invert_yaxis()
ax.set_xlim(0, 1.0); ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xticklabels([f"{int(100*t)} %" for t in np.arange(0, 1.01, 0.2)])
grid(ax, "x")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUES[2 - j], label=e) for j, e in enumerate(eds)], loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=3, frameon=False, title="Qué parte de la ventaja histórica se mantiene", title_fontsize=9)
titulo(fig, "Probabilidad en cada etapa del proceso", PRINCIPAL + " · " + f"{NP:,}".replace(",", ".") + " trayectorias simuladas por escenario")
fig.savefig(os.path.join(OUT, "1_embudo.png"), dpi=170); plt.close(fig)

# 2) Resultado de la evaluación por plan (ventaja histórica y 50 %)
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True); fig.subplots_adjust(top=0.78, left=0.25, right=0.97, bottom=0.12, wspace=0.16)
cats = [("pasa_ambas", "Pasa las dos fases", BLUE), ("falla_f1", "Falla en la fase 1", ORANGE), ("falla_f2", "Falla en la fase 2", YELLOW), ("sin_resolver", f"Sin resolver en 3 años", GRAY)]
for ax, ed in zip(axs, eds[:2]):
    for i, pl in enumerate(PLANES):
        left = 0
        for col, lab, c in cats:
            v = fila(pl, ed)[col]
            ax.barh(i, v, left=left, color=c, height=0.62, edgecolor=BG, linewidth=2, zorder=3)
            if v >= 0.045: ax.text(left + v / 2, i, pct(v), ha="center", va="center", fontsize=9, color="white" if c in (BLUE, ORANGE) else INK)
            left += v
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"]); ax.set_title(ed, fontsize=10.5, color=INK2, loc="left")
    ax.set_yticks(range(len(PLANES))); ax.set_yticklabels(list(PLANES)); ax.invert_yaxis(); grid(ax, "x")
axs[0].legend(handles=[Patch(color=c, label=l) for _, l, c in cats], loc="upper center", bbox_to_anchor=(1.05, -0.07), ncol=4, frameon=False)
titulo(fig, "Cómo acaba la evaluación según el plan", "Reparto de resultados de las 20.000 trayectorias. Fallar = romper el límite de pérdida diaria (5 %) o el máximo (10 %)")
fig.savefig(os.path.join(OUT, "2_resultados_por_plan.png"), dpi=170); plt.close(fig)

# 3) Tiempo hasta estar fondeado (acumulada)
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2)); fig.subplots_adjust(top=0.78, left=0.07, right=0.93, bottom=0.12, wspace=0.40)
xs = np.arange(0, CAP + 1, 5); cols = {list(PLANES)[0]: BLUE, list(PLANES)[1]: AQUA, list(PLANES)[2]: ORANGE, list(PLANES)[3]: GRAY}
def cdf(d): t2 = d["t2"]; return np.array([(((t2 >= 0) & (t2 <= x)).mean()) for x in xs])
for pl in PLANES:
    y = cdf(D[pl][eds[0]]); axs[0].plot(xs, 100 * y, color=cols[pl], lw=2.2)
    axs[0].text(xs[-1] + 12, 100 * y[-1], pl.split(":")[0], color=INK, fontsize=9, va="center")
axs[0].set_title("Los cuatro planes · ventaja histórica", fontsize=10.5, color=INK2, loc="left")
for ed, c, ls in zip(eds, [BLUES[2], BLUES[1], BLUES[0]], ["-", "-", "-"]):
    y = cdf(D[PRINCIPAL][ed]); axs[1].plot(xs, 100 * y, color=c, lw=2.2); axs[1].text(xs[-1] + 12, 100 * y[-1], ed.split(" (")[0], color=INK, fontsize=9, va="center")
axs[1].set_title(f"{PRINCIPAL.split(':')[0]} · según la ventaja que se mantenga", fontsize=10.5, color=INK2, loc="left")
for ax in axs:
    ax.set_xlim(0, CAP + 260); ax.set_ylim(0, 100); ax.set_xticks([0, 120, 240, 365, 540, 730, 1095]); ax.set_xticklabels(["0", "4 m", "8 m", "1 año", "1,5 a", "2 años", "3 años"])
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"]); grid(ax)
axs[0].set_ylabel("probabilidad acumulada de estar fondeado")
titulo(fig, "Cuánto se tarda en conseguir la cuenta fondeada", "Probabilidad acumulada de haber pasado las dos fases en cada momento. El tramo plano es el techo de cada plan")
fig.savefig(os.path.join(OUT, "3_tiempo_hasta_fondeada.png"), dpi=170); plt.close(fig)

# 4) Cobros en la cuenta fondeada (12 meses)
fig, axs = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True); fig.subplots_adjust(top=0.78, left=0.07, right=0.97, bottom=0.12, wspace=0.06)
bins = np.arange(0, 66, 3)
for ax, ed in zip(axs, eds[:2]):
    d = D[PRINCIPAL][ed]; fm = d["t2"] >= 0; c = 100 * d["cob"][fm]
    cnt, _ = np.histogram(np.clip(c, 0, 64.9), bins=bins); w = cnt / cnt.sum()
    ax.bar(bins[:-1] + 1.5, 100 * w, width=2.55, color=BLUE, zorder=3)
    med, mean = np.median(c), c.mean()
    ax.axvline(med, color=INK, lw=1.4, ls="--", zorder=4); ax.text(med + 0.6, ax.get_ylim()[1] * 0.0 + 0.8, "", color=INK)
    ax.set_title(ed, fontsize=10.5, color=INK2, loc="left"); ax.set_xlabel("cobrado en 12 meses (% de la cuenta, tras el reparto del 80 %)")
    ax.set_xticks(range(0, 65, 10)); ax.set_xticklabels([f"{x} %" for x in range(0, 65, 10)]); grid(ax)
    s = (f"Cuentas fondeadas que acaban cobrando algo: {pct((c > 0).mean())}\nCuentas que sobreviven 12 meses: {pct((d['out'][fm] == 8).mean())}\n"
         f"Cobro mediano: {median_txt if False else ''}{med:.1f} % · medio: {mean:.1f} %").replace(".", ",")
    ax.text(0.98, 0.97, s, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
axs[0].set_ylabel("% de las cuentas fondeadas"); axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %"))
titulo(fig, "Qué se cobra una vez fondeado (k fijo 0,35)", "Distribución del cobro total en 12 meses, solo entre cuentas que llegan a fondearse. La raya discontinua es la mediana")
fig.savefig(os.path.join(OUT, "4_cobros_fondeada.png"), dpi=170); plt.close(fig)

# 5) Resultado económico por intento y probabilidad de lograrlo en n intentos
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2)); fig.subplots_adjust(top=0.78, left=0.08, right=0.95, bottom=0.14, wspace=0.22)
d = D[PRINCIPAL][eds[0]]; net = -CUOTA + CUOTA * (d["npay"] > 0) + CUENTA * d["cob"]
fallo = (net <= -CUOTA + 1).mean()
bins = np.arange(-1000, 40001, 1500); hv = net[net > -CUOTA + 1]
cnt, _ = np.histogram(np.clip(hv, -999, 39999), bins=bins)
axs[0].bar(bins[:-1] + 750, 100 * cnt / len(net), width=1350, color=BLUE, zorder=3)
axs[0].bar([-540], [100 * fallo], width=1350, color=ORANGE, zorder=3)
axs[0].set_ylim(0, 100 * fallo * 1.28); axs[0].text(-300, 100 * fallo * 1.02, f"{pct(fallo)} pierden\n{int(CUOTA)} €", ha="left", fontsize=9, va="bottom")
axs[0].axvline(net.mean(), color=INK, ls="--", lw=1.4, zorder=4)
axs[0].text(net.mean() + 600, 100 * fallo * 0.55, f"media\n{net.mean():,.0f} €".replace(",", "."), fontsize=9)
axs[0].set_xlim(-2500, 40000); axs[0].set_xlabel("resultado neto de un intento en cuenta de 100 000 € (€, tras cuota y reparto)")
axs[0].set_xticks(range(0, 40001, 10000)); axs[0].set_xticklabels([f"{x//1000} k" for x in range(0, 40001, 10000)])
axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %")); axs[0].set_ylabel("% de los intentos"); grid(axs[0])
axs[0].set_title("Un intento · plan rápido, ventaja histórica", fontsize=10.5, color=INK2, loc="left")
ns = np.arange(1, 6)
for pl in list(PLANES)[:3]:
    for ed, ls in ((eds[0], "-"), (eds[1], ":")):
        p = fila(pl, ed)["pasa_ambas"]; y = 100 * (1 - (1 - p) ** ns)
        axs[1].plot(ns, y, color=cols[pl], lw=2.2, ls=ls, marker="o", ms=5)
    p = fila(pl, eds[0])["pasa_ambas"]; axs[1].text(5.12, 100 * (1 - (1 - p) ** 5) + {0: -4.5, 1: 0.0, 2: -1.0}[list(PLANES).index(pl)], pl.split(":")[0], fontsize=9, va="center")
axs[1].set_xlim(0.8, 6.6); axs[1].set_ylim(0, 102); axs[1].set_xticks(ns); axs[1].set_xlabel("número de intentos (cada uno con su cuota)")
axs[1].set_yticks([0, 25, 50, 75, 100]); axs[1].set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"]); grid(axs[1])
axs[1].set_ylabel("probabilidad de lograr al menos una cuenta fondeada")
axs[1].set_title("Línea continua: ventaja histórica · punteada: mitad de ventaja", fontsize=10.5, color=INK2, loc="left")
titulo(fig, "Dinero: qué esperar de un intento y de varios", f"Cuota supuesta de {int(CUOTA)} € por cuenta de 100.000 € (reembolsable con el primer cobro). No incluye impuestos")
fig.savefig(os.path.join(OUT, "5_dinero_e_intentos.png"), dpi=170); plt.close(fig)
print("OK")
