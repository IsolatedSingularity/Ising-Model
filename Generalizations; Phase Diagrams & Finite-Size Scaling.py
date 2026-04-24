#%% Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from cycler import cycler
import random as rd
from scipy.optimize import curve_fit


#%% Tokyo Night Storm Theme
PALETTE = {
    "bg":     "#1a1b26",
    "panel":  "#24283b",
    "fg":     "#c0caf5",
    "muted":  "#a9b1d6",
    "subtle": "#565f89",
    "blue":   "#7aa2f7",
    "cyan":   "#7dcfff",
    "purple": "#bb9af7",
    "red":    "#f7768e",
    "green":  "#9ece6a",
    "yellow": "#e0af68",
    "orange": "#ff9e64",
}
CYCLE = [PALETTE[k] for k in ("blue", "cyan", "purple", "red", "green", "yellow", "orange")]

def applyTokyoNight():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["bg"],
        "savefig.facecolor":  PALETTE["bg"],
        "axes.edgecolor":     PALETTE["subtle"],
        "axes.labelcolor":    PALETTE["fg"],
        "axes.titlecolor":    PALETTE["fg"],
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "text.color":         PALETTE["fg"],
        "grid.color":         PALETTE["subtle"],
        "grid.linestyle":     "--",
        "grid.alpha":         0.4,
        "axes.prop_cycle":    cycler(color=CYCLE),
        "legend.facecolor":   PALETTE["panel"],
        "legend.edgecolor":   PALETTE["subtle"],
        "legend.labelcolor":  PALETTE["fg"],
        "font.family":        "sans-serif",
        "font.size":          10,
    })

applyTokyoNight()

# Custom spin colormap: spin-down = blue, spin-up = red
spinCmap = LinearSegmentedColormap.from_list(
    "ising",
    [PALETTE["blue"], PALETTE["bg"], PALETTE["red"]],
    N=256,
)


#%% Simulation Parameters
critTemp  = 2.269    # Onsager critical temperature (units J/k_B)
eqSteps   = 500      # burn-in Monte Carlo sweeps before measurement
measSteps = 300      # measurement sweeps after equilibration


#%% Core Monte Carlo Step (Metropolis)
def metropolisSweep(lattice, T, size):
    """One full Monte Carlo sweep: N^2 single-spin Metropolis flip attempts."""
    for _ in range(size * size):
        i = rd.randint(0, size - 1)
        j = rd.randint(0, size - 1)
        S = lattice[i, j]
        topNier    = lattice[(i - 1) % size, j]
        botNier    = lattice[(i + 1) % size, j]
        leftNier   = lattice[i, (j - 1) % size]
        rightNier  = lattice[i, (j + 1) % size]
        dE = 2 * S * (topNier + botNier + leftNier + rightNier)
        if dE <= 0:
            lattice[i, j] = -S
        elif rd.random() < np.exp(-dE / T):
            lattice[i, j] = -S
    return lattice


#%% Generating Equilibrium Snapshots at Three Temperatures
print("Generating lattice snapshots...")
snapshotSize = 60
snapshotTemps = [1.0, critTemp, 3.5]   # ordered, critical, disordered
snapshotLabels = [r"$T \ll T_c$  (ordered)", r"$T \approx T_c$  (critical)", r"$T \gg T_c$  (disordered)"]
snapshotLattices = []

for T in snapshotTemps:
    lat = np.ones((snapshotSize, snapshotSize), dtype=int)  # all spin up
    for _ in range(eqSteps + measSteps):
        lat = metropolisSweep(lat, T, snapshotSize)
    snapshotLattices.append(lat.copy())


#%% Phase Diagram: m(T) and chi(T) for Multiple Lattice Sizes
print("Computing phase diagram...")
sizes       = [20, 30, 40, 60]
temperatures = np.linspace(1.5, 3.5, 25)
phaseMag    = {}   # size -> array of |m|
phaseChi    = {}   # size -> array of chi

for size in sizes:
    magArr = []
    chiArr = []
    for T in temperatures:
        lat = np.ones((size, size), dtype=int)
        for _ in range(eqSteps):
            lat = metropolisSweep(lat, T, size)
        mSamples = []
        for _ in range(measSteps):
            lat = metropolisSweep(lat, T, size)
            mSamples.append(abs(lat.mean()))
        mSamples = np.array(mSamples)
        magArr.append(mSamples.mean())
        chiArr.append((size ** 2) * (mSamples.var()))   # chi = N * Var(m)
    phaseMag[size] = np.array(magArr)
    phaseChi[size] = np.array(chiArr)
    print(f"  done size={size}")


#%% Finite-Size Scaling: chi_max and its location
chiPeakTemp  = {}
chiPeakValue = {}
for size in sizes:
    peakIdx = np.argmax(phaseChi[size])
    chiPeakTemp[size]  = temperatures[peakIdx]
    chiPeakValue[size] = phaseChi[size][peakIdx]


#%% External Field Hysteresis at T < T_c
print("Computing hysteresis loop...")
hystSize = 40
hystTemp = 1.8
fieldValues = np.linspace(-2.0, 2.0, 40)
hystMag = []

lat = np.ones((hystSize, hystSize), dtype=int)  # start fully ordered

def metropolisFieldSweep(lattice, T, size, H):
    """Metropolis sweep with external field H: dE includes -2*S*H term."""
    for _ in range(size * size):
        i = rd.randint(0, size - 1)
        j = rd.randint(0, size - 1)
        S = lattice[i, j]
        topNier   = lattice[(i - 1) % size, j]
        botNier   = lattice[(i + 1) % size, j]
        leftNier  = lattice[i, (j - 1) % size]
        rightNier = lattice[i, (j + 1) % size]
        dE = 2 * S * (topNier + botNier + leftNier + rightNier) + 2 * S * H
        if dE <= 0:
            lattice[i, j] = -S
        elif rd.random() < np.exp(-dE / T):
            lattice[i, j] = -S
    return lattice

# Sweep field up then down
fieldSweep = np.concatenate([fieldValues, fieldValues[::-1]])
for H in fieldSweep:
    for _ in range(80):
        lat = metropolisFieldSweep(lat, hystTemp, hystSize, H)
    hystMag.append(lat.mean())
hystMag = np.array(hystMag)


#%% Plotting: Figure 1 - Lattice Snapshots at Three Temperatures
print("Plotting Figure 1: lattice snapshots...")
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle("Ising Lattice Configurations", fontsize=15, color=PALETTE["fg"])

for ax, lat, lbl in zip(axes1, snapshotLattices, snapshotLabels):
    im = ax.imshow(lat, cmap=spinCmap, vmin=-1, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_title(lbl, color=PALETTE["fg"], fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig1.colorbar(im, ax=axes1.ravel().tolist(), fraction=0.015, pad=0.02)
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(["↓  −1", "0", "+1  ↑"])
cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])

fig1.tight_layout()
fig1.savefig("Plots/ising_snapshots.jpg", bbox_inches="tight", dpi=300)
plt.show()


#%% Plotting: Figure 2 - Phase Diagram m(T) and chi(T)
print("Plotting Figure 2: phase diagram...")
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("2D Ising Model: Phase Diagram", fontsize=15, color=PALETTE["fg"])

sizeColors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["purple"], PALETTE["orange"]]

for size, col in zip(sizes, sizeColors):
    lbl = f"$N = {size}$"
    ax2a.plot(temperatures, phaseMag[size], color=col, linewidth=2, label=lbl)
    ax2b.plot(temperatures, phaseChi[size], color=col, linewidth=2, label=lbl)

ax2a.axvline(critTemp, color=PALETTE["red"], linewidth=1.5, linestyle="--",
             label=r"$T_c = 2.269$")
ax2b.axvline(critTemp, color=PALETTE["red"], linewidth=1.5, linestyle="--",
             label=r"$T_c = 2.269$")

ax2a.set_title(r"Magnetisation per Spin  $\langle |m| \rangle$", color=PALETTE["fg"])
ax2a.set_xlabel(r"Temperature $T\;[J/k_B]$", color=PALETTE["fg"])
ax2a.set_ylabel(r"$\langle |m| \rangle$", color=PALETTE["fg"])
ax2a.legend()
ax2a.grid(True)

ax2b.set_title(r"Magnetic Susceptibility  $\chi = N\,\mathrm{Var}(m)$", color=PALETTE["fg"])
ax2b.set_xlabel(r"Temperature $T\;[J/k_B]$", color=PALETTE["fg"])
ax2b.set_ylabel(r"$\chi$", color=PALETTE["fg"])
ax2b.legend()
ax2b.grid(True)

fig2.tight_layout()
fig2.savefig("Plots/ising_phase_diagram.jpg", bbox_inches="tight", dpi=300)
plt.show()


#%% Plotting: Figure 3 - Finite-Size Scaling of chi_peak
print("Plotting Figure 3: finite-size scaling...")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Finite-Size Scaling of the Susceptibility Peak", fontsize=15, color=PALETTE["fg"])

logN    = np.log(np.array(sizes, dtype=float))
logPeak = np.log(np.array([chiPeakValue[s] for s in sizes], dtype=float))
slope, intercept = np.polyfit(logN, logPeak, 1)   # chi_max ~ N^(gamma/nu)

ax3a.plot(np.array(sizes), [chiPeakTemp[s] for s in sizes],
          "o-", color=PALETTE["cyan"], linewidth=2, markersize=7, label="Simulated")
ax3a.axhline(critTemp, color=PALETTE["red"], linewidth=1.5, linestyle="--",
             label=r"$T_c^\infty = 2.269$")
ax3a.set_title(r"Peak Location  $T^*(N) \to T_c^\infty$", color=PALETTE["fg"])
ax3a.set_xlabel("System size $N$", color=PALETTE["fg"])
ax3a.set_ylabel(r"$T^*(N)$", color=PALETTE["fg"])
ax3a.legend()
ax3a.grid(True)

ax3b.loglog(np.array(sizes), [chiPeakValue[s] for s in sizes],
            "o", color=PALETTE["purple"], markersize=8, label="Simulated")
NLine = np.linspace(min(sizes), max(sizes), 200)
ax3b.loglog(NLine, np.exp(intercept) * NLine ** slope, color=PALETTE["yellow"],
            linewidth=2, linestyle="--", label=f"fit  slope = {slope:.2f}")
ax3b.set_title(r"Peak Height  $\chi_\mathrm{max}(N) \sim N^{\gamma/\nu}$", color=PALETTE["fg"])
ax3b.set_xlabel("System size $N$", color=PALETTE["fg"])
ax3b.set_ylabel(r"$\chi_\mathrm{max}$", color=PALETTE["fg"])
ax3b.legend()
ax3b.grid(True, which="both")

fig3.tight_layout()
fig3.savefig("Plots/ising_fss.jpg", bbox_inches="tight", dpi=300)
plt.show()


#%% Plotting: Figure 4 - Hysteresis Loop
print("Plotting Figure 4: hysteresis loop...")
fig4, ax4 = plt.subplots(figsize=(9, 6))
fig4.suptitle(f"Ising Hysteresis Loop  ($T = {hystTemp}$, $N = {hystSize}$)",
              fontsize=15, color=PALETTE["fg"])

nField  = len(fieldSweep)
halfIdx = nField // 2

# Up-sweep
ax4.plot(fieldSweep[:halfIdx], hystMag[:halfIdx],
         color=PALETTE["blue"], linewidth=2.5, label="Field increasing", marker="o",
         markersize=3)
# Down-sweep
ax4.plot(fieldSweep[halfIdx:], hystMag[halfIdx:],
         color=PALETTE["red"], linewidth=2.5, label="Field decreasing", marker="o",
         markersize=3, linestyle="--")

ax4.axhline(0, color=PALETTE["subtle"], linewidth=1, linestyle=":")
ax4.axvline(0, color=PALETTE["subtle"], linewidth=1, linestyle=":")
ax4.set_xlabel("External Field $H$", color=PALETTE["fg"])
ax4.set_ylabel("Magnetisation per Spin $m$", color=PALETTE["fg"])
ax4.legend()
ax4.grid(True)

fig4.tight_layout()
fig4.savefig("Plots/ising_hysteresis.jpg", bbox_inches="tight", dpi=300)
plt.show()

print("Done.")
