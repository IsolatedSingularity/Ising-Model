#%% Importing Modules
import os
import random as rd

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

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
    """Apply Tokyo Night Storm dark theme to all subsequent matplotlib figures."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "axes.edgecolor":    PALETTE["subtle"],
        "axes.labelcolor":   PALETTE["fg"],
        "axes.titlecolor":   PALETTE["fg"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["fg"],
        "grid.color":        PALETTE["subtle"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "axes.prop_cycle":   cycler(color=CYCLE),
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["subtle"],
        "legend.labelcolor": PALETTE["fg"],
        "font.family":       "sans-serif",
        "font.size":         10,
    })


# Custom spin colormap: spin-down = blue, spin-up = red
SPIN_CMAP = LinearSegmentedColormap.from_list(
    "ising",
    [PALETTE["blue"], PALETTE["bg"], PALETTE["red"]],
    N=256,
)


# â”€â”€ Ising Lattice â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class IsingLattice:
    """
    Single 2D ferromagnetic Ising lattice with Metropolis spin-flip dynamics.

    Hamiltonian (no external field):

        H = -J * sum_{<i,j>} s_i * s_j,   s_i in {+1, -1}

    Periodic boundary conditions on all four sides.
    The Onsager critical temperature (infinite lattice) is:

        T_c = 2 / ln(1 + sqrt(2)) * J/k_B ~ 2.269 J/k_B

    Metropolis acceptance probability for a single-spin flip with energy cost dE:

        W(dE) = min(1, exp(-dE / T))

    Parameters
    ----------
    size : lattice side length N (lattice is N x N)
    T    : temperature in units of J / k_B
    """

    CRIT_TEMP = 2.269   # Onsager solution

    def __init__(self, size, T):
        self.size = size
        self.T = T
        # Start fully ordered (all spins up)
        self.lattice = np.ones((size, size), dtype=int)

    def metropolisSweep(self):
        """
        One full Metropolis sweep: N^2 single-spin flip attempts drawn
        uniformly at random with periodic boundary conditions.
        """
        size = self.size
        T = self.T
        lat = self.lattice
        for _ in range(size * size):
            i = rd.randint(0, size - 1)
            j = rd.randint(0, size - 1)
            S = lat[i, j]
            dE = 2 * S * (
                lat[(i - 1) % size, j]
                + lat[(i + 1) % size, j]
                + lat[i, (j - 1) % size]
                + lat[i, (j + 1) % size]
            )
            if dE <= 0 or rd.random() < np.exp(-dE / T):
                lat[i, j] = -S

    def metropolisFieldSweep(self, H):
        """
        Metropolis sweep with external field H. Energy cost includes field term:

            dE = 2*S*(sum of neighbors) + 2*S*H
        """
        size = self.size
        T = self.T
        lat = self.lattice
        for _ in range(size * size):
            i = rd.randint(0, size - 1)
            j = rd.randint(0, size - 1)
            S = lat[i, j]
            dE = (
                2 * S * (
                    lat[(i - 1) % size, j]
                    + lat[(i + 1) % size, j]
                    + lat[i, (j - 1) % size]
                    + lat[i, (j + 1) % size]
                )
                + 2 * S * H
            )
            if dE <= 0 or rd.random() < np.exp(-dE / T):
                lat[i, j] = -S

    def magnetization(self):
        """Return the absolute magnetization per spin: |<m>| = |sum(s_i)| / N^2."""
        return abs(self.lattice.mean())

    def equilibrate(self, nSweeps):
        """Run nSweeps Metropolis sweeps as burn-in (no measurement)."""
        for _ in range(nSweeps):
            self.metropolisSweep()

    def measure(self, nSweeps):
        """
        Run nSweeps sweeps and collect magnetization samples.
        Returns an ndarray of shape (nSweeps,).
        """
        samples = np.empty(nSweeps)
        for idx in range(nSweeps):
            self.metropolisSweep()
            samples[idx] = self.magnetization()
        return samples


# â”€â”€ Phase Diagram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PhaseDiagram:
    """
    Equilibrium magnetization <|m|>(T) and susceptibility chi(T) for multiple
    lattice sizes. The susceptibility is estimated from magnetization fluctuations:

        chi = N^2 * Var(|m|)

    The peak of chi shifts toward T_c as N -> inf. Comparing sizes reveals the
    approach to the thermodynamic critical point.

    Parameters
    ----------
    sizes        : list of lattice sizes to simulate
    temperatures : 1D array of temperatures to sweep
    eqSteps      : burn-in sweeps per (size, T) point
    measSteps    : measurement sweeps per (size, T) point
    """

    def __init__(self, sizes, temperatures, eqSteps=500, measSteps=300):
        self.sizes = sizes
        self.temperatures = temperatures
        self.eqSteps = eqSteps
        self.measSteps = measSteps
        self.mag = {}   # size -> ndarray(len(temperatures),)
        self.chi = {}   # size -> ndarray(len(temperatures),)

    def compute(self):
        """Run the full sweep for each (size, T) combination."""
        print("Computing phase diagram...")
        for size in self.sizes:
            magArr = np.empty(len(self.temperatures))
            chiArr = np.empty(len(self.temperatures))
            for tIdx, T in enumerate(self.temperatures):
                lat = IsingLattice(size, T)
                lat.equilibrate(self.eqSteps)
                samples = lat.measure(self.measSteps)
                magArr[tIdx] = samples.mean()
                chiArr[tIdx] = (size ** 2) * samples.var()
            self.mag[size] = magArr
            self.chi[size] = chiArr
            print(f"  done size={size}")
        return self


# â”€â”€ Finite-Size Scaling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FiniteSizeScaling:
    """
    Extract the finite-size critical exponent from the susceptibility peak.

    For a finite lattice of side N, the susceptibility maximum scales as:

        chi_max(N) ~ N^{gamma/nu}

    where gamma and nu are the critical exponents of the 2D Ising universality
    class (gamma/nu = 7/4 = 1.75 exactly). The peak location T*(N) converges
    to T_c as N^{-1/nu} with nu = 1.

    Parameters
    ----------
    phaseDiagram : PhaseDiagram instance (already computed)
    """

    def __init__(self, phaseDiagram):
        self.pd = phaseDiagram
        self.peakTemp = {}    # size -> float
        self.peakChi = {}     # size -> float
        self.slope = None
        self.intercept = None

    def computePeaks(self):
        """Locate the susceptibility maximum for each lattice size."""
        for size in self.pd.sizes:
            peakIdx = np.argmax(self.pd.chi[size])
            self.peakTemp[size] = self.pd.temperatures[peakIdx]
            self.peakChi[size] = self.pd.chi[size][peakIdx]
        return self

    def fit(self):
        """
        Log-log linear fit: ln(chi_max) = (gamma/nu) * ln(N) + const.
        Stores slope (= gamma/nu estimate) and intercept.
        """
        logN = np.log(np.array(self.pd.sizes, dtype=float))
        logPeak = np.log(np.array([self.peakChi[s] for s in self.pd.sizes]))
        self.slope, self.intercept = np.polyfit(logN, logPeak, 1)
        return self


# â”€â”€ Hysteresis Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class HysteresisLoop:
    """
    Magnetic hysteresis loop m(H) at a fixed sub-critical temperature.

    The external field is swept from H_min to H_max and back. At T < T_c the
    Ising model is in the ordered phase and exhibits a non-zero coercive field:
    the magnetization does not reverse until the external field is large enough
    to overcome the domain-wall energy, producing a closed hysteresis loop.

    Parameters
    ----------
    size          : lattice side length
    T             : temperature (should satisfy T < T_c ~ 2.269)
    fieldValues   : 1D array of field values for the up-sweep (reversed for down-sweep)
    nRelaxSweeps  : Metropolis sweeps at each field value before measurement
    """

    def __init__(self, size=40, T=1.8, fieldValues=None, nRelaxSweeps=80):
        self.size = size
        self.T = T
        self.fieldValues = (
            fieldValues if fieldValues is not None
            else np.linspace(-2.0, 2.0, 40)
        )
        self.nRelaxSweeps = nRelaxSweeps
        self.fieldSweep = None
        self.magnetization = None

    def compute(self):
        """Run the full up-then-down field sweep and record magnetization."""
        print("Computing hysteresis loop...")
        lat = IsingLattice(self.size, self.T)
        self.fieldSweep = np.concatenate([self.fieldValues, self.fieldValues[::-1]])
        mag = np.empty(len(self.fieldSweep))
        for fIdx, H in enumerate(self.fieldSweep):
            for _ in range(self.nRelaxSweeps):
                lat.metropolisFieldSweep(H)
            mag[fIdx] = lat.lattice.mean()
        self.magnetization = mag
        return self


# â”€â”€ Visualizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class IsingVisualizer:
    """
    Produces the four standard diagnostic figures for the Ising generalization suite.

    Figures
    -------
    1. Lattice spin-configuration heatmaps at three temperatures
    2. Phase diagram: <|m|>(T) and chi(T) for multiple lattice sizes
    3. Finite-size scaling of the susceptibility peak
    4. Magnetic hysteresis loop
    """

    def __init__(self, palette=None, spinCmap=None, outputDir="Plots"):
        self.palette = palette if palette is not None else PALETTE
        self.spinCmap = spinCmap if spinCmap is not None else SPIN_CMAP
        self.outputDir = outputDir
        os.makedirs(outputDir, exist_ok=True)

    def _savePath(self, filename):
        return os.path.join(self.outputDir, filename)

    def plotSnapshots(self, lattices, temperatures, labels):
        """Figure 1: Spin-configuration heatmaps at three temperatures."""
        p = self.palette
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Ising Lattice Configurations", fontsize=15)

        for ax, lat, lbl in zip(axes, lattices, labels):
            im = ax.imshow(lat, cmap=self.spinCmap, vmin=-1, vmax=1,
                           interpolation="nearest", aspect="equal")
            ax.set_title(lbl, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["down  -1", "0", "+1  up"])
        cbar.ax.yaxis.set_tick_params(color=p["muted"])

        fig.tight_layout()
        path = self._savePath("ising_snapshots.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)

    def plotPhaseDiagram(self, pd):
        """Figure 2: <|m|>(T) and chi(T) for multiple lattice sizes."""
        p = self.palette
        sizeColors = [p["blue"], p["cyan"], p["purple"], p["orange"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("2D Ising Model: Phase Diagram", fontsize=15)

        for size, col in zip(pd.sizes, sizeColors):
            lbl = f"$N = {size}$"
            ax1.plot(pd.temperatures, pd.mag[size], color=col, linewidth=2, label=lbl)
            ax2.plot(pd.temperatures, pd.chi[size], color=col, linewidth=2, label=lbl)

        for ax in (ax1, ax2):
            ax.axvline(IsingLattice.CRIT_TEMP, color=p["red"], linewidth=1.5,
                       linestyle="--", label=r"$T_c = 2.269$")
            ax.legend()
            ax.grid(True)

        ax1.set_title(r"Magnetisation per Spin  $\langle |m| \rangle$")
        ax1.set_xlabel(r"Temperature $T\;[J/k_B]$")
        ax1.set_ylabel(r"$\langle |m| \rangle$")

        ax2.set_title(r"Susceptibility  $\chi = N^2\,\mathrm{Var}(m)$")
        ax2.set_xlabel(r"Temperature $T\;[J/k_B]$")
        ax2.set_ylabel(r"$\chi$")

        fig.tight_layout()
        path = self._savePath("ising_phase_diagram.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)

    def plotFiniteSizeScaling(self, fss):
        """Figure 3: Susceptibility peak location and log-log height scaling."""
        p = self.palette
        pd = fss.pd
        sizes = np.array(pd.sizes)
        peakTemps = np.array([fss.peakTemp[s] for s in pd.sizes])
        peakChis = np.array([fss.peakChi[s] for s in pd.sizes])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Finite-Size Scaling of the Susceptibility Peak", fontsize=15)

        ax1.plot(sizes, peakTemps, "o-", color=p["cyan"], linewidth=2, markersize=7,
                 label="Simulated $T^*(N)$")
        ax1.axhline(IsingLattice.CRIT_TEMP, color=p["red"], linewidth=1.5,
                    linestyle="--", label=r"$T_c^\infty = 2.269$")
        ax1.set_title(r"Peak Location  $T^*(N) \to T_c^\infty$")
        ax1.set_xlabel("System size $N$")
        ax1.set_ylabel(r"$T^*(N)$")
        ax1.legend()
        ax1.grid(True)

        ax2.loglog(sizes, peakChis, "o", color=p["purple"], markersize=8,
                   label="Simulated $\\chi_{\\max}$")
        NLine = np.linspace(sizes.min(), sizes.max(), 200)
        ax2.loglog(NLine, np.exp(fss.intercept) * NLine ** fss.slope,
                   color=p["yellow"], linewidth=2, linestyle="--",
                   label=f"fit  $\\gamma/\\nu \\approx {fss.slope:.2f}$  (exact = 1.75)")
        ax2.set_title(r"Peak Height  $\chi_\mathrm{max}(N) \sim N^{\gamma/\nu}$")
        ax2.set_xlabel("System size $N$")
        ax2.set_ylabel(r"$\chi_\mathrm{max}$")
        ax2.legend()
        ax2.grid(True, which="both")

        fig.tight_layout()
        path = self._savePath("ising_fss.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)

    def plotHysteresis(self, hyst):
        """Figure 4: Hysteresis loop m(H) showing coercive field."""
        p = self.palette
        halfIdx = len(hyst.fieldSweep) // 2

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.suptitle(
            f"Ising Hysteresis Loop  ($T = {hyst.T}$, $N = {hyst.size}$)",
            fontsize=15,
        )

        ax.plot(hyst.fieldSweep[:halfIdx], hyst.magnetization[:halfIdx],
                color=p["blue"], linewidth=2.5, marker="o", markersize=3,
                label="Field increasing")
        ax.plot(hyst.fieldSweep[halfIdx:], hyst.magnetization[halfIdx:],
                color=p["red"], linewidth=2.5, marker="o", markersize=3,
                linestyle="--", label="Field decreasing")

        ax.axhline(0, color=p["subtle"], linewidth=1, linestyle=":")
        ax.axvline(0, color=p["subtle"], linewidth=1, linestyle=":")
        ax.set_xlabel("External Field $H$")
        ax.set_ylabel("Magnetisation per Spin $m$")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        path = self._savePath("ising_hysteresis.jpg")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"  saved {path}")
        plt.close(fig)


# â”€â”€ Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    applyTokyoNight()

    scriptDir = os.path.dirname(os.path.abspath(__file__))
    plotsDir = os.path.join(scriptDir, "Plots")

    # Lattice snapshots at three temperatures
    print("Generating lattice snapshots...")
    snapshotSize = 60
    snapshotTemps = [1.0, IsingLattice.CRIT_TEMP, 3.5]
    snapshotLabels = [
        r"$T \ll T_c$  (ordered)",
        r"$T \approx T_c$  (critical)",
        r"$T \gg T_c$  (disordered)",
    ]
    snapshotLattices = []
    for T in snapshotTemps:
        lat = IsingLattice(snapshotSize, T)
        lat.equilibrate(500)
        lat.measure(300)
        snapshotLattices.append(lat.lattice.copy())

    # Phase diagram and finite-size scaling
    pd = PhaseDiagram(
        sizes=[20, 30, 40, 60],
        temperatures=np.linspace(1.5, 3.5, 25),
        eqSteps=500,
        measSteps=300,
    )
    pd.compute()

    fss = FiniteSizeScaling(pd)
    fss.computePeaks().fit()

    # Hysteresis loop
    hyst = HysteresisLoop(
        size=40, T=1.8,
        fieldValues=np.linspace(-2.0, 2.0, 40),
        nRelaxSweeps=80,
    )
    hyst.compute()

    # Produce all four figures
    viz = IsingVisualizer(palette=PALETTE, spinCmap=SPIN_CMAP, outputDir=plotsDir)
    print("Plotting Figure 1: lattice snapshots...")
    viz.plotSnapshots(snapshotLattices, snapshotTemps, snapshotLabels)
    print("Plotting Figure 2: phase diagram...")
    viz.plotPhaseDiagram(pd)
    print("Plotting Figure 3: finite-size scaling...")
    viz.plotFiniteSizeScaling(fss)
    print("Plotting Figure 4: hysteresis loop...")
    viz.plotHysteresis(hyst)

    print("Done.")



