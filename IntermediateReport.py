"""
================================================================================
CMS DIELECTRON STATISTICAL ANALYSIS  --  INTERMEDIATE REPORT CODE
================================================================================

This single script reproduces every figure and table in the intermediate report

    "CMS Dielectron Statistical Analysis: Intermediate Report"
    Sai Nandan, Dimitri Lykopoulos, Shazil
    MATH 448/648, Computational Statistics, Spring 2026

It is organised into five logical blocks matching the report's structure:

    1.  Setup            -- imports, paths, plot style, helpers
    2.  Data             -- load CSV, define samples used by each question
    3.  Question 1       -- MLE, CIs, window sensitivity
    4.  Question 2       -- chi-square goodness-of-fit on lineshape models
    5.  Question 3       -- t-tests, covariance, correlation, partial r

A reader can run it top-to-bottom with no arguments:

    python3 cms_dielectron_analysis.py

All figures are written as both .pdf (for the LaTeX report) and .png (for
quick viewing) into a single output directory.

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
Standard scientific Python stack:
    numpy   >= 1.20
    pandas  >= 1.3
    matplotlib >= 3.4
    scipy   >= 1.7

If any of these are missing, install them with:
    pip install numpy pandas matplotlib scipy

--------------------------------------------------------------------------------
DATA
--------------------------------------------------------------------------------
The input is the CMS Open Data electron-collision CSV.

    Source: https://opendata.cern.ch/record/304
    Mirror: https://www.kaggle.com/datasets/fedesoriano/cern-electron-collision-data

The CSV has one row per dielectron event with the following columns:

    Run, Event       -- categorical run number and event ID
    E1, px1, py1, pz1, pt1, eta1, phi1, Q1     -- electron 1 kinematics
    E2, px2, py2, pz2, pt2, eta2, phi2, Q2     -- electron 2 kinematics
    M                -- pre-computed invariant mass [GeV]

Edit the INPUT_CSV constant below to point to your local copy.

--------------------------------------------------------------------------------
REPRODUCIBILITY
--------------------------------------------------------------------------------
- The bootstrap (Q1) uses numpy's default_rng with seed=448. Results are
  bit-for-bit reproducible across runs on the same machine.
- All figures use a fixed serif+STIX font, 11pt body, navy/wine/olive
  colour palette, and inward ticks. They should compile identically on
  matplotlib >= 3.4.
- The chi-square statistic in Q2 uses Pearson's form (O - E)^2 / E,
  matching Eq. (3) of the report. Note that scipy.optimize.curve_fit with
  sigma=sqrt(O) minimises the Neyman form (O - E)^2 / O; the script
  recomputes Pearson chi-square explicitly after the fit converges. The
  two differ by about 5 units of chi-square on this sample.
================================================================================
"""

# ==============================================================================
# 1.  SETUP
# ==============================================================================
# Imports.  Everything we need is in the standard scientific Python stack;
# no external particle-physics libraries are used.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, special, stats
from pathlib import Path

# ---- File paths.  Edit INPUT_CSV to point to your local copy of the data. ----
INPUT_CSV = "/mnt/user-data/uploads/dielectron.csv"
OUT       = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Physical constants from the Particle Data Group. ----
M_Z_PDG = 91.1876   # Z boson mass        [GeV]
GAMMA_Z = 2.4955    # Z natural width     [GeV]

# ---- Plot style.  Applied globally so every figure has the same look. ----
# We use the STIX math font for compatibility with LaTeX equations in captions;
# serif body font; inward tick marks on all four sides; minor ticks visible;
# transparent backgrounds for figures so they slot into the report cleanly.
plt.rcParams.update({
    "font.family":         "serif",
    "mathtext.fontset":    "stix",
    "font.serif":          ["STIXGeneral", "DejaVu Serif"],
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.linewidth":      0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.major.size":    4.5,
    "ytick.major.size":    4.5,
    "xtick.minor.size":    2.5,
    "ytick.minor.size":    2.5,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.minor.width":   0.6,
    "ytick.minor.width":   0.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.unicode_minus":  True,
    "figure.dpi":          120,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.04,
    "legend.frameon":      False,
    "legend.fontsize":     10,
})

# ---- Colour palette.  Three accent colours plus a neutral slate grey. ----
# These are chosen for adequate contrast in both print and on-screen, and
# to remain distinguishable when printed in greyscale.
NAVY  = "#1f3a5f"  # primary plotting colour
WINE  = "#7a1f3d"  # secondary; used for fit curves and the PDG reference line
OLIVE = "#5a6730"  # tertiary; second model in Q2, third group in Q3
SLATE = "#3b3b3b"  # neutral; gridlines, baseline rules, error bars


def save(fig, name):
    """Save a figure as both PDF (for the report) and PNG (for quick viewing)."""
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png}}")


# ==============================================================================
# 2.  DATA
# ==============================================================================
# Load the CSV and define the three samples used throughout the report:
#   (a) full           - all events with M present                    (Q2 contingency tests, never used here)
#   (b) opposite-sign  - Q1 * Q2 < 0; the physically meaningful subset (Q3)
#   (c) Z candidates   - opposite-sign with 80 <= M <= 100 GeV         (Q1, Q2)
print("=" * 70)
print("Loading data and defining analysis samples")
print("=" * 70)

df = pd.read_csv(INPUT_CSV).dropna(subset=["M"])
df.columns = [c.strip() for c in df.columns]  # tolerate stray whitespace

# Sample (b): opposite-sign electron pairs.
# Multiplying the two charges gives -1 for opposite-sign and +1 for same-sign.
opp = df[df.Q1 * df.Q2 < 0].copy()

# Sample (c): Z candidates = opposite-sign events in the Z-peak window.
# The window [80, 100] GeV is wide enough to contain the peak with a few full
# widths on each side and narrow enough to limit Drell-Yan contamination.
z_cand = opp[(opp.M >= 80) & (opp.M <= 100)].copy()

N_total = len(df)
N_opp   = len(opp)
N_z     = len(z_cand)
print(f"  Full dataset       (rows with M present) : N = {N_total:>7,}")
print(f"  Opposite-sign      (Q1 * Q2 < 0)          : N = {N_opp:>7,}")
print(f"  Z candidates       (OS + 80 <= M <= 100)  : N = {N_z:>7,}")


# ==============================================================================
# 3.  EXPLORATORY ANALYSIS (Section 2 of the report)
# ==============================================================================
# Three figures: full mass spectrum, opposite-vs-same-sign overlay, eta map.
# All three use the FULL dataset (no charge cut applied) so the reader sees
# the data as it comes.
print()
print("=" * 70)
print("EDA figures (report Section 2)")
print("=" * 70)


# ---- Figure 1(a): full M_ee spectrum on log-log axes. ----
# The log-log view makes the J/psi (3.1 GeV), Upsilon (~9.5 GeV), Z (~91 GeV)
# resonances and the Drell-Yan continuum all visible at the same time.
fig, ax = plt.subplots(figsize=(5.2, 3.8))
# Log-spaced bins for a log-x histogram (otherwise bin widths are misleading).
bins_log = np.logspace(np.log10(df.M.min()), np.log10(df.M.max()), 100)
ax.hist(df.M, bins=bins_log, histtype="step", color=NAVY, lw=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{ee}\ \mathrm{[GeV]}$")
ax.set_ylabel(r"$\mathrm{Events\ /\ bin}$")
save(fig, "01_mass_spectrum_full")


# ---- Figure 1(b): opposite-sign vs same-sign on the same axes. ----
# This is the figure that motivates the OS-only selection used downstream:
# the resonance peaks show up only in OS, so the SS spectrum is a clean
# data-driven background template.
same = df[df.Q1 * df.Q2 > 0]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.hist(opp.M,  bins=bins_log, histtype="step", color=NAVY,  lw=1.0,
        label=r"$Q_{1}Q_{2}=-1$")
ax.hist(same.M, bins=bins_log, histtype="step", color=SLATE, lw=1.0, ls="--",
        label=r"$Q_{1}Q_{2}=+1$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{ee}\ \mathrm{[GeV]}$")
ax.set_ylabel(r"$\mathrm{Events\ /\ bin}$")
ax.legend(loc="upper left")
save(fig, "06_charge_sign_comparison")


# ---- Figure 1(c): two-dimensional eta_1 vs eta_2 map. ----
# Shows the CMS detector's geometry directly: the dark cross at |eta| ~ 1.5
# is the gap between the barrel and endcap calorimeters where reconstruction
# fails, and the bright corners are events where both electrons go forward.
fig, ax = plt.subplots(figsize=(4.4, 4.0))
h = ax.hist2d(df.eta1, df.eta2, bins=80,
              range=[[-3, 3], [-3, 3]],
              norm="log", cmap="magma")
fig.colorbar(h[3], ax=ax, label=r"$\mathrm{Events\ /\ bin}$")
ax.set_xlabel(r"$\eta_{1}$")
ax.set_ylabel(r"$\eta_{2}$")
ax.set_aspect("equal")
save(fig, "04_eta_correlation")


# ==============================================================================
# 4.  QUESTION 1: MLE OF THE Z MASS (Section 3 of the report)
# ==============================================================================
# Uses the Z-candidate sample.  Three outputs:
#   - Fig. r02 : Z-window mass histogram with mu_hat and PDG marked
#   - Fig. 14  : forest plot of Wald + bootstrap CIs at 90/95/99%
#   - Fig. 15  : MLE drift as a function of window half-width Delta
print()
print("=" * 70)
print("QUESTION 1: maximum-likelihood estimation of the Z mass")
print("=" * 70)

M = z_cand.M.values
N = len(M)

# --- MLE point estimate ---
# For a Gaussian likelihood, the MLE of the mean is the sample mean
# (closed-form maximum of the log-likelihood in mu).  The unbiased
# variance estimator divides by N-1; the standard error of the mean
# is sigma / sqrt(N).
mu_hat    = M.mean()
sigma_hat = M.std(ddof=1)
SE        = sigma_hat / np.sqrt(N)
print(f"\n  mu_hat   = {mu_hat:.4f} GeV")
print(f"  sigma    = {sigma_hat:.4f} GeV")
print(f"  SE(mu)   = {SE:.4f} GeV")


# --- Wald confidence intervals ---
# These are the standard "estimate +/- z * SE" intervals.  z is the
# (1 - alpha/2) quantile of the standard normal; CLT guarantees this
# is approximately correct at large N even if the underlying data are
# not Gaussian.
def wald_ci(level):
    """Two-sided Wald confidence interval at the given confidence level."""
    z = stats.norm.ppf(0.5 + level / 2)
    return mu_hat - z * SE, mu_hat + z * SE


# --- Bootstrap confidence intervals ---
# Draw B resamples of size N with replacement, compute the mean of each,
# and take the empirical (alpha/2, 1 - alpha/2) quantiles.  This makes no
# Gaussianity assumption; if it agrees with the Wald interval, that's
# itself a piece of evidence that the CLT regime is reached.
rng = np.random.default_rng(seed=448)   # fixed seed for reproducibility
B = 5000
boot_means = np.array([rng.choice(M, size=N, replace=True).mean()
                       for _ in range(B)])

def boot_ci(level):
    """Two-sided percentile bootstrap CI at the given confidence level."""
    lo_pct = (1 - level) / 2
    hi_pct = 1 - lo_pct
    return float(np.quantile(boot_means, lo_pct)), \
           float(np.quantile(boot_means, hi_pct))


# --- Hypothesis test against PDG ---
# Two-sided Z-test of H0: mu = M_Z_PDG.  Because we substitute the sample
# sigma for the unknown population sigma, the strictly correct reference
# distribution is Student's t_{N-1}, but for N-1 = 6668 the t and normal
# distributions agree to four decimal places.  We quote the normal p-value.
z_stat       = (mu_hat - M_Z_PDG) / SE
p_two_sided  = 2 * stats.norm.sf(abs(z_stat))
print(f"\n  Hypothesis test vs PDG ({M_Z_PDG} GeV):")
print(f"    Z statistic = {z_stat:.4f}")
print(f"    two-sided p = {p_two_sided:.4e}")

# --- Print CI table ---
print(f"\n  Confidence intervals:")
print(f"    Level     Wald CI [GeV]               Bootstrap CI [GeV]")
for level in [0.90, 0.95, 0.99]:
    w_lo, w_hi = wald_ci(level)
    b_lo, b_hi = boot_ci(level)
    print(f"    {int(level*100)}%      [{w_lo:.4f}, {w_hi:.4f}]      "
          f"[{b_lo:.4f}, {b_hi:.4f}]")


# --- Window sensitivity sweep ---
# Repeat the MLE on windows of varying half-width Delta around M_Z_PDG.
# For each Delta, we count events, compute mu_hat, SE, the 95% CI half-width,
# and the deviation from the PDG value in standard-error units.  The result
# is the central physical finding of Q1: the bias is set by the window.
deltas = np.array([2, 3, 4, 6, 8, 10, 12, 15])
sweep_rows = []
for d in deltas:
    win = opp[(opp.M >= M_Z_PDG - d) & (opp.M <= M_Z_PDG + d)]
    n   = len(win)
    mu  = win.M.mean()
    sd  = win.M.std(ddof=1)
    se  = sd / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    sweep_rows.append({
        "Delta":              int(d),
        "N":                  n,
        "mu":                 mu,
        "SE":                 se,
        "CI95_half":          t_crit * se,
        "deviation_sigma":    (mu - M_Z_PDG) / se,
    })
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT / "window_sweep.csv", index=False)
print("\n  Window sensitivity sweep:")
print(sweep_df.to_string(index=False))


# --- Figure r02: Z-candidate mass distribution with mu_hat and PDG marked ---
fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.hist(M, bins=40, histtype="step", color=NAVY, lw=1.0)
ax.axvline(M_Z_PDG, color=WINE,  ls="--", lw=1.0,
           label=fr"$M_{{Z}}={M_Z_PDG:.2f}\ \mathrm{{GeV}}$")
ax.axvline(mu_hat,  color=SLATE, ls="-",  lw=1.0,
           label=fr"$\hat{{\mu}}={mu_hat:.4f}\ \mathrm{{GeV}}$")
ax.set_xlabel(r"$M_{ee}\ \mathrm{[GeV]}$")
ax.set_ylabel(r"$\mathrm{Events\ /\ bin}$")
ax.set_xlim(80, 100)
ax.legend(loc="upper left")
ax.minorticks_off()
save(fig, "r02_z_candidate_mass")


# --- Figure 14: CI forest plot ---
# Six intervals stacked vertically: three Wald (top block) and three bootstrap
# (bottom block), each in ascending confidence-level order (90% at top of
# each block, 99% at bottom).  The PDG line is drawn as a vertical reference.
# Visual agreement between Wald and bootstrap is itself a CLT diagnostic.
fig, ax = plt.subplots(figsize=(6.0, 4.0))
rows = []
for level in [0.90, 0.95, 0.99]:                  # 90% first => top of block
    lo, hi = wald_ci(level)
    rows.append(("Wald",      level, lo, hi, NAVY))
for level in [0.90, 0.95, 0.99]:
    lo, hi = boot_ci(level)
    rows.append(("Bootstrap", level, lo, hi, OLIVE))
# We want row 0 of the list ("90% Wald") at the TOP of the plot, so we
# place it at the largest y value and count down.  Matplotlib draws y from
# bottom to top by default, so we invert the y-positions.
n_rows = len(rows)
for i, (method, level, lo, hi, c) in enumerate(rows):
    y = n_rows - 1 - i      # 0 -> top, n_rows-1 -> bottom
    ax.errorbar([mu_hat], [y],
                xerr=[[mu_hat - lo], [hi - mu_hat]],
                fmt="o", color=c, ecolor=c,
                ms=5, lw=0, elinewidth=1.0, capsize=3)
ax.axvline(M_Z_PDG, color=WINE, ls="--", lw=1.0,
           label=r"$M_{Z}\ \mathrm{(PDG)}$")
ax.set_yticks([n_rows - 1 - i for i in range(n_rows)])
ax.set_yticklabels([f"{int(lvl*100)}% {method}"
                    for method, lvl, *_ in rows], fontsize=10)
ax.set_xlabel(r"$\hat{\mu}\ \mathrm{[GeV]}$")
ax.set_xlim(89.7, 91.3)
ax.legend(loc="lower right")
ax.minorticks_off()
save(fig, "14_q1_ci_forest")


# --- Figure 15: window sensitivity plot ---
# Points are mu_hat for each Delta with 95% CI error bars; dashed line is
# the PDG value.  The bias's monotone drift is the headline of Q1.
fig, ax = plt.subplots(figsize=(5.8, 4.0))
ax.errorbar(sweep_df.Delta, sweep_df["mu"],
            yerr=sweep_df.CI95_half,
            fmt="o-", color=NAVY, ecolor=SLATE,
            ms=5, lw=1.0, elinewidth=0.9, capsize=3,
            label=r"$\hat{\mu}\,\pm\,95\%\ \mathrm{CI}$")
ax.axhline(M_Z_PDG, color=WINE, ls="--", lw=1.0,
           label=r"$M_{Z}\ \mathrm{(PDG)}$")
ax.set_xlabel(r"$\mathrm{Window\ half\!-\!width}\ \Delta\ \mathrm{[GeV]}$")
ax.set_ylabel(r"$\hat{\mu}\ \mathrm{[GeV]}$")
ax.set_xticks(deltas)
ax.legend(loc="lower left")
ax.minorticks_off()
save(fig, "15_q1_window_sensitivity")


# ==============================================================================
# 5.  QUESTION 2: CHI-SQUARE GOODNESS-OF-FIT (Section 4 of the report)
# ==============================================================================
# Bin the Z-window M_ee spectrum into 40 bins of 0.5 GeV and fit two
# candidate models by weighted least squares:
#   Model G:  Gaussian peak    + linear background
#   Model V:  Voigt    peak    + linear background     (gamma fixed to PDG)
# Both have 5 free parameters (A, mu, sigma, b0, b1) so dof = 40 - 5 = 35.
print()
print("=" * 70)
print("QUESTION 2: chi-square goodness-of-fit on lineshape models")
print("=" * 70)

# Bin the spectrum.  We use 40 equal-width bins of 0.5 GeV over [80, 100] GeV;
# the smallest bin count is 35, comfortably above the Cochran threshold (~5)
# for the chi-square asymptotic distribution to apply.
bin_edges   = np.linspace(80, 100, 41)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
O, _        = np.histogram(z_cand.M, bins=bin_edges)
sigma_O     = np.sqrt(np.maximum(O, 1))    # Poisson errors, floor 1 to avoid div-by-zero


def gaussian_linear(M_, A, mu, sigma, b0, b1):
    """Gaussian peak + linear background, evaluated at mass M_."""
    peak = A * np.exp(-0.5 * ((M_ - mu) / sigma) ** 2)
    bg   = b0 + b1 * (M_ - 90.0)
    return peak + bg


def voigt_profile(M_, mu, sigma, gamma):
    """Voigt profile = convolution of Gaussian(sigma) with Lorentzian(gamma).
    Computed via the Faddeeva function w(z) = exp(-z^2) * erfc(-i*z),
    which scipy provides as special.wofz."""
    z = ((M_ - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(special.wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def voigt_linear(M_, A, mu, sigma, b0, b1):
    """Voigt peak (Lorentzian half-width fixed to PDG) + linear background."""
    peak = A * voigt_profile(M_, mu, sigma, GAMMA_Z / 2)
    bg   = b0 + b1 * (M_ - 90.0)
    return peak + bg


def fit_lineshape(model, p0, name, n_params=5):
    """Fit a lineshape model by weighted least squares.

    Returns the best-fit parameters, their statistical uncertainties from
    the covariance matrix, the fitted curve at the bin centers, the
    standardised residuals (O - E) / sqrt(E), the Pearson chi-square,
    the degrees of freedom, and the p-value.

    Note: scipy.optimize.curve_fit with sigma=sqrt(O) minimises the Neyman
    chi-square (O - E)^2 / O.  The report uses Pearson chi-square
    (O - E)^2 / E (Eq. 3), so we recompute that value explicitly.  The
    fitted parameters are identical for both forms.
    """
    popt, pcov = optimize.curve_fit(model, bin_centers, O,
                                    p0=p0, sigma=sigma_O,
                                    absolute_sigma=True, maxfev=10000)
    perr   = np.sqrt(np.diag(pcov))
    fitted = model(bin_centers, *popt)
    # Pearson chi-square per Eq. (3) of the report.
    resid_std = (O - fitted) / np.sqrt(np.maximum(fitted, 1))
    chi2      = float(np.sum(resid_std ** 2))
    dof       = len(O) - n_params
    p_value   = 1 - stats.chi2.cdf(chi2, dof)
    print(f"\n  {name}:")
    print(f"    chi2/dof = {chi2:.2f}/{dof} = {chi2/dof:.3f}, p = {p_value:.4e}")
    for pname, pv, pe in zip(["A", "mu", "sigma", "b0", "b1"], popt, perr):
        print(f"    {pname:>5} = {pv:8.4f} +/- {pe:.4f}")
    return popt, perr, fitted, resid_std, chi2, dof, p_value


# Initial guesses chosen visually near the peak in r02_z_candidate_mass.
gauss_popt, gauss_perr, gauss_fit, gauss_res, gauss_chi2, gauss_dof, gauss_p = \
    fit_lineshape(gaussian_linear, p0=[340, 91, 3, 60, -3],
                  name="Gaussian + linear bg")
voigt_popt, voigt_perr, voigt_fit, voigt_res, voigt_chi2, voigt_dof, voigt_p = \
    fit_lineshape(voigt_linear,    p0=[2400, 91, 2.3, 50, -3],
                  name="Voigt (gamma fixed) + linear bg")


# --- Figures 17 and 18: fit + residual plots ---
# Each figure has two stacked panels: the data + fit on top, and the
# standardised residuals (O - E) / sqrt(E) below.  Reference lines at +-2
# in the residual panel mark approximate 2-sigma deviations.
def plot_fit(model_fn, popt, residuals, label, colour, fname):
    fig, axes = plt.subplots(
        2, 1, figsize=(5.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    ax_top, ax_bot = axes
    # Top panel: data points with Poisson error bars, plus a smooth fit curve.
    ax_top.errorbar(bin_centers, O, yerr=sigma_O,
                    fmt="o", color=NAVY, ms=3.5, lw=0,
                    elinewidth=0.8, capsize=2, label="Data")
    M_smooth = np.linspace(80, 100, 400)
    ax_top.plot(M_smooth, model_fn(M_smooth, *popt),
                color=colour, lw=1.2, label=label)
    ax_top.set_ylabel(r"$\mathrm{Events}\ /\ 0.5\ \mathrm{GeV}$")
    ax_top.legend(loc="upper left")
    ax_top.minorticks_off()
    # Bottom panel: standardised residuals as a bar chart.
    ax_bot.bar(bin_centers, residuals,
               width=0.5 * 0.92, color=colour, alpha=0.6,
               edgecolor=colour, linewidth=0.4)
    ax_bot.axhline( 0, color=SLATE, lw=0.7)
    ax_bot.axhline( 2, color=SLATE, lw=0.5, ls=":")
    ax_bot.axhline(-2, color=SLATE, lw=0.5, ls=":")
    ax_bot.set_xlabel(r"$M_{ee}\ \mathrm{[GeV]}$")
    ax_bot.set_ylabel(r"$(O-E)/\sqrt{E}$")
    ax_bot.set_xlim(80, 100)
    ax_bot.minorticks_off()
    save(fig, fname)

plot_fit(gaussian_linear, gauss_popt, gauss_res,
         "Gaussian + linear", WINE,  "17_q2_gaussian_fit")
plot_fit(voigt_linear,    voigt_popt, voigt_res,
         "Voigt + linear",    OLIVE, "18_q2_voigt_fit")


# --- Figure 19: chi^2/dof comparison ---
# Two bars (Gaussian and Voigt) plus a dashed reference at chi^2/dof = 1
# and a shaded +/-1 sigma band of the chi^2/dof distribution at this dof.
fig, ax = plt.subplots(figsize=(4.6, 3.6))
chi2_dof_vals = [gauss_chi2 / gauss_dof, voigt_chi2 / voigt_dof]
ax.bar(["Gaussian", "Voigt"], chi2_dof_vals,
       color=[WINE, OLIVE], width=0.5,
       edgecolor="black", linewidth=0.5)
ax.axhline(1.0, color=SLATE, lw=0.8, ls="--",
           label=r"$\chi^{2}/\mathrm{dof}=1$")
# Reference band: +/- 1 sigma of chi^2/dof at this dof.
# For chi^2_nu / nu, the standard deviation is sqrt(2/nu).
ref_sd = np.sqrt(2 / gauss_dof)
ax.axhspan(1 - ref_sd, 1 + ref_sd, color=SLATE, alpha=0.15, lw=0)
ax.set_ylabel(r"$\chi^{2}/\mathrm{dof}$")
ax.set_ylim(0, max(chi2_dof_vals) * 1.1)
ax.legend(loc="upper right")
ax.minorticks_off()
save(fig, "19_q2_chi2_comparison")


# --- LaTeX table of fit results (Table 2 in the report). ---
# Includes parameter values and uncertainties for both models, plus the
# goodness-of-fit summary.  Written with \input{} compatibility in mind.
table_tex = rf"""\begin{{table}}[t]
\centering
\caption{{Goodness-of-fit results for the Z-window invariant-mass spectrum
($N = 6{{,}}669$ opposite-sign events, 40 equal-width bins of $0.5$~GeV).
The Gaussian and Voigt models include a common linear background
$b_0 + b_1(M_{{ee}} - 90)$. The Voigt Lorentzian half-width at half-maximum
is fixed to $\Gamma_Z / 2 = 1.2478$~GeV from the PDG natural width. The peak
amplitude $A$ is the Gaussian peak height for Model~G and the integrated
signal yield for Model~V. Quoted uncertainties are from the covariance matrix
returned by the weighted least-squares fit.}}
\label{{tab:q2_fits}}
\begin{{tabular}}{{lcc}}
\toprule
Parameter & Gaussian + linear bg & Voigt + linear bg \\
\midrule
$A$               & ${gauss_popt[0]:.2f} \pm {gauss_perr[0]:.2f}$ & ${voigt_popt[0]:.2f} \pm {voigt_perr[0]:.2f}$ \\
$\mu$ [GeV]       & ${gauss_popt[1]:.4f} \pm {gauss_perr[1]:.4f}$ & ${voigt_popt[1]:.4f} \pm {voigt_perr[1]:.4f}$ \\
$\sigma$ [GeV]    & ${gauss_popt[2]:.4f} \pm {gauss_perr[2]:.4f}$ & ${voigt_popt[2]:.4f} \pm {voigt_perr[2]:.4f}$ \\
$b_0$             & ${gauss_popt[3]:.2f} \pm {gauss_perr[3]:.2f}$ & ${voigt_popt[3]:.2f} \pm {voigt_perr[3]:.2f}$ \\
$b_1$ [GeV$^{{-1}}$] & ${gauss_popt[4]:.4f} \pm {gauss_perr[4]:.4f}$ & ${voigt_popt[4]:.4f} \pm {voigt_perr[4]:.4f}$ \\
\midrule
$\chi^{{2}}$         & ${gauss_chi2:.2f}$ & ${voigt_chi2:.2f}$ \\
dof                  & ${gauss_dof}$ & ${voigt_dof}$ \\
$\chi^{{2}}/$dof     & ${gauss_chi2/gauss_dof:.2f}$ & ${voigt_chi2/voigt_dof:.2f}$ \\
$p$-value            & ${gauss_p:.1e}$ & ${voigt_p:.1e}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
with open(OUT / "q2_table_fits.tex", "w") as f:
    f.write(table_tex)
print(f"\n  wrote q2_table_fits.tex")


# ==============================================================================
# 6.  QUESTION 3: COVARIANCE, CORRELATION, t-TESTS (Section 5 of the report)
# ==============================================================================
# Uses the OPPOSITE-SIGN sample (not Z candidates).  The reason: Q3 is about
# the joint structure of (E1, E2, M_ee), and restricting to a narrow Z window
# would freeze M_ee approximately constant and wash out the correlations.
print()
print("=" * 70)
print("QUESTION 3: covariance, correlation, t-tests on the OS sample")
print("=" * 70)

# Stack the three continuous variables into one (N, 3) matrix.
X = opp[["E1", "E2", "M"]].values

# Sample covariance and correlation matrices using the standard unbiased
# (N-1) divisor.  numpy.cov returns the (3, 3) covariance matrix.
cov_mat = np.cov(X, rowvar=False, ddof=1)
cor_mat = np.corrcoef(X, rowvar=False)
print("\n  Sample covariance matrix (GeV^2):")
print(pd.DataFrame(cov_mat, index=["E1","E2","M"], columns=["E1","E2","M"]).round(2))
print("\n  Sample correlation matrix:")
print(pd.DataFrame(cor_mat, index=["E1","E2","M"], columns=["E1","E2","M"]).round(4))


# --- Partial correlations from the precision matrix ---
# If R is the correlation matrix, the precision matrix P = R^{-1} encodes
# partial correlations directly:
#     r(i, j | rest)  =  -P[i, j] / sqrt(P[i, i] * P[j, j]).
# This is the cleanest way to compute partials when you have a small number
# of variables; for larger systems it generalises to graphical-model methods.
P = np.linalg.inv(cor_mat)

def partial_r(i, j):
    return -P[i, j] / np.sqrt(P[i, i] * P[j, j])

print("\n  Pairwise vs partial correlations:")
print("    Pair                Raw r     Partial r   t_partial    p")
N_opp = len(opp)
labels_idx = [(0, 1, 2, "(E1, E2)|M_ee"),
              (0, 2, 1, "(E1, M_ee)|E2"),
              (1, 2, 0, "(E2, M_ee)|E1")]
for i, j, k, label in labels_idx:
    r_raw  = cor_mat[i, j]
    r_part = partial_r(i, j)
    # t statistic for a partial correlation: one extra dof is subtracted
    # because we conditioned on one variable.
    df_p = N_opp - 2 - 1
    t_p  = r_part * np.sqrt(df_p / max(1 - r_part ** 2, 1e-300))
    p_p  = 2 * stats.t.sf(abs(t_p), df=df_p)
    print(f"    {label:<18}  {r_raw:+.4f}   {r_part:+.4f}   {t_p:+8.2f}   {p_p:.2e}")


# --- Two-sample Welch t-tests ---
# Welch's t-test (unequal-variance two-sample t-test) does NOT assume the
# two groups have the same variance, which is appropriate here because the
# group sizes and spreads differ.
# Group A: central (both electrons in |eta| < 1.5) vs forward (else).
cen_mask = (opp.eta1.abs() < 1.5) & (opp.eta2.abs() < 1.5)
m_cen = opp.M[ cen_mask].values
m_for = opp.M[~cen_mask].values
t_eta = stats.ttest_ind(m_cen, m_for, equal_var=False)

# Group B: barrel vs endcap, classified by the LEADING electron's |eta|.
# "Leading" here means the electron carrying more energy in the event.
lead_eta = np.where(opp.E1 >= opp.E2, opp.eta1.abs(), opp.eta2.abs())
m_bar = opp.M[lead_eta <  1.4].values
m_end = opp.M[lead_eta >= 1.4].values
t_det = stats.ttest_ind(m_bar, m_end, equal_var=False)

print("\n  Welch two-sample t-tests:")
print(f"    A (central vs forward) : t = {t_eta.statistic:+.2f}, p = {t_eta.pvalue:.2e}")
print(f"      means: central = {m_cen.mean():.4f} GeV (N = {len(m_cen):,})")
print(f"             forward = {m_for.mean():.4f} GeV (N = {len(m_for):,})")
print(f"    B (barrel  vs endcap ) : t = {t_det.statistic:+.2f}, p = {t_det.pvalue:.2e}")
print(f"      means: barrel  = {m_bar.mean():.4f} GeV (N = {len(m_bar):,})")
print(f"             endcap  = {m_end.mean():.4f} GeV (N = {len(m_end):,})")


# --- Figure 4: joint distributions of (E1, E2, M_ee) ---
# Three hexbin panels arranged horizontally.  Hexbin is essential at
# N ~ 57k: a literal scatter would be unreadable because of overplotting.
# Log colour scale to show both the bright cores and the faint tails.
# Gridsize 50 matches the report's hexagon size; the densest hexagons
# top out around 10^3 events with this resolution.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
hexbin_opts = dict(gridsize=50, cmap="magma", mincnt=1, bins="log")

# Panel 1: E1 vs E2.  Shows the energy-budget structure: low-energy events
# dominate, and simultaneously-high (E1, E2) configurations are rare.
hb1 = axes[0].hexbin(opp.E1, opp.E2, extent=[0, 250, 0, 250], **hexbin_opts)
axes[0].set_xlabel(r"$E_{1}\ \mathrm{[GeV]}$")
axes[0].set_ylabel(r"$E_{2}\ \mathrm{[GeV]}$")
fig.colorbar(hb1, ax=axes[0], label=r"$\mathrm{Events\ /\ hexagon}$")

# Panel 2: M_ee vs E1.  The bright horizontal band at M_ee ~ 91 GeV is the
# Z resonance.  The diagonal upper envelope is the kinematic bound M_ee <= E1 + E2.
hb2 = axes[1].hexbin(opp.E1, opp.M, extent=[0, 250, 0, 120], **hexbin_opts)
axes[1].set_xlabel(r"$E_{1}\ \mathrm{[GeV]}$")
axes[1].set_ylabel(r"$M_{ee}\ \mathrm{[GeV]}$")
fig.colorbar(hb2, ax=axes[1], label=r"$\mathrm{Events\ /\ hexagon}$")

# Panel 3: M_ee vs E2.  Same physics as panel 2 with the two electrons
# swapped.  Symmetric because (E1, E2) is arbitrary ordering.
hb3 = axes[2].hexbin(opp.E2, opp.M, extent=[0, 250, 0, 120], **hexbin_opts)
axes[2].set_xlabel(r"$E_{2}\ \mathrm{[GeV]}$")
axes[2].set_ylabel(r"$M_{ee}\ \mathrm{[GeV]}$")
fig.colorbar(hb3, ax=axes[2], label=r"$\mathrm{Events\ /\ hexagon}$")

for ax in axes:
    ax.minorticks_off()
plt.tight_layout()
save(fig, "25_q3_joint_distributions")


# --- Figure 5(a): correlation matrix heatmap ---
# Diverging red-blue colormap centered at zero so positive and negative
# correlations are visually distinguishable.  Numerical values printed in
# each cell.  White text on dark backgrounds for legibility.
fig, ax = plt.subplots(figsize=(4.0, 3.6))
im = ax.imshow(cor_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
labels = [r"$E_{1}$", r"$E_{2}$", r"$M_{ee}$"]
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.tick_params(top=False, right=False)
for i in range(3):
    for j in range(3):
        val    = cor_mat[i, j]
        colour = "white" if abs(val) > 0.55 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                color=colour, fontsize=11)
ax.minorticks_off()
cbar = fig.colorbar(im, ax=ax, pad=0.04, fraction=0.045)
cbar.set_label(r"$\mathrm{Pearson}\ r$")
cbar.outline.set_linewidth(0.6)
save(fig, "22_q3_corr_matrix")


# --- Figure 5(b): raw vs partial correlations ---
# Three pairs of bars showing the raw correlation (navy) vs the partial
# correlation conditioning on the third variable (wine).  The headline
# finding is the (E1, E2)|M_ee bar: raw r ~ -0.06 grows to partial -0.22
# once M_ee is controlled for.
pairs = [
    (0, 1, 2, r"$E_{1},E_{2}\,|\,M_{ee}$"),
    (0, 2, 1, r"$E_{1},M_{ee}\,|\,E_{2}$"),
    (1, 2, 0, r"$E_{2},M_{ee}\,|\,E_{1}$"),
]
x_pos    = np.arange(len(pairs))
bar_w    = 0.36
raw_vals = [cor_mat[i, j]   for i, j, _, _ in pairs]
par_vals = [partial_r(i, j) for i, j, _, _ in pairs]

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.bar(x_pos - bar_w/2, raw_vals, bar_w,
       color=NAVY, label=r"$\mathrm{raw}\ r$",
       edgecolor="black", linewidth=0.5)
ax.bar(x_pos + bar_w/2, par_vals, bar_w,
       color=WINE, label=r"$\mathrm{partial}\ r$",
       edgecolor="black", linewidth=0.5)
ax.axhline(0, color=SLATE, lw=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([lab for *_, lab in pairs], fontsize=10)
ax.set_ylabel(r"$\mathrm{Correlation}$")
ax.set_ylim(-0.35, 0.5)
ax.legend(loc="upper left")
ax.minorticks_off()
save(fig, "23_q3_partial_vs_raw")


# --- Figure 5(c): group means with 95% CIs for the two t-tests ---
# Four points: central, forward, barrel, endcap.  Vertical dotted line
# separates the two t-tests visually.  Labels "t-test A" and "t-test B"
# at the top group the points by test.
def mean_and_halfwidth(x, level=0.95):
    """Return (mean, 95% CI half-width) for a numerical 1-D array."""
    n      = len(x)
    mean   = x.mean()
    se     = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.5 + level/2, df=n-1)
    return mean, t_crit * se

groups = [
    ("central\n($|\\eta_{1,2}|<1.5$)",     m_cen, NAVY),
    ("forward",                            m_for, WINE),
    ("barrel\n(leading $|\\eta|<1.4$)",    m_bar, NAVY),
    ("endcap",                             m_end, WINE),
]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
x_positions = np.array([0.0, 1.0, 2.5, 3.5])
for x_, (label, data_, colour) in zip(x_positions, groups):
    mean, hw = mean_and_halfwidth(data_)
    ax.errorbar([x_], [mean], yerr=[hw],
                fmt="o", color=colour, ecolor=SLATE,
                ms=6, lw=0, elinewidth=1.0, capsize=3)
ax.set_xticks(x_positions)
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel(r"$\langle M_{ee}\rangle\ \mathrm{[GeV]}$")
ax.axvline(1.75, color=SLATE, lw=0.4, ls=":")
ax.set_ylim(25, 38)
ax.text(0.5, 37.5, r"$t$-test A", ha="center", fontsize=10, va="top")
ax.text(3.0, 37.5, r"$t$-test B", ha="center", fontsize=10, va="top")
ax.minorticks_off()
save(fig, "24_q3_ttest_means")


# ==============================================================================
# 7.  DONE
# ==============================================================================
print()
print("=" * 70)
print(f"All outputs written to: {OUT.resolve()}")
print("=" * 70)