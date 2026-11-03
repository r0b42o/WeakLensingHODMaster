"""NN-basierte Theorie fuer Galaxy-Galaxy-Lensing (G2L) und Likelihood fuer den Fit.

Pipeline pro Parametervektor:
  (Kosmologie, HOD je Lens-Bin) -> NN -> P_gm(k, z)
  -> Limber: C_kg(ell) je (Lens-Bin, Source-Bin) -> Hankel: gamma_t(theta)

Zwei Emulator-Varianten (automatisch erkannt an der Zahl der NN-Eingaenge):
  * 11 Eingaenge (Variante B, Standard): NN kennt z. Es liegt in `NN_GM/gm_emulator.npz`
    (erzeugt von `export_gm_emulator.py` aus dem trainierten Keras-Modell in `NN_GM/`).
  * 10 Eingaenge (Variante A, alt): NN nur bei z=0, es wird mit D(z)^2 skaliert (nur grobe Naeherung).
"""
import glob
import os
import re

import numpy as np
import pyccl as ccl
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf, jv

TRAPZ = getattr(np, "trapezoid", None) or np.trapz

BASE = "/home/rhoffmann/MA"
TOMO_DIR = f"{BASE}/Tomobins"
NN_DIR = f"{BASE}/Modelling/NN_GM"
EMU_FILE_Z = f"{NN_DIR}/gm_emulator.npz"
EMU_FILE_Z0 = f"{BASE}/Modelling/gm_emulator.npz"
EMU_FILE = EMU_FILE_Z if os.path.exists(EMU_FILE_Z) else EMU_FILE_Z0
MEAS_FILE = f"{BASE}/Modelling/Compare/ggl_measurements.npz"
COV_FILE = f"{BASE}/G2L/onecov_run/output/covariance_list.dat"

# Reihenfolge der Lens-Bins in der OneCovariance-Rechnung (Index 1..5, siehe G2LShearProfile.ipynb)
ONECOV_LENS_ORDER = ["BGS_0.1-0.2", "BGS_0.2-0.3", "BGS_0.3-0.4", "LRG_0.4-0.6", "LRG_0.6-0.8"]

try:
    C_OVER_H0_H = ccl.physical_constants.CLIGHT_HMPC  # c/H0 in Mpc/h
except AttributeError:
    C_OVER_H0_H = 2997.92458

# ---------------------------------------------------------------------------
# Parameter und Priors (= Trainingsintervalle des NN, siehe GenerateNNTrainData.py)
# ---------------------------------------------------------------------------
COSMO_NAMES = ["Om_c", "Om_b", "h", "sigma_8", "n_s"]
HOD_NAMES = ["alpha", "sigma", "log10_Mth", "log10_Mprime", "beta"]
PRIOR_COSMO = {"Om_c": (0.1, 0.6), "Om_b": (0.04, 0.06), "h": (0.64, 0.82),
               "sigma_8": (0.6, 1.0), "n_s": (0.84, 1.1)}
# Massen: flach in log10(M) ueber das Trainingsintervall (Mth 1e10-1e15, Mprime 1e10-1e17)
PRIOR_HOD = {"alpha": (0.0, 1.0), "sigma": (0.01, 1.0), "log10_Mth": (10.0, 15.0),
             "log10_Mprime": (10.0, 17.0), "beta": (0.1, 2.0)}
Z_TRAIN_RANGE = (0.0, 1.1)  # z-Intervall des Trainingsdatensatzes (GenerateNNTrainData.py)
LABEL_COSMO = [r"$\Omega_c$", r"$\Omega_b$", r"$h$", r"$\sigma_8$", r"$n_s$"]
LABEL_HOD = [r"$\alpha$", r"$\sigma$", r"$\log M_{\rm th}$", r"$\log M'$", r"$\beta$"]

# Fiduzialer Startpunkt (alte Einzelkonfiguration aus G2LTheory)
FIDUCIAL_COSMO = [0.28, 0.05, 0.67, 0.8, 0.96]
FIDUCIAL_HOD = [1.0, 0.2, 12.0, np.log10(20e12), 1.0]


class ParamSpace:
    """Globale Kosmologie (5) + eigene HOD (5) je Lens-Bin."""

    def __init__(self, lens_names):
        self.lens_names = list(lens_names)
        self.n_lens = len(self.lens_names)
        self.names = list(COSMO_NAMES) + [f"{l}:{p}" for l in self.lens_names for p in HOD_NAMES]
        self.labels = list(LABEL_COSMO) + [
            f"{lab}\n{l.replace('_', ' ')}" for l in self.lens_names for lab in LABEL_HOD
        ]
        lo = [PRIOR_COSMO[n][0] for n in COSMO_NAMES] + [PRIOR_HOD[p][0] for _ in self.lens_names for p in HOD_NAMES]
        hi = [PRIOR_COSMO[n][1] for n in COSMO_NAMES] + [PRIOR_HOD[p][1] for _ in self.lens_names for p in HOD_NAMES]
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.ndim = len(self.names)

    def split(self, theta):
        theta = np.asarray(theta, dtype=float)
        return theta[:5], theta[5:].reshape(self.n_lens, 5)

    def in_prior(self, theta):
        theta = np.asarray(theta)
        return bool(np.all((theta >= self.lo) & (theta <= self.hi)))

    def fiducial(self):
        return np.array(FIDUCIAL_COSMO + FIDUCIAL_HOD * self.n_lens, dtype=float)

    def draw_prior(self, n, rng=None):
        rng = np.random.default_rng(rng)
        return self.lo + (self.hi - self.lo) * rng.random((n, self.ndim))


# ---------------------------------------------------------------------------
# n(z)
# ---------------------------------------------------------------------------
def load_nz(tomo_dir=TOMO_DIR, zmax=2.0, nz=2001):
    """Liest die n(z) der KiDS-Source-Bins und DESI-Lens-Bins auf ein gemeinsames Gitter."""
    zs = np.linspace(0, zmax, nz)

    def on_grid(z, n):
        n = np.interp(zs, z, n, left=0, right=0)
        return n / TRAPZ(n, zs)

    sources = {}
    for f in sorted(glob.glob(os.path.join(tomo_dir, "KiDS", "source_bin*_nz.ascii"))):
        k = int(re.search(r"source_bin(\d+)_nz", f).group(1))
        sources[k] = np.loadtxt(f, unpack=True)
    lenses = {}
    for f in glob.glob(os.path.join(tomo_dir, "DESI", "*_nz.ascii")):
        lenses[os.path.basename(f).replace("_nz.ascii", "")] = np.loadtxt(f, unpack=True)
    lenses = dict(sorted(lenses.items(), key=lambda kv: TRAPZ(kv[1][0] * kv[1][1], kv[1][0])))
    return (zs, {k: on_grid(*v) for k, v in lenses.items()},
            {k: on_grid(*v) for k, v in sources.items()})


# ---------------------------------------------------------------------------
# NN-Emulator (reines NumPy, Gewichte aus export_gm_emulator.py)
# ---------------------------------------------------------------------------
def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


_SELU_ALPHA, _SELU_SCALE = 1.6732632423543772, 1.0507009873554805
ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0),
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))),
    "selu": lambda x: _SELU_SCALE * np.where(x > 0, x, _SELU_ALPHA * np.expm1(np.minimum(x, 0))),
    "gelu": _gelu,
    "swish": lambda x: x * _sigmoid(x),
    "sigmoid": _sigmoid,
    "tanh": np.tanh,
}


def get_activation(name):
    """'leaky_relu:0.3' (Steigung fuer x<0) oder Name aus ACTIVATIONS."""
    name = str(name)
    if name.startswith("leaky_relu"):
        alpha = float(name.split(":")[1]) if ":" in name else 0.3
        return lambda x: np.where(x > 0, x, alpha * x)
    return ACTIVATIONS[name]


class GMEmulator:
    """Galaxie-Materie-Leistungsspektrum auf 50 k-Werten (1/Mpc), k = geomspace(1e-2, 1e2, 50).

    Eingaenge: [Om_c, Om_b, h, sigma_8, n_s, alpha, sigma, log10 Mth, log10 Mprime, beta] (+ z bei Variante B).
    """

    def __init__(self, path=EMU_FILE):
        d = np.load(path)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]
        self.act = [get_activation(a) for a in d["activations"]]
        self.feat_mean, self.feat_sigma = d["feat_mean"], d["feat_sigma"]
        self.n_in = self.feat_mean.shape[1]
        self.has_z = self.n_in == 11
        self.z_range = tuple(d["z_range"]) if "z_range" in d.files else Z_TRAIN_RANGE
        # (3, 1, 50) -> (1, 3, 50); Reihenfolge der 3 Komponenten: 1h, 2h, total
        self.targ_mean = d["targ_mean"].transpose(1, 0, 2)
        self.targ_sigma = d["targ_sigma"].transpose(1, 0, 2)
        self.ks = d["ks"]
        self.lnk = np.log(self.ks)

    def ln_pk(self, features):
        """features: (N, n_in). Rueckgabe: ln P in (N, 3, 50) fuer (1h, 2h, total)."""
        h = (np.atleast_2d(features) - self.feat_mean) / self.feat_sigma
        for W, b, a in zip(self.W, self.b, self.act):
            h = a(h @ W + b)
        return h.reshape(-1, 3, 50) * self.targ_sigma + self.targ_mean


def _interp_extrap(x, xp, fp):
    """Kubischer Spline innerhalb von xp, ausserhalb Potenzgesetz mit der mittleren Steigung der Randpunkte."""
    spline = CubicSpline(xp, fp)
    out = spline(np.clip(x, xp[0], xp[-1]))
    n_fit = 5
    slope_lo = np.polyfit(xp[:n_fit], fp[:n_fit], 1)[0]
    slope_hi = np.polyfit(xp[-n_fit:], fp[-n_fit:], 1)[0]
    lo, hi = x < xp[0], x > xp[-1]
    out = np.where(lo, fp[0] + slope_lo * (x - xp[0]), out)
    out = np.where(hi, fp[-1] + slope_hi * (x - xp[-1]), out)
    return out


# ---------------------------------------------------------------------------
# Theorie
# ---------------------------------------------------------------------------
class G2LTheory:
    """gamma_t(theta) fuer alle (Lens, Source)-Kombinationen aus einem Parametervektor.

    combos        : Liste (lens_name, source_bin)
    thetas_arcmin : Liste von Arrays, Winkel je Kombination [arcmin]
    """

    def __init__(self, combos, thetas_arcmin, zs, n_lenses, n_sources, emulator=None,
                 ell_range=(0.5, 3e6), n_ell_coarse=110, n_ell_fine=12000, dz_lens=0.004,
                 taper=100.0, growth_scaling=True, n_z_nodes=8):
        self.combos = list(combos)
        self.lens_names = list(dict.fromkeys(l for l, _ in self.combos))
        self.emu = emulator if emulator is not None else GMEmulator()
        self.zs, self.n_lenses, self.n_sources = zs, n_lenses, n_sources
        self.growth_scaling = growth_scaling
        self.n_z_nodes = n_z_nodes

        # z-Gitter der Lens-Integration (nur dort, wo n_l(z) nennenswert ist)
        zmax_l = max(zs[np.where(n > 1e-4 * n.max())[0][-1]] for n in n_lenses.values()) + 0.05
        n_zl = int(np.ceil(zmax_l / dz_lens)) + 1
        self.z_l = np.linspace(0, zmax_l, n_zl)[1:]
        dz = self.z_l[1] - self.z_l[0]
        self.w_l = np.full(self.z_l.size, dz)
        self.w_l[-1] *= 0.5
        self.nl_z = {l: np.interp(self.z_l, zs, n_lenses[l]) for l in self.lens_names}

        # z-Stuetzstellen des NN je Lens-Bin: nur dort, wo n_l(z) relevant ist (ln P wird dazwischen kubisch interpoliert)
        self.z_nodes = {}
        z_max_nn = self.emu.z_range[1] if self.emu.has_z else np.inf
        for l in self.lens_names:
            sig = np.where(n_lenses[l] > 1e-4 * n_lenses[l].max())[0]
            lo = max(zs[sig[0]] - 0.02, self.emu.z_range[0] if self.emu.has_z else 0.0)
            hi = min(zs[sig[-1]] + 0.02, z_max_nn)
            self.z_nodes[l] = np.linspace(lo, hi, n_z_nodes)

        # ell-Gitter und Hankel-Kern gamma_t = int dln(ell) ell^2 C(ell) J_2(ell theta) / (2 pi).
        # Der Integrand oszilliert fuer ell*theta >> 1; die glatte Daempfung exp(-(ell*theta/taper)^2)
        # vermeidet Aliasing (Fehler < 0.1 % gegen ein 1e6-Punkte-Referenzgitter ohne Daempfung bis 1e8).
        self.ell_c = np.geomspace(*ell_range, n_ell_coarse)
        self.ell_f = np.geomspace(*ell_range, n_ell_fine)
        self.ln_ell_c, self.ln_ell_f = np.log(self.ell_c), np.log(self.ell_f)
        w = np.full(n_ell_fine, self.ln_ell_f[1] - self.ln_ell_f[0])
        w[[0, -1]] *= 0.5
        self.kernels = []
        for th in thetas_arcmin:
            th_rad = np.deg2rad(np.asarray(th) / 60.0)
            K = np.empty((th_rad.size, n_ell_fine))
            for i, t in enumerate(th_rad):
                K[i] = (w * self.ell_f**2 * jv(2, self.ell_f * t) / (2 * np.pi)
                        * np.exp(-(self.ell_f * t / taper) ** 2))
            self.kernels.append(K)
        self.n_data = int(sum(K.shape[0] for K in self.kernels))

    # -- Hintergrund ------------------------------------------------------------
    def background(self, cosmo_vals):
        Om_c, Om_b, h, s8, ns = cosmo_vals
        cosmo = ccl.Cosmology(Omega_c=Om_c, Omega_b=Om_b, h=h, sigma8=s8, n_s=ns)
        chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + self.zs))
        chi_l = np.interp(self.z_l, self.zs, chi)
        if self.growth_scaling and not self.emu.has_z:
            D = ccl.growth_factor(cosmo, 1 / (1 + self.z_l)) / ccl.growth_factor(cosmo, 1.0)
        else:
            D = np.ones_like(self.z_l)
        return {"chi": chi, "chi_l": chi_l, "D2": D**2, "Om": Om_c + Om_b, "h": h}

    def lensing_kernel(self, bg, src):
        """q_s(z) = 3/2 Om (1+z) / (c/H0)^2 * int_z^inf dz' n_s(z') (1 - chi(z)/chi(z')) auf dem Lens-z-Gitter."""
        chi = bg["chi"]
        inv = np.divide(1.0, chi, out=np.zeros_like(chi), where=chi > 0)
        ns = self.n_sources[src]
        cum_n = cumulative_trapezoid(ns, self.zs, initial=0)
        cum_ni = cumulative_trapezoid(ns * inv, self.zs, initial=0)
        Q = (cum_n[-1] - cum_n) - chi * (cum_ni[-1] - cum_ni)
        Q_l = np.interp(self.z_l, self.zs, Q)
        c_h0 = C_OVER_H0_H / bg["h"]
        return 1.5 * bg["Om"] * (1 + self.z_l) / c_h0**2 * Q_l

    # -- Leistungsspektrum --------------------------------------------------------
    def pk_matrix(self, ln_pk0, bg):
        """P_gm(ell / chi(z), z) auf dem (z_l, ell_coarse)-Gitter.

        Nur fuer Variante A (NN nur bei z=0): P(k,z) = P(k,0) * D(z)^2, grobe Naeherung.
        Variante B (NN mit z) nutzt `pk_matrix_z`."""
        lnk = np.log(self.ell_c[None, :] / bg["chi_l"][:, None])
        ln_p = _interp_extrap(lnk, self.emu.lnk, ln_pk0)
        return np.exp(ln_p) * bg["D2"][:, None]

    def nn_nodes(self, cosmo_vals, hod_vals):
        """Ein gemeinsamer NN-Vorwaertslauf fuer alle Lens-Bins und z-Knoten: dict {lens: ln P_total (n_nodes, 50)}."""
        rows = [np.column_stack([np.tile(np.concatenate([cosmo_vals, hod_vals[i]]), (self.z_nodes[l].size, 1)),
                                 self.z_nodes[l]]) for i, l in enumerate(self.lens_names)]
        ln_pk = self.emu.ln_pk(np.vstack(rows))[:, 2, :]
        out, pos = {}, 0
        for l, r in zip(self.lens_names, rows):
            out[l] = ln_pk[pos:pos + len(r)]
            pos += len(r)
        return out

    def pk_matrix_z(self, lens, ln_pk, bg):
        """Variante B: P_gm(ell / chi(z), z) auf dem (z_l, ell_coarse)-Gitter fuer einen Lens-Bin aus ln_pk = nn_nodes()[lens].

        Das NN wird nur an `n_z_nodes` Rotverschiebungen ausgewertet; ln P wird bei festem ell
        kubisch in z auf das feine z-Gitter interpoliert (ausserhalb des Knotenbereichs konstant fortgesetzt)."""
        zn = self.z_nodes[lens]
        chi_n = np.interp(zn, self.zs, bg["chi"])
        ln_p_nodes = np.empty((zn.size, self.ell_c.size))
        for j in range(zn.size):
            ln_p_nodes[j] = _interp_extrap(np.log(self.ell_c / chi_n[j]), self.emu.lnk, ln_pk[j])
        z_eval = np.clip(self.z_l, zn[0], zn[-1])
        return np.exp(CubicSpline(zn, ln_p_nodes, axis=0)(z_eval))

    def c_kg(self, pmat, bg, lens, src):
        weight = self.w_l * self.nl_z[lens] * self.lensing_kernel(bg, src) / bg["chi_l"]
        return weight @ pmat

    def gamma_t_from_c(self, c_coarse):
        """c_coarse: (n_combos, n_ell_coarse) -> Liste von gamma_t-Arrays."""
        c_fine = np.exp(CubicSpline(self.ln_ell_c, np.log(c_coarse), axis=1)(self.ln_ell_f))
        return [K @ c for K, c in zip(self.kernels, c_fine)]

    # -- Oeffentliche Schnittstelle -------------------------------------------------
    def predict(self, cosmo_vals, hod_vals, pmats=None):
        """Rueckgabe: (Liste gamma_t je Kombination, dict C_kg_coarse je Kombination).

        hod_vals: (n_lens, 5) in der Reihenfolge self.lens_names.
        pmats   : optional dict {lens: P-Matrix}, ersetzt das NN (z. B. echtes Halomodell zur Validierung)."""
        bg = self.background(cosmo_vals)
        if pmats is None:
            hod_vals = np.asarray(hod_vals)
            if self.emu.has_z:
                nodes = self.nn_nodes(np.asarray(cosmo_vals), hod_vals)
                pmats = {l: self.pk_matrix_z(l, nodes[l], bg) for l in self.lens_names}
            else:
                feats = np.hstack([np.tile(cosmo_vals, (len(self.lens_names), 1)), hod_vals])
                ln_pk0 = self.emu.ln_pk(feats)[:, 2, :]
                pmats = {l: self.pk_matrix(ln_pk0[i], bg) for i, l in enumerate(self.lens_names)}
        c = np.array([self.c_kg(pmats[l], bg, l, s) for l, s in self.combos])
        return self.gamma_t_from_c(c), dict(zip(self.combos, c))

    def model_vector(self, theta, space):
        cosmo_vals, hod_vals = space.split(theta)
        order = [space.lens_names.index(l) for l in self.lens_names]
        gts, _ = self.predict(cosmo_vals, hod_vals[order])
        return np.concatenate(gts)


# ---------------------------------------------------------------------------
# Daten + Kovarianz
# ---------------------------------------------------------------------------
class GGLData:
    """gamma_t-Messungen und volle OneCovariance-Kovarianz (theta und Bins) fuer alle Kombinationen."""

    def __init__(self, meas_file=MEAS_FILE, cov_file=COV_FILE, theta_min=None, theta_max=None,
                 lens_order=ONECOV_LENS_ORDER):
        meas = np.load(meas_file)
        self.combos, self.theta, self.gt, self.sig_diag, self.slices = [], [], [], [], []
        masks, pos = [], 0
        for key in meas.files:
            if not key.startswith("gammat_"):
                continue
            lens, sb = key[len("gammat_"):].rsplit("_S", 1)
            th = meas[f"theta_{lens}_S{sb}"]
            m = np.ones(th.size, bool)
            if theta_min is not None:
                m &= th >= theta_min
            if theta_max is not None:
                m &= th <= theta_max
            if not m.any():
                continue
            self.combos.append((lens, int(sb)))
            masks.append(m)
            self.theta.append(th[m])
            self.gt.append(meas[key][m])
            self.sig_diag.append(meas[f"sigma_onecov_{lens}_S{sb}"][m])
            self.slices.append(slice(pos, pos + int(m.sum())))
            pos += int(m.sum())
        self.lens_names = list(dict.fromkeys(l for l, _ in self.combos))
        self.d = np.concatenate(self.gt)
        self.sig = np.concatenate(self.sig_diag)
        self.n = self.d.size
        self.cov = self._build_cov(cov_file, masks, lens_order)
        self._cho = cho_factor(self.cov, lower=True)

    def _build_cov(self, cov_file, masks, lens_order):
        rows = []
        with open(cov_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.split()
                rows.append((float(p[1]), int(p[5]), int(p[6]), float(p[2]), int(p[7]), int(p[8]), float(p[9])))
        rows = np.array(rows)
        th_unique = np.unique(rows[:, 0])
        # Position im Datenvektor: (lens_index, source, theta-Bin) -> Index
        lookup = {}
        for (lens, sb), m, sl in zip(self.combos, masks, self.slices):
            idx = np.arange(sl.start, sl.stop)
            bins = np.where(m)[0]
            for b, i in zip(bins, idx):
                lookup[(lens_order.index(lens) + 1, sb, b)] = i
        cov = np.zeros((self.n, self.n))
        th_bin = {t: i for i, t in enumerate(th_unique)}
        for t1, li, sj, t2, lk, sl_, c in rows:
            a = lookup.get((int(li), int(sj), th_bin[t1]))
            b = lookup.get((int(lk), int(sl_), th_bin[t2]))
            if a is not None and b is not None:
                cov[a, b] = c
        if not np.allclose(cov, cov.T, rtol=1e-6, atol=0):
            cov = 0.5 * (cov + cov.T)
        # Messwerte sind m-kalibriert und boost-korrigiert; sigma_onecov enthaelt diese Faktoren s = M * boost
        s = self.sig / np.sqrt(np.diag(cov))
        if not (0.5 < s.min() and s.max() < 3.0):
            raise ValueError(f"Zuordnung Lens-Bin/Kovarianz pruefen, Skalierung s in [{s.min():.2f}, {s.max():.2f}]")
        self.cov_scale = s
        return cov * np.outer(s, s)

    def chi2(self, model):
        r = self.d - model
        return float(r @ cho_solve(self._cho, r))


class G2LPosterior:
    """ln p(theta | Daten) mit flachen Priors und Gauss-Likelihood mit voller Kovarianz."""

    def __init__(self, data, theory, space):
        self.data, self.theory, self.space = data, theory, space

    def lnlike(self, theta):
        return -0.5 * self.data.chi2(self.theory.model_vector(theta, self.space))

    def __call__(self, theta):
        if not self.space.in_prior(theta):
            return -np.inf
        try:
            ll = self.lnlike(theta)
        except Exception:
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf


def build(theta_min=None, theta_max=None, **theory_kwargs):
    """Alles zusammenbauen: Daten, Theorie, Parameterraum, Posterior."""
    zs, n_lenses, n_sources = load_nz()
    data = GGLData(theta_min=theta_min, theta_max=theta_max)
    theory = G2LTheory(data.combos, data.theta, zs, n_lenses, n_sources, **theory_kwargs)
    space = ParamSpace(data.lens_names)
    return data, theory, space, G2LPosterior(data, theory, space)
