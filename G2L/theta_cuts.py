"""Angular cuts and plotting helpers for the GGL measurement notebook."""

import json
import warnings
from pathlib import Path

import numpy as np


BOOST_MIN = 0.95
BOOST_MAX = 1.05
THETA_MAX_ARCMIN = 150.0


def determine_cut(theta, boost, theta_max=THETA_MAX_ARCMIN):
    """Select the smallest effective bin radius with a valid 5% boost."""
    theta = np.asarray(theta, dtype=float)
    boost = np.asarray(boost, dtype=float)
    if theta.ndim != 1 or theta.shape != boost.shape:
        raise ValueError("theta and boost must be equally sized 1D arrays")
    if not np.isfinite(theta_max) or theta_max <= 0:
        raise ValueError("theta_max must be finite and positive")
    eligible = (
        np.isfinite(theta) & (theta > 0) & (theta <= theta_max)
        & np.isfinite(boost) & (boost >= BOOST_MIN) & (boost <= BOOST_MAX)
    )
    if not eligible.any():
        return None
    return float(np.min(theta[eligible]))


def cut_mask(theta, cut):
    theta = np.asarray(theta, dtype=float)
    if cut["theta_min_arcmin"] is None:
        return np.zeros(theta.shape, dtype=bool)
    return (
        np.isfinite(theta) & (theta >= cut["theta_min_arcmin"])
        & (theta <= cut["theta_max_arcmin"])
    )


def make_cuts(results, lens_cats, random_cats, output_path):
    """Calculate and save cuts without changing any measured arrays."""
    if not results:
        raise ValueError("No lens-source measurements available for angular cuts")
    cuts = {}
    entries = []
    for (lens, source), (theta, _, _, _, ng, rg) in results.items():
        lens_sum = lens_cats[lens].sumw
        random_sum = random_cats[lens].sumw
        if not np.isfinite(lens_sum) or not np.isfinite(random_sum) or lens_sum <= 0 or random_sum <= 0:
            raise ValueError(f"Invalid catalogue weight for {lens}, source {source}")
        lens_weight = np.asarray(ng.weight, dtype=float) / lens_sum
        random_weight = np.asarray(rg.weight, dtype=float) / random_sum
        valid = (
            np.isfinite(lens_weight) & np.isfinite(random_weight)
            & (lens_weight > 0) & (random_weight > 0)
        )
        boost = np.full(lens_weight.shape, np.nan)
        np.divide(lens_weight, random_weight, out=boost, where=valid)
        theta_min = determine_cut(theta, boost)
        cut = {
            "lens": lens,
            "source_bin": int(source),
            "theta_min_arcmin": theta_min,
            "theta_max_arcmin": THETA_MAX_ARCMIN,
            "status": "ok" if theta_min is not None else "no_valid_bin",
        }
        cuts[(lens, source)] = cut
        entries.append(cut)
        if theta_min is None:
            warnings.warn(f"{lens} x source {source}: kein gueltiges Boost-Bin bis 150 arcmin", stacklevel=2)
        elif np.any(cut_mask(theta, cut) & ~valid):
            warnings.warn(f"{lens} x source {source}: ungueltiger Boost innerhalb des Winkelbereichs", stacklevel=2)
    config = {
        "version": 1,
        "theta_unit": "arcmin",
        "theta_definition": "TreeCorr meanr effective bin radius",
        "bounds_inclusive": True,
        "selection": "first finite boost in [boost_min, boost_max] at theta <= theta_max; later deviations are retained",
        "boost_min": BOOST_MIN,
        "boost_max": BOOST_MAX,
        "combinations": entries,
    }
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, allow_nan=False)
        file.write("\n")
    return cuts


def plot_cut_errorbar(ax, theta, y, cut, *, yerr=None, x=None, **kwargs):
    """Plot selected bins normally and excluded bins as grey markers."""
    theta = np.asarray(theta)
    x = theta if x is None else np.asarray(x)
    y = np.asarray(y)
    mask = cut_mask(theta, cut)
    errors = None if yerr is None else np.asarray(yerr)
    line = ax.errorbar(x[mask], y[mask], yerr=None if errors is None else errors[..., mask], **kwargs)
    grey = dict(kwargs)
    grey.update(color="0.65", ecolor="0.65", linestyle="none", alpha=0.7, label="_nolegend_")
    ax.errorbar(x[~mask], y[~mask], yerr=None if errors is None else errors[..., ~mask], **grey)
    return line
