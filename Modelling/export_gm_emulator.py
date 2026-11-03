"""Exportiert das trainierte Galaxie-Materie-NN (Keras) als reine NumPy-Gewichte.

Damit laeuft der Emulator in jeder Umgebung ohne TensorFlow (z. B. im Kernel 'MA').

In den Ordner NN_GM kopieren (aus G3LHalo_python):
  * das beste Modell            ../NN_builds/NNz_<Zeitstempel>.keras
  * Processing_vectors_parameter_<Datensatz>_sl.json   (aus ../Data/Rescaling/)
  * Processing_vectors_data_<Datensatz>_sl.json

Aufruf (Umgebung mit TensorFlow, z. B. System-Python):
    python export_gm_emulator.py                       # nimmt die neuesten Dateien in NN_GM
    python export_gm_emulator.py --model NN_GM/NNz_X.keras --dir NN_GM

Schreibt NN_GM/gm_emulator.npz, das g2l_nn.py automatisch bevorzugt.
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2l_nn import NN_DIR, Z_TRAIN_RANGE, get_activation

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=NN_DIR)
ap.add_argument("--model", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()


def newest(pattern):
    files = sorted(glob.glob(os.path.join(args.dir, pattern)), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"Nichts gefunden: {os.path.join(args.dir, pattern)}")
    return files[-1]


model_path = args.model or newest("*.keras")
feat_path = newest("Processing_vectors_parameter_*_sl.json")
targ_path = newest("Processing_vectors_data_*_sl.json")
out = args.out or os.path.join(args.dir, "gm_emulator.npz")
print("Modell:", model_path, "\nFeature-Vektoren:", feat_path, "\nTarget-Vektoren:", targ_path)

model = keras.models.load_model(model_path, compile=False)

weights, biases, activations = [], [], []
for layer in model.layers:
    if isinstance(layer, keras.layers.Dense):
        w, b = layer.get_weights()
        weights.append(w)
        biases.append(b)
        activations.append(layer.get_config()["activation"])
    elif isinstance(layer, keras.layers.LeakyReLU):
        # Optuna-Modell: Dense(linear) gefolgt von LeakyReLU
        if activations[-1] != "linear":
            raise NotImplementedError("LeakyReLU nach nichtlinearem Dense")
        activations[-1] = f"leaky_relu:{float(layer.get_config()['alpha'])}"
    elif isinstance(layer, keras.layers.Dropout):
        continue  # bei der Inferenz inaktiv
    else:
        raise NotImplementedError(f"Layer-Typ {type(layer)} wird nicht unterstuetzt")

with open(feat_path) as f:
    feat = json.load(f)
with open(targ_path) as f:
    targ = json.load(f)
feat_mean = np.array(feat["mean"]).reshape(1, -1)
n_in = model.input_shape[1]
if feat_mean.shape[1] != n_in:
    raise ValueError(f"Modell hat {n_in} Eingaenge, Feature-Vektoren {feat_mean.shape[1]}")

payload = {
    "n_layers": np.array(len(weights)),
    "activations": np.array(activations),
    "feat_mean": feat_mean,
    "feat_sigma": np.array(feat["sigma"]).reshape(1, -1),
    "targ_mean": np.array(targ["mean"]),    # (3, 1, 50): 1h, 2h, total
    "targ_sigma": np.array(targ["sigma"]),
    "ks": np.geomspace(1e-2, 1e2),
    "z_range": np.array(Z_TRAIN_RANGE),
    "source_model": np.array(os.path.basename(model_path)),
}
for i, (w, b) in enumerate(zip(weights, biases)):
    payload[f"W{i}"] = w
    payload[f"b{i}"] = b
np.savez(out, **payload)
print("gespeichert:", out, "| Eingaenge:", n_in, "| Aktivierungen:", sorted(set(activations)))

# Kontrolle: NumPy-Vorwaertslauf gegen Keras (auf standardisierten Eingaben, wie im Training)
acts = [get_activation(a) for a in activations]
rng = np.random.default_rng(0)
x = rng.normal(size=(16, n_in))
ref = model.predict(x, verbose=0)
h = x
for w, b, a in zip(weights, biases, acts):
    h = a(h @ w + b)
print("max. Abweichung NumPy vs. Keras:", np.max(np.abs(h - ref)))
