import json
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from routes import predict_acd, predict_ace, predict_bcd, predict_bce

# racine du projet (ce fichier est dans src/) : les données sont dans data/, le modèle dans models/
ROOT = Path(__file__).resolve().parent.parent

# 1. Chargement et extraction des durées réelles
records = []
with open(ROOT / "data" / "traffic.jsonl", "r") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line.strip()))

df = pd.DataFrame(records)
df["dep_dt"] = df["depature"].apply(lambda s: datetime.strptime(s, "%H:%M"))
df["arr_dt"] = df["arrival"].apply(lambda s: datetime.strptime(s, "%H:%M"))
df["duration_min"] = (df["arr_dt"] - df["dep_dt"]).dt.total_seconds() / 60.0
df["t"] = df["dep_dt"].dt.hour + df["dep_dt"].dt.minute / 60.0

# 2. Les fonctions mathématiques calquées sur la forme réelle de chaque route sont dans routes.py


# 3. Optimisation locale Fortuna
def fit_model(func, t_vals, y_vals, init_theta, step_sizes, n_steps=50000):
    curr_theta = np.array(init_theta, dtype=float)
    curr_loss = np.mean((func(t_vals, curr_theta) - y_vals) ** 2)
    best_theta = curr_theta.copy()
    best_loss = curr_loss

    for _ in range(n_steps):
        cand = curr_theta + np.random.normal(0, step_sizes)
        cand_loss = np.mean((func(t_vals, cand) - y_vals) ** 2)
        if cand_loss < curr_loss:
            curr_loss = cand_loss
            curr_theta = cand
            if cand_loss < best_loss:
                best_loss = cand_loss
                best_theta = cand.copy()

    return best_theta, np.sqrt(best_loss)


# 4. Entraînement des 4 routes
models = {}

# Route A->C->D (cuvette)
sub_acd = df[df["road"] == "A->C->D"]
models["A->C->D"], rmse_acd = fit_model(
    predict_acd, sub_acd["t"].values, sub_acd["duration_min"].values,
    init_theta=[100.0, 25.0, 11.5, 1.8], step_sizes=[0.1, 0.1, 0.05, 0.05]
)
print(f"[A->C->D] RMSE: {rmse_acd:.2f} min")

# Route A->C->E (plate)
sub_ace = df[df["road"] == "A->C->E"]
models["A->C->E"], rmse_ace = fit_model(
    predict_ace, sub_ace["t"].values, sub_ace["duration_min"].values,
    init_theta=[98.0], step_sizes=[0.1]
)
print(f"[A->C->E] RMSE: {rmse_ace:.2f} min")

# Route B->C->D (dents de scie + cuvette)
sub_bcd = df[df["road"] == "B->C->D"]
models["B->C->D"], rmse_bcd = fit_model(
    predict_bcd, sub_bcd["t"].values, sub_bcd["duration_min"].values,
    init_theta=[45.0, 60.0, 0.25, 2.5], step_sizes=[0.1, 0.1, 0.01, 0.05]
)
print(f"[B->C->D] RMSE: {rmse_bcd:.2f} min")

# Route B->C->E (dents de scie régulières)
sub_bce = df[df["road"] == "B->C->E"]
models["B->C->E"], rmse_bce = fit_model(
    predict_bce, sub_bce["t"].values, sub_bce["duration_min"].values,
    init_theta=[70.0, 60.0, 0.25], step_sizes=[0.1, 0.1, 0.01]
)
print(f"[B->C->E] RMSE: {rmse_bce:.2f} min")

# 5. Sauvegarde du fichier binaire
with open(ROOT / "models" / "traffic_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("models/traffic_models.pkl généré avec succès !")
