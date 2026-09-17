# Notations
# ---------
# t : l'heure de départ en heures décimales, dans un tableau numpy (ex. 13:17 -> 13 + 17/60 = 13.283).
#
# theta : les paramètres du modèle, c'est-à-dire les nombres que Fortuna ajuste.
#   La formule d'une route est fixée à la main, seul theta change pendant l'entraînement.
#   Chaque route a son propre theta, un tableau numpy de floats dont la taille dépend de la formule
#   (10 paramètres au total) :
#     A->C->D : [quad_w, base]                    parabole centrée sur 11h30
#     A->C->E : [const]                           constante
#     B->C->D : [base, saw_amp, shift, quad_w]    dents de scie + parabole centrée sur 11h30
#     B->C->E : [base, saw_amp, shift]            dents de scie
#   base, const et saw_amp sont en minutes ; shift en heures (shift ≈ 0.25 : le "ferry" part
#   vers le quart de chaque heure) ; quad_w en minutes par heure².
#   Les 4 theta sont rangés dans un dict {route: theta}, sauvegardé dans traffic_models.pkl,
#   par exemple {"A->C->E": array([97.84]), "B->C->E": array([72.53, 59.53, 0.26]), ...}.
#
# y_hat (ŷ) : la sortie d'une fonction predict_xxx(t, theta), soit la durée du trajet PRÉDITE en minutes.
#   C'est un tableau numpy de même taille que t : une durée prédite par heure de départ.
#   - ici, t contient toutes les heures de départ d'une route, et ŷ = func(t_vals, theta) est comparé
#     à y, les durées RÉELLES (arrivée - départ, colonne duration_min) : la perte est mean((ŷ - y)²) ;
#   - dans best_route (routes.py), t ne contient qu'une seule heure, donc ŷ n'a qu'une valeur, lue avec [0].

import json
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Parsing with Datetime & Feature Extraction
# ---------------------------------------------------------
records = []
with open("data/traffic.jsonl", "r") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line.strip()))

df = pd.DataFrame(records)

# Parse departure and arrival timestamps to calculate exact trip duration
df["dep_dt"] = df["depature"].apply(lambda s: datetime.strptime(s, "%H:%M"))
df["arr_dt"] = df["arrival"].apply(lambda s: datetime.strptime(s, "%H:%M"))
df["duration_min"] = (df["arr_dt"] - df["dep_dt"]).dt.total_seconds() / 60.0

# Continuous decimal hours (e.g., 07:30 -> 7.5)
df["t"] = df["dep_dt"].dt.hour + df["dep_dt"].dt.minute / 60.0

# ---------------------------------------------------------
# 2. Mathematical Topologies per Route (10 Total Parameters)
# ---------------------------------------------------------

# Route A->C->D: Centered 2nd-degree parabola (2 parameters)
# theta = [quad_w, base]
def predict_acd(t, theta):
    quad_w, base = theta
    return quad_w * ((t - 11.5) ** 2) + base

# Route A->C->E: Flat constant line (1 parameter)
# theta = [const]
def predict_ace(t, theta):
    return np.full_like(t, theta[0])

# Route B->C->D: Hourly sawtooth wave + rush-hour quadratic bowl (4 parameters)
# theta = [base, saw_amp, shift, quad_w]
def predict_bcd(t, theta):
    base, saw_amp, shift, quad_w = theta
    saw = 1.0 - ((t - shift) % 1.0)
    curve = quad_w * ((t - 11.5) ** 2)
    return base + saw_amp * saw + curve

# Route B->C->E: Pure hourly sawtooth wave (3 parameters)
# theta = [base, saw_amp, shift]
def predict_bce(t, theta):
    base, saw_amp, shift = theta
    saw = 1.0 - ((t - shift) % 1.0)
    return base + saw_amp * saw

# ---------------------------------------------------------
# 3. Fortuna Local Search Optimizer
# ---------------------------------------------------------
def fit_model(func, t_vals, y_vals, init_theta, step_sizes, n_steps=50000):
    curr_theta = np.array(init_theta, dtype=float)
    curr_loss = np.mean((func(t_vals, curr_theta) - y_vals) ** 2)
    best_theta = curr_theta.copy()
    best_loss = curr_loss

    for _ in range(n_steps):
        # Gaussian mutation bounded by step_sizes
        cand = curr_theta + np.random.normal(0, step_sizes)
        cand_loss = np.mean((func(t_vals, cand) - y_vals) ** 2)

        # Greedy acceptance step
        if cand_loss < curr_loss:
            curr_loss = cand_loss
            curr_theta = cand
            if cand_loss < best_loss:
                best_loss = cand_loss
                best_theta = cand.copy()

    return best_theta, np.sqrt(best_loss)

# ---------------------------------------------------------
# 4. Training Across All 4 Routes
# ---------------------------------------------------------
models = {}

# 4.1 Route A->C->D (2 parameters)
sub_acd = df[df["road"] == "A->C->D"]
models["A->C->D"], rmse_acd = fit_model(
    predict_acd, sub_acd["t"].values, sub_acd["duration_min"].values,
    init_theta=[2.5, 75.0], step_sizes=[0.05, 0.1]
)
print(f"[A->C->D] Fitted (2 params) — RMSE: {rmse_acd:.2f} min")

# 4.2 Route A->C->E (1 parameter)
sub_ace = df[df["road"] == "A->C->E"]
models["A->C->E"], rmse_ace = fit_model(
    predict_ace, sub_ace["t"].values, sub_ace["duration_min"].values,
    init_theta=[98.0], step_sizes=[0.1]
)
print(f"[A->C->E] Fitted (1 param)  — RMSE: {rmse_ace:.2f} min")

# 4.3 Route B->C->D (4 parameters)
sub_bcd = df[df["road"] == "B->C->D"]
models["B->C->D"], rmse_bcd = fit_model(
    predict_bcd, sub_bcd["t"].values, sub_bcd["duration_min"].values,
    init_theta=[45.0, 60.0, 0.25, 2.5], step_sizes=[0.1, 0.1, 0.01, 0.05]
)
print(f"[B->C->D] Fitted (4 params) — RMSE: {rmse_bcd:.2f} min")

# 4.4 Route B->C->E (3 parameters)
sub_bce = df[df["road"] == "B->C->E"]
models["B->C->E"], rmse_bce = fit_model(
    predict_bce, sub_bce["t"].values, sub_bce["duration_min"].values,
    init_theta=[70.0, 60.0, 0.25], step_sizes=[0.1, 0.1, 0.01]
)
print(f"[B->C->E] Fitted (3 params) — RMSE: {rmse_bce:.2f} min")

# ---------------------------------------------------------
# 5. Serialization via Pickle
# ---------------------------------------------------------
with open("models/traffic_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("All 4 models successfully serialized to traffic_models.pkl (Total: 10 parameters)")
