# Une formule par route, calquée sur la forme réelle observée dans les données.
# Seul endroit où les formules sont définies : importé par train.py (entraînement),
# evaluate.py (gain) et knut_knut_app.py (prédiction).
# Les notations (t, theta, y_hat) sont expliquées en haut de train.py.

import numpy as np


def predict_acd(t, theta):
    quad_w, base = theta
    return quad_w * ((t - 11.5) ** 2) + base


def predict_ace(t, theta):
    const = theta[0]
    return np.full_like(t, const)


def predict_bcd(t, theta):
    base, saw_amp, shift, quad_w = theta
    saw = 1.0 - ((t - shift) % 1.0)
    curve = quad_w * ((t - 11.5) ** 2)
    return base + saw_amp * saw + curve


def predict_bce(t, theta):
    base, saw_amp, shift = theta
    saw = 1.0 - ((t - shift) % 1.0)
    return base + saw_amp * saw


# Pour retrouver la bonne fonction à partir du nom de la route
PREDICT = {
    "A->C->D": predict_acd,
    "A->C->E": predict_ace,
    "B->C->D": predict_bcd,
    "B->C->E": predict_bce,
}


def best_route(models, hour, minute):
    # models : {route: theta} chargé depuis traffic_models.pkl
    # Calcule la durée prédite des 4 routes pour un départ hour:minute et garde la plus courte.
    # Renvoie (meilleure route, sa durée en minutes, dict de toutes les durées)
    t = np.array([hour + minute / 60.0])

    durations = {}
    for road, theta in models.items():
        durations[road] = float(PREDICT[road](t, theta)[0])

    best = min(durations, key=durations.get)
    return best, durations[best], durations
