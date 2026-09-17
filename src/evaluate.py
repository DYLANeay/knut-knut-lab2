# Combien de minutes le modèle fait gagner aux chauffeurs de Knut Knut.
# Sans le modèle : la durée réelle des trajets de l'historique (les routes étaient prises au hasard).
# Avec le modèle : pour chaque trajet, à la même heure de départ, la durée de la route conseillée.

import json
import pickle
import numpy as np
from routes import PREDICT

durations = []
hours = []
with open("data/traffic.jsonl") as f:
    for line in f:
        trip = json.loads(line)
        dep_h, dep_m = map(int, trip["depature"].split(":"))
        arr_h, arr_m = map(int, trip["arrival"].split(":"))
        durations.append((arr_h * 60 + arr_m) - (dep_h * 60 + dep_m))
        hours.append(dep_h + dep_m / 60.0)

y = np.array(durations)
t = np.array(hours)

with open("models/traffic_models.pkl", "rb") as f:
    models = pickle.load(f)

# Une ligne par route, une colonne par trajet : durée prédite de chaque route à chaque heure de départ
predictions = np.array([PREDICT[road](t, theta) for road, theta in models.items()])
with_model = predictions.min(axis=0)

without = y.mean()
with_ = with_model.mean()
print(f"Sans le modèle : {without:.1f} min par trajet")
print(f"Avec le modèle : {with_:.1f} min par trajet")
print(f"Gain           : {without - with_:.1f} min par trajet ({(without - with_) / without:.0%})")
hours_saved = (without - with_) * len(y) / 60
print(f"Sur les {len(y)} trajets : {hours_saved:.0f} heures gagnées")

# Coût horaire d'un chauffeur
HOURLY_RATE = 250  # kr
print(f"Argent économisé : {hours_saved * HOURLY_RATE:.0f} kr (à {HOURLY_RATE} kr de l'heure)")
