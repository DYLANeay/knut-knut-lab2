import pickle
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask
from flask import request

from routes import best_route

app = Flask(__name__)

# racine du projet (ce fichier est dans src/)
ROOT = Path(__file__).resolve().parent.parent

# Les paramètres des 4 routes, entraînés par train.py (chargés une seule fois au démarrage)
with open(ROOT / "models" / "traffic_models.pkl", "rb") as f:
    MODELS = pickle.load(f)


def get_the_best_route_as_a_text_informatic(dep_hour, dep_min):
    roads = ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]

    # on lit l'heure et les minutes envoyées par le formulaire (ce sont des textes)
    try:
        hour = int(dep_hour)
        minute = int(dep_min)
    except (TypeError, ValueError):
        return """
        <p>Invalid departure time: please write the minutes as a number between 0 and 59.</p>
        <p><a href="/">Back</a></p>
        """
    if minute < 0 or minute > 59:
        return """
        <p>Invalid departure time: please write the minutes as a number between 0 and 59.</p>
        <p><a href="/">Back</a></p>
        """

    # durée prédite de chaque route, on garde la plus courte
    best_road, est_travel_time, durations = best_route(MODELS, hour, minute)

    # heure d'arrivée estimée = départ + durée prédite
    departure_time = datetime(2000, 1, 1, hour, minute)
    arrival_time = departure_time + timedelta(minutes=round(est_travel_time))

    all_routes = ""
    for road in roads:
        all_routes += "{}: {} minutes <br>".format(road, round(durations[road]))

    out = """
    <p>
    Departure time: {} <br>
    Best travel route: {} <br>
    Estimated travel time of {} minutes. <br>
    Estimated arrival time: {} </p>
    <p>All routes: <br>
    {} </p>
    <p><a href="/">Back</a></p>
    """.format(departure_time.strftime("%H:%M"), best_road, round(est_travel_time),
               arrival_time.strftime("%H:%M"), all_routes)

    return out


@app.route('/')
def get_departure_time():
    return """
    	<h3>Knut Knut Transport AS</h3>
        <form action="/get_best_route" method="get">
            <label for="hour">Hour:</label>
            <select name="hour" id="hour">
                <option value="06">06</option>
                <option value="07">07</option>
                <option value="08">08</option>
                <option value="09">09</option>
                <option value="10">10</option>
                <option value="11">11</option>
                <option value="12">12</option>
                <option value="13">13</option>
                <option value="14">14</option>
                <option value="15">15</option>
                <option value="16">16</option>
            </select>

            <label for="mins">Mins:</label>
            <input type="text" name="mins" size="2"/>
            <input type="submit">
        </form>
    """


@app.route("/get_best_route")
def get_route():
    departure_h = request.args.get('hour')
    departure_m = request.args.get('mins')

    route_info = get_the_best_route_as_a_text_informatic(departure_h, departure_m)
    return route_info


if __name__ == '__main__':
    print("<starting>")
    app.run()
    print("<done>")
