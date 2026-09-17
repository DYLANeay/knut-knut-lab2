import os
import pickle
import numpy as np
from flask import Flask, request

# Calculate dynamic absolute paths relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_models.pkl")

# Initialize Flask with absolute static directory path
app = Flask(__name__, static_folder=STATIC_DIR)

# Load trained Fortuna model parameters (10 parameters total across 4 routes)
with open(MODEL_PATH, "rb") as f:
    ROAD_MODELS = pickle.load(f)

# Clean, top-aligned styling with original light theme
BASE_STYLE = """
<style>
    :root {
        --primary: #1e3a8a;
        --success: #15803d;
        --surface: #ffffff;
        --bg: #f8fafc;
        --text: #0f172a;
        --muted: #64748b;
        --border: #e2e8f0;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: var(--bg);
        color: var(--text);
        margin: 0;
        padding: 40px 20px;
        display: flex;
        justify-content: center;
    }
    .wrapper {
        display: flex;
        align-items: flex-start;
        justify-content: center;
        width: 100%;
        max-width: 1100px;
        gap: 28px;
    }
    .side-driver {
        width: 250px;
        height: auto;
        max-height: 480px;
        object-fit: contain;
        user-select: none;
    }
    .card {
        background: var(--surface);
        max-width: 540px;
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
        border: 1px solid var(--border);
        padding: 32px;
        box-sizing: border-box;
    }
    h2, h3, h4 {
        margin-top: 0;
        color: var(--primary);
    }
    .form-group {
        display: flex;
        gap: 16px;
        align-items: center;
        margin: 24px 0;
    }
    select, input[type="text"] {
        padding: 10px 14px;
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 15px;
        background: #fff;
    }
    .btn {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 11px 20px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.15s ease;
        text-decoration: none;
        display: inline-block;
    }
    .btn:hover { background-color: #1d4ed8; }
    .recommendation-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .badge-route {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--success);
    }
    .alternative-item {
        padding: 12px 14px;
        border-left: 3px solid #cbd5e1;
        background: #f8fafc;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
    }
    .alternative-item b { color: #1e293b; }
    .delay-reason { color: var(--muted); font-size: 0.9em; margin-top: 4px; }

    @media (max-width: 900px) {
        .side-driver { display: none; }
    }
</style>
"""


def predict_single_road(road, t):
    """Predict travel duration using the specific topology per route."""
    theta = ROAD_MODELS[road]

    if road == "A->C->D":
        quad_w, base = theta
        return quad_w * ((t - 11.5) ** 2) + base

    elif road == "A->C->E":
        return float(theta[0])

    elif road == "B->C->D":
        base, saw_amp, shift, quad_w = theta
        saw = 1.0 - ((t - shift) % 1.0)
        curve = quad_w * ((t - 11.5) ** 2)
        return base + saw_amp * saw + curve

    elif road == "B->C->E":
        base, saw_amp, shift = theta
        saw = 1.0 - ((t - shift) % 1.0)
        return base + saw_amp * saw


def explain_delay(road, duration, best_duration, h_float):
    """Generate operational explanation for travel time discrepancies."""
    delay = int(round(duration - best_duration))

    if road.startswith("B->") and (h_float < 10.0 or h_float > 13.5):
        return f"+{delay} min — Periodic boarding delay or queue bottleneck at node B."

    if road.endswith("->D") and (h_float < 9.5 or h_float > 15.5):
        return f"+{delay} min — Severe rush-hour urban congestion approaching node D."

    if road.endswith("->E"):
        return f"+{delay} min — Greater physical distance via node E despite steady traffic flow."

    return f"+{delay} min — Lower average traffic throughput on this corridor."


def get_the_best_route_as_a_text_informatic(dep_hour, dep_min):
    roads = ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]

    h = float(dep_hour) if dep_hour else 8.0
    m = float(dep_min) if dep_min and dep_min.isdigit() else 0.0
    t = h + m / 60.0

    predictions = {r: float(predict_single_road(r, t)) for r in roads}
    sorted_roads = sorted(predictions.items(), key=lambda item: item[1])
    best_road, best_time = sorted_roads[0]
    best_time_int = int(round(best_time))

    alternatives_html = ""
    for road, dur in sorted_roads[1:]:
        dur_int = int(round(dur))
        reason = explain_delay(road, dur, best_time, t)
        alternatives_html += f"""
        <div class="alternative-item">
            <div><b>{road}</b> — <b>{dur_int} min</b></div>
            <div class="delay-reason">{reason}</div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fast & Furious AI - Best Route Optimizer</title>
        {BASE_STYLE}
    </head>
    <body>
        <div class="wrapper">
            <img src="/static/dom.png" alt="Dominic Toretto" class="side-driver">

            <div class="card">
                <h2>Fast & Furious AI</h2>
                <p style="color: var(--muted); margin-bottom: 20px;">Optimal Dispatch & Route Guidance</p>

                <div class="recommendation-box">
                    <span style="font-size: 0.85em; text-transform: uppercase; font-weight: bold; color: var(--success); letter-spacing: 0.05em;">
                        Fastest Route
                    </span>
                    <div class="badge-route">{best_road}</div>
                    <p style="margin: 8px 0 0 0;">
                        Departure: <b>{int(h):02d}:{int(m):02d}</b> &nbsp;|&nbsp; 
                        Estimated Travel Time: <b>{best_time_int} minutes</b>
                    </p>
                </div>

                <h4 style="margin-bottom: 12px;">Alternative Options</h4>
                {alternatives_html}

                <div style="margin-top: 28px;">
                    <a href="/" class="btn">New Search</a>
                </div>
            </div>

            <img src="/static/brian.png" alt="Brian O'Conner" class="side-driver">
        </div>
    </body>
    </html>
    """


@app.route('/')
def get_departure_time():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fast & Furious AI - Best Route Optimizer</title>
        {BASE_STYLE}
    </head>
    <body>
        <div class="wrapper">
            <img src="/static/dom.png" alt="Dominic Toretto" class="side-driver">

            <div class="card">
                <h2>Fast & Furious AI</h2>
                <p style="color: var(--muted);">Select your planned departure time to determine the fastest corridor.</p>

                <form action="/get_best_route" method="get">
                    <div class="form-group">
                        <div>
                            <label for="hour" style="display:block; font-size: 0.85em; font-weight:600; margin-bottom: 6px;">HOUR</label>
                            <select name="hour" id="hour">
                                <option value="06" selected>06</option>
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
                        </div>

                        <div>
                            <label for="mins" style="display:block; font-size: 0.85em; font-weight:600; margin-bottom: 6px;">MINUTES</label>
                            <input type="text" name="mins" id="mins" size="3" maxlength="2" value="00"/>
                        </div>
                    </div>

                    <input type="submit" class="btn" value="Find Best Route">
                </form>
            </div>

            <img src="/static/brian.png" alt="Brian O'Conner" class="side-driver">
        </div>
    </body>
    </html>
    """


@app.route("/get_best_route")
def get_route():
    departure_h = request.args.get('hour')
    departure_m = request.args.get('mins')
    return get_the_best_route_as_a_text_informatic(departure_h, departure_m)


if __name__ == '__main__':
    print("<starting>")
    app.run(debug=False)
    print("<done>")