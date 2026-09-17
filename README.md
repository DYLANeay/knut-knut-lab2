# knut-knut-lab2

IKT110, hand-in 1, task 6: a small Flask app that tells the Knut Knut drivers which of the four
routes to take depending on the departure time.

## Setup

You need [uv](https://docs.astral.sh/uv/). Then:

```bash
uv sync
```

## Files

- `data/traffic.jsonl` the trips from Canvas (route, departure, arrival)
- `src/routes.py` one formula per route (`predict_acd`, `predict_ace`, `predict_bcd`, `predict_bce`) and `best_route`
- `src/train.py` reads the trips, fits each formula with Fortuna and writes `models/traffic_models.pkl`
- `models/traffic_models.pkl` the trained parameters, loaded by the app
- `src/knut_knut_app.py` the web app

## Running

Everything is run from the root of the repo.

Train the model (a few seconds, writes `models/traffic_models.pkl`):

```bash
uv run python src/train.py
```

Start the app, then open http://127.0.0.1:5000:

```bash
uv run python src/knut_knut_app.py
```

## The model

Looking at the trips route by route:

- the routes to E take about 98 minutes all day, so `A->C->E` is a constant;
- the routes to D are slow in the morning and the afternoon and fast around noon, so `A->C->D`
  is a bowl (a cosine centred on noon);
- the routes from B follow a sawtooth that repeats every hour: something like a ferry leaves
  around quarter past, and the waiting time for the next one is added. `B->C->E` is a constant
  plus that sawtooth, `B->C->D` is the bowl plus the sawtooth.

Each formula is fitted on its own trips with a local Fortuna search (random steps around the
current best, kept only when the error goes down). For a departure time the app computes the four
durations and picks the shortest.
