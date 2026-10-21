"""The web server. Run it with:  python3 app.py   then open http://127.0.0.1:8000

Games are kept in memory (they disappear when the server restarts). You sit in
seat 0 and the other three seats are bots. Whenever your turn ends, the server
lets the bots play until it's your turn again.

(Port 8000 instead of Flask's usual 5000, because on a Mac port 5000 is often
taken by AirPlay.)
"""

import os
import uuid

from flask import Flask, jsonify, render_template, request

from czech_cards.bots import take_turn
from czech_cards.game import OVER, Game, GameError
from czech_cards.rules import is_meld, meld_points

app = Flask(__name__)

HUMAN_SEAT = 0
GAMES: dict[str, Game] = {}


def run_bots(game: Game) -> None:
    while game.phase != OVER and game.current != HUMAN_SEAT:
        take_turn(game, game.current)


def state(game_id: str) -> dict:
    data = GAMES[game_id].view_for(HUMAN_SEAT)
    data["game_id"] = game_id
    return data


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/new")
def new_game():
    game = Game(["You", "Jana", "Petr", "Eva"])
    game_id = uuid.uuid4().hex[:8]
    GAMES[game_id] = game
    run_bots(game)
    return jsonify(state(game_id))


@app.get("/api/game/<game_id>")
def get_game(game_id):
    if game_id not in GAMES:
        return jsonify(error="No such game."), 404
    return jsonify(state(game_id))


@app.post("/api/game/<game_id>/check")
def check_groups(game_id):
    """Is each group of cards a run or set, and what is it worth? (For the staging area.)"""
    game = GAMES.get(game_id)
    if game is None:
        return jsonify(error="No such game."), 404
    by_id = {c.id: c for c in game.hands[HUMAN_SEAT]}
    results = []
    for group in (request.get_json(silent=True) or {}).get("groups", []):
        cards = [by_id[i] for i in group if i in by_id]
        valid = len(cards) == len(group) and is_meld(cards)
        results.append({"valid": valid, "points": meld_points(cards) if valid else 0})
    return jsonify(results=results)


@app.post("/api/game/<game_id>/action")
def act(game_id):
    game = GAMES.get(game_id)
    if game is None:
        return jsonify(error="No such game."), 404
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    try:
        if action == "draw":
            game.draw_from_pile(HUMAN_SEAT)
        elif action == "take":
            game.take_discard(HUMAN_SEAT)
        elif action == "put_back":
            game.put_back(HUMAN_SEAT)
        elif action == "lay_down":
            groups = [[int(i) for i in group] for group in body.get("groups", [])]
            game.lay_down(HUMAN_SEAT, groups)
        elif action == "lay_off":
            game.lay_off(HUMAN_SEAT, int(body["card_id"]), int(body["meld"]))
        elif action == "discard":
            game.discard(HUMAN_SEAT, int(body["card_id"]))
        else:
            return jsonify(error=f"Unknown action {action!r}."), 400
    except (GameError, KeyError, TypeError, ValueError) as e:
        return jsonify(error=str(e), **state(game_id)), 400
    run_bots(game)
    return jsonify(state(game_id))


if __name__ == "__main__":
    # Locally this listens only on your own computer (127.0.0.1) with Flask's
    # debug mode on. The Dockerfile sets HOST=0.0.0.0 so the game can be
    # reached from outside the container, and turns debug mode off.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8000)),
        debug=os.environ.get("DEBUG", "1") == "1",
    )