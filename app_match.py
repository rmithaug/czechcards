"""Czech Cards as a MATCH: one deal per player, lowest running total wins.

Run it with:  python3 app_match.py   then open http://127.0.0.1:8000

Jana, Petr and Eva are the learning bot, using whatever it has learned so far
(the weights in match_weights.json, saved by train_match.py). Before you've
trained it, they play with rough untrained guesses. It uses the same game
page as app.py, plus a scoreboard and a "Next deal" button
(templates/match.html and static/match.js).
"""

import os
import uuid

from flask import Flask, jsonify, render_template, request

from czech_cards import smart_bot
from czech_cards.game import OVER, GameError
from czech_cards.match import Match
from czech_cards.rules import is_meld, meld_points

app = Flask(__name__)

HUMAN_SEAT = 0
MATCHES: dict[str, Match] = {}

weights = smart_bot.load_weights()
bot_turn = smart_bot.make_player(weights)


def run_bots(match: Match) -> None:
    """Let the bots play until it's the human's turn or the deal is over."""
    game = match.game
    while game.phase != OVER and game.current != HUMAN_SEAT:
        bot_turn(game, game.current)
    if game.phase == OVER:
        match.score_deal()


def state(match_id: str) -> dict:
    data = MATCHES[match_id].view_for(HUMAN_SEAT)
    data["game_id"] = match_id
    return data


@app.get("/")
def index():
    return render_template("match.html")


@app.post("/api/new")
def new_match():
    match = Match(["You", "Jana", "Petr", "Eva"])
    match_id = uuid.uuid4().hex[:8]
    MATCHES[match_id] = match
    run_bots(match)
    return jsonify(state(match_id))


@app.get("/api/game/<match_id>")
def get_match(match_id):
    if match_id not in MATCHES:
        return jsonify(error="No such game."), 404
    return jsonify(state(match_id))


@app.post("/api/game/<match_id>/check")
def check_groups(match_id):
    """Is each group of cards a run or set, and what is it worth? (For the staging area.)"""
    match = MATCHES.get(match_id)
    if match is None:
        return jsonify(error="No such game."), 404
    by_id = {c.id: c for c in match.game.hands[HUMAN_SEAT]}
    results = []
    for group in (request.get_json(silent=True) or {}).get("groups", []):
        cards = [by_id[i] for i in group if i in by_id]
        valid = len(cards) == len(group) and is_meld(cards)
        results.append({"valid": valid, "points": meld_points(cards) if valid else 0})
    return jsonify(results=results)


@app.post("/api/game/<match_id>/next_deal")
def next_deal(match_id):
    match = MATCHES.get(match_id)
    if match is None:
        return jsonify(error="No such game."), 404
    try:
        match.start_next_deal()
    except ValueError as e:
        return jsonify(error=str(e), **state(match_id)), 400
    run_bots(match)
    return jsonify(state(match_id))


@app.post("/api/game/<match_id>/action")
def act(match_id):
    match = MATCHES.get(match_id)
    if match is None:
        return jsonify(error="No such game."), 404
    game = match.game
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
        elif action == "swap_joker":
            game.swap_joker(HUMAN_SEAT, int(body["card_id"]), int(body["meld"]))
        elif action == "discard":
            game.discard(HUMAN_SEAT, int(body["card_id"]))
        else:
            return jsonify(error=f"Unknown action {action!r}."), 400
    except (GameError, KeyError, TypeError, ValueError) as e:
        return jsonify(error=str(e), **state(match_id)), 400
    run_bots(match)
    return jsonify(state(match_id))


if __name__ == "__main__":
    print("The bots are using these weights:", weights)
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 8000)),
        debug=os.environ.get("DEBUG", "1") == "1",
    )
