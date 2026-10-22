"""Watch the learning bot play one deal against three old bots, move by move.

Run it from the project folder:

    python3 watch_game.py               the bot as trained so far (match_weights.json)
    python3 watch_game.py --untrained   the bot before any training
    python3 watch_game.py --blank       a bot that knows nothing (every dial at 0)
    python3 watch_game.py --seed 7      a particular deal (use the same seed to compare bots)
    python3 watch_game.py --speed 0     no pauses

Every time the learning bot ("Learner") throws a card away, you see what its
card tracker knows, and the keep score it gave each card in its hand, with
two of the tracker's odds:
  draw  = % chance its next draw makes a run or set with that card
  risk  = % chance the next player could use that card if it's thrown to them
It throws away the card with the lowest score. Watch how the scores change
once another player is down to a few cards.
"""

import argparse
import random
import time

from czech_cards import bots, smart_bot
from czech_cards.game import OVER, Game
from czech_cards.match import hand_score
from czech_cards.tracker import FACE_OF_LABEL, Tracker

SUIT_ORDER = {"H": 0, "S": 1, "D": 2, "C": 3, None: 4}


def show_hand(cards) -> str:
    cards = sorted(cards, key=lambda c: (SUIT_ORDER[c.suit], c.rank))
    return " ".join(c.label() for c in cards)


def face_label(face: tuple) -> str:
    return next(label for label, f in FACE_OF_LABEL.items() if f == face)


def show_tracker(tracker, names: list[str]) -> None:
    tracker.update()
    gone = tracker.gone()
    print(f"  Tracker: {tracker.unseen_total()} cards unseen (pile + other hands)."
          + (f" All copies gone: {' '.join(face_label(f) for f in gone)}" if gone else ""))
    for seat, thrown in tracker.thrown_by.items():
        if seat != tracker.seat and thrown:
            print(f"    {names[seat]} has thrown: {' '.join(face_label(f) for f in thrown[-8:])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the learning bot play.")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--blank", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speed", type=float, default=0.6, help="seconds between moves")
    args = parser.parse_args()

    if args.blank:
        weights, who = {name: 0.0 for name in smart_bot.DEFAULT_WEIGHTS}, "a blank bot"
        weights.update(joker_layoff_hand=1.0, lay_down_when_left=1.0, lay_off_when_left=1.0)
    elif args.untrained:
        weights, who = dict(smart_bot.DEFAULT_WEIGHTS), "the untrained bot"
    else:
        weights, who = smart_bot.load_weights(), "the trained bot"
    seed = args.seed if args.seed is not None else random.randrange(10_000)

    names = ["Learner", "Jana", "Petr", "Eva"]
    game = Game(names, rng=random.Random(seed))
    tracker = Tracker(game, 0, weights["trust_discards"])   # the bot's memory for this deal
    print(f"Deal {seed}: Learner is {who}. Jana, Petr and Eva are the old bots.\n")
    shown = 0
    while game.phase != OVER:
        seat = game.current
        if seat == 0:
            print(f"\n  Learner's hand: {show_hand(game.hands[0])}")
            show_tracker(tracker, names)
            scores = smart_bot.take_turn(game, 0, weights, tracker)
        else:
            bots.take_turn(game, seat)
            scores = None
        for line in game.log[shown:]:
            print("  " + line if seat == 0 else line)
        shown = len(game.log)
        if scores:
            ranked = sorted(scores.items(), key=lambda item: item[1]["score"])
            print("  Keep scores, lowest first (it throws away the first one):")
            print("    " + ",  ".join(f"{label} {d['score']:g} (draw {d['draw']}%, risk {d['risk']}%)"
                                     for label, d in ranked[:6]))
        time.sleep(args.speed)

    print("\nEnd of the deal. Points left in each hand (lower is better):")
    for seat, name in enumerate(names):
        points = 0 if seat == game.winner else hand_score(game.hands[seat])
        print(f"  {name:8} {points:3}   {show_hand(game.hands[seat])}")


if __name__ == "__main__":
    main()
