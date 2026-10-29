"""Watch the scoreboard bot play a whole match, and see what it was thinking.

Run it from the project folder:

    python3 watch_score.py                 a whole match (12 deals) with the best bot
    python3 watch_score.py --deals 1       just one deal
    python3 watch_score.py --seed 900000   a particular match (same seed = same cards)
    python3 watch_score.py --pause         wait for Enter after each deal
    python3 watch_score.py --speed 0       no pauses between moves
    python3 watch_score.py --file score_bot_v1.pt     any saved scoreboard bot

On each of its turns you see:
  * its hand and the scoreboard
  * its "feeling": the network's own guess of how things will go from here
    (0 is an ordinary result, above 0 is good, below 0 is bad)
  * every choice it made and how sure it was: take the discard? lay down
    now or wait? lay off this card or keep it? and its top picks to throw
"""

import argparse
import random
import time

import torch

from czech_cards import bots
from czech_cards.game import OVER
from czech_cards.match import Match
from czech_cards.nn_bot import FACE_INDEX, FACE_LABELS, LAY_DOWN, TAKE, THROW, _melds_to_lay_down
from czech_cards.score_bot import LAY_OFF_CARD, MODEL_FILE, ScorePlayer, load_score_net
from watch_nn import show_hand

NAMES = ["Network", "Jana", "Petr", "Eva"]


def yes_no(action: int, p_yes: float) -> str:
    sure = p_yes if action else 1 - p_yes
    return f"{'yes' if action else 'no'} ({sure:.0%} sure)"


class Commentator(ScorePlayer):
    """The scoreboard bot, plus a note of what it thought at every choice."""

    def __init__(self, net):
        super().__init__(net)
        self.notes: list[str] = []
        self.feeling = 0.0

    def _odds(self, game, seat):
        with torch.no_grad():
            return self.net(self._observe(game, seat))

    def decide(self, game, seat, kind, face=None):
        yes, lay_off, _, _ = self._odds(game, seat)
        if kind == LAY_DOWN:
            melds = "  ".join(show_hand(m) for m in _melds_to_lay_down(game, seat))
        action = super().decide(game, seat, kind, face)
        if kind == TAKE:
            p = torch.sigmoid(yes[0]).item()
            self.notes.append(f"Take the {game.top_discard().label()} from the discard pile? "
                              f"{yes_no(action, p)}")
        elif kind == LAY_DOWN:
            p = torch.sigmoid(yes[1]).item()
            self.notes.append(f"Lay down {melds} now? {yes_no(action, p)}"
                              + ("" if action else "  <- keeps them hidden for now"))
        elif kind == LAY_OFF_CARD:
            p = torch.sigmoid(lay_off[FACE_INDEX[face]]).item()
            self.notes.append(f"Lay off the {FACE_LABELS[FACE_INDEX[face]]}? {yes_no(action, p)}"
                              + ("" if action else "  <- holds it back"))
        elif kind == THROW:
            odds = sorted(self.last_throw_odds.items(), key=lambda item: -item[1])
            self.notes.append("Throw away: " + ",  ".join(f"{label} {p:.0%}" for label, p in odds[:4]))
        return action

    def __call__(self, game, seat):
        self.feeling = self._odds(game, seat)[3].item()
        self.notes = []
        super().__call__(game, seat)


def show_totals(match: Match) -> str:
    return ",  ".join(f"{name} {total}" for name, total in zip(match.names, match.totals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the scoreboard bot play a match.")
    parser.add_argument("--file", default=MODEL_FILE, help="which saved bot to watch")
    parser.add_argument("--deals", type=int, default=12, help="deals in the match")
    parser.add_argument("--seed", type=int, default=None, help="which match (same seed = same cards)")
    parser.add_argument("--speed", type=float, default=0.2, help="seconds between moves")
    parser.add_argument("--pause", action="store_true", help="wait for Enter after each deal")
    args = parser.parse_args()

    player = Commentator(load_score_net(args.file))
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    match = Match(NAMES, rng=random.Random(seed))
    match.num_deals = args.deals
    player.match = match
    print(f"Match {seed}: Network is the bot from {args.file}. "
          f"Jana, Petr and Eva are the simple bots.")

    while True:
        game = match.game
        print(f"\n===== {game.log[0]}  Totals so far: {show_totals(match)} =====")
        shown = 1
        while game.phase != OVER:
            seat = game.current
            if seat == 0:
                print(f"\n  Network's hand: {show_hand(game.hands[0])}")
                player(game, 0)
                print(f"  Its feeling about how things will go: {player.feeling:+.2f}")
                for note in player.notes:
                    print(f"    {note}")
            else:
                bots.take_turn(game, seat)
            for line in game.log[shown:]:
                print(f"  -> {line}" if seat == 0 else line)
            shown = len(game.log)
            time.sleep(args.speed)

        scores = match.score_deal()
        print("\nEnd of the deal. Points left in each hand (lower is better):")
        for seat, name in enumerate(NAMES):
            print(f"  {name:8} {scores[seat]:3}   {show_hand(game.hands[seat])}")
        print(f"Totals: {show_totals(match)}")
        if match.over:
            break
        if args.pause:
            input("\nPress Enter for the next deal...")
        match.start_next_deal()

    winners = " and ".join(NAMES[s] for s in match.winners())
    print(f"\nThe match is over. Winner: {winners}.")


if __name__ == "__main__":
    main()