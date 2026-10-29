"""Watch the neural-network bot play one deal against three simple bots.

Run it from the project folder (after some training, so nn_bot.pt exists):

    python3 watch_nn.py               the trained network (nn_bot.pt)
    python3 watch_nn.py --untrained   a brand-new network (random moves)
    python3 watch_nn.py --seed 7      a particular deal (same seed = same cards)
    python3 watch_nn.py --speed 0     no pauses

On each of its turns you see its hand and how sure it was about which card
to throw away: the network gives every card a chance, and the bot throws
the one with the highest chance.
"""

import argparse
import random
import time

from czech_cards import bots
from czech_cards.game import OVER, Game
from czech_cards.match import hand_score
from czech_cards.nn_bot import CardNet, NetPlayer, load_net

SUIT_ORDER = {"H": 0, "S": 1, "D": 2, "C": 3, None: 4}


def show_hand(cards) -> str:
    cards = sorted(cards, key=lambda c: (SUIT_ORDER[c.suit], c.rank))
    return " ".join(c.label() for c in cards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the network bot play.")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speed", type=float, default=0.6, help="seconds between moves")
    args = parser.parse_args()

    net = CardNet().eval() if args.untrained else load_net()
    player = NetPlayer(net)
    seed = args.seed if args.seed is not None else random.randrange(10_000)
    names = ["Network", "Jana", "Petr", "Eva"]
    game = Game(names, rng=random.Random(seed))
    print(f"Deal {seed}: Network is the {'untrained' if args.untrained else 'trained'} bot. "
          f"Jana, Petr and Eva are the simple bots.\n")

    shown = 0
    while game.phase != OVER:
        seat = game.current
        if seat == 0:
            print(f"\n  Network's hand: {show_hand(game.hands[0])}")
            player(game, 0)
        else:
            bots.take_turn(game, seat)
        for line in game.log[shown:]:
            print("  " + line if seat == 0 else line)
        shown = len(game.log)
        if seat == 0 and player.last_throw_odds:
            odds = sorted(player.last_throw_odds.items(), key=lambda item: -item[1])
            print("  Chance it gave each card of being the one to throw: "
                  + ",  ".join(f"{label} {p:.0%}" for label, p in odds[:5]))
        time.sleep(args.speed)

    print("\nEnd of the deal. Points left in each hand (lower is better):")
    for seat, name in enumerate(names):
        points = 0 if seat == game.winner else hand_score(game.hands[seat])
        print(f"  {name:8} {points:3}   {show_hand(game.hands[seat])}")


if __name__ == "__main__":
    main()