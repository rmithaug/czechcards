"""Play full family-length matches: the network against three simple bots.

Run it from the project folder (after training, so nn_bot.pt exists):

    python3 play_matches.py                 200 matches of 3 full deal cycles (12 deals)
    python3 play_matches.py --cycles 1      shorter matches: 1 cycle (4 deals)
    python3 play_matches.py --matches 500   more matches, a more exact result
    python3 play_matches.py --bot score     the best scoreboard bot (score_bot_best.pt)
    python3 play_matches.py --bot score --file score_bot_v1.pt     any saved scoreboard bot

A match works like your family games: everyone deals once per cycle, every
deal the cards left in your hand are added to your running total (the one
who goes out scores 0, a leftover joker counts 12), and after the last deal
the lowest total wins.

It prints the first match deal by deal, so you can see the scoreboard, and
then how often the network won a whole match. A fair share is 25%.
"""

import argparse
import random

from czech_cards import bots
from czech_cards.match import Match
from czech_cards.nn_bot import load_player

NAMES = ["Network", "Jana", "Petr", "Eva"]


def play_match(network, seed: int, cycles: int) -> tuple[Match, int]:
    """One match. The network sits in a different seat each match."""
    seat = seed % len(NAMES)
    names = NAMES[-seat:] + NAMES[:-seat]        # puts "Network" in that seat
    match = Match(names, rng=random.Random(seed))
    match.num_deals = cycles * len(NAMES)        # e.g. 3 cycles x 4 players = 12 deals
    players = [bots.take_turn] * len(NAMES)
    players[seat] = network
    if hasattr(network, "match"):                # the scoreboard bot needs to see the match
        network.match = match
    match.play_out(players)
    return match, seat


def show_scoreboard(match: Match) -> None:
    print("Deal  " + "".join(f"{name:>11}" for name in match.names))
    totals = [0] * len(match.names)
    for number, scores in enumerate(match.deal_scores, 1):
        totals = [t + s for t, s in zip(totals, scores)]
        print(f"{number:4}  " + "".join(f" {s:>4} ({t:>3})" for s, t in zip(scores, totals)))
    winners = ", ".join(match.names[s] for s in match.winners())
    print(f"Winner: {winners}  (each column: points that deal, running total in brackets)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play full matches against the simple bots.")
    parser.add_argument("--matches", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=3, help="full deal cycles per match")
    parser.add_argument("--seed", type=int, default=777_000)
    parser.add_argument("--bot", choices=["nn", "score"], default="nn",
                        help="nn = nn_bot.pt, score = the scoreboard bot")
    parser.add_argument("--file", help="which saved bot to load (default: the usual file)")
    args = parser.parse_args()

    if args.bot == "score":
        from czech_cards.score_bot import load_score_player
        network = load_score_player(args.file) if args.file else load_score_player()
    else:
        network = load_player(args.file) if args.file else load_player()
    deals = args.cycles * len(NAMES)
    won = ahead = 0.0
    for m in range(args.matches):
        match, seat = play_match(network, args.seed + m, args.cycles)
        if m == 0:
            print(f"Example match ({deals} deals):\n")
            show_scoreboard(match)
        winners = match.winners()
        if seat in winners:
            won += 1 / len(winners)                 # a shared first place counts as part of a win
        others = [t for s, t in enumerate(match.totals) if s != seat]
        ahead += sum(others) / len(others) - match.totals[seat]
        if (m + 1) % 50 == 0:
            print(f"  ...{m + 1} matches played")

    print(f"\nOver {args.matches} matches of {deals} deals each, against 3 simple bots:")
    print(f"  The network won {won / args.matches:.1%} of matches (a fair share is 25%).")
    print(f"  Its final total was {ahead / args.matches:.0f} points lower than the "
          f"others' average.")


if __name__ == "__main__":
    main()