"""Teach the learning bot to play Czech Cards matches, by evolution.

Run it from the project folder:

    python3 train_match.py              train (or keep training where you stopped)
    python3 train_match.py --fresh      start over from rough, untrained guesses
    python3 train_match.py --blank      start over from a bot that knows NOTHING (every dial at 0)
    python3 train_match.py --compare    don't train, just measure how good the bots are

Stop it any time with Ctrl+C. The best bot so far is saved after every
generation in match_weights.json, and the next run carries on from there.

How it works, in plain words:
  1. Make a "population" of 16 bots. Each one is the learning bot from
     czech_cards/smart_bot.py with its dials set a bit differently.
  2. Every bot plays the same matches, each against three old bots from
     bots.py. Its score ("fitness") is how many points it finishes AHEAD of
     the others: their average total minus its own total. Lower totals win,
     so more points ahead is better.
  3. The best 3 bots go through to the next generation unchanged.
  4. The other 13 places go to "children". Pick two good parents, give the
     child each dial from one parent or the other ("crossover"), then nudge a
     few dials at random ("mutation"). Most nudges make things worse, but
     now and then one makes the bot better, and that bot gets to have
     children of its own.
  5. Repeat. Good dial settings spread through the population; bad ones die
     out. Each generation plays brand-new deals, so a bot can't stay on top
     just by getting lucky once.

Each generation prints a bar (longer is better) and the dials that have
changed most since the start, so you can see which strategies it's learning.
"""

import argparse
import json
import multiprocessing
import random
import signal
import time
from datetime import date

from czech_cards import bots, smart_bot
from czech_cards.match import Match

GENES = list(smart_bot.DEFAULT_WEIGHTS)

# How far one mutation can move each dial.
MUTATION_SIZE = {
    "in_meld": 2.0, "same_rank": 1.0, "same_suit_next": 1.0, "same_suit_gap": 1.0,
    "duplicate": 1.0, "live_outs": 0.2, "points_before_out": 0.1, "points_after_out": 0.1,
    "fits_table": 1.0, "helps_next_player": 2.0,
    "take_for_meld": 1.0, "take_for_table": 1.0, "joker_layoff_hand": 2.0,
    "lay_down_when_left": 3.0, "lay_off_when_left": 3.0,
    "points_when_close": 0.3, "stop_holding_at": 0.3, "swap_for_jokers": 1.0,
    "draw_chance": 0.2, "risk_next_player": 0.1, "trust_discards": 0.2,
}

# The range each dial has to stay in.
LIMITS = {name: (-20.0, 20.0) for name in GENES}
for name in ("joker_layoff_hand", "lay_down_when_left", "lay_off_when_left"):
    LIMITS[name] = (1.0, 13.0)
LIMITS["stop_holding_at"] = (0.0, 2.0)
LIMITS["trust_discards"] = (0.0, 1.0)

# What each dial is about, in a few words (for the "learning:" lines).
ABOUT = {
    "in_meld": "keeping cards that are already in a run or set",
    "same_rank": "keeping pairs of the same number",
    "same_suit_next": "keeping cards next to each other in a suit",
    "same_suit_gap": "keeping cards one apart in a suit (5 and 7)",
    "duplicate": "keeping a second copy of the same card",
    "live_outs": "keeping cards whose partners are still out there",
    "points_before_out": "keeping high cards before coming out",
    "points_after_out": "keeping high cards after coming out",
    "fits_table": "keeping cards that fit a meld on the table",
    "helps_next_player": "not throwing cards the next player can use",
    "take_for_meld": "taking the discard to make a new meld",
    "take_for_table": "taking the discard to add to the table",
    "joker_layoff_hand": "how few cards before adding jokers to the table",
    "lay_down_when_left": "how long to hold runs/sets back",
    "lay_off_when_left": "how long to hold back cards that fit the table",
    "points_when_close": "keeping high cards when someone is about to go out",
    "stop_holding_at": "how soon to play everything when someone is close",
    "swap_for_jokers": "swapping cards for jokers on the table",
    "draw_chance": "keeping cards with a good chance of being completed by my next draw",
    "risk_next_player": "not throwing cards the next player is likely to use",
    "trust_discards": "believing what other players' throws give away",
}


# ----- playing matches --------------------------------------------------------

def play_match(weights: dict | None, seed: int) -> tuple[float, float]:
    """One match: the bot being tested against three old bots.
    weights=None tests an old bot instead (for comparison).
    Returns (points ahead of the others' average, share of first place)."""
    match = Match(["A", "B", "C", "D"], rng=random.Random(seed))
    seat = seed % 4          # a different seat each match, so nobody gets an unfair seat
    players = [bots.take_turn] * 4
    if weights is not None:
        players[seat] = smart_bot.make_player(weights)
    match.play_out(players)
    others = [t for s, t in enumerate(match.totals) if s != seat]
    ahead = sum(others) / len(others) - match.totals[seat]
    winners = match.winners()
    return ahead, (1 / len(winners) if seat in winners else 0.0)


def _measure(job: tuple) -> tuple[float, float]:
    weights, seeds = job
    results = [play_match(weights, seed) for seed in seeds]
    return (sum(a for a, _ in results) / len(seeds), sum(w for _, w in results) / len(seeds))


def _ignore_ctrl_c() -> None:
    # Only the main program should react to Ctrl+C, not every helper process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def measure_all(pool, population: list, seeds: list[int]) -> list[tuple[float, float]]:
    """Every bot plays the same matches. Uses all your computer's cores at once."""
    return pool.map(_measure, [(w, seeds) for w in population])


# ----- evolution --------------------------------------------------------------

def clamp(name: str, value: float) -> float:
    low, high = LIMITS[name]
    return round(min(high, max(low, value)), 3)


def mutate(weights: dict, rng: random.Random, chance: float, scale: float = 1.0) -> dict:
    """Nudge some of the dials at random."""
    child = dict(weights)
    for name in GENES:
        if rng.random() < chance:
            child[name] = clamp(name, child[name] + rng.gauss(0, MUTATION_SIZE[name] * scale))
    return child


def crossover(mum: dict, dad: dict, rng: random.Random) -> dict:
    """A child gets each dial from one parent or the other."""
    return {name: (mum if rng.random() < 0.5 else dad)[name] for name in GENES}


def pick_parent(population: list[dict], scores: list[float], rng: random.Random) -> dict:
    """Tournament: pick 3 bots at random, and the best of them becomes a parent."""
    contenders = rng.sample(range(len(population)), 3)
    return population[max(contenders, key=lambda i: scores[i])]


def starting_weights(fresh: bool, blank: bool) -> tuple[dict, str]:
    if blank:
        return {name: clamp(name, 0.0) for name in GENES}, "a blank bot (every dial at 0)"
    if smart_bot.WEIGHTS_FILE.exists() and not fresh:
        return smart_bot.load_weights(), "where you stopped last time"
    return dict(smart_bot.DEFAULT_WEIGHTS), "rough, untrained guesses"


def bar(points: float) -> str:
    """One block for every 2 points ahead."""
    return "█" * min(40, max(0, round(points / 2)))


def biggest_changes(start: dict, now: dict, how_many: int = 3) -> list[str]:
    """The dials that moved most since the start (measured in mutation-sized steps)."""
    moved = sorted(GENES, key=lambda n: abs(now[n] - start[n]) / MUTATION_SIZE[n], reverse=True)
    lines = []
    for name in moved[:how_many]:
        if abs(now[name] - start[name]) < MUTATION_SIZE[name] / 2:
            break
        lines.append(f"{ABOUT[name]}: {start[name]:g} -> {now[name]:g}")
    return lines


def save(weights: dict, ahead: float, wins: float, note: str) -> None:
    result = {
        "weights": weights,
        "points_ahead_per_match": round(ahead, 2),
        "match_win_rate_vs_old_bots": round(wins, 4),
        "note": note,
        "saved_on": date.today().isoformat(),
    }
    smart_bot.WEIGHTS_FILE.write_text(json.dumps(result, indent=2) + "\n")


def train(generations: int, size: int, matches: int, final_matches: int, elite: int,
          seed: int, fresh: bool, blank: bool) -> None:
    rng = random.Random(seed)
    start_weights, where = starting_weights(fresh, blank)
    print(f"Starting from {where}.")
    print(f"{size} bots per generation, {matches} matches each. Press Ctrl+C to stop.\n")
    population = [dict(start_weights)]
    while len(population) < size:
        population.append(mutate(start_weights, rng, chance=0.5, scale=1.5))
    next_seed = seed * 1_000_000 + 500_000
    begin = time.time()

    with multiprocessing.Pool(initializer=_ignore_ctrl_c) as pool:
        try:
            for gen in range(1, generations + 1):
                seeds = list(range(next_seed, next_seed + matches))
                next_seed += matches
                results = measure_all(pool, population, seeds)
                if gen == 1:
                    ahead0, wins0 = results[0]
                    print(f"The starting bot: {ahead0:+6.1f} points ahead, wins {wins0:5.1%} "
                          f"of matches  {bar(ahead0)}\n")
                order = sorted(range(size), key=lambda i: results[i][0], reverse=True)
                population = [population[i] for i in order]
                results = [results[i] for i in order]
                scores = [ahead for ahead, _ in results]
                best_ahead, best_wins = results[0]
                print(f"Generation {gen:3}: best {best_ahead:+6.1f} points ahead, wins "
                      f"{best_wins:5.1%} of matches  {bar(best_ahead):25} "
                      f"({time.time() - begin:.0f}s)")
                for line in biggest_changes(start_weights, population[0]):
                    print(f"      learning: {line}")
                save(population[0], best_ahead, best_wins, f"best of generation {gen}")

                if gen == generations:
                    break
                children = population[:elite]          # the best go through unchanged
                while len(children) < size:
                    child = crossover(pick_parent(population, scores, rng),
                                      pick_parent(population, scores, rng), rng)
                    children.append(mutate(child, rng, chance=0.3))
                population = children
        except KeyboardInterrupt:
            print("\nStopped. The best bot so far is saved in match_weights.json.")
            return

        # Final match: the top 5 and the starting bot play lots of brand-new deals,
        # so we know which one is REALLY best and not just lucky.
        finalists = population[:5] + [dict(start_weights)]
        seeds = list(range(next_seed, next_seed + final_matches))
        print(f"\nFinal check: {len(finalists)} bots x {final_matches} matches each...")
        try:
            final = measure_all(pool, finalists, seeds)
        except KeyboardInterrupt:
            print("\nStopped. The best bot so far is saved in match_weights.json.")
            return

    for i, (ahead, wins) in enumerate(final):
        label = "the starting bot" if i == len(final) - 1 else f"finalist {i + 1}"
        print(f"  {label:17} {ahead:+6.1f} points ahead, wins {wins:5.1%} of matches  {bar(ahead)}")
    best = max(range(len(final)), key=lambda i: final[i][0])
    save(finalists[best], final[best][0], final[best][1], f"final check, {final_matches} matches")
    print(f"\nSaved the best bot in match_weights.json. Against three old bots it finishes "
          f"{final[best][0]:+.1f} points ahead per match and wins {final[best][1]:.1%} of "
          f"matches (a fair share is 25%).")


def compare(matches: int) -> None:
    tests = [
        ("Old bot (bots.py)", None),
        ("Learning bot, untrained", smart_bot.DEFAULT_WEIGHTS),
        ("Learning bot, trained", smart_bot.load_weights()),
    ]
    print(f"Each bot plays {matches} matches against 3 old bots (the same deals for all).\n")
    chunks = [list(range(s, min(s + 50, 800_000_000 + matches)))
              for s in range(800_000_000, 800_000_000 + matches, 50)]
    with multiprocessing.Pool(initializer=_ignore_ctrl_c) as pool:
        for label, weights in tests:
            parts = pool.map(_measure, [(weights, c) for c in chunks])
            ahead = sum(a * len(c) for (a, _), c in zip(parts, chunks)) / matches
            wins = sum(w * len(c) for (_, w), c in zip(parts, chunks)) / matches
            print(f"  {label:25} {ahead:+6.1f} points ahead per match, "
                  f"wins {wins:5.1%} of matches  {bar(ahead)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach the learning bot by evolution.")
    parser.add_argument("--fresh", action="store_true", help="start over from rough guesses")
    parser.add_argument("--blank", action="store_true", help="start over from a bot that knows nothing")
    parser.add_argument("--compare", action="store_true", help="only measure the bots")
    parser.add_argument("--generations", type=int, default=20, help="how many rounds of evolution")
    parser.add_argument("--population", type=int, default=16, help="how many bots per generation")
    parser.add_argument("--matches", type=int, default=300, help="matches each bot plays per generation")
    parser.add_argument("--final-matches", type=int, default=1000, help="matches in the final check")
    parser.add_argument("--elite", type=int, default=3, help="best bots kept unchanged each generation")
    parser.add_argument("--seed", type=int, default=None, help="pick the deals (random if left out)")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1000)
    if args.compare:
        compare(args.final_matches)
    else:
        train(args.generations, args.population, args.matches, args.final_matches,
              args.elite, args.seed, args.fresh, args.blank)


if __name__ == "__main__":
    main()
