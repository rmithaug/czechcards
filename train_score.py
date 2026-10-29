"""Train the scoreboard bot on whole matches, by self-play (the stronger version).

Run it from the project folder:

    python3 train_score.py                 train (carries on from score_bot_latest.pt if
                                           it exists, otherwise starts a brand-new network)
    python3 train_score.py --fresh         throw away the saved progress and start over
    python3 train_score.py --hours 8       stop by itself after 8 hours (overnight)
    python3 train_score.py --check         just measure the best saved bot, no training

The bot learns only from playing; nobody gives it hints about the game.
Four things make it a stronger learner than the first version:

  1. A bigger brain: 3 layers of 512 neurons (--hidden, --layers) instead of
     2 layers of 256, so it has room for subtler instincts.
  2. PPO: after each batch of games it studies the same decisions several
     times (--epochs), but never moves its odds too far in one go (--clip).
     More learning from every game, without lurching around.
  3. A league: it doesn't only play copies of itself. Some seats are older
     versions of itself and some are simple bots, so it can't get stuck on a
     trick that only fools its current self.
  4. Keep the best: every check that beats the best so far is saved to
     score_bot_best.pt, so a bad stretch late in training never costs you
     the good bot.

Each decision is graded on its own deal and the deals after it (later deals
count less: --discount), plus whether the seat won the whole match
(--match-weight). Decisions aren't blamed for deals that came before them.

The check plays full matches against three simple bots (bots.py), the same
matches play_matches.py uses, so the numbers can be compared directly.
"""

import argparse
import copy
import os
import random
import time
from pathlib import Path

import torch

from czech_cards import bots
from czech_cards.game import OVER
from czech_cards.match import Match
from czech_cards.nn_bot import MODEL_FILE as NN_MODEL_FILE
from czech_cards.score_bot import (LATEST_FILE, MODEL_FILE, ScoreNet, ScorePlayer,
                                   from_card_net, judge, load_score_net)

NAMES = ["A", "B", "C", "D"]
POINTS_SCALE = 50      # deal rewards are divided by this, so they're around -1 to 1
CPUS = os.cpu_count() or 1


def points_ahead(scores: list[int], seat: int) -> float:
    others = [s for i, s in enumerate(scores) if i != seat]
    return sum(others) / len(others) - scores[seat]


def new_match(seed: int, cycles: int, names=NAMES) -> Match:
    match = Match(names, rng=random.Random(seed))
    match.num_deals = cycles * len(names)
    return match


def deal_returns(rewards: list[float], bonus: float, discount: float) -> list[float]:
    """The grade for each deal's decisions: that deal's result, plus later deals'
    results counting less and less, plus the match result (which also counts
    less the further away the end of the match is)."""
    grades, later = [], 0.0
    last = len(rewards) - 1
    for k in range(last, -1, -1):
        later = rewards[k] + discount * later
        grades.append(later + bonus * discount ** (last - k))
    return grades[::-1]


# ----- play and score -------------------------------------------------------------

def pick_seats(learner: ScorePlayer, league: list, rng: random.Random,
               simple_share: float, past_share: float) -> list:
    """Who sits where: the learner, an older version of it, or a simple bot."""
    seats = []
    for _ in NAMES:
        r = rng.random()
        if r < simple_share:
            seats.append(bots.take_turn)
        elif r < simple_share + past_share and league:
            seats.append(ScorePlayer(rng.choice(league), explore=True))
        else:
            seats.append(learner)
    if learner not in seats:
        seats[rng.randrange(len(NAMES))] = learner
    return seats


def play_batch(net: ScoreNet, matches: int, cycles: int, rng: random.Random,
               match_weight: float, discount: float, league: list | None = None,
               simple_share: float = 0.0, past_share: float = 0.0) -> list:
    """The network plays whole matches (against itself, its older versions and
    simple bots). Returns every decision the network made, with its grade."""
    learner = ScorePlayer(net, explore=True, record=True)
    rows = []
    for _ in range(matches):
        match = new_match(rng.randrange(2**31), cycles)
        seats = pick_seats(learner, league or [], rng, simple_share, past_share)
        for player in seats:
            if isinstance(player, ScorePlayer):
                player.match = match
        mine = [s for s, p in enumerate(seats) if p is learner]
        steps = {s: [] for s in mine}          # per learner seat: decisions of each deal
        rewards = {s: [] for s in mine}        # per learner seat: result of each deal
        while True:
            game = match.game
            while game.phase != OVER:
                seats[game.current](game, game.current)
            scores = match.score_deal()
            for s in mine:
                steps[s].append(learner.take_steps(game, s))
                rewards[s].append(points_ahead(scores, s) / POINTS_SCALE)
            if match.over:
                break
            match.start_next_deal()
        winners = match.winners()
        for s in mine:
            won = 1 / len(winners) if s in winners else 0.0
            bonus = match_weight * (won * 4 - 1) / 3      # win +1, loss -1/3
            for deal_steps, grade in zip(steps[s], deal_returns(rewards[s], bonus, discount)):
                for step in deal_steps:
                    rows.append((*step, grade))
    return rows


# ----- learn (PPO) --------------------------------------------------------------------

def learn(net: ScoreNet, optimizer, rows: list, entropy_bonus: float,
          epochs: int = 4, clip: float = 0.2, minibatch: int = 4096) -> dict:
    obs = torch.stack([r[0] for r in rows])
    kinds = torch.tensor([r[1] for r in rows])
    actions = torch.tensor([r[2] for r in rows])
    masks = torch.stack([r[3] for r in rows])
    faces = torch.tensor([r[4] for r in rows])
    old_log_prob = torch.tensor([r[5] for r in rows])
    grades = torch.tensor([r[6] for r in rows])

    with torch.no_grad():
        value = net(obs)[3]
    advantage = grades - value                    # better or worse than it expected?
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    stats = {"value_loss": 0.0, "entropy": 0.0}
    for _ in range(epochs):
        order = torch.randperm(len(rows))
        for start in range(0, len(rows), minibatch):
            i = order[start:start + minibatch]
            log_prob, entropy, value = judge(net, obs[i], kinds[i], actions[i], masks[i], faces[i])
            ratio = torch.exp(log_prob - old_log_prob[i])     # how much the odds have moved
            policy_loss = -torch.min(ratio * advantage[i],
                                     ratio.clamp(1 - clip, 1 + clip) * advantage[i]).mean()
            value_loss = ((value - grades[i]) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss - entropy_bonus * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            stats = {"value_loss": value_loss.item(), "entropy": entropy.mean().item()}
    return stats


# ----- measure against the simple bots ----------------------------------------------

def check(net: ScoreNet, matches: int, cycles: int, seed: int = 777_000) -> tuple[float, float]:
    """Full matches against three simple bots (best moves, no exploring).
    Returns (share of matches won, final total below the others' average)."""
    player = ScorePlayer(net, explore=False)
    won = ahead = 0.0
    for m in range(matches):
        match = new_match(seed + m, cycles)
        seat = (seed + m) % len(NAMES)
        player.match = match
        players = [bots.take_turn] * len(NAMES)
        players[seat] = player
        match.play_out(players)
        winners = match.winners()
        won += 1 / len(winners) if seat in winners else 0.0
        ahead += points_ahead(match.totals, seat)
    return won / matches, ahead / matches


def bar(share: float) -> str:
    return "█" * max(0, round(share * 40))


def save_best(net: ScoreNet, step: int, won: float, ahead: float, path: str) -> None:
    torch.save({"net": net.state_dict(), "config": net.config,
                "round": step, "wins": won, "ahead": ahead}, path)


def save_latest(net, optimizer, step, history, best, path) -> None:
    torch.save({"net": net.state_dict(), "config": net.config, "optimizer": optimizer.state_dict(),
                "step": step, "history": history, "best": best}, path)


def start(args):
    """The network and optimizer to train, picking up saved progress if there is any."""
    if Path(args.latest).exists() and not args.fresh:
        saved = torch.load(args.latest, weights_only=False)
        net = ScoreNet(**saved["config"])
        net.load_state_dict(saved["net"])
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        optimizer.load_state_dict(saved["optimizer"])
        print(f"Picking up from {args.latest} (already trained for {saved['step']} rounds, "
              f"{net.config['layers']} layers of {net.config['hidden']}).")
        return net, optimizer, saved["step"], saved["history"], saved["best"]
    if args.start == "nn":
        if (args.hidden, args.layers) != (256, 2):
            raise SystemExit("--start nn only works with --hidden 256 --layers 2 "
                             "(the size of nn_bot.pt).")
        net = from_card_net(torch.load(NN_MODEL_FILE, weights_only=False)["net"])
        print(f"Starting from your trained {NN_MODEL_FILE}.")
    else:
        net = ScoreNet(args.hidden, args.layers)
        print(f"Starting from a brand-new network ({args.layers} layers of {args.hidden}): "
              f"its moves are random.")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    return net, optimizer, 0, [], None


def train(args) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    net, optimizer, step, history, best = start(args)
    deals = args.cycles * len(NAMES)

    torch.set_num_threads(1)
    net.eval()
    if best is None:
        won, ahead = check(net, args.check_matches, args.cycles)
        best = ahead
        save_best(net, step, won, ahead, args.out)
        print(f"Against 3 simple bots ({deals}-deal matches): wins {won:5.1%} of matches, "
              f"finishes {ahead:+.0f} points ahead  {bar(won)}")
    print(f"Best so far: {best:+.0f} points ahead (kept in {args.out}).\n")
    print(f"Each round: {args.matches} matches, then {args.epochs} passes of learning. "
          f"Press Ctrl+C to stop (it saves first).\n")

    league = [copy.deepcopy(net).eval()]
    begin = time.time()
    deadline = begin + args.hours * 3600 if args.hours else None
    try:
        for _ in range(args.rounds):
            if deadline and time.time() > deadline:
                print("Time's up.")
                break
            torch.set_num_threads(1)               # playing: one card decision at a time
            net.eval()
            rows = play_batch(net, args.matches, args.cycles, rng, args.match_weight,
                              args.discount, league, args.simple_share, args.past_share)
            torch.set_num_threads(CPUS)            # learning: big batches, use every core
            net.train()
            stats = learn(net, optimizer, rows, args.entropy, args.epochs, args.clip,
                          args.minibatch)
            step += 1
            if step % args.league_every == 0:
                league.append(copy.deepcopy(net).eval())
                league = league[-args.league_size:]
            if step % args.check_every == 0:
                torch.set_num_threads(1)
                net.eval()
                won, ahead = check(net, args.check_matches, args.cycles)
                history.append({"round": step, "wins": round(won, 4), "ahead": round(ahead, 1)})
                note = ""
                if ahead > best:
                    best = ahead
                    save_best(net, step, won, ahead, args.out)
                    note = "  ← new best, saved"
                print(f"Round {step:5}: wins {won:5.1%} of matches, {ahead:+5.0f} points ahead  "
                      f"{bar(won):20} (unsure-ness {stats['entropy']:.2f}, "
                      f"{(time.time() - begin) / 60:.0f} min){note}")
                save_latest(net, optimizer, step, history, best, args.latest)
    except KeyboardInterrupt:
        print("\nStopping.")
    save_latest(net, optimizer, step, history, best, args.latest)
    print(f"Trained for {step} rounds. Best bot ({best:+.0f} points ahead) is in {args.out}; "
          f"progress is in {args.latest}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scoreboard bot on whole matches.")
    parser.add_argument("--fresh", action="store_true", help="start over, ignoring saved progress")
    parser.add_argument("--start", choices=["blank", "nn"], default="blank",
                        help="blank = a random network (pure self-learning), nn = copy nn_bot.pt")
    parser.add_argument("--check", action="store_true", help="only measure the best saved bot")
    parser.add_argument("--rounds", type=int, default=100_000, help="most learning rounds")
    parser.add_argument("--hours", type=float, default=0, help="stop after this many hours")
    parser.add_argument("--hidden", type=int, default=512, help="neurons per layer")
    parser.add_argument("--layers", type=int, default=3, help="how many layers")
    parser.add_argument("--matches", type=int, default=8, help="matches played per round")
    parser.add_argument("--cycles", type=int, default=3, help="deal cycles per match")
    parser.add_argument("--match-weight", type=float, default=0.3,
                        help="how much winning the match counts, next to the deal points")
    parser.add_argument("--discount", type=float, default=0.5,
                        help="how much each later deal counts toward a decision's grade")
    parser.add_argument("--epochs", type=int, default=4, help="learning passes over each batch")
    parser.add_argument("--clip", type=float, default=0.2, help="how far odds may move per round")
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--simple-share", type=float, default=0.2,
                        help="chance a seat is a simple bot")
    parser.add_argument("--past-share", type=float, default=0.3,
                        help="chance a seat is an older version of the network")
    parser.add_argument("--league-every", type=int, default=50,
                        help="rounds between keeping a copy for the league")
    parser.add_argument("--league-size", type=int, default=10, help="older versions kept")
    parser.add_argument("--check-every", type=int, default=50, help="rounds between checks")
    parser.add_argument("--check-matches", type=int, default=200, help="matches per check")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate: how big each nudge is")
    parser.add_argument("--entropy", type=float, default=0.01, help="bonus for trying new things")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=MODEL_FILE, help="where the best bot is kept")
    parser.add_argument("--latest", default=LATEST_FILE, help="where training progress is kept")
    args = parser.parse_args()
    if args.check:
        torch.set_num_threads(1)
        won, ahead = check(load_score_net(args.out), 200, args.cycles)
        print(f"Over 200 matches of {args.cycles * len(NAMES)} deals against 3 simple bots: "
              f"wins {won:.1%} of matches (a fair share is 25%), final total {ahead:+.0f} "
              f"points ahead of the others' average.")
    else:
        train(args)


if __name__ == "__main__":
    main()