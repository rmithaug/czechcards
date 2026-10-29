"""Teach the neural-network bot to play by letting it play against itself.

Run it from the project folder:

    python3 train_nn.py                 keep training (picks up from nn_bot.pt)
    python3 train_nn.py --fresh         start again from a brand-new, random network
    python3 train_nn.py --check         just measure the saved bot, no training

How it learns (this is called reinforcement learning, self-play):

  1. PLAY. All four seats are the same network. It plays a batch of deals
     against itself, picking moves at random but weighted by what it
     currently thinks is good, so it keeps trying new things.
  2. SCORE. When a deal ends, every seat gets a reward: how many points it
     finished ahead of the other three (the one who went out has 0 points,
     everyone else has the cards left in their hand).
  3. LEARN. Every decision that led to a good reward is made a bit more
     likely, and every decision that led to a bad one a bit less likely.
     The network also learns to guess its own reward (the "value"), so it
     can tell "this went better than expected" from plain luck.
  4. Repeat. Every so often it plays deals against three simple bots
     (bots.py) to show how good it has got, and saves itself to nn_bot.pt.

Press Ctrl+C any time. The network is saved before it stops.
"""

import argparse
import random
import time
from pathlib import Path

import torch
from torch.distributions import Bernoulli, Categorical

from czech_cards import bots
from czech_cards.game import OVER, Game
from czech_cards.match import hand_score
from czech_cards.nn_bot import MODEL_FILE, THROW, CardNet, NetPlayer

NAMES = ["A", "B", "C", "D"]
POINTS_SCALE = 50      # rewards are divided by this, so they're around -1 to 1


def deal_scores(game: Game) -> list[int]:
    return [0 if seat == game.winner else hand_score(game.hands[seat])
            for seat in range(game.num_players)]


def points_ahead(scores: list[int], seat: int) -> float:
    others = [s for i, s in enumerate(scores) if i != seat]
    return sum(others) / len(others) - scores[seat]


# ----- 1 + 2: play and score --------------------------------------------------

def play_batch(net: CardNet, deals: int, rng: random.Random) -> list:
    """The network plays `deals` deals against itself. Returns every decision
    it made, each with the reward its seat got at the end of that deal."""
    player = NetPlayer(net, explore=True, record=True)
    rows = []
    for _ in range(deals):
        game = Game(NAMES, rng=random.Random(rng.randrange(2**31)))
        while game.phase != OVER:
            player(game, game.current)
        scores = deal_scores(game)
        for seat in range(game.num_players):
            reward = points_ahead(scores, seat) / POINTS_SCALE
            for obs, kind, action, mask in player.take_steps(game, seat):
                rows.append((obs, kind, action, mask, reward))
    return rows


# ----- 3: learn ----------------------------------------------------------------

def learn(net: CardNet, optimizer, rows: list, entropy_bonus: float) -> dict:
    obs = torch.stack([r[0] for r in rows])
    kinds = torch.tensor([r[1] for r in rows])
    actions = torch.tensor([r[2] for r in rows])
    masks = torch.stack([r[3] for r in rows])
    rewards = torch.tensor([r[4] for r in rows])

    yes_no, throw, value = net(obs)
    is_throw = kinds == THROW

    # How likely the network was to make each decision it made.
    yes_logit = yes_no.gather(1, kinds.clamp(max=2).unsqueeze(1)).squeeze(1)
    yes_dist = Bernoulli(logits=yes_logit)
    throw_dist = Categorical(logits=throw.masked_fill(~masks, -1e9))
    log_prob = torch.where(is_throw, throw_dist.log_prob(actions),
                           yes_dist.log_prob(actions.clamp(max=1).float()))
    entropy = torch.where(is_throw, throw_dist.entropy(), yes_dist.entropy())

    # Better than expected -> make that decision more likely, worse -> less likely.
    advantage = rewards - value.detach()
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    policy_loss = -(log_prob * advantage).mean()
    value_loss = ((value - rewards) ** 2).mean()          # learn to guess the reward
    loss = policy_loss + 0.5 * value_loss - entropy_bonus * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return {"value_loss": value_loss.item(), "entropy": entropy.mean().item()}


# ----- 4: measure against the simple bots --------------------------------------

def check(net: CardNet, deals: int, seed: int = 123_456) -> tuple[float, float]:
    """The network (best moves, no exploring) against three bots from bots.py.
    Same deals every time, so the numbers can be compared fairly.
    Returns (points ahead per deal, share of deals won)."""
    player = NetPlayer(net, explore=False)
    ahead = wins = 0.0
    for d in range(deals):
        game = Game(NAMES, rng=random.Random(seed + d))
        seat = d % 4                       # a different seat each deal
        while game.phase != OVER:
            (player if game.current == seat else bots.take_turn)(game, game.current)
        ahead += points_ahead(deal_scores(game), seat)
        wins += game.winner == seat
    return ahead / deals, wins / deals


def bar(points: float) -> str:
    return "█" * max(0, round(points))


def save(net, optimizer, step, history, path) -> None:
    torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "history": history}, path)


def train(args) -> None:
    torch.set_num_threads(1)           # a small network is fastest on one core
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    net = CardNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    step, history = 0, []
    if Path(args.out).exists() and not args.fresh:
        saved = torch.load(args.out, weights_only=False)
        net.load_state_dict(saved["net"])
        optimizer.load_state_dict(saved["optimizer"])
        step, history = saved["step"], saved["history"]
        print(f"Picking up from {args.out} (already trained for {step} rounds).")
    else:
        print("Starting from a brand-new network: its moves are random.")

    ahead, wins = check(net, args.check_deals)
    print(f"Against 3 simple bots: {ahead:+5.1f} points ahead per deal, wins {wins:5.1%} "
          f"of deals (a fair share is 25%)  {bar(ahead)}\n")
    print(f"Each round: {args.deals} deals of self-play, then one learning step. "
          f"Press Ctrl+C to stop.\n")

    begin = time.time()
    try:
        for _ in range(args.rounds):
            net.train()
            rows = play_batch(net, args.deals, rng)
            stats = learn(net, optimizer, rows, args.entropy)
            step += 1
            if step % args.check_every == 0:
                net.eval()
                ahead, wins = check(net, args.check_deals)
                history.append({"round": step, "ahead": round(ahead, 2), "wins": round(wins, 4)})
                print(f"Round {step:5}: {ahead:+5.1f} points ahead, wins {wins:5.1%}  "
                      f"{bar(ahead):20} (unsure-ness {stats['entropy']:.2f}, "
                      f"{time.time() - begin:.0f}s)")
                save(net, optimizer, step, history, args.out)
    except KeyboardInterrupt:
        print("\nStopping.")
    save(net, optimizer, step, history, args.out)
    print(f"Saved the network in {args.out} after {step} rounds of training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neural-network bot by self-play.")
    parser.add_argument("--fresh", action="store_true", help="start from a new random network")
    parser.add_argument("--check", action="store_true", help="only measure the saved bot")
    parser.add_argument("--rounds", type=int, default=2000, help="how many learning steps")
    parser.add_argument("--deals", type=int, default=16, help="self-play deals per round")
    parser.add_argument("--check-every", type=int, default=25, help="rounds between checks")
    parser.add_argument("--check-deals", type=int, default=200, help="deals per check")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate: how big each nudge is")
    parser.add_argument("--entropy", type=float, default=0.01, help="bonus for trying new things")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=MODEL_FILE)
    args = parser.parse_args()
    if args.check:
        torch.set_num_threads(1)
        net = CardNet()
        net.load_state_dict(torch.load(args.out, weights_only=False)["net"])
        net.eval()
        ahead, wins = check(net, 1000)
        print(f"Over 1000 deals against 3 simple bots: {ahead:+.1f} points ahead per deal, "
              f"wins {wins:.1%} of deals (a fair share is 25%).")
    else:
        train(args)


if __name__ == "__main__":
    main()