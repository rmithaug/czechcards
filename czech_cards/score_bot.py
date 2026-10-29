"""The scoreboard bot: a neural network that plays the whole match, not one deal.

It's the network bot from nn_bot.py with two upgrades:

  1. It sees the scoreboard: how far ahead or behind each other player is,
     whether it's leading, and how many deals are left in the match.
  2. It decides card by card whether to lay a card off, so it can choose to
     stop with 3 cards in hand instead of always laying off everything.

The network's size can be chosen (how many layers, how many "neurons" per
layer). A bigger network has more room to learn instincts like card counting,
but needs more games to fill that room.

train_score.py teaches it by self-play over whole matches.

To know the score, the player needs the match it's playing in:
    player = load_score_player()
    player.match = match        # before each match
"""

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from .game import DRAW, Game
from .match import Match
from .nn_bot import (FACE_INDEX, FACE_LABELS, FACES, LAY_DOWN, N, TAKE, THROW, Memory,
                     _can_use, _fits_table, _melds_to_lay_down, _play_taken_card, observe)
from .nn_bot import OBS_SIZE as CARD_OBS_SIZE

SCORE_SIZE = 5                        # the scoreboard numbers (see scoreboard below)
OBS_SIZE = CARD_OBS_SIZE + SCORE_SIZE  # 437 numbers
LAY_OFF_CARD = 2                       # "lay off this card?" (one question per card)

MODEL_FILE = "score_bot_best.pt"       # the best version training has found
LATEST_FILE = "score_bot_latest.pt"    # where training is right now (to carry on later)


# ----- the network ------------------------------------------------------------

class ScoreNet(nn.Module):
    """437 numbers in. Out: two yes/no answers (take the discard? lay down?),
    a lay-off answer for every card face, a score for every card face to throw
    away, and a guess of how well things will go.

    hidden = neurons per layer, layers = how many layers deep."""

    def __init__(self, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.config = {"hidden": hidden, "layers": layers}
        parts, size = [], OBS_SIZE
        for _ in range(layers):
            parts += [nn.Linear(size, hidden), nn.ReLU()]
            size = hidden
        self.body = nn.Sequential(*parts)
        self.yes_no = nn.Linear(hidden, 2)    # take? lay down?
        self.lay_off = nn.Linear(hidden, N)   # lay off this card now?
        self.throw = nn.Linear(hidden, N)     # which card to throw away
        self.value = nn.Linear(hidden, 1)     # how well it expects things to go

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.yes_no(h), self.lay_off(h), self.throw(h), self.value(h).squeeze(-1)


def judge(net: ScoreNet, obs, kinds, actions, masks, faces):
    """For a batch of past decisions: how likely the network is NOW to make each
    of them (log-probability), how unsure it is (entropy), and its value guess."""
    yes_no, lay_off, throw, value = net(obs)
    is_throw = kinds == THROW
    yes_logit = yes_no.gather(1, kinds.clamp(max=1).unsqueeze(1)).squeeze(1)
    off_logit = lay_off.gather(1, faces.unsqueeze(1)).squeeze(1)
    yes_dist = Bernoulli(logits=torch.where(kinds == LAY_OFF_CARD, off_logit, yes_logit))
    throw_dist = Categorical(logits=throw.masked_fill(~masks, -1e9))
    log_prob = torch.where(is_throw, throw_dist.log_prob(actions),
                           yes_dist.log_prob(actions.clamp(max=1).float()))
    entropy = torch.where(is_throw, throw_dist.entropy(), yes_dist.entropy())
    return log_prob, entropy, value


def from_card_net(saved: dict) -> ScoreNet:
    """A ScoreNet (the standard size) that starts out playing exactly like a
    trained nn_bot.pt network. The scoreboard inputs start switched off."""
    net = ScoreNet()
    with torch.no_grad():
        first = net.body[0]
        first.weight.zero_()
        first.weight[:, :CARD_OBS_SIZE] = saved["body.0.weight"]
        first.bias.copy_(saved["body.0.bias"])
        net.body[2].weight.copy_(saved["body.2.weight"])
        net.body[2].bias.copy_(saved["body.2.bias"])
        net.yes_no.weight.copy_(saved["yes_no.weight"][:2])
        net.yes_no.bias.copy_(saved["yes_no.bias"][:2])
        net.lay_off.weight.copy_(saved["yes_no.weight"][2].expand(N, -1))   # same answer for
        net.lay_off.bias.fill_(saved["yes_no.bias"][2].item())               # every card at first
        net.throw.weight.copy_(saved["throw.weight"])
        net.throw.bias.copy_(saved["throw.bias"])
        net.value.weight.copy_(saved["value.weight"])
        net.value.bias.copy_(saved["value.bias"])
    return net


def load_score_net(path: str = MODEL_FILE) -> ScoreNet:
    saved = torch.load(path, weights_only=False)
    net = ScoreNet(**saved.get("config", {"hidden": 256, "layers": 2}))
    net.load_state_dict(saved["net"])
    net.eval()
    return net


# ----- what the bot can see -----------------------------------------------------

def scoreboard(match: Match | None, game: Game, seat: int) -> list[float]:
    """The match situation as 5 numbers (all 0 when it isn't playing a match):
       3 x how far ahead I am of each other player (their total minus mine, /100),
       whether I'm leading, and how many deals are left after this one (/12)."""
    x = [0.0] * SCORE_SIZE
    if match is None or match.game is not game:
        return x
    mine = match.totals[seat]
    for k in range(1, 4):
        other = (seat + k) % match.num_players
        x[k - 1] = (match.totals[other] - mine) / 100
    x[3] = 1.0 if mine == min(match.totals) else 0.0
    x[4] = (match.num_deals - match.deal_number) / 12
    return x


# ----- the player ---------------------------------------------------------------

class ScorePlayer:
    """Plays turns with a ScoreNet. Use it like any bot: player(game, seat).
    Set player.match to the match it's playing, so it can see the scoreboard.

    explore=True  picks moves at random, weighted by the network's odds
                  (needed while training, so it tries new things).
    explore=False always picks the network's favourite move (for real games).
    record=True   remembers every decision, so training can learn from it.
    """

    def __init__(self, net: ScoreNet, explore: bool = False, record: bool = False):
        self.net = net
        self.explore = explore
        self.record = record
        self.match: Match | None = None
        self.memories: dict[int, Memory] = {}
        self.steps: dict[tuple[int, int], list] = {}
        self.last_throw_odds: dict[str, float] = {}

    def _observe(self, game: Game, seat: int) -> torch.Tensor:
        memory = self.memories.get(seat)
        if memory is None or memory.game is not game:       # a new deal: start fresh
            memory = self.memories[seat] = Memory(game)
        memory.update()
        cards = observe(game, seat, memory)
        return torch.cat([cards, torch.tensor(scoreboard(self.match, game, seat))])

    def decide(self, game: Game, seat: int, kind: int, face: tuple | None = None) -> int:
        obs = self._observe(game, seat)
        mask = torch.ones(N, dtype=torch.bool)
        face_index = FACE_INDEX[face] if face is not None else 0
        with torch.no_grad():
            yes_no, lay_off, throw, _ = self.net(obs)
            if kind == THROW:
                mask = torch.zeros(N, dtype=torch.bool)
                for card in game.hands[seat]:
                    mask[FACE_INDEX[card.face]] = True
                dist = Categorical(logits=throw.masked_fill(~mask, -1e9))
                odds = dist.probs
                self.last_throw_odds = {FACE_LABELS[i]: odds[i].item() for i in range(N) if mask[i]}
                choice = dist.sample() if self.explore else odds.argmax()
            else:
                logit = lay_off[face_index] if kind == LAY_OFF_CARD else yes_no[kind]
                dist = Bernoulli(logits=logit)
                choice = dist.sample() if self.explore else (logit > 0).float()
            log_prob = dist.log_prob(choice).item()
        action = int(choice.item())
        if self.record:
            self.steps.setdefault((id(game), seat), []).append(
                (obs, kind, action, mask, face_index, log_prob))
        return action

    def take_steps(self, game: Game, seat: int) -> list:
        """Hand over (and forget) the decisions this seat made in this deal."""
        return self.steps.pop((id(game), seat), [])

    def __call__(self, game: Game, seat: int) -> None:
        # 1. Draw: take the discard only if it could be played, and the network says yes.
        if game.phase == DRAW:
            top = game.top_discard()
            if (top is not None and game.came_out[seat] and not game.put_back_this_turn
                    and _can_use(game, seat, top) and self.decide(game, seat, TAKE)):
                game.take_discard(seat)
                if not _play_taken_card(game, seat):
                    game.put_back(seat)
                    game.draw_from_pile(seat)
            else:
                game.draw_from_pile(seat)

        # 2. Lay down runs and sets, if allowed and the network says now.
        melds = _melds_to_lay_down(game, seat)
        if melds and self.decide(game, seat, LAY_DOWN):
            game.lay_down(seat, [[c.id for c in m] for m in melds])

        # 3. Lay off card by card: the network says yes or no to each card that fits.
        if game.came_out[seat]:
            asked: set[int] = set()
            while len(game.hands[seat]) > 1:
                card = next((c for c in game.hands[seat] if c.id not in asked
                             and _fits_table(game, c) is not None), None)
                if card is None:
                    break
                asked.add(card.id)
                if self.decide(game, seat, LAY_OFF_CARD, card.face):
                    game.lay_off(seat, card.id, _fits_table(game, card))

        # 4. Throw away the card the network picks.
        face = FACES[self.decide(game, seat, THROW)]
        card = next(c for c in game.hands[seat] if c.face == face)
        game.discard(seat, card.id)


def load_score_player(path: str = MODEL_FILE) -> ScorePlayer:
    """A trained scoreboard bot. Remember to set player.match before each match."""
    return ScorePlayer(load_score_net(path))