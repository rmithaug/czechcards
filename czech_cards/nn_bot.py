"""A bot whose decisions come from a neural network (PyTorch).

Nobody tells this bot a strategy. The rules stay in plain code (finding runs
and sets, what's allowed), and the network makes the four choices that
decide how well you play:

  TAKE      take the top discard? (only offered when it could be played)
  LAY_DOWN  lay down my runs and sets now, or keep them hidden?
  LAY_OFF   add cards to the table now, or hold on to them?
  THROW     which card to throw away

What the network sees (only what a person in that seat could see), turned
into 432 numbers:
  * my hand, and which of my cards are already in a run or set
  * the cards on the table and in the discard pile, and the top discard
  * for each other player: what they've thrown away, how many cards they
    hold, and whether they've come out
  * whether I've come out, and how big the pile is

At first its choices are random. train_nn.py makes it play thousands of
deals against itself and nudges it toward the choices that ended deals with
fewer points in hand.
"""

import torch
from torch import nn
from torch.distributions import Categorical

from .bots import find_melds
from .cards import Card, build_double_deck
from .game import DRAW, Game
from .rules import OPENING_POINTS, is_meld, meld_points

# Every different card face: 52 cards plus the joker.
FACES = sorted({c.face for c in build_double_deck()}, key=lambda f: (f[0] or "Z", f[1]))
FACE_INDEX = {face: i for i, face in enumerate(FACES)}
FACE_OF_LABEL = {c.label(): c.face for c in build_double_deck()}
FACE_LABELS = [next(label for label, f in FACE_OF_LABEL.items() if f == face) for face in FACES]
N = len(FACES)                       # 53

OBS_SIZE = 5 * N + 3 * (N + 2) + 2   # 432 numbers

TAKE, LAY_DOWN, LAY_OFF, THROW = range(4)

MODEL_FILE = "nn_bot.pt"


# ----- the network ------------------------------------------------------------

class CardNet(nn.Module):
    """432 numbers in. Out: three yes/no answers, a score for every card face,
    and a guess of how well this deal will go."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.yes_no = nn.Linear(hidden, 3)   # take? lay down? lay off?
        self.throw = nn.Linear(hidden, N)    # which card to throw away
        self.value = nn.Linear(hidden, 1)    # how well it expects the deal to go

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.yes_no(h), self.throw(h), self.value(h).squeeze(-1)


def load_net(path: str = MODEL_FILE) -> CardNet:
    net = CardNet()
    saved = torch.load(path, weights_only=False)
    net.load_state_dict(saved["net"])
    net.eval()
    return net


# ----- what the bot can see -----------------------------------------------------

class Memory:
    """Who threw away what, read from the game log (like watching the table)."""

    def __init__(self, game: Game):
        self.game = game
        self.thrown = [[] for _ in range(game.num_players)]
        self._read = 0

    def update(self) -> None:
        for line in self.game.log[self._read:]:
            for seat, name in enumerate(self.game.names):
                start = "You throw away " if name == "You" else f"{name} throws away "
                if line.startswith(start):
                    label = line[len(start):].rstrip(".")
                    if label in FACE_OF_LABEL:
                        self.thrown[seat].append(FACE_OF_LABEL[label])
        self._read = len(self.game.log)


def observe(game: Game, seat: int, memory: Memory) -> torch.Tensor:
    """Turn everything this seat can see into a list of 432 numbers."""
    x = [0.0] * OBS_SIZE

    def count(block: int, faces) -> None:
        for face in faces:
            x[block * N + FACE_INDEX[face]] += 0.5      # 2 copies of a card -> 1.0

    hand = game.hands[seat]
    count(0, (c.face for c in hand))
    count(1, (c.face for m in find_melds(hand) for c in m))
    count(2, (c.face for m in game.table for c in m["cards"]))
    count(3, (c.face for c in game.discards))
    top = game.top_discard()
    if top is not None:
        x[4 * N + FACE_INDEX[top.face]] = 1.0

    start = 5 * N
    for k in range(1, game.num_players):                 # next player, then the one after...
        other = (seat + k) % game.num_players
        for face in memory.thrown[other]:
            x[start + FACE_INDEX[face]] = min(1.0, x[start + FACE_INDEX[face]] + 0.5)
        x[start + N] = len(game.hands[other]) / 13
        x[start + N + 1] = 1.0 if game.came_out[other] else 0.0
        start += N + 2
    x[-2] = 1.0 if game.came_out[seat] else 0.0
    x[-1] = len(game.pile) / 106
    return torch.tensor(x)


# ----- rule helpers (plain code, no learning) -----------------------------------

def _fits_table(game: Game, card: Card) -> int | None:
    for index, meld in enumerate(game.table):
        if is_meld(meld["cards"] + [card]):
            return index
    return None


def _meld_using(hand: list[Card], card: Card) -> list[Card] | None:
    """A run or set from `hand` that includes `card`, keeping a card to throw away."""
    for meld in find_melds(hand):
        if card in meld and len(hand) - len(meld) >= 1:
            return meld
    return None


def _can_use(game: Game, seat: int, card: Card) -> bool:
    hand = game.hands[seat] + [card]
    return (_fits_table(game, card) is not None) or _meld_using(hand, card) is not None


def _play_taken_card(game: Game, seat: int) -> bool:
    card = game.must_play
    meld = _meld_using(game.hands[seat], card)
    if meld is not None:
        game.lay_down(seat, [[c.id for c in meld]])
        return True
    index = _fits_table(game, card)
    if index is not None and len(game.hands[seat]) > 1:
        game.lay_off(seat, card.id, index)
        return True
    return False


def _melds_to_lay_down(game: Game, seat: int) -> list[list[Card]]:
    hand = game.hands[seat]
    melds = find_melds(hand)
    while melds and sum(len(m) for m in melds) >= len(hand):
        melds.pop()                                       # keep a card to throw away
    if not game.came_out[seat] and sum(meld_points(m) for m in melds) < OPENING_POINTS:
        return []
    return melds


def _lay_off_all(game: Game, seat: int) -> None:
    added = True
    while added and len(game.hands[seat]) > 1:
        added = False
        for card in list(game.hands[seat]):
            index = _fits_table(game, card)
            if index is not None:
                game.lay_off(seat, card.id, index)
                added = True
                break


# ----- the player ---------------------------------------------------------------

class NetPlayer:
    """Plays turns with a CardNet. Use it like any bot: player(game, seat).

    explore=True  picks moves at random, weighted by the network's odds
                  (needed while training, so it tries new things).
    explore=False always picks the network's favourite move (for real games).
    record=True   remembers every decision, so train_nn.py can learn from it.
    """

    def __init__(self, net: CardNet, explore: bool = False, record: bool = False):
        self.net = net
        self.explore = explore
        self.record = record
        self.memories: dict[int, Memory] = {}
        self.steps: dict[tuple[int, int], list] = {}
        self.last_throw_odds: dict[str, float] = {}       # for watch_nn.py

    def _memory(self, game: Game, seat: int) -> Memory:
        memory = self.memories.get(seat)
        if memory is None or memory.game is not game:       # a new deal: start fresh
            memory = self.memories[seat] = Memory(game)
        memory.update()
        return memory

    def decide(self, game: Game, seat: int, kind: int) -> int:
        obs = observe(game, seat, self._memory(game, seat))
        mask = torch.ones(N, dtype=torch.bool)
        with torch.no_grad():
            yes_no, throw, _ = self.net(obs)
        if kind == THROW:
            mask = torch.zeros(N, dtype=torch.bool)
            for card in game.hands[seat]:
                mask[FACE_INDEX[card.face]] = True
            logits = throw.masked_fill(~mask, -1e9)
            odds = torch.softmax(logits, -1)
            self.last_throw_odds = {FACE_LABELS[i]: odds[i].item() for i in range(N) if mask[i]}
            action = Categorical(logits=logits).sample().item() if self.explore else logits.argmax().item()
        else:
            p_yes = torch.sigmoid(yes_no[kind]).item()
            action = int(torch.rand(()).item() < p_yes) if self.explore else int(p_yes > 0.5)
        if self.record:
            self.steps.setdefault((id(game), seat), []).append((obs, kind, action, mask))
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

        # 3. Lay off onto the table, if anything fits and the network says now.
        if (game.came_out[seat] and len(game.hands[seat]) > 1
                and any(_fits_table(game, c) is not None for c in game.hands[seat])
                and self.decide(game, seat, LAY_OFF)):
            _lay_off_all(game, seat)

        # 4. Throw away the card the network picks.
        face = FACES[self.decide(game, seat, THROW)]
        card = next(c for c in game.hands[seat] if c.face == face)
        game.discard(seat, card.id)


def load_player(path: str = MODEL_FILE) -> NetPlayer:
    """The trained bot, ready to play: take_turn = load_player()."""
    return NetPlayer(load_net(path))