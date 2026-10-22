"""A card tracker: what one player can know about the cards, and the odds.

It only uses what a person sitting in that seat could know:
  * their own hand,
  * every card on the table and in the discard pile,
  * who threw away what (read from the game log, like watching the table).

Everything else is "unseen": it's either in the face-down pile or in another
player's hand, and the tracker can't tell which. There are two copies of
every card (two decks) and two jokers, so for every card it knows how many
copies are still unseen. That gives realistic odds:

  * the chance that my next draw from the pile is a card I need, and
  * the chance that another player holds a certain card, which tells me how
    risky it is to throw them something.

Reading throws: if Jana threw away the 8♥, she probably isn't saving hearts
near the 8, or other 8s. How much to believe that is a dial
("trust_discards", from 0 = ignore it to 1 = she certainly has none).

When the pile runs out and the discards are shuffled back in, those cards
become unseen again, and the odds take that into account automatically.
"""

from collections import Counter

from .cards import SUITS, build_double_deck
from .game import Game
from .rules import is_meld

COPIES = Counter(c.face for c in build_double_deck())   # 2 of every card, and 2 jokers
JOKER = (None, 0)
FACE_OF_LABEL = {c.label(): c.face for c in build_double_deck()}


CARD_OF_FACE = {c.face: c for c in build_double_deck()}


def _faces_that_complete(face_a: tuple, face_b: tuple) -> list[tuple]:
    """Cards that would make a run or set of three with these two."""
    a, b = CARD_OF_FACE[face_a], CARD_OF_FACE[face_b]
    found = [f for f, card in CARD_OF_FACE.items()
             if f not in (JOKER, face_a, face_b) and is_meld([a, b, card])]
    return found + [JOKER] if found else []   # a joker fits wherever a real card does


# Worked out once: for every pair of cards, which third cards complete them.
_PAIRS: dict[tuple, list[tuple]] = {}


def completing(face_a: tuple, face_b: tuple) -> list[tuple]:
    key = tuple(sorted((face_a, face_b), key=str))
    if key not in _PAIRS:
        _PAIRS[key] = _faces_that_complete(*key)
    return _PAIRS[key]


def _partner_pairs(face: tuple) -> list[tuple[tuple, tuple]]:
    """Pairs of other cards that would make a run or set of three with `face`."""
    suit, rank = face
    others = [s for s in SUITS if s != suit]
    pairs = [((a, rank), (b, rank))                     # sets: two more of the same number
             for i, a in enumerate(others) for b in others[i + 1:]]
    spots = [rank] + ([14] if rank == 1 else [])       # an Ace can be low or high
    for spot in spots:                                  # runs: two neighbours in the suit
        for d1, d2 in ((-2, -1), (-1, 1), (1, 2)):
            q1, q2 = spot + d1, spot + d2
            if 1 <= q1 <= 14 and 1 <= q2 <= 14:
                pairs.append(((suit, 1 if q1 == 14 else q1), (suit, 1 if q2 == 14 else q2)))
    return pairs


class Tracker:
    def __init__(self, game: Game, seat: int, trust_discards: float = 0.5):
        self.game = game
        self.seat = seat
        self.trust = max(0.0, min(1.0, trust_discards))
        self.thrown_by: dict[int, list[tuple]] = {s: [] for s in range(game.num_players)}
        self._read = 0

    def update(self) -> None:
        """Read the new lines of the game log to see who threw away what."""
        for line in self.game.log[self._read:]:
            for seat, name in enumerate(self.game.names):
                for start in (f"{name} throws away ", "You throw away " if name == "You" else None):
                    if start and line.startswith(start):
                        label = line[len(start):].rstrip(".")
                        if label in FACE_OF_LABEL:
                            self.thrown_by[seat].append(FACE_OF_LABEL[label])
        self._read = len(self.game.log)

    # ----- counting ----------------------------------------------------------

    def seen(self) -> Counter:
        counts = Counter(c.face for c in self.game.hands[self.seat])
        for meld in self.game.table:
            counts.update(c.face for c in meld["cards"])
        counts.update(c.face for c in self.game.discards)
        return counts

    def unseen(self, face: tuple, seen: Counter | None = None) -> int:
        """How many copies of this card could still be in the pile or someone's hand."""
        seen = seen if seen is not None else self.seen()
        return max(0, COPIES[face] - seen[face])

    def unseen_total(self) -> int:
        others = sum(len(h) for s, h in enumerate(self.game.hands) if s != self.seat)
        return len(self.game.pile) + others

    def gone(self) -> list[tuple]:
        """Cards whose every copy is on the table or in the discards (none left anywhere)."""
        seen = self.seen()
        mine = Counter(c.face for c in self.game.hands[self.seat])
        return sorted(f for f in COPIES if f != JOKER and seen[f] - mine[f] >= COPIES[f])

    # ----- odds ----------------------------------------------------------------

    def draw_chance(self, faces: list[tuple], seen: Counter | None = None) -> float:
        """Chance that my next draw from the pile is one of these cards."""
        seen = seen if seen is not None else self.seen()
        total = self.unseen_total()
        if total == 0:
            return 0.0
        return min(1.0, sum(self.unseen(f, seen) for f in set(faces)) / total)

    def _doesnt_want(self, player: int, face: tuple) -> bool:
        """Did this player throw away something close to this card?"""
        for suit, rank in self.thrown_by[player]:
            if rank == face[1] or (suit == face[0] and abs(rank - face[1]) <= 2):
                return True
        return False

    def holds_chance(self, player: int, face: tuple, seen: Counter | None = None) -> float:
        """Chance that `player` holds at least one copy of this card.

        Every unseen card is equally likely to be anywhere unseen, so each copy
        is in their hand with chance (their cards / all unseen cards). Then we
        lower it if they threw away something close to it.
        """
        seen = seen if seen is not None else self.seen()
        total = self.unseen_total()
        copies = self.unseen(face, seen)
        if total == 0 or copies == 0:
            return 0.0
        each = len(self.game.hands[player]) / total
        if face != JOKER and self._doesnt_want(player, face):
            each *= 1 - self.trust
        return 1 - (1 - min(1.0, each)) ** copies

    def use_chance(self, player: int, face: tuple, seen: Counter | None = None) -> float:
        """Chance that `player` could make a new run or set with this card,
        because they hold two cards that go with it (or one and a joker)."""
        seen = seen if seen is not None else self.seen()
        miss_all = 1.0
        joker = self.holds_chance(player, JOKER, seen)
        for a, b in _partner_pairs(face):
            pa, pb = self.holds_chance(player, a, seen), self.holds_chance(player, b, seen)
            both = pa * pb
            with_joker = joker * (pa + pb - pa * pb)
            miss_all *= 1 - min(1.0, both + with_joker)
        return 1 - miss_all
