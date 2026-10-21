"""Cards and the double deck.

The game uses two standard 52-card decks: ranks A-K in 4 suits, two copies
of each (104 cards), plus the 2 printed jokers (one from each deck), which are wild.
"""

import random
from dataclasses import dataclass

SUITS = ["H", "S", "D", "C"]
SUIT_SYMBOLS = {"H": "♥", "S": "♠", "D": "♦", "C": "♣"}
RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


@dataclass(frozen=True)
class Card:
    id: int            # unique 0-105, so the two copies of a card can be told apart
    suit: str | None   # None for jokers
    rank: int          # 1 (Ace) to 13 (King); 0 for jokers

    @property
    def is_joker(self) -> bool:
        return self.suit is None

    @property
    def face(self) -> tuple[str, int]:
        """(suit, rank) without the id, so both copies compare equal."""
        return (self.suit, self.rank)

    def label(self) -> str:
        if self.is_joker:
            return "Joker"
        return RANK_NAMES.get(self.rank, str(self.rank)) + SUIT_SYMBOLS[self.suit]

    def __repr__(self) -> str:
        return f"{self.label()}#{self.id}"


def build_double_deck() -> list[Card]:
    """Two 52-card decks plus their two jokers: 106 cards."""
    cards = []
    next_id = 0
    for _copy in range(2):
        for suit in SUITS:
            for rank in range(1, 14):
                cards.append(Card(next_id, suit, rank))
                next_id += 1
        cards.append(Card(next_id, None, 0))  # this deck's joker
        next_id += 1
    return cards


def shuffled_deck(rng: random.Random | None = None) -> list[Card]:
    deck = build_double_deck()
    (rng or random).shuffle(deck)
    return deck
