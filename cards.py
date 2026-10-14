##czech cards

import random
from dataclasses import dataclass

suits = ["H", "D", "C", "S"]
suitSymbols = {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}
rankNames = {1: "A", 11: "J", 12: "Q", 13: "K"}

@dataclass(frozen = True)
class Card:
    id: int
    suit: str | None
    rank: int

    @property
    def is_joker(self) -> bool:
        return self.suit is None

    def label(self) -> str:
        if self.is_joker:
            return "Joker"
        return rankNames.get(self.rank, str(self.rank)) + suitSymbols[self.suit]
    

def build_double_deck() -> list[Card]:
        cards = []
        next_id = 0
        for _deck in range(2):
            for suit in suits:
                for rank in range(1, 14):
                    cards.append(Card(next_id, suit, rank))
                    next_id += 1
            cards.append(Card(next_id, None, 0))
            next_id += 1

        return cards





