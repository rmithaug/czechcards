# Step 1 in VS Code: make the cards and deal a hand

By the end of this step you'll run a program that shuffles two decks and
deals you 14 cards.

## 1. Set up the folder

1. Make a new empty folder somewhere, e.g. `Documents/czech-cards`.
2. In VS Code: **File → Open Folder…** and pick it.
3. Install the **Python** extension if VS Code asks (or from the Extensions
   tab on the left, search "Python", publisher Microsoft).
4. Open a terminal inside VS Code: **Terminal → New Terminal**. Check Python
   is there:

   ```
   python --version
   ```

   (On Mac, use `python3` everywhere instead of `python`.) You need 3.10 or
   newer. If it says "not found", install it from python.org and restart VS Code.

## 2. Make the files

In the Explorer on the left, create:

```
czech-cards/
├── czech_cards/
│   ├── __init__.py     (leave it empty)
│   └── cards.py        (paste from below)
└── try_it.py           (paste from below)
```

The empty `__init__.py` tells Python that `czech_cards` is a package you can import from.

### `czech_cards/cards.py`

```python
"""Cards and the double deck.

The game uses two 52-card decks: A-K in 4 suits, two of each, plus the
2 printed jokers. 106 cards in all.
"""

import random
from dataclasses import dataclass

SUITS = ["H", "S", "D", "C"]
SUIT_SYMBOLS = {"H": "♥", "S": "♠", "D": "♦", "C": "♣"}
RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


@dataclass(frozen=True)
class Card:
    id: int            # 0-105, so the two copies of a card can be told apart
    suit: str | None   # None means joker
    rank: int          # 1 (Ace) to 13 (King); 0 for jokers

    @property
    def is_joker(self) -> bool:
        return self.suit is None

    def label(self) -> str:
        if self.is_joker:
            return "Joker"
        return RANK_NAMES.get(self.rank, str(self.rank)) + SUIT_SYMBOLS[self.suit]


def build_double_deck() -> list[Card]:
    """Two 52-card decks plus their two jokers: 106 cards."""
    cards = []
    next_id = 0
    for _deck in range(2):
        for suit in SUITS:
            for rank in range(1, 14):
                cards.append(Card(next_id, suit, rank))
                next_id += 1
        cards.append(Card(next_id, None, 0))  # this deck's joker
        next_id += 1
    return cards

```

### `try_it.py`

```python
import random

from czech_cards.cards import build_double_deck

deck = build_double_deck()
print("Cards in the deck:", len(deck))  # should be 106

random.shuffle(deck)

# Deal yourself 14 cards and sort them by suit, then number.
hand = [deck.pop() for _ in range(14)]
hand.sort(key=lambda c: (c.suit or "Z", c.rank))
print("Your hand:", " ".join(card.label() for card in hand))
print("Cards left:", len(deck))
```

## 3. Run it

In the VS Code terminal:

```
python try_it.py
```

You should see something like:

```
Cards in the deck: 106
Your hand: 3♥ 9♥ Q♥ 2♠ 2♠ 10♠ K♠ A♦ 5♦ 8♦ 4♣ 6♣ J♣ Joker
Cards left: 92
```

Run it a few times; it's different each shuffle. If the suit symbols show as
`?` on Windows, run `chcp 65001` in the terminal first.

## 4. Play with it (this is how you learn)

* Change `range(14)` to `range(15)` to deal the starting player's hand.
* Deal 4 hands (one per player) and print each one.
* Count how many jokers you were dealt: `sum(card.is_joker for card in hand)`.

When that all makes sense, move on to `STEP2.md`. The finished version of this file is
`czech_cards/cards.py` in the main project if you want to compare.
