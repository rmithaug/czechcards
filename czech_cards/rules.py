"""Czech Cards rules: what counts as a run or a set, and what it's worth.

Melds:
  * Run:  3 or more cards of the SAME suit in a row, e.g. 4♥ 5♥ 6♥.
          The Ace can be low (A-2-3) or high (Q-K-A), but a run never wraps
          around (K-A-2 is not allowed).
  * Set:  3 or 4 cards of the SAME rank in DIFFERENT suits, e.g. 7♥ 7♠ 7♣.

Jokers are wild: a joker can stand in for any card, so it can be the third 7
in a set or fill the gap in 5♥ _ 7♥. A meld still needs at least one real card.

Swapping a joker: if a joker on the table stands for one exact card and you
hold that card, you can put your card in its place and take the joker. In a
run a joker always stands for one exact card. In a set it only does once the
set has all 4 suits, because before that it could still be either missing suit.

Points (used for the 40-point opening):
  * 2-10 are worth their number, J/Q/K are worth 10.
  * An Ace is worth 11, or 1 when it sits at the bottom of a run (A-2-3).
  * A joker is worth whatever card it stands in for.
"""

from .cards import Card

OPENING_POINTS = 40
JOKER_HAND_POINTS = 25  # what a joker left in your hand counts against you


def rank_value(rank: int) -> int:
    """Points for a card of this rank in a set, or left in your hand."""
    if rank == 1:
        return 11
    return 10 if rank >= 11 else rank


def run_value(position: int) -> int:
    """Points for a spot in a run. Position 1 is a low Ace, 14 is a high Ace."""
    if position == 1:
        return 1
    if position == 14:
        return 11
    return 10 if position >= 11 else position


def card_points(card: Card) -> int:
    """What a card in your hand is worth."""
    return JOKER_HAND_POINTS if card.is_joker else rank_value(card.rank)


def _real_cards(cards: list[Card]) -> list[Card]:
    return [c for c in cards if not c.is_joker]


def is_set(cards: list[Card]) -> bool:
    real = _real_cards(cards)
    if not 3 <= len(cards) <= 4 or not real:
        return False
    same_rank = len({c.rank for c in real}) == 1
    different_suits = len({c.suit for c in real}) == len(real)
    return same_rank and different_suits


def run_layout(cards: list[Card]) -> list[int] | None:
    """The spots a run covers, low to high (14 = high Ace), or None if it isn't a run.

    When the jokers could go in more than one place, they go as high as
    possible, since that's worth the most.
    """
    real = _real_cards(cards)
    n = len(cards)
    if n < 3 or n > 13 or not real:
        return None
    if len({c.suit for c in real}) != 1:
        return None

    best = None
    for ace_high in (False, True):
        ranks = sorted(14 if (c.rank == 1 and ace_high) else c.rank for c in real)
        if len(set(ranks)) != len(ranks):
            return None  # the same card twice can't be in one run
        first_start = max(1, ranks[-1] - n + 1)
        last_start = min(ranks[0], 14 - n + 1)
        for start in range(first_start, last_start + 1):
            layout = list(range(start, start + n))
            if best is None or layout[-1] > best[-1]:
                best = layout
    return best


def is_run(cards: list[Card]) -> bool:
    return run_layout(cards) is not None


def is_meld(cards: list[Card]) -> bool:
    return is_set(cards) or is_run(cards)


def meld_points(cards: list[Card]) -> int:
    """What a meld is worth. Jokers count as the card they stand in for."""
    if is_set(cards):
        rank = _real_cards(cards)[0].rank
        return rank_value(rank) * len(cards)
    layout = run_layout(cards)
    return 0 if layout is None else sum(run_value(p) for p in layout)


def arrange_meld(cards: list[Card]) -> list[Card]:
    """Put a meld in display order, with each joker in the gap it fills."""
    if is_set(cards):
        return sorted(cards, key=lambda c: (c.is_joker, c.suit or ""))
    layout = run_layout(cards)
    if layout is None:
        return list(cards)
    jokers = [c for c in cards if c.is_joker]
    at = {}
    for c in _real_cards(cards):
        at[14 if (c.rank == 1 and 14 in layout) else c.rank] = c
    return [at[p] if p in at else jokers.pop() for p in layout]


def joker_stands_for(cards: list[Card]) -> dict[int, str]:
    """For display: card id of each joker -> the card it stands in for, like '6♥'."""
    from .cards import RANK_NAMES, SUIT_SYMBOLS
    result = {}
    if is_set(cards):
        # Until the set has all 4 suits, a joker could still be any missing suit.
        rank = _real_cards(cards)[0].rank
        missing = [s for s in "HSDC" if s not in {c.suit for c in cards}]
        label = RANK_NAMES.get(rank, str(rank)) + "/".join(SUIT_SYMBOLS[s] for s in missing)
        for joker in (c for c in cards if c.is_joker):
            result[joker.id] = label
        return result
    layout = run_layout(cards)
    if layout is None:
        return result
    suit = _real_cards(cards)[0].suit
    for card, pos in zip(arrange_meld(cards), layout):
        if card.is_joker:
            rank = 1 if pos == 14 else pos
            result[card.id] = RANK_NAMES.get(rank, str(rank)) + SUIT_SYMBOLS[suit]
    return result


def joker_swap_target(cards: list[Card], card: Card) -> Card | None:
    """The joker in this meld that `card` could replace, or None.

    In a run, a joker can be swapped for exactly the card it stands in for.
    In a set, only once the set has 4 cards: then every suit is spoken for,
    so the joker is "locked in" as the one suit that's missing.
    """
    if card.is_joker or not any(c.is_joker for c in cards):
        return None
    if is_set(cards):
        real = _real_cards(cards)
        if len(cards) == 4 and card.rank == real[0].rank and card.suit not in {c.suit for c in real}:
            return next(c for c in cards if c.is_joker)
        return None
    layout = run_layout(cards)
    if layout is None or card.suit != _real_cards(cards)[0].suit:
        return None
    for placed, pos in zip(arrange_meld(cards), layout):
        if placed.is_joker and card.rank == (1 if pos == 14 else pos):
            return placed
    return None