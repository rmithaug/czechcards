"""A basic computer player, used as a simple baseline for the learning bots.

It plays by the rules but has no strategy. Each turn it:
  1. Draws from the face-down pile. It never takes the discard.
  2. Looks for sets, then runs, in its hand (using jokers to finish
     nearly-made ones) and lays them down, but only if it has come out
     already or they add up to 40 or more.
  3. Once it has come out, adds any cards it can to melds on the table.
  4. Throws away a random card that isn't part of a set or run. Jokers are
     never thrown away while there's another card to throw.
It always keeps one card back, since every turn has to end with a throw.
"""

from .cards import Card
from .game import DRAW, Game, GameError
from .rules import OPENING_POINTS, is_meld, meld_points


def _sets(cards: list[Card]) -> list[list[Card]]:
    found = []
    for rank in range(1, 14):
        one_per_suit = {}
        for c in cards:
            if c.rank == rank and c.suit not in one_per_suit:
                one_per_suit[c.suit] = c
        if len(one_per_suit) >= 3:
            found.append(list(one_per_suit.values()))
    return found


def _runs(cards: list[Card]) -> list[list[Card]]:
    found = []
    used: set[int] = set()
    for suit in "HSDC":
        at = {}
        for c in cards:
            if c.suit == suit:
                at.setdefault(c.rank, c)
        if 1 in at:
            at[14] = at[1]  # the same Ace can also sit above the King
        streak: list[Card] = []
        for pos in range(1, 16):
            card = at.get(pos)
            if card is None or card.id in used or card in streak:
                if len(streak) >= 3:
                    found.append(streak)
                    used.update(c.id for c in streak)
                streak = []
                if card is not None and card.id not in used:
                    streak = [card]
                continue
            streak.append(card)
    return found


def find_melds(hand: list[Card]) -> list[list[Card]]:
    """Runs and sets in a hand. Not perfect, but quick and decent."""
    real = [c for c in hand if not c.is_joker]
    jokers = [c for c in hand if c.is_joker]

    melds = []
    left = list(real)
    for finder in (_sets, _runs):
        for meld in finder(left):
            melds.append(meld)
            ids = {c.id for c in meld}
            left = [c for c in left if c.id not in ids]

    # Use each joker to finish the best nearly-made meld (two cards that need one more).
    for joker in jokers:
        best = None
        for i, a in enumerate(left):
            for b in left[i + 1:]:
                trial = [a, b, joker]
                if is_meld(trial) and (best is None or meld_points(trial) > meld_points(best)):
                    best = trial
        if best is None:
            break
        melds.append(best)
        ids = {c.id for c in best}
        left = [c for c in left if c.id not in ids]
    return melds


def _lay_down(game: Game, seat: int) -> None:
    """Lay down every set and run we found, keeping one card to throw away."""
    melds = find_melds(game.hands[seat])
    while melds and sum(len(m) for m in melds) >= len(game.hands[seat]):
        melds.pop()  # too many: hold one meld back
    total = sum(meld_points(m) for m in melds)
    if melds and (game.came_out[seat] or total >= OPENING_POINTS):
        try:
            game.lay_down(seat, [[c.id for c in m] for m in melds])
        except GameError:
            pass


def _lay_off(game: Game, seat: int) -> None:
    """Add whatever fits onto the table, keeping one card to throw away."""
    added = True
    while added and len(game.hands[seat]) > 1:
        added = False
        for card in list(game.hands[seat]):
            for index, meld in enumerate(game.table):
                if is_meld(meld["cards"] + [card]):
                    game.lay_off(seat, card.id, index)
                    added = True
                    break
            if added or len(game.hands[seat]) <= 1:
                break


def _random_throw(game: Game, seat: int) -> Card:
    """A random card that isn't in a set or run (and isn't a joker, if possible)."""
    hand = game.hands[seat]
    in_melds = {c.id for m in find_melds(hand) for c in m}
    for choices in ([c for c in hand if c.id not in in_melds and not c.is_joker],
                    [c for c in hand if not c.is_joker],
                    hand):
        if choices:
            return game.rng.choice(choices)


def take_turn(game: Game, seat: int) -> None:
    if game.phase == DRAW:
        game.draw_from_pile(seat)                       # 1. draw
    _lay_down(game, seat)                               # 2. lay down sets and runs
    if game.came_out[seat]:
        _lay_off(game, seat)                            # 3. add to the table
    game.discard(seat, _random_throw(game, seat).id)    # 4. throw a random spare card