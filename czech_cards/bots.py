"""A simple computer player.

The bot only uses what a person in its seat could see. Each turn it:
  1. Draws. Once it has come out, it takes the top discard only if it can
     play it straight away (the rules say a taken discard must be played
     that turn); otherwise it draws from the pile.
  2. Looks for runs and sets in its hand (using jokers to finish nearly-made
     ones) and lays them down, but only if it has come out already or they
     add up to 40 or more.
  3. Once it has come out, adds any cards it can to melds on the table.
  4. Throws away its "loneliest" card: the one with the fewest cards of the
     same rank or close by in the same suit. It always keeps one card back
     for this, since every turn has to end with a throw.
"""

from .cards import Card
from .game import DRAW, Game, GameError
from .rules import OPENING_POINTS, card_points, is_meld, meld_points


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


def loneliness(card: Card, hand: list[Card]) -> tuple:
    """Higher = more alone. Bots throw away the loneliest card."""
    if card.is_joker:
        return (-100, 0)  # never throw a joker if there's anything else
    friends = 0
    for other in hand:
        if other.id == card.id or other.is_joker:
            continue
        if other.rank == card.rank and other.suit != card.suit:
            friends += 2
        elif other.suit == card.suit and 0 < abs(other.rank - card.rank) <= 2:
            friends += 3 - abs(other.rank - card.rank)
    return (-friends, card_points(card))


def _fits_table(game: Game, card: Card) -> bool:
    return any(is_meld(m["cards"] + [card]) for m in game.table)


def _meld_using(hand: list[Card], card: Card) -> list[Card] | None:
    """A meld from `hand` that includes `card`, leaving at least one card to throw away."""
    for meld in find_melds(hand):
        if card in meld and len(hand) - len(meld) >= 1:
            return meld
    return None


def _can_use(game: Game, seat: int, card: Card) -> bool:
    """Could we play this card straight away if we picked it up?"""
    hand = game.hands[seat] + [card]
    return (_fits_table(game, card) and len(hand) >= 2) or _meld_using(hand, card) is not None


def _play_taken_card(game: Game, seat: int) -> None:
    """We took the discard, so it has to go on the table this turn."""
    card = game.must_play
    meld = _meld_using(game.hands[seat], card)
    if meld is not None:
        game.lay_down(seat, [[c.id for c in meld]])
        return
    for index, table_meld in enumerate(game.table):
        if is_meld(table_meld["cards"] + [card]):
            game.lay_off(seat, card.id, index)
            return
    game.put_back(seat)  # shouldn't happen, but never get stuck
    game.draw_from_pile(seat)


def take_turn(game: Game, seat: int) -> None:
    # 1. Draw.
    if game.phase == DRAW:
        top = game.top_discard()
        if top is not None and game.came_out[seat] and _can_use(game, seat, top):
            game.take_discard(seat)
            _play_taken_card(game, seat)
        else:
            game.draw_from_pile(seat)

    # 2. Lay down runs and sets (always keeping one card to throw away).
    melds = find_melds(game.hands[seat])
    while melds and sum(len(m) for m in melds) >= len(game.hands[seat]):
        melds.pop()  # too many: hold one meld back
    total = sum(meld_points(m) for m in melds)
    if melds and (game.came_out[seat] or total >= OPENING_POINTS):
        try:
            game.lay_down(seat, [[c.id for c in m] for m in melds])
        except GameError:
            pass

    # 3. Add whatever fits onto the table, again keeping one card back.
    if game.came_out[seat]:
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

    # 4. Throw away the loneliest card. (Throwing away the last card wins.)
    hand = game.hands[seat]
    worst = max(hand, key=lambda c: loneliness(c, hand))
    game.discard(seat, worst.id)