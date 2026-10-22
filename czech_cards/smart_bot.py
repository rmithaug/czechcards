"""A computer player that can LEARN: its choices are steered by numbers ("weights").

The bot in bots.py follows fixed rules of thumb. This one works differently:

  1. It finds runs and sets carefully. It tries every way of splitting its
     hand into runs and sets (jokers included) and picks the split that gets
     rid of the most cards.

  2. Every other choice is made by adding up weights, like dials on a mixing
     desk. To decide which card to throw away, it gives every card in its
     hand a "keep score":

         keep score = same_rank         x (cards of the same number it holds)
                    + same_suit_next    x (cards right next to it in its suit)
                    + points_when_close x (its points, if someone is about to go out)
                    + ...

     and throws away the card with the lowest score.

Nobody sets the dials by hand. train_match.py finds good settings by letting
bots play thousands of matches and breeding the winners (a "genetic
algorithm"), then saves them in match_weights.json. load_weights() reads that
file, and falls back to DEFAULT_WEIGHTS (rough, untrained guesses) if it
isn't there yet.

It also keeps a card tracker (tracker.py): which cards are gone, who threw
away what, and the odds that a card is still out there. Two dials use those
odds: "draw_chance" and "risk_next_player".
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from .cards import SUITS, Card
from .game import DRAW, Game, GameError
from .rules import OPENING_POINTS, card_points, is_meld, joker_swap_target, meld_points
from .tracker import Tracker, completing

# What each weight means. A positive weight means "this is a reason to KEEP
# the card"; a negative one means "this is a reason to throw it away".
DEFAULT_WEIGHTS = {
    "in_meld": 8.0,             # card is already part of a run or set in my hand
    "same_rank": 2.0,           # for each card I hold with the same number, other suit
    "same_suit_next": 2.0,      # for each card I hold next to it in the same suit (5♥ 6♥)
    "same_suit_gap": 1.0,       # for each card I hold two away in the same suit (5♥ 7♥)
    "duplicate": -1.0,          # I hold the exact same card twice
    "live_outs": 0.2,           # for each card still unseen that would go with it
    "points_before_out": 0.05,  # x its points, before I've come out (helps reach 40)
    "points_after_out": 0.0,    # x its points, after I've come out
    "fits_table": 0.5,          # it could go onto a meld on the table later
    "helps_next_player": 3.0,   # the next player has come out and could take it and play it
    # Decisions that aren't about throwing away (above 0 means "yes"):
    "take_for_meld": 1.0,       # take the top discard when it makes a new run or set with my cards
    "take_for_table": 1.0,      # take the top discard when it only fits a meld on the table
    "joker_layoff_hand": 3.0,   # only add a joker to the table when I hold this many cards or fewer
    "lay_down_when_left": 13.0, # after coming out, lay down new runs/sets only if that leaves me
                                # this many cards or fewer (holding them back means other
                                # players can't add their cards to them)
    "lay_off_when_left": 13.0,  # the same, for adding my cards to melds on the table
    # For match play, where leftover cards count against you. "How close" is
    # 1 when another player who has come out holds 1 card, 0.5 with 2 cards, 0.33 with 3...
    "points_when_close": 0.0,   # x its points x how close another player is to going out
    "stop_holding_at": 2.0,     # when "how close" reaches this, put everything I can on the
                                # table (2 means never, since "how close" is at most 1)
    "swap_for_jokers": 1.0,     # swap my cards for jokers on the table (above 0 means yes)
    # From the card tracker (tracker.py). Both are in percent, 0 to 100.
    "draw_chance": 0.3,         # x the % chance my next draw makes a run/set with this card
    "risk_next_player": 0.1,    # x the % chance the next player could use it in a new run/set
    "trust_discards": 0.5,      # how much to believe "they threw an 8♥, so they don't want hearts
                                # near 8" (0 = ignore it, 1 = completely sure)
}

WEIGHTS_FILE = Path(__file__).resolve().parent.parent / "match_weights.json"


def load_weights(path: Path | str = WEIGHTS_FILE) -> dict:
    """The trained weights from match_weights.json, or the defaults if it's missing."""
    weights = dict(DEFAULT_WEIGHTS)
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return weights
    for name, value in data.get("weights", data).items():
        if name in weights:
            weights[name] = float(value)
    return weights


# ----- finding the best runs and sets --------------------------------------

def _candidates(hand: list[Card]) -> list[tuple]:
    """Every run or set the hand could make, as (faces, jokers needed, size, points).

    A "face" is (suit, rank) without the card's id, so it doesn't matter which
    of the two copies of a card we hold.
    """
    real = [c for c in hand if not c.is_joker]
    jokers = [c for c in hand if c.is_joker]
    by_face = {}
    for c in real:
        by_face.setdefault(c.face, c)

    found = {}

    def add(faces: list[tuple], num_jokers: int) -> None:
        key = (frozenset(faces), num_jokers)
        if key in found:
            return
        cards = [by_face[f] for f in faces] + jokers[:num_jokers]
        if is_meld(cards):
            found[key] = (tuple(faces), num_jokers, len(cards), meld_points(cards))

    # Sets: 3 or 4 of the same number, different suits.
    for rank in range(1, 14):
        faces = [(s, rank) for s in SUITS if (s, rank) in by_face]
        for size in range(1, len(faces) + 1):
            for combo in combinations(faces, size):
                for j in range(len(jokers) + 1):
                    if 3 <= size + j <= 4:
                        add(list(combo), j)

    # Runs: any stretch of one suit, with jokers filling the gaps.
    for suit in SUITS:
        ranks = {r for (s, r) in by_face if s == suit}
        spots = ranks | ({14} if 1 in ranks else set())   # an Ace can also sit above the King
        for start in range(1, 13):
            for end in range(start + 2, min(start + 12, 14) + 1):
                present = [p for p in range(start, end + 1) if p in spots]
                missing = (end - start + 1) - len(present)
                if missing > len(jokers):
                    break
                if present:
                    add([(suit, 1 if p == 14 else p) for p in present], missing)
    return list(found.values())


def best_melds(hand: list[Card], need_points: int = 0, must_include: Card | None = None,
               keep: int = 1, fewest: bool = False, max_steps: int = 20000) -> list[list[Card]]:
    """The runs and sets to lay down from `hand`, or [] if there are none.

    Picks the split that lays down the most cards (then the most points), or
    with fewest=True the fewest cards that still do the job. Always keeps
    `keep` cards in hand, and only accepts splits worth at least `need_points`
    that use `must_include` (when given).
    """
    cands = _candidates(hand)
    if not cands:
        return []
    counts = Counter(c.face for c in hand if not c.is_joker)
    num_jokers = sum(c.is_joker for c in hand)
    limit = len(hand) - keep
    need_face = must_include.face if must_include is not None and not must_include.is_joker else None
    need_joker = must_include is not None and must_include.is_joker

    best = {"key": None, "pick": None}
    steps = [0]

    def consider(chosen: list[int], used: int, jokers_used: int) -> None:
        if not chosen:
            return
        points = sum(cands[k][3] for k in chosen)
        if points < need_points:
            return
        if need_face is not None and not any(need_face in cands[k][0] for k in chosen):
            return
        if need_joker and jokers_used == 0:
            return
        key = (-used if fewest else used, points, -jokers_used)
        if best["key"] is None or key > best["key"]:
            best["key"], best["pick"] = key, list(chosen)

    def search(start: int, chosen: list[int], used: int, jokers_left: int) -> None:
        steps[0] += 1
        consider(chosen, used, num_jokers - jokers_left)
        if steps[0] > max_steps:
            return
        for k in range(start, len(cands)):
            faces, nj, size, _ = cands[k]
            if used + size > limit or nj > jokers_left:
                continue
            if all(counts[f] > 0 for f in faces):
                for f in faces:
                    counts[f] -= 1
                chosen.append(k)
                search(k, chosen, used + size, jokers_left - nj)
                chosen.pop()
                for f in faces:
                    counts[f] += 1

    search(0, [], 0, num_jokers)
    if best["pick"] is None:
        return []

    # Turn faces back into real cards, using `must_include` itself where it fits.
    pool: dict[tuple, list[Card]] = {}
    for c in hand:
        if not c.is_joker:
            pool.setdefault(c.face, []).append(c)
    jokers = [c for c in hand if c.is_joker]
    if must_include is not None:
        if must_include.is_joker:
            jokers.remove(must_include)
            jokers.insert(0, must_include)
        else:
            pool[must_include.face].remove(must_include)
            pool[must_include.face].insert(0, must_include)
    melds = []
    for k in best["pick"]:
        faces, nj, _, _ = cands[k]
        melds.append([pool[f].pop(0) for f in faces] + [jokers.pop(0) for _ in range(nj)])
    return melds


# ----- what the bot can see --------------------------------------------------

def closeness(game: Game, seat: int) -> float:
    """How close another player is to going out: 1 / (fewest cards held by
    anyone else who has come out), or 0 if nobody else has come out."""
    counts = [len(game.hands[s]) for s in range(game.num_players)
              if s != seat and game.came_out[s]]
    return 1 / max(1, min(counts)) if counts else 0.0


def _fits_table(game: Game, card: Card) -> int | None:
    """Index of a table meld this card could be added to, or None."""
    for index, meld in enumerate(game.table):
        if is_meld(meld["cards"] + [card]):
            return index
    return None


def _neighbours(rank: int, distance: int) -> list[int]:
    """Ranks `distance` away in a run. The Ace (1) is next to both 2 and K."""
    spots = [rank] + ([14] if rank == 1 else [])
    found = set()
    for spot in spots:
        for other in (spot - distance, spot + distance):
            if 1 <= other <= 14:
                found.add(1 if other == 14 else other)
    found.discard(rank)
    return sorted(found)


def card_features(game: Game, seat: int, hand: list[Card],
                  tracker: Tracker | None = None) -> dict[int, dict]:
    """For every card in `hand`, the facts the keep score is built from."""
    tracker = tracker or Tracker(game, seat)
    tracked = tracker.seen()
    in_meld = {c.id for m in best_melds(hand, keep=1) for c in m}
    seen = Counter(c.face for c in hand)
    for meld in game.table:
        seen.update(c.face for c in meld["cards"])
    seen.update(c.face for c in game.discards)

    def unseen(suit: str, rank: int) -> int:
        return max(0, 2 - seen[(suit, rank)])

    next_seat = game.next_seat(seat)
    next_can_take = game.came_out[next_seat]
    came_out = game.came_out[seat]
    faces = Counter(c.face for c in hand if not c.is_joker)
    close = closeness(game, seat)

    features = {}
    for card in hand:
        if card.is_joker:
            continue
        others = [o for o in hand if o.id != card.id and not o.is_joker]
        same_rank = len({o.suit for o in others if o.rank == card.rank and o.suit != card.suit})
        near = {o.rank for o in others if o.suit == card.suit}
        next_to = sum(r in near for r in _neighbours(card.rank, 1))
        gap = sum(r in near for r in _neighbours(card.rank, 2))
        live = sum(unseen(s, card.rank) for s in SUITS if s != card.suit)
        live += sum(unseen(card.suit, r) for d in (1, 2) for r in _neighbours(card.rank, d))
        fits = _fits_table(game, card) is not None
        points = card_points(card)
        # Cards that would turn this one and another card in my hand into a run or set.
        outs = {f for o in others if o.face != card.face for f in completing(card.face, o.face)}
        draw = 100 * tracker.draw_chance(list(outs), tracked)
        risk = 100 * tracker.use_chance(next_seat, card.face, tracked) if next_can_take else 0.0
        features[card.id] = {
            "in_meld": 1.0 if card.id in in_meld else 0.0,
            "same_rank": same_rank,
            "same_suit_next": next_to,
            "same_suit_gap": gap,
            "duplicate": 1.0 if faces[card.face] > 1 else 0.0,
            "live_outs": live,
            "points_before_out": 0 if came_out else points,
            "points_after_out": points if came_out else 0,
            "fits_table": 1.0 if fits else 0.0,
            "helps_next_player": 1.0 if (fits and next_can_take) else 0.0,
            "points_when_close": points * close,
            "draw_chance": draw,
            "risk_next_player": risk,
        }
    return features


def keep_score(features: dict, weights: dict) -> float:
    return sum(weights[name] * value for name, value in features.items())


# ----- the turn --------------------------------------------------------------

def _wants_discard(game: Game, seat: int, weights: dict) -> bool:
    top = game.top_discard()
    if top is None or not game.came_out[seat] or game.put_back_this_turn:
        return False
    if best_melds(game.hands[seat] + [top], must_include=top):
        return weights["take_for_meld"] > 0
    if _fits_table(game, top) is not None:
        return weights["take_for_table"] > 0
    return False


def _play_taken_card(game: Game, seat: int) -> bool:
    """We took the discard, so it has to go on the table this turn."""
    card = game.must_play
    melds = best_melds(game.hands[seat], must_include=card)
    if melds:
        game.lay_down(seat, [[c.id for c in m] for m in melds])
        return True
    index = _fits_table(game, card)
    if index is not None and len(game.hands[seat]) > 1:
        game.lay_off(seat, card.id, index)
        return True
    return False


def _lay_down(game: Game, seat: int, weights: dict) -> None:
    hand = game.hands[seat]
    melds = best_melds(hand)
    if not melds:
        return
    left = len(hand) - sum(len(m) for m in melds)
    hold_back = (left > weights["lay_down_when_left"]
                 and closeness(game, seat) < weights["stop_holding_at"])
    if not game.came_out[seat]:
        # Coming out: if we're holding cards back, show only as much as it takes to reach 40.
        melds = best_melds(hand, need_points=OPENING_POINTS, fewest=hold_back)
    elif hold_back:
        return
    if melds:
        game.lay_down(seat, [[c.id for c in m] for m in melds])


def _play_swapped_joker(game: Game, seat: int, joker: Card) -> None:
    """We took this joker off the table, so it has to go back down this turn.
    Best is new runs/sets with cards from our hand; otherwise onto a table meld."""
    melds = best_melds(game.hands[seat], must_include=joker)
    if melds:
        game.lay_down(seat, [[c.id for c in m] for m in melds])
        return
    index = _fits_table(game, joker)
    if index is not None:
        game.lay_off(seat, joker.id, index)


def _swap_jokers(game: Game, seat: int, weights: dict) -> None:
    """Trade cards from our hand for jokers on the table, playing each joker straight away."""
    if not game.came_out[seat] or weights["swap_for_jokers"] <= 0:
        return
    swapped = True
    while swapped and len(game.hands[seat]) > 1:
        swapped = False
        for card in list(game.hands[seat]):
            for index, meld in enumerate(game.table):
                if joker_swap_target(meld["cards"], card) is None:
                    continue
                try:
                    joker = game.swap_joker(seat, card.id, index)
                except GameError:
                    continue      # nowhere to play the joker this turn
                _play_swapped_joker(game, seat, joker)
                swapped = True
                break
            if swapped:
                break


def _lay_off(game: Game, seat: int, weights: dict) -> None:
    if not game.came_out[seat]:
        return
    fitting = [c for c in game.hands[seat] if _fits_table(game, c) is not None]
    if (len(game.hands[seat]) - len(fitting) > weights["lay_off_when_left"]
            and closeness(game, seat) < weights["stop_holding_at"]):
        return
    added = True
    while added and len(game.hands[seat]) > 1:
        added = False
        hand = game.hands[seat]
        for card in sorted(hand, key=lambda c: c.is_joker):   # real cards first
            if card.is_joker and len(hand) > weights["joker_layoff_hand"]:
                continue
            index = _fits_table(game, card)
            if index is not None:
                game.lay_off(seat, card.id, index)
                added = True
                break


def _discard(game: Game, seat: int, weights: dict, tracker: Tracker) -> dict[str, dict]:
    """Throw away the card with the lowest keep score.
    Returns, for every card, its keep score and the tracker's odds (for watch_game.py)."""
    hand = game.hands[seat]
    features = card_features(game, seat, hand, tracker)
    if not features:          # only jokers left
        game.discard(seat, hand[0].id)
        return {}
    scores = {c.id: keep_score(features[c.id], weights) for c in hand if not c.is_joker}
    worst = min((c for c in hand if not c.is_joker),
                key=lambda c: (scores[c.id], -card_points(c)))
    details = {c.label(): {"score": round(scores[c.id], 1),
                           "draw": round(features[c.id]["draw_chance"]),
                           "risk": round(features[c.id]["risk_next_player"])}
               for c in hand if not c.is_joker}
    game.discard(seat, worst.id)
    return details


def take_turn(game: Game, seat: int, weights: dict | None = None,
              tracker: Tracker | None = None) -> dict[str, dict]:
    """Play one whole turn for `seat`: draw, lay down, swap jokers, lay off, throw away.

    Pass the same tracker every turn of a deal so it remembers who threw what
    (make_player does this for you). Returns what _discard returns."""
    weights = weights or DEFAULT_WEIGHTS
    tracker = tracker or Tracker(game, seat, weights["trust_discards"])
    tracker.update()
    if game.phase == DRAW:
        if _wants_discard(game, seat, weights):
            game.take_discard(seat)
            if not _play_taken_card(game, seat):
                game.put_back(seat)
                game.draw_from_pile(seat)
        else:
            game.draw_from_pile(seat)
    _lay_down(game, seat, weights)
    _swap_jokers(game, seat, weights)
    _lay_off(game, seat, weights)
    return _discard(game, seat, weights, tracker)


def make_player(weights: dict | None = None):
    """A take_turn(game, seat) function that always uses these weights,
    with its own card tracker that it keeps for the whole deal."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    trackers: dict[int, Tracker] = {}

    def play(game: Game, seat: int) -> dict[str, dict]:
        tracker = trackers.get(seat)
        if tracker is None or tracker.game is not game:     # a new deal: start a new tracker
            tracker = trackers[seat] = Tracker(game, seat, weights["trust_discards"])
        return take_turn(game, seat, weights, tracker)
    return play
