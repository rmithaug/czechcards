"""The game itself: dealing, turns, laying down, laying off and winning.

This module knows nothing about the web or about bots. Every action names the
seat that is acting, and the game checks that it's really that seat's turn.
That's what will let real people sit in the other seats later.

A turn (seats go 0 -> 1 -> 2 -> 3 -> 0):
  1. Draw one card (so you hold 13): from the face-down pile, or (once
     you've come out) the top card of the discard pile. A card taken from
     the discards has to be played in a run or set THAT turn. If you can't,
     put it back and draw from the pile instead.
  2. Optionally lay down runs and sets. Your FIRST lay-down ("coming out")
     has to be worth 40 points or more, all in one go.
  3. Once you've come out, optionally add cards to any run or set on the
     table ("laying off").
  4. Throw one card onto the discard pile. EVERY turn ends this way, so you
     always have to keep at least one card back to throw away.
You win by throwing away your last card. When the pile runs out, the
discard pile (all but its top card) is shuffled into a new pile.
"""

import random

from .cards import Card, shuffled_deck
from .rules import (OPENING_POINTS, arrange_meld, card_points, is_meld,
                    joker_stands_for, meld_points)

DRAW, PLAY, OVER = "draw", "play", "over"
HAND_SIZE = 12     # everyone is dealt 12 and draws a 13th each turn
MAX_TURNS = 2000  # safety net so a game can never run forever


class GameError(Exception):
    """Raised when a player tries something that isn't allowed."""


class Game:
    def __init__(self, names: list[str], rng: random.Random | None = None, starter: int | None = None):
        if not 2 <= len(names) <= 4:
            raise ValueError("Czech Cards is played with 2 to 4 players")
        self.rng = rng or random.Random()
        self.names = list(names)
        self.num_players = len(names)
        self.log: list[str] = []

        deck = shuffled_deck(self.rng)
        self.current = self.rng.randrange(self.num_players) if starter is None else starter
        self.hands: list[list[Card]] = []
        for seat in range(self.num_players):
            self.hands.append([deck.pop() for _ in range(HAND_SIZE)])

        self.pile: list[Card] = deck
        self.discards: list[Card] = []
        self.table: list[dict] = []          # each meld: {"owner": seat, "cards": [...]}
        self.came_out = [False] * self.num_players
        self.phase = DRAW
        self.turns = 0
        self.winner: int | None = None
        self.must_play: Card | None = None   # a card taken from the discards this turn
        self.put_back_this_turn = False       # after putting it back you must use the pile

        self._say(f"Everyone gets {HAND_SIZE} cards. {self._who(self.current, 'starts')}.")

    # ----- helpers ---------------------------------------------------------

    def _say(self, text: str) -> None:
        self.log.append(text)

    def _who(self, seat: int, verb: str) -> str:
        """Builds "Jana draws" or, for the human, "You draw"."""
        name = self.names[seat]
        if name == "You":
            verb = verb[:-2] if verb.endswith(("shes", "ses")) else verb[:-1]
            return f"You {verb}"
        return f"{name} {verb}"

    def next_seat(self, seat: int) -> int:
        return (seat + 1) % self.num_players

    def top_discard(self) -> Card | None:
        return self.discards[-1] if self.discards else None

    def _check_turn(self, seat: int, phase: str) -> None:
        if self.phase == OVER:
            raise GameError("The game is over.")
        if seat != self.current:
            raise GameError("It's not your turn.")
        if self.phase != phase:
            raise GameError("Draw a card first." if self.phase == DRAW
                            else "You already drew this turn.")

    def _cards_from_hand(self, seat: int, card_ids: list[int]) -> list[Card]:
        if len(set(card_ids)) != len(card_ids):
            raise GameError("You picked the same card twice.")
        by_id = {c.id: c for c in self.hands[seat]}
        missing = [i for i in card_ids if i not in by_id]
        if missing:
            raise GameError("That card isn't in your hand.")
        return [by_id[i] for i in card_ids]

    def _remove_from_hand(self, seat: int, cards: list[Card]) -> None:
        ids = {c.id for c in cards}
        self.hands[seat] = [c for c in self.hands[seat] if c.id not in ids]

    def _played(self, card_ids: list[int]) -> None:
        """Cards went onto the table. If one was the discard you took, you've used it."""
        if self.must_play is not None and self.must_play.id in card_ids:
            self.must_play = None

    def _check_for_win(self, seat: int) -> bool:
        if not self.hands[seat]:
            self.phase = OVER
            self.winner = seat
            if self.names[seat] == "You":
                self._say("You threw away your last card. You win!")
            else:
                self._say(f"{self.names[seat]} threw away their last card and wins!")
            return True
        return False

    def _refill_pile(self) -> None:
        """Shuffle the discards (all but the top card) into a new pile."""
        if len(self.discards) <= 1:
            return
        top = self.discards.pop()
        self.pile = self.discards
        self.rng.shuffle(self.pile)
        self.discards = [top]
        self._say("The pile ran out, so the discards were shuffled into a new pile.")

    # ----- actions ---------------------------------------------------------

    def draw_from_pile(self, seat: int) -> Card:
        self._check_turn(seat, DRAW)
        if not self.pile:
            self._refill_pile()
        if not self.pile:
            raise GameError("There are no cards left to draw.")
        card = self.pile.pop()
        self.hands[seat].append(card)
        self.phase = PLAY
        self._say(f"{self._who(seat, 'draws')} from the pile.")
        return card

    def take_discard(self, seat: int) -> Card:
        self._check_turn(seat, DRAW)
        if not self.came_out[seat]:
            raise GameError("You can only take the discard after you've come out.")
        if self.put_back_this_turn:
            raise GameError("You put that card back, so draw from the pile this time.")
        if not self.discards:
            raise GameError("There's no discard to take.")
        card = self.discards.pop()
        self.hands[seat].append(card)
        self.phase = PLAY
        self.must_play = card
        self._say(f"{self._who(seat, 'takes')} {card.label()} from the discards.")
        return card

    def put_back(self, seat: int) -> None:
        """Changed your mind about the discard you took: put it back, then draw from the pile."""
        self._check_turn(seat, PLAY)
        card = self.must_play
        if card is None or card not in self.hands[seat]:
            raise GameError("There's nothing to put back.")
        self._remove_from_hand(seat, [card])
        self.discards.append(card)
        self.must_play = None
        self.put_back_this_turn = True
        self.phase = DRAW
        self._say(f"{self._who(seat, 'puts')} {card.label()} back.")

    def lay_down(self, seat: int, groups: list[list[int]]) -> None:
        """Lay down one or more melds, each given as a list of card ids."""
        self._check_turn(seat, PLAY)
        if not groups or any(not g for g in groups):
            raise GameError("Pick some cards to lay down.")
        all_ids = [i for g in groups for i in g]
        self._cards_from_hand(seat, all_ids)  # checks they're all yours, no repeats
        melds = [self._cards_from_hand(seat, g) for g in groups]
        for meld in melds:
            if not is_meld(meld):
                raise GameError(f"{' '.join(c.label() for c in meld)} isn't a run or a set.")
        if len(all_ids) >= len(self.hands[seat]):
            raise GameError("Keep at least one card: every turn has to end by throwing one away.")
        total = sum(meld_points(m) for m in melds)
        if not self.came_out[seat] and total < OPENING_POINTS:
            raise GameError(f"That's {total} points. To come out you need "
                            f"{OPENING_POINTS} or more, all at once.")

        for meld in melds:
            self._remove_from_hand(seat, meld)
            self.table.append({"owner": seat, "cards": arrange_meld(meld)})
            self._say(f"{self._who(seat, 'lays')} down "
                      f"{' '.join(c.label() for c in arrange_meld(meld))}.")
        if not self.came_out[seat]:
            self.came_out[seat] = True
            self._say(f"{self._who(seat, 'comes')} out with {total} points.")
        self._played(all_ids)

    def lay_off(self, seat: int, card_id: int, meld_index: int) -> None:
        """Add one card from your hand to a meld on the table."""
        self._check_turn(seat, PLAY)
        if not self.came_out[seat]:
            raise GameError("You can only add to the table after you've come out.")
        if not 0 <= meld_index < len(self.table):
            raise GameError("There's no such meld on the table.")
        (card,) = self._cards_from_hand(seat, [card_id])
        if len(self.hands[seat]) <= 1:
            raise GameError("Keep your last card: every turn has to end by throwing one away.")
        meld = self.table[meld_index]
        new_cards = meld["cards"] + [card]
        if not is_meld(new_cards):
            raise GameError(f"{card.label()} doesn't fit there.")
        self._remove_from_hand(seat, [card])
        meld["cards"] = arrange_meld(new_cards)
        self._say(f"{self._who(seat, 'adds')} {card.label()} to "
                  f"{' '.join(c.label() for c in meld['cards'])}.")
        self._played([card.id])

    def discard(self, seat: int, card_id: int) -> Card:
        self._check_turn(seat, PLAY)
        (card,) = self._cards_from_hand(seat, [card_id])
        if self.must_play is not None:
            raise GameError(f"You took {self.must_play.label()} from the discards, so you have to "
                            f"use it in a run or set this turn (or put it back).")
        self._remove_from_hand(seat, [card])
        self.discards.append(card)
        self._say(f"{self._who(seat, 'throws')} away {card.label()}.")
        if self._check_for_win(seat):
            return card
        self.current = self.next_seat(seat)
        self.phase = DRAW
        self.put_back_this_turn = False
        self.turns += 1
        if self.turns >= MAX_TURNS or (not self.pile and len(self.discards) <= 1):
            self.phase = OVER
            self._say("No cards left to play with. Nobody wins this round.")
        return card

    # ----- what a player is allowed to see --------------------------------

    def view_for(self, seat: int) -> dict:
        """Everything `seat` may see, as plain data (ready to send as JSON).

        Other players' hands are hidden; only their card counts are shown.
        """
        def card_data(card: Card | None, stands_for: str | None = None):
            if card is None:
                return None
            data = {"id": card.id, "suit": card.suit, "rank": card.rank,
                    "label": card.label(), "joker": card.is_joker}
            if stands_for:
                data["stands_for"] = stands_for
            return data

        game_over = self.phase == OVER
        table = []
        for meld in self.table:
            stands = joker_stands_for(meld["cards"])
            table.append({
                "owner": self.names[meld["owner"]],
                "cards": [card_data(c, stands.get(c.id)) for c in meld["cards"]],
                "points": meld_points(meld["cards"]),
            })
        return {
            "seat": seat,
            "phase": self.phase,
            "current": self.current,
            "your_turn": self.current == seat and not game_over,
            "hand": [card_data(c) for c in self.hands[seat]],
            "came_out": self.came_out[seat],
            "must_play": card_data(self.must_play) if self.current == seat else None,
            "opening_points": OPENING_POINTS,
            "pile_count": len(self.pile),
            "top_discard": card_data(self.top_discard()),
            "can_take": (self.came_out[seat] and bool(self.discards)
                         and not self.put_back_this_turn),
            "table": table,
            "players": [
                {
                    "name": name,
                    "card_count": len(self.hands[i]),
                    "came_out": self.came_out[i],
                    # at the end, show what everyone was stuck holding
                    "hand": [card_data(c) for c in self.hands[i]] if game_over else None,
                    "hand_points": sum(card_points(c) for c in self.hands[i]) if game_over else None,
                }
                for i, name in enumerate(self.names)
            ],
            "winner": self.winner,
            "log": self.log[-60:],
        }