"""A match: several deals with a running total. The lowest total wins.

  * A match has one deal per player, so with 4 players there are 4 deals.
    The dealer moves one seat to the left each deal, and the player after
    the dealer starts.
  * When a deal ends, the player who threw away their last card scores 0.
    Everyone else scores the cards left in their hand: 2-10 count their
    number, J/Q/K count 10, an Ace counts 11, and a joker counts 12. It
    doesn't matter whether you came out or not.
  * If a deal ends with nobody going out, everyone scores their hand.
  * After the last deal, the lowest total wins (a tie means shared first place).

This works on top of game.py, which plays one deal and knows nothing about
matches.
"""

import random

from .cards import Card
from .game import OVER, Game
from .rules import rank_value

LEFTOVER_JOKER_POINTS = 12


def leftover_points(card: Card) -> int:
    """What a card left in your hand costs you at the end of a deal."""
    return LEFTOVER_JOKER_POINTS if card.is_joker else rank_value(card.rank)


def hand_score(hand: list[Card]) -> int:
    return sum(leftover_points(c) for c in hand)


class Match:
    def __init__(self, names: list[str], rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.names = list(names)
        self.num_players = len(names)
        self.num_deals = self.num_players
        self.first_dealer = self.rng.randrange(self.num_players)
        self.totals = [0] * self.num_players
        self.deal_scores: list[list[int]] = []   # one list of scores per finished deal
        self.deal_number = 0
        self.game: Game | None = None
        self.start_next_deal()

    @property
    def dealer(self) -> int:
        return (self.first_dealer + self.deal_number - 1) % self.num_players

    @property
    def deal_scored(self) -> bool:
        return len(self.deal_scores) == self.deal_number

    @property
    def over(self) -> bool:
        return self.deal_number == self.num_deals and self.deal_scored

    def start_next_deal(self) -> None:
        if self.deal_number and not self.deal_scored:
            raise ValueError("Finish this deal first.")
        if self.deal_number == self.num_deals:
            raise ValueError("The match is over.")
        self.deal_number += 1
        starter = (self.dealer + 1) % self.num_players
        self.game = Game(self.names, rng=self.rng, starter=starter)
        self.game.log.insert(0, f"Deal {self.deal_number} of {self.num_deals}. "
                                f"{self.names[self.dealer]} deals.")

    def score_deal(self) -> list[int]:
        """Add up the deal once it's over. Safe to call more than once."""
        if self.game.phase != OVER:
            raise ValueError("This deal isn't over yet.")
        if not self.deal_scored:
            scores = [0 if seat == self.game.winner else hand_score(self.game.hands[seat])
                      for seat in range(self.num_players)]
            self.deal_scores.append(scores)
            self.totals = [t + s for t, s in zip(self.totals, scores)]
        return self.deal_scores[-1]

    def winners(self) -> list[int]:
        """The seats with the lowest total (more than one if there's a tie)."""
        best = min(self.totals)
        return [seat for seat, total in enumerate(self.totals) if total == best]

    def play_out(self, players: list) -> None:
        """Play the whole match with players[seat] = a take_turn(game, seat) function."""
        while True:
            while self.game.phase != OVER:
                players[self.game.current](self.game, self.game.current)
            self.score_deal()
            if self.over:
                return
            self.start_next_deal()

    def view_for(self, seat: int) -> dict:
        """The deal as `seat` sees it, plus the match scores."""
        data = self.game.view_for(seat)
        deal_over = self.game.phase == OVER
        for i, player in enumerate(data["players"]):
            player["total"] = self.totals[i]
            if deal_over:   # show the match score, not game.py's own count
                player["hand_points"] = 0 if i == self.game.winner else hand_score(self.game.hands[i])
        data["match"] = {
            "deal": self.deal_number,
            "deals": self.num_deals,
            "dealer": self.dealer,
            "totals": self.totals,
            "deal_scores": self.deal_scores,
            "over": self.over,
            "winners": self.winners() if self.over else [],
        }
        return data
