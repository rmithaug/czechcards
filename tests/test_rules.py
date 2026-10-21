"""Run with:  python3 -m unittest"""

import random
import unittest

from czech_cards.bots import find_melds, take_turn
from czech_cards.cards import Card, build_double_deck
from czech_cards.game import DRAW, OVER, PLAY, Game, GameError
from czech_cards.rules import arrange_meld, is_meld, is_run, is_set, meld_points

_ids = iter(range(1000, 100000))


def hand(text: str) -> list[Card]:
    """hand("AH 2H 3H JK") -> cards. JK is a joker."""
    cards = []
    for token in text.split():
        if token == "JK":
            cards.append(Card(next(_ids), None, 0))
            continue
        rank = {"A": 1, "J": 11, "Q": 12, "K": 13}.get(token[:-1]) or int(token[:-1])
        cards.append(Card(next(_ids), token[-1], rank))
    return cards


class DeckTests(unittest.TestCase):
    def test_double_deck_has_106_cards(self):
        deck = build_double_deck()
        self.assertEqual(len(deck), 106)
        self.assertEqual(sum(c.is_joker for c in deck), 2)


class MeldTests(unittest.TestCase):
    def test_runs(self):
        self.assertTrue(is_run(hand("4D 5D 6D")))
        self.assertTrue(is_run(hand("AS 2S 3S")))
        self.assertTrue(is_run(hand("QS KS AS")))
        self.assertFalse(is_run(hand("KS AS 2S")))   # no wrapping around
        self.assertFalse(is_run(hand("4D 5D 7D")))   # gap
        self.assertFalse(is_run(hand("4D 5H 6D")))   # mixed suits
        self.assertFalse(is_run(hand("4D 5D")))      # too short

    def test_sets(self):
        self.assertTrue(is_set(hand("7H 7S 7C")))
        self.assertTrue(is_set(hand("7H 7S 7C 7D")))
        self.assertFalse(is_set(hand("7H 7H 7C")))   # same suit twice
        self.assertFalse(is_set(hand("7H 7S 7C 7D 7H")))

    def test_joker_fills_a_gap_or_completes_a_set(self):
        self.assertTrue(is_run(hand("5H JK 7H")))
        self.assertTrue(is_run(hand("5H 6H JK")))
        self.assertTrue(is_set(hand("4H 4S JK")))
        self.assertTrue(is_meld(hand("9C JK JK")))
        self.assertFalse(is_meld(hand("5H JK 9H")))  # one joker can't fill two gaps
        self.assertFalse(is_meld(hand("JK JK")))

    def test_points(self):
        self.assertEqual(meld_points(hand("10H JH QH")), 30)
        self.assertEqual(meld_points(hand("AS 2S 3S")), 6)    # low Ace = 1
        self.assertEqual(meld_points(hand("QS KS AS")), 31)   # high Ace = 11
        self.assertEqual(meld_points(hand("AH AS AC")), 33)
        self.assertEqual(meld_points(hand("7H 7S JK")), 21)   # joker = the 7 it stands for
        self.assertEqual(meld_points(hand("5H 6H JK")), 18)   # joker goes high: 7♥

    def test_arrange_puts_joker_in_its_gap(self):
        labels = [c.label() for c in arrange_meld(hand("7H JK 5H"))]
        self.assertEqual(labels, ["5♥", "Joker", "7♥"])


def rigged_game(my_cards: str, starter: int = 0) -> Game:
    """A game where seat 0 holds exactly `my_cards`."""
    g = Game(["You", "Jana", "Petr", "Eva"], rng=random.Random(7), starter=starter)
    g.draw_from_pile(0)
    g.hands[0] = hand(my_cards)
    return g


class GameTests(unittest.TestCase):
    def test_deal(self):
        g = Game(["a", "b", "c", "d"], rng=random.Random(1), starter=2)
        self.assertEqual([len(h) for h in g.hands], [12, 12, 12, 12])
        self.assertEqual(len(g.pile), 106 - 48)
        self.assertEqual((g.current, g.phase), (2, DRAW))  # the starter draws too
        g.draw_from_pile(2)
        self.assertEqual(len(g.hands[2]), 13)

    def test_turn_order_is_enforced(self):
        g = Game(["a", "b", "c", "d"], rng=random.Random(1), starter=0)
        with self.assertRaises(GameError):
            g.discard(0, g.hands[0][0].id)  # have to draw first
        with self.assertRaises(GameError):
            g.draw_from_pile(1)  # not seat 1's turn
        g.draw_from_pile(0)
        g.discard(0, g.hands[0][0].id)
        self.assertEqual((g.current, g.phase), (1, DRAW))

    def test_coming_out_needs_40_in_one_go(self):
        g = rigged_game("10H JH QH 7S 7D 7C 2C 9D")
        run = [c.id for c in g.hands[0][:3]]      # 30 points
        trio = [c.id for c in g.hands[0][3:6]]    # 21 points
        with self.assertRaises(GameError):
            g.lay_down(0, [run])                  # 30 isn't enough
        g.lay_down(0, [run, trio])                # 51 together is fine
        self.assertTrue(g.came_out[0])
        self.assertEqual(len(g.table), 2)

    def test_after_coming_out_any_meld_goes(self):
        g = rigged_game("10H JH QH 7S 7D 7C 2C 3C 4C 9D")
        g.lay_down(0, [[c.id for c in g.hands[0][:3]], [c.id for c in g.hands[0][3:6]]])
        g.lay_down(0, [[c.id for c in g.hands[0][:3]]])  # 2-3-4♣ is only 9 points
        self.assertEqual(len(g.table), 3)

    def test_lay_off_onto_the_table(self):
        g = rigged_game("10H JH QH KH 7S 7D 7C 9H 2C")
        cards = g.hands[0]
        with self.assertRaises(GameError):
            g.lay_off(0, cards[7].id, 0)          # can't lay off before coming out
        g.lay_down(0, [[c.id for c in cards[:3]], [c.id for c in cards[4:7]]])
        g.lay_off(0, cards[3].id, 0)              # K♥ onto 10-J-Q♥
        g.lay_off(0, cards[7].id, 0)              # 9♥ onto the bottom
        self.assertEqual(len(g.table[0]["cards"]), 5)      # 9 10 J Q K♥
        with self.assertRaises(GameError):
            g.lay_off(0, cards[8].id, 0)          # 2♣ doesn't fit

    def test_take_discard_only_after_coming_out(self):
        g = Game(["a", "b", "c", "d"], rng=random.Random(3), starter=0)
        g.draw_from_pile(0)
        g.discard(0, g.hands[0][0].id)
        with self.assertRaises(GameError):
            g.take_discard(1)
        g.came_out[1] = True
        g.take_discard(1)
        self.assertEqual(len(g.hands[1]), 13)

    def test_win_by_emptying_your_hand(self):
        g = rigged_game("10H JH QH KH 7S 7D 7C 5D")
        g.lay_down(0, [[c.id for c in g.hands[0][:4]], [c.id for c in g.hands[0][4:7]]])
        g.discard(0, g.hands[0][0].id)
        self.assertEqual((g.phase, g.winner), (OVER, 0))

    def test_taken_discard_must_be_played_this_turn(self):
        g = rigged_game("8H 9H 2C 5D")
        g.came_out[0] = True
        g.discard(0, g.hands[0][3].id)            # throw 5♦; seats 1-3 take their turns
        for seat in (1, 2, 3):
            g.draw_from_pile(seat)
            g.discard(seat, g.hands[seat][0].id)
        g.discards.append(hand("10H")[0])         # pretend Eva threw 10♥
        g.take_discard(0)
        ten = g.must_play
        with self.assertRaises(GameError):
            g.discard(0, g.hands[0][0].id)        # can't throw until 10♥ is used
        g.lay_down(0, [[c.id for c in g.hands[0] if c.suit == "H"]])  # 8-9-10♥
        self.assertIsNone(g.must_play)
        self.assertIn(ten, g.table[-1]["cards"])
        g.discard(0, g.hands[0][0].id)

    def test_put_back_a_discard_you_cant_use(self):
        g = Game(["a", "b", "c", "d"], rng=random.Random(3), starter=0)
        g.draw_from_pile(0)
        thrown = g.discard(0, g.hands[0][0].id)
        g.came_out[1] = True
        g.take_discard(1)
        g.put_back(1)
        self.assertEqual(g.top_discard(), thrown)
        self.assertEqual(g.phase, DRAW)
        with self.assertRaises(GameError):
            g.take_discard(1)                     # not again this turn
        g.draw_from_pile(1)

    def test_must_keep_a_card_to_throw_away(self):
        g = rigged_game("10H JH QH 7S 7D 7C")
        with self.assertRaises(GameError):
            g.lay_down(0, [[c.id for c in g.hands[0][:3]], [c.id for c in g.hands[0][3:]]])
        g = rigged_game("10H JH QH 7S 7D 7C KH 9D")
        g.lay_down(0, [[c.id for c in g.hands[0][:3]], [c.id for c in g.hands[0][3:6]]])
        g.lay_off(0, g.hands[0][0].id, 0)         # K♥ onto the run, leaving 9♦
        with self.assertRaises(GameError):
            g.lay_off(0, g.hands[0][0].id, 0)     # 9♦ is the last card: must be thrown
        self.assertEqual(g.phase, PLAY)
        g.discard(0, g.hands[0][0].id)
        self.assertEqual((g.phase, g.winner), (OVER, 0))

    def test_pile_reshuffles_when_empty(self):
        g = Game(["a", "b", "c", "d"], rng=random.Random(5), starter=0)
        g.draw_from_pile(0)
        g.discards = g.pile[:10]
        g.pile = []
        g.discard(0, g.hands[0][0].id)
        top = g.top_discard()
        g.draw_from_pile(1)
        self.assertEqual(g.discards, [top])
        self.assertEqual(len(g.pile), 9)

    def test_bots_find_melds_with_jokers(self):
        melds = find_melds(hand("5H 6H 7H 9C 9D JK 2S 3S"))
        self.assertTrue(all(is_meld(m) for m in melds))
        self.assertEqual(sum(len(m) for m in melds), 6)  # 5-6-7♥ and 9♣ 9♦ joker

    def test_bots_play_complete_games(self):
        for seed in range(30):
            g = Game(["a", "b", "c", "d"], rng=random.Random(seed))
            while g.phase != OVER:
                take_turn(g, g.current)
            in_play = sum(map(len, g.hands)) + len(g.pile) + len(g.discards)
            on_table = sum(len(m["cards"]) for m in g.table)
            self.assertEqual(in_play + on_table, 106)
            self.assertTrue(all(is_meld(m["cards"]) for m in g.table))


if __name__ == "__main__":
    unittest.main()