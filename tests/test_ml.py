"""Tests for the learning bot and match play. Run with:  python3 -m unittest"""

import random
import unittest

from czech_cards import bots, smart_bot
from czech_cards.game import OVER, Game
from czech_cards.match import Match, hand_score
from czech_cards.tracker import Tracker
from czech_cards.rules import OPENING_POINTS, is_meld, meld_points
from tests.test_rules import hand


class BestMeldsTests(unittest.TestCase):
    def test_finds_the_split_that_uses_the_most_cards(self):
        # 7♥ could go in the set of 7s or the heart run; the best split uses both.
        cards = hand("5H 6H 7H 7S 7D 7C 9S 9D 9C KS")
        melds = smart_bot.best_melds(cards)
        self.assertTrue(all(is_meld(m) for m in melds))
        self.assertEqual(sum(len(m) for m in melds), 9)

    def test_always_keeps_a_card_to_throw_away(self):
        cards = hand("5H 6H 7H 9S 9D 9C")
        melds = smart_bot.best_melds(cards)
        self.assertLessEqual(sum(len(m) for m in melds), 5)

    def test_coming_out_needs_40_points(self):
        cards = hand("2H 3H 4H 5S 5D 5C 9C")   # only 24 points
        self.assertEqual(smart_bot.best_melds(cards, need_points=OPENING_POINTS), [])
        cards = hand("JH QH KH 5S 5D 5C 9C")   # 45 points
        melds = smart_bot.best_melds(cards, need_points=OPENING_POINTS)
        self.assertGreaterEqual(sum(meld_points(m) for m in melds), OPENING_POINTS)

    def test_uses_a_joker_to_fill_a_gap(self):
        cards = hand("4H 6H JK 9S")
        melds = smart_bot.best_melds(cards)
        self.assertEqual(len(melds), 1)
        self.assertTrue(is_meld(melds[0]))

    def test_must_include_the_taken_card(self):
        cards = hand("5H 6H 7H 9S 9D")
        taken = hand("9C")[0]
        melds = smart_bot.best_melds(cards + [taken], must_include=taken)
        self.assertTrue(any(taken in m for m in melds))


class SmartBotGameTests(unittest.TestCase):
    def test_plays_complete_games_against_old_bots(self):
        me = smart_bot.make_player(smart_bot.DEFAULT_WEIGHTS)
        for seed in range(20):
            game = Game(["A", "B", "C", "D"], rng=random.Random(seed))
            players = [me, bots.take_turn, me, bots.take_turn]
            while game.phase != OVER:
                players[game.current](game, game.current)  # raises if it breaks a rule

    def test_swaps_for_a_joker_and_plays_it(self):
        me = smart_bot.make_player(smart_bot.DEFAULT_WEIGHTS)
        swaps = 0
        for seed in range(40):
            game = Game(["A", "B", "C", "D"], rng=random.Random(seed))
            while game.phase != OVER:
                me(game, game.current)   # raises if a swapped joker isn't played
            swaps += sum("swaps" in line for line in game.log)
        self.assertGreater(swaps, 0)

    def test_load_weights_falls_back_to_defaults(self):
        self.assertEqual(smart_bot.load_weights("no_such_file.json"), smart_bot.DEFAULT_WEIGHTS)


class MatchTests(unittest.TestCase):
    def test_leftover_points(self):
        # 2-10 their number, J/Q/K 10, Ace 11, joker 12.
        self.assertEqual(hand_score(hand("2H 10S JD QC KH AS JK")), 2 + 10 + 10 + 10 + 10 + 11 + 12)

    def test_one_deal_per_player_and_the_dealer_moves(self):
        match = Match(["A", "B", "C"], rng=random.Random(1))
        dealers = []
        while True:
            dealers.append(match.dealer)
            self.assertEqual(match.game.current, (match.dealer + 1) % 3)  # left of the dealer starts
            while match.game.phase != OVER:
                bots.take_turn(match.game, match.game.current)
            match.score_deal()
            if match.over:
                break
            match.start_next_deal()
        self.assertEqual(match.deal_number, 3)
        self.assertEqual(sorted(dealers), [0, 1, 2])

    def test_scoring_a_deal(self):
        match = Match(["A", "B", "C", "D"], rng=random.Random(2))
        while match.game.phase != OVER:
            bots.take_turn(match.game, match.game.current)
        scores = match.score_deal()
        match.score_deal()   # scoring twice mustn't count twice
        for seat in range(4):
            expected = 0 if seat == match.game.winner else hand_score(match.game.hands[seat])
            self.assertEqual(scores[seat], expected)
        self.assertEqual(match.totals, scores)

    def test_cant_skip_ahead(self):
        match = Match(["A", "B"], rng=random.Random(3))
        with self.assertRaises(ValueError):
            match.start_next_deal()   # this deal isn't finished

    def test_lowest_total_wins(self):
        match = Match(["A", "B", "C", "D"], rng=random.Random(4))
        match.play_out([smart_bot.make_player(), bots.take_turn, bots.take_turn, bots.take_turn])
        self.assertTrue(match.over)
        self.assertEqual(len(match.deal_scores), 4)
        for seat in match.winners():
            self.assertEqual(match.totals[seat], min(match.totals))


if __name__ == "__main__":
    unittest.main()


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.game = Game(["A", "B", "C", "D"], rng=random.Random(5))
        self.game.hands[0] = hand("7H 8H KS")
        self.game.table = [{"owner": 1, "cards": hand("9H 10H JH")}]
        self.game.discards = hand("9H 2C")   # the second 9♥ is gone too
        self.tracker = Tracker(self.game, 0, trust_discards=1.0)

    def test_counts_unseen_copies(self):
        self.assertEqual(self.tracker.unseen(("H", 7)), 1)   # I hold one of the two
        self.assertEqual(self.tracker.unseen(("H", 9)), 0)   # table + discards
        self.assertEqual(self.tracker.unseen(("S", 5)), 2)
        self.assertIn(("H", 9), self.tracker.gone())

    def test_draw_chance(self):
        total = self.tracker.unseen_total()
        # 6♥ has 2 unseen copies, 9♥ none.
        self.assertAlmostEqual(self.tracker.draw_chance([("H", 6), ("H", 9)]), 2 / total)

    def test_reads_who_threw_what(self):
        self.game.log.append("B throws away 8D.")    # not a real label: ignored
        self.game.log.append("B throws away 8♦.")
        self.tracker.update()
        self.assertEqual(self.tracker.thrown_by[1], [("D", 8)])

    def test_a_throw_lowers_the_odds_of_holding_nearby_cards(self):
        before = self.tracker.holds_chance(1, ("D", 9))
        self.game.log.append("B throws away 8♦.")
        self.tracker.update()
        self.assertLess(self.tracker.holds_chance(1, ("D", 9)), before)
        self.assertEqual(self.tracker.holds_chance(1, ("D", 9)), 0.0)   # fully trusted

