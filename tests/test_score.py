"""Tests for the scoreboard bot. Run with: python3 -m unittest"""

import os
import random
import tempfile
import unittest

import torch

from czech_cards import bots
from czech_cards.game import Game
from czech_cards.match import Match
from czech_cards.nn_bot import N, THROW, CardNet, Memory, observe
from czech_cards.score_bot import (OBS_SIZE, ScoreNet, ScorePlayer, from_card_net, judge,
                                   load_score_net, scoreboard)
from train_score import check, deal_returns, learn, play_batch, save_best


class TestScoreNet(unittest.TestCase):
    def test_shapes(self):
        for net in (ScoreNet(), ScoreNet(hidden=64, layers=3)):
            yes_no, lay_off, throw, value = net(torch.zeros(4, OBS_SIZE))
            self.assertEqual(tuple(yes_no.shape), (4, 2))
            self.assertEqual(tuple(lay_off.shape), (4, N))
            self.assertEqual(tuple(throw.shape), (4, N))
            self.assertEqual(tuple(value.shape), (4,))

    def test_starts_out_playing_like_the_trained_card_net(self):
        card_net = CardNet()
        net = from_card_net(card_net.state_dict())
        game = Game(["A", "B", "C", "D"], rng=random.Random(2))
        cards = observe(game, 0, Memory(game))
        scores = torch.tensor([0.3, -0.2, 0.1, 1.0, 0.5])     # any scoreboard
        _, card_throw, _ = card_net(cards)
        _, _, throw, _ = net(torch.cat([cards, scores]))
        self.assertTrue(torch.allclose(card_throw, throw, atol=1e-5))

    def test_saving_keeps_the_size(self):
        net = ScoreNet(hidden=32, layers=3)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "best.pt")
            save_best(net, 0, 0.0, 0.0, path)
            self.assertEqual(load_score_net(path).config, {"hidden": 32, "layers": 3})

    def test_judge_agrees_with_the_odds_used_while_playing(self):
        torch.manual_seed(0)
        net = ScoreNet(hidden=32)
        player = ScorePlayer(net, explore=True, record=True)
        game = Game(["A", "B", "C", "D"], rng=random.Random(3))
        seat = game.current
        player(game, seat)
        steps = player.take_steps(game, seat)
        self.assertTrue(any(s[1] == THROW for s in steps))
        obs, kinds, actions, masks, faces, old = zip(*steps)
        log_prob, _, _ = judge(net, torch.stack(obs), torch.tensor(kinds), torch.tensor(actions),
                               torch.stack(masks), torch.tensor(faces))
        self.assertTrue(torch.allclose(log_prob, torch.tensor(old), atol=1e-5))


class TestScoreboard(unittest.TestCase):
    def test_numbers(self):
        match = Match(["A", "B", "C", "D"], rng=random.Random(1))
        match.num_deals = 12
        match.totals = [50, 80, 30, 50]
        x = scoreboard(match, match.game, 0)
        self.assertEqual(x[:3], [0.3, -0.2, 0.0])     # B is 30 behind me, C 20 ahead, D level
        self.assertEqual(x[3], 0.0)                   # I'm not leading
        self.assertAlmostEqual(x[4], 11 / 12)         # 11 deals left after this one

    def test_zeros_outside_a_match(self):
        game = Game(["A", "B", "C", "D"], rng=random.Random(1))
        self.assertEqual(scoreboard(None, game, 0), [0.0] * 5)


class TestGrades(unittest.TestCase):
    def test_later_deals_count_less_and_earlier_ones_not_at_all(self):
        grades = deal_returns([1.0, 0.0, 2.0], bonus=0.3, discount=0.5)
        self.assertAlmostEqual(grades[2], 2.0 + 0.3)
        self.assertAlmostEqual(grades[1], 0.0 + 0.5 * 2.0 + 0.5 * 0.3)
        self.assertAlmostEqual(grades[0], 1.0 + 0.25 * 2.0 + 0.25 * 0.3)


class TestPlaying(unittest.TestCase):
    def test_plays_a_whole_match_by_the_rules(self):
        torch.manual_seed(0)
        player = ScorePlayer(ScoreNet(), explore=True)
        match = Match(["A", "B", "C", "D"], rng=random.Random(5))
        player.match = match
        match.play_out([player, bots.take_turn, bots.take_turn, bots.take_turn])
        self.assertTrue(match.over)

    def test_league_play_and_one_round_of_learning(self):
        torch.manual_seed(0)
        net = ScoreNet(hidden=64, layers=3)
        league = [ScoreNet(hidden=64, layers=3).eval()]
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        rows = play_batch(net, 2, 1, random.Random(0), 0.3, 0.5, league,
                          simple_share=0.3, past_share=0.4)
        self.assertTrue(rows)
        self.assertEqual(len(rows[0]), 7)
        before = [p.clone() for p in net.parameters()]
        learn(net, optimizer, rows, 0.01, epochs=2, minibatch=256)
        self.assertTrue(any((a != b).any() for a, b in zip(before, net.parameters())))

    def test_check_gives_numbers(self):
        won, ahead = check(ScoreNet().eval(), 2, 1)
        self.assertTrue(0 <= won <= 1)


if __name__ == "__main__":
    unittest.main()