"""Tests for the neural-network bot. Run with: python3 -m unittest"""

import random
import unittest

import torch

from czech_cards import bots
from czech_cards.game import OVER, Game
from czech_cards.nn_bot import N, OBS_SIZE, THROW, CardNet, Memory, NetPlayer, observe
from train_nn import check, learn, play_batch


class TestNetwork(unittest.TestCase):
    def test_shapes(self):
        net = CardNet()
        yes_no, throw, value = net(torch.zeros(5, OBS_SIZE))
        self.assertEqual(tuple(yes_no.shape), (5, 3))
        self.assertEqual(tuple(throw.shape), (5, N))
        self.assertEqual(tuple(value.shape), (5,))

    def test_observation_only_shows_my_hand(self):
        game = Game(["A", "B", "C", "D"], rng=random.Random(3))
        obs = observe(game, 0, Memory(game))
        self.assertEqual(len(obs), OBS_SIZE)
        self.assertAlmostEqual(obs[:N].sum().item(), len(game.hands[0]) / 2)


class TestPlaying(unittest.TestCase):
    def test_plays_whole_deals_by_the_rules(self):
        torch.manual_seed(0)
        player = NetPlayer(CardNet(), explore=True)
        for seed in range(5):
            game = Game(["A", "B", "C", "D"], rng=random.Random(seed))
            while game.phase != OVER:
                (player if game.current == 0 else bots.take_turn)(game, game.current)
            self.assertEqual(game.phase, OVER)

    def test_only_throws_cards_it_holds(self):
        player = NetPlayer(CardNet(), record=True)
        game = Game(["A", "B", "C", "D"], rng=random.Random(1), starter=0)
        player(game, 0)
        throws = [s for s in player.take_steps(game, 0) if s[1] == THROW]
        self.assertEqual(len(throws), 1)
        obs, _, action, mask = throws[0]
        self.assertTrue(mask[action])


class TestTraining(unittest.TestCase):
    def test_one_round_of_learning(self):
        torch.manual_seed(0)
        net = CardNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        rows = play_batch(net, 2, random.Random(0))
        self.assertTrue(rows)
        before = [p.clone() for p in net.parameters()]
        learn(net, optimizer, rows, 0.01)
        self.assertTrue(any((a != b).any() for a, b in zip(before, net.parameters())))
    def test_check_gives_numbers(self):
        ahead, wins = check(CardNet().eval(), 4)
        self.assertTrue(0 <= wins <= 1)


if __name__ == "__main__":
    unittest.main()