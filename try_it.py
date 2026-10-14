import random

from cards import Card, build_double_deck, wild_from_indicator

deck = build_double_deck()

print("Cards in the Deck:", len(deck))

random.shuffle(deck)

for card in deck:
    print(card.label())
