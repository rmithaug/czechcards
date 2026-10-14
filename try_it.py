import random

from cards import Card, build_double_deck

deck = build_double_deck()

print("Cards in the Deck:", len(deck))

random.shuffle(deck)

# for card in deck:
#     print(card.label())


myhand = deck[:14]

print(myhand)
for card in myhand:
    print(card.label())