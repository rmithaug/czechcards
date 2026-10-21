# Step 2: play a round in the terminal

Your deck works, so now let's make a game you can actually play. It runs in
the VS Code terminal: you against three bots (Jana, Petr and Eva), taking
turns until the pile runs out. Nobody can win yet. Checking for a win is
Step 3, where you'll build runs and sets. For now, jokers are just ordinary
cards.

## First, one fix in your try_it.py

```python
myhand = deck[:14]
```

This **copies** the first 14 cards, but they also stay in the deck, so it's
easy to deal the same card to two people by mistake. The safer way is
`deck.pop()`, which **takes** the last card out of the deck and hands it to
you:

```python
myhand = [deck.pop() for _ in range(14)]
print(len(deck))  # 92: the 14 cards really left the deck
```

The game below deals this way.

## How a turn works

The game needs to remember four things, and each one is just a list or a number:

| Name           | What it is                                               |
|----------------|----------------------------------------------------------|
| `hands`        | a list of 4 hands, one per player (yours is `hands[0]`)  |
| `pile`         | the face-down cards left after dealing                   |
| `discard_pile` | the cards thrown away, newest on top (`discard_pile[-1]`)|
| `turn`         | a counter that goes up by 1 every turn                   |

Whose turn it is comes from `turn % 4` (the remainder after dividing by 4),
which goes 0, 1, 2, 3, 0, 1, 2, 3, and so on. Seat 0 is you.

On each turn you draw one card (from the pile, or the card the last player
threw away), then throw one card away. You start with 15 cards, so on your
very first turn you only throw one away.

## Make `game.py`

Create a new file `game.py` next to `cards.py` and paste this in:

```python
import random

from cards import build_double_deck

PLAYERS = ["You", "Jana", "Petr", "Eva"]


def deal():
    """Shuffle, give everyone 14 cards, and give you (the starter) a 15th."""
    deck = build_double_deck()
    random.shuffle(deck)
    hands = []
    for _player in PLAYERS:
        hand = [deck.pop() for _ in range(14)]
        hands.append(hand)
    hands[0].append(deck.pop())
    return hands, deck  # whatever is left in the deck is the draw pile


def sort_hand(hand):
    """Jokers go last; everything else by suit, then by number."""
    hand.sort(key=lambda card: (card.suit or "Z", card.rank))


def show_hand(hand):
    for number, card in enumerate(hand, start=1):
        print(f"  {number:>2}: {card.label()}")


def ask_number(question, lowest, highest):
    """Keep asking until the player types a number in range."""
    while True:
        answer = input(question).strip()
        if answer.isdigit() and lowest <= int(answer) <= highest:
            return int(answer)
        print(f"Please type a number from {lowest} to {highest}.")


def human_turn(hand, pile, discard_pile, first_turn):
    if not first_turn:
        top = discard_pile[-1]
        print("\nYour turn. Your hand:")
        sort_hand(hand)
        show_hand(hand)
        print(f"The last card thrown away was {top.label()}.")
        choice = ask_number(f"  1 = draw from the pile, 2 = take the {top.label()}: ", 1, 2)
        if choice == 1:
            card = pile.pop()
            print(f"You drew {card.label()}.")
        else:
            card = discard_pile.pop()
        hand.append(card)
    else:
        print("\nYou start, so you have 15 cards and just throw one away.")

    sort_hand(hand)
    show_hand(hand)
    number = ask_number("Which card do you throw away? ", 1, len(hand))
    return hand.pop(number - 1)


def bot_turn(name, hand, pile):
    """A very dumb bot: draw from the pile, throw away a random card."""
    hand.append(pile.pop())
    card = hand.pop(random.randrange(len(hand)))
    print(f"{name} draws from the pile and throws away {card.label()}.")
    return card


def play():
    hands, pile = deal()
    discard_pile = []
    turn = 0

    while pile:
        seat = turn % len(PLAYERS)
        if seat == 0:
            card = human_turn(hands[0], pile, discard_pile, first_turn=(turn == 0))
        else:
            card = bot_turn(PLAYERS[seat], hands[seat], pile)
        discard_pile.append(card)
        turn += 1

    print("\nThe pile is empty. Nobody won this round.")


if __name__ == "__main__":
    play()
```

## Run it

```
python3 game.py
```

You'll see your hand numbered 1 to 15. Type a number and press Enter to throw
that card away, then watch the bots take their turns. A full round gives you
about a dozen turns. Press **Ctrl+C** to quit early.

## What the new bits mean

* **`return hands, deck`** sends back two things at once. You unpack them the
  same way when you call it: `hands, pile = deal()`.
* **`while pile:`** keeps looping as long as the pile still has cards. An
  empty list counts as "false", so the loop stops when the pile runs out.
* **`ask_number`** keeps asking until you type something valid, so a typo
  doesn't crash the game. `answer.isdigit()` checks that it's a number before
  `int(answer)` turns it into one.
* **`enumerate(hand, start=1)`** gives you each card *and* a counting number,
  which is how your hand gets numbered.
* **`hand.pop(number - 1)`** removes the card you picked. It's `- 1` because
  you see the cards numbered from 1, but Python counts from 0.
* **`if __name__ == "__main__":`** means "only start the game when this file
  is run directly". Later, when other files import from `game.py`, the game
  won't start on its own.

## Try these changes

1. Make the bots smarter: if the card on top of the discard pile has the same
   rank as a card in their hand, they take it instead of drawing.
2. Count the turns and print "Round over after N turns" at the end.
3. Let the player choose their own name with `input("Your name? ")`.

## Commit it

In Source Control, type a message like `Playable round in the terminal`,
click **Commit**, then **Sync Changes**.

## Next: Step 3

Step 3 is checking for a win. You'll write `is_set(cards)` and
`is_run(cards)`, then check whether a hand can be split into sets and runs.
That's also where jokers will fit in naturally.
