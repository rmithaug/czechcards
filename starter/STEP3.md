# Step 3: runs, sets, coming out with 40, and reshuffling

Now the game has real rules and someone can win. You'll add:

* **Runs and sets**, and how many points they're worth.
* **Coming out:** your first lay-down has to be worth **40 points or more**.
* **Taking the discard:** until you've come out, you can only draw from the
  face-down pile.
* **Adding to the table:** once you've come out, you can put your cards onto
  any run or set on the table (yours or anyone's).
* **Reshuffling:** when the pile runs out, the discards (except the top card)
  get shuffled into a new pile.
* **Winning:** the first player to get rid of every card wins.

Jokers are still ordinary cards for now. They come in Step 4.

## The rules I used (tell me if Grandma's differ)

* **Run:** 3 or more cards of the same suit in a row, like 4♦ 5♦ 6♦. The Ace
  can go low (A-2-3) or high (Q-K-A), but not around the corner (K-A-2).
* **Set:** 3 or 4 cards with the same number, all different suits, like 7♥ 7♠ 7♣.
* **Points:** number cards are worth their number, J/Q/K are 10, and an Ace is 11
  (but only 1 at the bottom of a run, A-2-3).
* You can come out with several melds at once to reach 40, for example
  10♥ J♥ Q♥ (30) plus 7♣ 7♦ 7♠ (21) is 51.

## Two new ideas in the code

**1. A player is now a dictionary.** Each player needs to remember three
things, so instead of separate lists we bundle them together:

```python
player = {"name": "Jana", "hand": [...], "came_out": False}
player["came_out"] = True   # change one part of it
```

**2. The table** is a list of every meld laid down, each with the name of who
laid it: `[("Jana", [7♥, 7♠, 7♣]), ("You", [4♦, 5♦, 6♦])]`.

## File 1: `melds.py` (new)

This file only answers questions about cards: *Is this a run? Is it a set?
How many points?* It never prints anything or asks the player anything,
which makes it easy to test, and later the web version can use it unchanged.

```python
# Runs, sets, and how many points they're worth. (No jokers yet.)


def card_points(card):
    """A = 11, J/Q/K = 10, everything else is its number."""
    if card.rank == 1:
        return 11
    if card.rank >= 11:
        return 10
    return card.rank


def is_set(cards):
    """3 or 4 cards with the same number, all different suits. Example: 7♥ 7♠ 7♣"""
    if len(cards) < 3 or len(cards) > 4:
        return False
    ranks = {card.rank for card in cards}
    suits = {card.suit for card in cards}
    return len(ranks) == 1 and len(suits) == len(cards)


def run_ranks(cards):
    """The ranks of a run in order, with a high Ace counted as 14 (Q-K-A)."""
    ranks = sorted(card.rank for card in cards)
    if 1 in ranks and 13 in ranks:
        ranks.remove(1)
        ranks.append(14)
    return ranks


def is_run(cards):
    """3 or more cards of the same suit in a row. Example: 4♦ 5♦ 6♦"""
    if len(cards) < 3:
        return False
    if len({card.suit for card in cards}) != 1:
        return False
    ranks = run_ranks(cards)
    for i in range(1, len(ranks)):
        if ranks[i] != ranks[i - 1] + 1:
            return False
    return True


def is_meld(cards):
    return is_set(cards) or is_run(cards)


def meld_points(cards):
    """Add up a meld. An Ace at the bottom of a run (A-2-3) only counts 1."""
    total = sum(card_points(card) for card in cards)
    if is_run(cards) and run_ranks(cards)[0] == 1:
        total -= 10
    return total


def find_melds(hand):
    """Look for melds in a hand (used by the bots). Simple, not perfect:
    first it grabs sets, then runs from whatever is left."""
    left = list(hand)
    found = []

    # Sets: for each rank, take one card of each suit.
    for rank in range(1, 14):
        one_per_suit = {}
        for card in left:
            if card.rank == rank and card.suit not in one_per_suit:
                one_per_suit[card.suit] = card
        if len(one_per_suit) >= 3:
            meld = list(one_per_suit.values())
            found.append(meld)
            for card in meld:
                left.remove(card)

    # Runs: walk each suit from low to high, collecting cards in a row.
    for suit in {card.suit for card in left}:
        in_suit = sorted((c for c in left if c.suit == suit), key=lambda c: c.rank)
        streak = []
        for card in in_suit:
            if streak and card.rank == streak[-1].rank:
                continue  # skip the second copy of the same card
            if streak and card.rank != streak[-1].rank + 1:
                if len(streak) >= 3:
                    found.append(streak)
                streak = []
            streak.append(card)
        if len(streak) >= 3:
            found.append(streak)
    return found
```

How `is_run` works: it sorts the numbers (`[4, 5, 6]`) and then checks that
each one is exactly 1 more than the one before it. For Q-K-A, `run_ranks`
turns the Ace into 14 so it becomes `[12, 13, 14]`.

`find_melds` is only for the bots. It isn't perfect (sometimes a smarter split
exists), but it's simple and good enough for now.

**Try it** in the terminal before going on:

```
python3 -c "from cards import Card; from melds import *; print(is_run([Card(1,'D',4), Card(2,'D',5), Card(3,'D',6)]))"
```

That should print `True`. Change the 6 to a 7 and it should print `False`.

## File 2: `game.py` (replace the whole file)

```python
import random

from cards import build_double_deck
from melds import card_points, find_melds, is_meld, meld_points

NAMES = ["You", "Jana", "Petr", "Eva"]
COME_OUT_POINTS = 40


def deal():
    """Shuffle, give everyone 14 cards, and give you (the starter) a 15th."""
    deck = build_double_deck()
    random.shuffle(deck)
    players = []
    for name in NAMES:
        hand = [deck.pop() for _ in range(14)]
        # Each player is a small dictionary: their name, their cards,
        # and whether they've come out yet.
        players.append({"name": name, "hand": hand, "came_out": False})
    players[0]["hand"].append(deck.pop())
    return players, deck  # whatever is left in the deck is the draw pile


def refill_pile(pile, discard_pile):
    """The pile is empty: keep the top discard, shuffle the rest into a new pile."""
    top = discard_pile.pop()
    pile.extend(discard_pile)
    discard_pile.clear()
    discard_pile.append(top)
    random.shuffle(pile)
    print("\n(The pile ran out, so the discards were shuffled into a new pile.)")


def sort_hand(hand):
    """Jokers go last; everything else by suit, then by number."""
    hand.sort(key=lambda card: (card.suit or "Z", card.rank))


def show_hand(hand):
    for number, card in enumerate(hand, start=1):
        print(f"  {number:>2}: {card.label()}")


def show_cards(cards):
    return " ".join(card.label() for card in cards)


def show_table(table):
    if not table:
        print("Nothing on the table yet.")
        return
    print("On the table:")
    for number, (owner, meld) in enumerate(table, start=1):
        print(f"  [{number}] {show_cards(meld)}   ({owner})")


def ask_number(question, lowest, highest):
    """Keep asking until the player types a number in range."""
    while True:
        answer = input(question).strip()
        if answer.isdigit() and lowest <= int(answer) <= highest:
            return int(answer)
        print(f"Please type a number from {lowest} to {highest}.")


def ask_melds(hand, came_out):
    """Ask which cards to lay down. Returns a list of melds (maybe empty)."""
    print("\nLay down runs or sets? Type the card numbers, like 1 2 3")
    print("For more than one at once, put a comma between them: 1 2 3, 7 8 9")
    if not came_out:
        print(f"You haven't come out yet, so they must add up to {COME_OUT_POINTS} or more.")
    while True:
        answer = input("Cards to lay down (or just press Enter to skip): ").strip()
        if answer == "":
            return []

        # Turn "1 2 3, 7 8 9" into [[1, 2, 3], [7, 8, 9]]
        groups = []
        for part in answer.split(","):
            words = part.split()
            if not all(word.isdigit() and 1 <= int(word) <= len(hand) for word in words):
                groups = None
                break
            groups.append([int(word) for word in words])
        if groups is None:
            print(f"Use card numbers from 1 to {len(hand)}.")
            continue
        all_numbers = [n for group in groups for n in group]
        if len(all_numbers) != len(set(all_numbers)):
            print("You used the same card twice.")
            continue

        melds = [[hand[n - 1] for n in group] for group in groups]
        bad = [meld for meld in melds if not is_meld(meld)]
        if bad:
            print(f"{show_cards(bad[0])} isn't a run or a set. Try again.")
            continue
        total = sum(meld_points(meld) for meld in melds)
        if not came_out and total < COME_OUT_POINTS:
            print(f"That's only {total} points. You need {COME_OUT_POINTS} to come out.")
            continue
        return melds


def sort_meld(meld):
    """Keep runs in order on the table (a high Ace goes after the King)."""
    ace_high = any(card.rank == 13 for card in meld)
    meld.sort(key=lambda card: 14 if (card.rank == 1 and ace_high) else card.rank)


def lay_down(player, melds, table):
    for meld in melds:
        sort_meld(meld)
        for card in meld:
            player["hand"].remove(card)
        table.append((player["name"], meld))
    if melds and not player["came_out"]:
        player["came_out"] = True
        total = sum(meld_points(meld) for meld in melds)
        print(f"{player['name']} came out with {total} points!")


def ask_add_to_table(hand, table):
    """After coming out you can add cards to any run or set on the table."""
    while hand and table:
        print()
        show_table(table)
        sort_hand(hand)
        show_hand(hand)
        answer = input("Add a card to the table? Type the card number then the [table] number, like 4 2 (Enter to skip): ").strip()
        if answer == "":
            return
        words = answer.split()
        if len(words) != 2 or not all(word.isdigit() for word in words):
            print("Type two numbers: your card, then the table meld.")
            continue
        card_number, meld_number = int(words[0]), int(words[1])
        if not (1 <= card_number <= len(hand) and 1 <= meld_number <= len(table)):
            print("One of those numbers is out of range.")
            continue
        card = hand[card_number - 1]
        owner, meld = table[meld_number - 1]
        if not is_meld(meld + [card]):
            print(f"{card.label()} doesn't fit on {show_cards(meld)}.")
            continue
        hand.remove(card)
        meld.append(card)
        sort_meld(meld)
        print(f"You add {card.label()} to {show_cards(meld)}.")


def human_turn(player, table, pile, discard_pile, first_turn):
    hand = player["hand"]
    print("\n" + "=" * 40)
    show_table(table)

    if first_turn:
        print("\nYou start, so you have 15 cards and don't draw this turn.")
    else:
        top = discard_pile[-1]
        print("\nYour hand:")
        sort_hand(hand)
        show_hand(hand)
        print(f"The last card thrown away was {top.label()}.")
        if player["came_out"]:
            choice = ask_number(f"  1 = draw from the pile, 2 = take the {top.label()}: ", 1, 2)
        else:
            print("You haven't come out yet, so you draw from the pile.")
            choice = 1
        if choice == 1:
            card = pile.pop()
            print(f"You drew {card.label()}.")
        else:
            card = discard_pile.pop()
        hand.append(card)

    sort_hand(hand)
    show_hand(hand)
    lay_down(player, ask_melds(hand, player["came_out"]), table)
    if player["came_out"]:
        ask_add_to_table(hand, table)
    if not hand:
        return None  # laid down every card

    sort_hand(hand)
    show_hand(hand)
    number = ask_number("Which card do you throw away? ", 1, len(hand))
    return hand.pop(number - 1)


def loneliness(card, hand):
    """How alone a card is: high = no other cards near it. Bots throw these away."""
    friends = 0
    for other in hand:
        if other is card:
            continue
        if other.rank == card.rank:
            friends += 1
        if other.suit == card.suit and abs(other.rank - card.rank) <= 2:
            friends += 1
    return (-friends, card_points(card))


def bot_turn(player, table, pile):
    """A simple bot: draw from the pile, lay down any melds it can, throw away its loneliest card."""
    hand = player["hand"]
    hand.append(pile.pop())

    melds = find_melds(hand)
    total = sum(meld_points(meld) for meld in melds)
    if melds and (player["came_out"] or total >= COME_OUT_POINTS):
        for meld in melds:
            print(f"{player['name']} lays down {show_cards(meld)}.")
        lay_down(player, melds, table)
    if player["came_out"]:
        # Try every card in hand against every meld on the table.
        for card in list(hand):
            for owner, meld in table:
                if len(hand) > 1 and is_meld(meld + [card]):
                    hand.remove(card)
                    meld.append(card)
                    sort_meld(meld)
                    print(f"{player['name']} adds {card.label()} to {show_cards(meld)}.")
                    break
    if not hand:
        return None

    card = max(hand, key=lambda c: loneliness(c, hand))
    hand.remove(card)
    print(f"{player['name']} draws from the pile and throws away {card.label()}.")
    return card


def play():
    players, pile = deal()
    discard_pile = []
    table = []  # every meld laid down, as (owner's name, cards)
    turn = 0

    while True:
        seat = turn % len(players)
        player = players[seat]

        if not pile:
            refill_pile(pile, discard_pile)

        if seat == 0:
            card = human_turn(player, table, pile, discard_pile, first_turn=(turn == 0))
        else:
            card = bot_turn(player, table, pile)
        if card is not None:
            discard_pile.append(card)

        if not player["hand"]:
            print(f"\n{player['name']} played every card and wins the round!")
            return
        turn += 1


if __name__ == "__main__":
    play()
```

## What changed since Step 2

* **`refill_pile`** runs at the start of a turn whenever the pile is empty.
  `pile.extend(discard_pile)` moves all the discards into the pile in one go,
  then `random.shuffle(pile)` mixes them up. The top discard stays put so the
  next player still has something to take.
* **`ask_melds`** reads answers like `1 2 3, 7 8 9`. `answer.split(",")` cuts
  it into `"1 2 3"` and `" 7 8 9"`, then `.split()` cuts each part into
  numbers. Then it checks, in order: real card numbers, no card used twice,
  every group is a run or set, and 40+ points if you haven't come out.
* **`lay_down`** moves the cards from the hand to the table and flips
  `came_out` to `True` the first time.
* **`ask_add_to_table`** lets you type `4 2` to put card 4 on table meld [2].
  The trick for checking it is to add the card and see if it's still a meld:
  `is_meld(meld + [card])`. That one line works for runs *and* sets.
* **`human_turn`** only offers "take the discard" if `player["came_out"]` is
  `True`.
* **`play`** now loops with `while True` and stops with `return` as soon as
  someone's hand is empty.
* **The bots** lay down whatever `find_melds` finds (if it's 40+ or they've
  already come out), add cards to the table, and throw away their "loneliest"
  card: the one with the fewest cards of the same number or nearby in its
  suit.

## Run it

```
python3 game.py
```

A turn now looks like this: see the table and your hand, draw, optionally lay
down melds, optionally add to the table, then throw a card away. Press Enter
to skip any question you don't need.

## Commit it

In Source Control, type `Runs, sets, coming out with 40, reshuffling`, then
click **Commit**, then **Sync Changes**.

## Next: Step 4, jokers

Now that runs and sets exist, a joker is "a card that can fill any one gap".
In Step 4 you'll teach `is_set` and `is_run` to count jokers and use them to
fill missing spots.
