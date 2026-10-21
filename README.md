# Czech Cards

A rummy card game from my family, played with two decks of cards and their
jokers. Play it in your browser against three computer players (Jana, Petr
and Eva).

Everything that matters is written in Python. The web page only shows the
cards and sends your clicks to the Python server.

## Run it on a Mac

You need Python 3.10 or newer (check with `python3 --version`).

1. Open this folder in VS Code (**File → Open Folder…**).
2. Open a terminal in VS Code (**Terminal → New Terminal**) and run these
   one at a time. You only need to do this the first time:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Start the game:

   ```bash
   python3 app.py
   ```

4. Open <http://127.0.0.1:8000> in your browser.

Press **Ctrl+C** in the terminal to stop the server. Next time, you only need
`source .venv/bin/activate` and then `python3 app.py`.

**Or with one key:** after step 2, press **F5** in VS Code and choose
"Play Czech Cards". If VS Code asks which Python to use, pick the one in
`.venv`.

**Run the tests:** `python3 -m unittest`

## The rules

* **Cards:** two 52-card decks plus their 2 jokers, so 106 cards. Four
  players. The starting player gets 15 cards and everyone else gets 14.
* **Your turn:** draw one card, lay down or add cards if you want, then throw
  one card onto the discard pile. The starting player skips the draw on their
  very first turn.
* **Melds:**
  * a **run** is 3 or more cards of the same suit in a row. The Ace can be low
    (A-2-3) or high (Q-K-A), but K-A-2 doesn't count;
  * a **set** is 3 or 4 cards of the same number, all in different suits.
* **Jokers are wild.** A joker can stand in for any card, like the third 7 in
  a set or the missing 6♥ in 5♥ _ 7♥.
* **Coming out:** you can't lay anything down until you have runs and sets
  worth **40 points or more**, and you lay them all down **together** in one
  turn. Number cards are worth their number, J/Q/K are worth 10, an Ace is
  worth 11 (but only 1 at the bottom of A-2-3), and a joker is worth the card
  it stands in for.
* **Before you've come out** you can only draw from the face-down pile.
  **After** you've come out you can also take the top discard, lay down any
  run or set, and add your cards to any meld on the table, yours or anyone's.
* **The pile runs out:** the discard pile (everything except its top card) is
  shuffled into a new pile.
* **Winning:** the first player to get rid of every card wins.

## Project layout

```
app.py                  Web server (Flask). The page talks to it.
czech_cards/cards.py    A card, and the 106-card double deck
czech_cards/rules.py    Runs, sets, jokers and points
czech_cards/game.py     Dealing, turns, coming out, laying off, winning
czech_cards/bots.py     The computer players
templates/index.html    The page
static/app.js           Draws the table and sends your moves
static/style.css        Colors and layout
tests/test_rules.py     Tests for the rules, the game and the bots
starter/                The step-by-step guides from earlier
```

## How it fits together

The game is built in layers, and each one only talks to the layer below it:

```
browser page (static/app.js)   shows cards, sends clicks
          │
web server (app.py)            turns clicks into game actions, runs the bots
          │
game (czech_cards/game.py)     whose turn it is, what's allowed
          │
rules (czech_cards/rules.py)   is this a run or a set? what's it worth?
```

The rules never know there's a website, which is why they're easy to test.

## Toward real multiplayer

The game is already set up for it. Every action in `czech_cards/game.py`
names the seat that's acting and is refused if it isn't that seat's turn, and
`Game.view_for(seat)` gives each player only what they're allowed to see. To
let friends play, the server needs to hand each person a seat (with a secret
token so nobody can play someone else's), and tell everyone when something
changes, for example with the Flask-SocketIO library. Bots can fill any empty
seats.
