// The browser side: draws the table from the server's state and sends moves.
// All the rules live in Python; this file only shows things and sends clicks.

const RANKS = { 1: "A", 11: "J", 12: "Q", 13: "K" };
const SUITS = { H: "♥", S: "♠", D: "♦", C: "♣" };

let state = null;        // last state from the server
let order = [];          // order of card ids in your hand (kept only in the browser)
let selected = new Set();// ids of selected cards
let groups = [];         // runs/sets you're putting together before laying down
let checks = [];         // the server's verdict on each group: {valid, points}
let dragId = null;

const $ = (id) => document.getElementById(id);

// ----- talking to the server -------------------------------------------

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (data.hand) setState(data);
  if (data.error) showMessage(data.error, "error");
  return data;
}

async function newGame() {
  order = []; selected.clear(); groups = []; checks = [];
  const data = await api("/api/new", {});
  try { localStorage.setItem("czechCardsGame", data.game_id); } catch (e) {}
}

const act = (action, extra = {}) => api(`/api/game/${state.game_id}/action`, { action, ...extra });

async function checkGroups() {
  if (!groups.length) { checks = []; return render(); }
  const data = await api(`/api/game/${state.game_id}/check`, { groups });
  checks = data.results || [];
  render();
}

// ----- drawing the page ------------------------------------------------

function cardEl(card, extraClass = "") {
  const el = document.createElement("div");
  if (!card) {
    el.className = "card empty " + extraClass;
    return el;
  }
  el.className = `card ${extraClass}`;
  if (card.joker) {
    el.classList.add("joker");
    el.innerHTML = "🃏<br>JOKER" + (card.stands_for ? `<small>= ${card.stands_for}</small>` : "");
  } else {
    el.classList.add("s-" + card.suit);
    el.innerHTML = `<span>${RANKS[card.rank] || card.rank}</span><span class="suit">${SUITS[card.suit]}</span>`;
  }
  el.dataset.id = card.id;
  return el;
}

function setState(data) {
  state = data;
  const ids = data.hand.map((c) => c.id);
  order = order.filter((id) => ids.includes(id));
  ids.forEach((id) => { if (!order.includes(id)) order.push(id); });
  selected = new Set([...selected].filter((id) => ids.includes(id)));
  const before = groups.length;
  groups = groups.filter((g) => g.every((id) => ids.includes(id)));
  if (groups.length !== before) checks = [];
  render();
}

function stagedIds() {
  return new Set(groups.flat());
}

function render() {
  const s = state;
  const myTurn = s.your_turn;
  const playing = myTurn && s.phase === "play";
  const byId = Object.fromEntries(s.hand.map((c) => [c.id, c]));

  // Opponents
  const opp = $("opponents");
  opp.replaceChildren();
  [1, 2, 3].forEach((seat) => {
    const p = s.players[seat];
    const box = document.createElement("div");
    box.className = "seat" + (s.current === seat && s.phase !== "over" ? " active" : "");
    box.innerHTML = `<div class="name">${p.name}${p.came_out ? '<span class="badge">out</span>' : ""}</div>
      <div class="info">${p.card_count} card${p.card_count === 1 ? "" : "s"}${p.hand_points !== null ? ` · ${p.hand_points} pts left` : ""}</div>`;
    if (p.hand && p.hand.length) {
      const r = document.createElement("div");
      r.className = "reveal";
      p.hand.forEach((c) => r.appendChild(cardEl(c)));
      box.appendChild(r);
    }
    opp.appendChild(box);
  });

  // Pile and discard
  $("pile-count").textContent = s.pile_count;
  const canDraw = myTurn && s.phase === "draw";
  $("pile").disabled = !canDraw;
  $("pile").classList.toggle("clickable", canDraw);
  const canTake = canDraw && s.can_take;
  const top = cardEl(s.top_discard, canTake ? "clickable" : "");
  top.title = canTake ? "Take this card (you'll have to play it this turn)"
            : (s.came_out ? "" : "You can take the discard once you've come out");
  if (canTake) top.onclick = () => act("take");
  $("discard-slot").replaceChildren(top);

  // Melds on the table
  const melds = $("melds");
  melds.replaceChildren();
  if (!s.table.length) {
    melds.innerHTML = '<div class="empty-note">Nobody has come out yet.</div>';
  }
  const canLayOff = playing && s.came_out && selected.size === 1;
  s.table.forEach((meld, index) => {
    const el = document.createElement("div");
    el.className = "meld" + (canLayOff ? " target" : "");
    const cards = document.createElement("div");
    cards.className = "cards";
    meld.cards.forEach((c) => cards.appendChild(cardEl(c)));
    el.appendChild(cards);
    const owner = document.createElement("div");
    owner.className = "owner";
    owner.textContent = meld.owner;
    el.appendChild(owner);
    if (canLayOff) {
      el.title = "Add your selected card here";
      el.onclick = () => act("lay_off", { card_id: [...selected][0], meld: index });
    }
    melds.appendChild(el);
  });

  // Staging area (groups you're building)
  $("staging").hidden = !groups.length;
  const groupsEl = $("groups");
  groupsEl.replaceChildren();
  let total = 0;
  let allValid = groups.length > 0;
  groups.forEach((g, i) => {
    const check = checks[i];
    const el = document.createElement("div");
    el.className = "group" + (check && !check.valid ? " bad" : "");
    el.title = "Click to put these cards back in your hand";
    const cards = document.createElement("div");
    cards.className = "cards";
    g.forEach((id) => cards.appendChild(cardEl(byId[id])));
    el.appendChild(cards);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = !check ? "checking…" : check.valid ? `${check.points} points` : "not a run or set";
    el.appendChild(meta);
    el.onclick = () => { groups.splice(i, 1); checks.splice(i, 1); render(); checkGroups(); };
    groupsEl.appendChild(el);
    if (check && check.valid) total += check.points; else allValid = false;
  });
  $("staging-total").textContent = total;
  const enough = s.came_out || total >= s.opening_points;
  $("lay-down").disabled = !(playing && allValid && enough);
  $("lay-down").textContent = s.came_out ? "Lay down" : `Come out (need ${s.opening_points})`;

  // Your hand
  const hand = $("hand");
  hand.replaceChildren();
  const staged = stagedIds();
  order.filter((id) => !staged.has(id)).forEach((id) => {
    const el = cardEl(byId[id], selected.has(id) ? "selected" : "");
    el.draggable = true;
    el.onclick = () => { selected.has(id) ? selected.delete(id) : selected.add(id); render(); };
    el.ondragstart = () => { el.classList.add("dragging"); dragId = id; };
    el.ondragend = () => el.classList.remove("dragging");
    el.ondragover = (e) => e.preventDefault();
    el.ondrop = (e) => { e.preventDefault(); moveCard(dragId, id); };
    hand.appendChild(el);
  });

  $("make-group").disabled = !(playing && selected.size >= 3);
  $("discard").disabled = !(playing && selected.size === 1 && !s.must_play);
  $("put-back").hidden = !(playing && s.must_play);

  // Message
  if (s.phase === "over") {
    if (s.winner === null) showMessage("Nobody won this round. Start a new game!");
    else if (s.winner === s.seat) showMessage("You threw away your last card. You win! 🎉", "win");
    else showMessage(`${s.players[s.winner].name} wins this round. Everyone's leftover cards are shown above.`);
  } else if (!myTurn) {
    showMessage(`${s.players[s.current].name} is playing…`);
  } else if (s.phase === "draw") {
    showMessage(s.can_take ? "Your turn: draw from the pile, or take the discard if you can play it right away."
              : s.came_out ? "Your turn: draw from the pile."
              : "Your turn: draw from the pile. (You can take discards once you've come out.)");
  } else if (s.must_play) {
    showMessage(`You took ${s.must_play.label}: play it in a run or set this turn, or put it back and draw from the pile.`);
  } else if (!s.came_out) {
    showMessage(`To come out, build runs/sets worth ${s.opening_points}+ and lay them down together. Then throw a card away.`);
  } else {
    showMessage("Lay down runs/sets or add to the table, then throw a card away.");
  }

  // Log
  $("log").replaceChildren(...s.log.map((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    return li;
  }));
  $("log").lastChild?.scrollIntoView({ block: "nearest" });
}

function showMessage(text, kind = "") {
  $("message").textContent = text;
  $("message").className = "message " + kind;
}

// ----- arranging the hand ------------------------------------------------

function moveCard(fromId, toId) {
  if (fromId === null || fromId === toId) return;
  order.splice(order.indexOf(fromId), 1);
  order.splice(order.indexOf(toId), 0, fromId);
  render();
}

function sortHand(key) {
  const byId = Object.fromEntries(state.hand.map((c) => [c.id, c]));
  const suitOrder = { H: 0, S: 1, D: 2, C: 3 };
  const k = (c) => c.joker ? [9, 0] : key === "suit" ? [suitOrder[c.suit], c.rank] : [c.rank, suitOrder[c.suit]];
  order.sort((a, b) => {
    const [a1, a2] = k(byId[a]), [b1, b2] = k(byId[b]);
    return a1 - b1 || a2 - b2;
  });
  render();
}

// ----- buttons -----------------------------------------------------------

$("new-game").onclick = newGame;
$("rules-button").onclick = () => $("rules").showModal();
$("pile").onclick = () => act("draw");
$("put-back").onclick = () => act("put_back");
$("discard").onclick = () => act("discard", { card_id: [...selected][0] });
$("make-group").onclick = () => {
  groups.push(order.filter((id) => selected.has(id)));
  selected.clear();
  render();
  checkGroups();
};
$("clear-groups").onclick = () => { groups = []; checks = []; render(); };
$("lay-down").onclick = async () => {
  const data = await act("lay_down", { groups });
  if (!data.error) { groups = []; checks = []; render(); }
};
$("sort-suit").onclick = () => sortHand("suit");
$("sort-rank").onclick = () => sortHand("rank");

// Pick up the last game after a page reload, or start a new one.
(async () => {
  let saved = null;
  try { saved = localStorage.getItem("czechCardsGame"); } catch (e) {}
  if (saved) {
    const data = await api(`/api/game/${saved}`);
    if (!data.error) return;
  }
  newGame();
})();