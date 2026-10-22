// Match play: adds the scoreboard and the "Next deal" button to the normal
// game page. app.js does all the usual work; this file runs after it and
// adds a bit more to its render() function.

const drawDeal = render;     // the normal page drawing from app.js

render = function () {
  drawDeal();
  drawScores();
};

function drawScores() {
  const s = state;
  const m = s.match;
  if (!m) return;
  const dealOver = s.phase === "over";

  $("deal-label").textContent = `Deal ${m.deal} of ${m.deals} · ${s.players[m.dealer].name} dealt`;
  $("next-deal").hidden = !(dealOver && !m.over);

  // Scoreboard: one column per player, one row per finished deal, then the totals.
  const lowest = Math.min(...m.totals);
  const rows = [`<tr><th></th>${s.players.map((p) => `<th>${p.name}</th>`).join("")}</tr>`];
  m.deal_scores.forEach((scores, i) => {
    rows.push(`<tr><td>Deal ${i + 1}</td>${scores.map((x) => `<td>${x}</td>`).join("")}</tr>`);
  });
  rows.push(`<tr class="total"><td>Total</td>${m.totals.map((t) =>
    `<td class="${m.deal_scores.length && t === lowest ? "best" : ""}">${t}</td>`).join("")}</tr>`);
  $("score-table").innerHTML = rows.join("");

  // Show each bot's running total next to its name.
  document.querySelectorAll("#opponents .seat").forEach((box, i) => {
    const info = box.querySelector(".info");
    if (info) info.textContent += ` · total ${m.totals[i + 1]}`;
  });

  if (!dealOver) return;
  const mine = m.deal_scores.length ? m.deal_scores[m.deal_scores.length - 1][s.seat] : 0;
  const whoWent = s.winner === null ? "Nobody went out"
                : s.winner === s.seat ? "You went out"
                : `${s.players[s.winner].name} went out`;
  if (m.over) {
    const names = m.winners.map((i) => s.players[i].name);
    const youWon = m.winners.includes(s.seat);
    showMessage(youWon
      ? `Match over: you win with ${m.totals[s.seat]} points! 🎉`
      : `Match over: ${names.join(" and ")} win${names.length === 1 ? "s" : ""} with ${lowest} points. You had ${m.totals[s.seat]}.`,
      youWon ? "win" : "");
  } else {
    showMessage(`${whoWent}. You scored ${mine} this deal. Press "Next deal" when you're ready.`,
      s.winner === s.seat ? "win" : "");
  }
}

$("next-deal").onclick = async () => {
  groups = []; checks = []; selected.clear();
  await api(`/api/game/${state.game_id}/next_deal`, {});
};
