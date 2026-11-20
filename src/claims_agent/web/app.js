"use strict";
// Everything that came from a caller, a model, the database or the API is untrusted:
// every element is built with createElement and filled with textContent. No markup is ever parsed from text.

const NETWORK_LINE = "Could not reach the server. Check that it is running, then try again.";
const TIMEOUT_LINE = "The server took too long to answer. Try again in a moment.";
const RECEIVED_LINE = "Your message was received.";
const WAITING_LINE = "An unsent message is waiting in another conversation.";
const UNEXPECTED_LINE = "The server sent a reply this page could not read.";
const ENDED_LINE = new Map([
  ["ESCALATED", "This conversation has been passed to a person, who will pick it up from here."],
  ["CLOSED", "This conversation has ended."],
]);
const RATE_FAILED_LINE = "Your rating could not be saved. Try again.";
const RATED_UP_LINE = "Rated helpful.";
const RATED_DOWN_LINE = "Rated not helpful.";
const MAX_DEPTH = 8;
const REQUEST_TIMEOUT_MS = 150000; // above the server's worst case of about two minutes
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Who decided each part of the payload. Fixed maps: API text never picks a tag or a class name.
const TAG = new Map([
  ["code", ["tag tag-code", "Code"]], ["fast", ["tag tag-fast", "Fast model"]],
  ["text", ["tag tag-text", "Text model"]], ["large", ["tag tag-large", "Larger model"]],
]);
const DECIDER = new Map([
  ["state_before", "code"], ["state_after", "code"], ["freedom", "code"], ["filler", "code"], ["flags", "code"],
  ["fired", "code"], ["parts", "code"], ["checks", "code"], ["memory", "code"], ["counts", "code"],
  ["claims_visible", "code"], ["outbox", "code"], ["handover_note", "code"], ["form", "fast"], ["stated", "text"],
]);
// reply_source: "live" is worded by the text model, "rewrite" by the larger model, the rest is code.
const REPLY_DECIDER = new Map([["live", "text"], ["rewrite", "large"]]);
const ORDER = ["state_before", "state_after", "freedom", "filler", "form", "stated", "flags", "fired", "parts",
  "reply_source", "checks", "memory", "counts", "claims_visible", "outbox", "handover_note"];

const $ = (id) => document.getElementById(id);
const els = {
  banner: $("banner"), needsScript: $("needs-script"), transcript: $("transcript"), ended: $("ended"),
  endedText: $("ended-text"), newBtn: $("new-conversation"), none: $("no-conversation"), retry: $("retry"),
  form: $("composer"), input: $("message"), send: $("send"), script: $("consent-script"), today: $("today"),
  apply: $("apply-controls"), note: $("controls-note"), picker: $("turn-picker"), body: $("inspector-body"),
  loadReport: $("load-report"), outbox: $("outbox"), draftNote: $("draft-note"),
  metaProblem: $("meta-problem"), metaText: $("meta-text"), reloadScripts: $("reload-scripts"),
};
els.needsScript.hidden = true; // the script loaded

// One owner for "what the page is doing": app.op is the token of the one running operation, or null.
// Only the operation that holds the current token may touch the busy state, the open conversation, the transcript,
// the banner or the controls. A superseded operation does nothing at all when it returns.
const app = { cid: null, state: null, turns: [], selected: null, op: null, addressDirty: false, metaOk: false, metaError: "", drafts: new Map(), carry: null, outboxSeq: 0 };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.classList.add(...className.split(" "));
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function showBanner(message) { els.banner.textContent = String(message); els.banner.hidden = false; }
function clearBanner() { els.banner.hidden = true; els.banner.textContent = ""; }

// ---- API --------------------------------------------------------------------------------------------------------
// Never throws. Returns {status, data, failure}: failure is "network", "timeout" or "aborted" (status 0), else "".
async function api(method, path, body, outer) {
  const ctl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctl.abort(); }, REQUEST_TIMEOUT_MS);
  if (outer) outer.addEventListener("abort", () => ctl.abort());
  try {
    const res = await fetch(path, {
      method, signal: ctl.signal,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    let data = null;
    try { data = await res.json(); } catch (e) {
      if (timedOut) return { status: 0, data: null, failure: "timeout" };
      if (outer && outer.aborted) return { status: 0, data: null, failure: "aborted" };
      data = null;
    }
    return { status: res.status, data, failure: "" };
  } catch (e) {
    return { status: 0, data: null, failure: timedOut ? "timeout" : (outer && outer.aborted ? "aborted" : "network") };
  } finally {
    clearTimeout(timer);
  }
}

// Any 2xx is success.
function ok(r) { return !!r && r.status >= 200 && r.status < 300; }
function hasMessage(r) {
  return !!(r && r.data && typeof r.data === "object" && typeof r.data.message === "string" && r.data.message);
}
function errorText(r) {
  if (r && r.failure === "timeout") return TIMEOUT_LINE;
  if (r && r.failure) return NETWORK_LINE;
  return hasMessage(r) ? r.data.message : UNEXPECTED_LINE;
}
// The pinned "nothing was recorded" shape: a 4xx with {error, message}.
function isCleanRefusal(r) {
  return !!(r && r.status >= 400 && r.status < 500 && hasMessage(r) && typeof r.data.error === "string");
}

// ---- Response validators: every body is checked before any of it is used ------------------------------------------
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const absent = (v) => v === undefined || v === null;
const optString = (v) => absent(v) || typeof v === "string";
const validControls = (c) => absent(c) || (isObj(c) && optString(c.consent_script) && optString(c.today));
const validInspector = (i) => absent(i) || isObj(i);
const optText = (v) => v === null || typeof v === "string"; // a stored turn's message or reply may be null
const validTurn = (t) => isObj(t) && optText(t.message) && optText(t.reply) && validInspector(t.inspector);
const validCreate = (d) => isObj(d) && typeof d.conversation_id === "string" && UUID.test(d.conversation_id) &&
  typeof d.state === "string" && validControls(d.controls);
const validGet = (d) => isObj(d) && typeof d.state === "string" && Array.isArray(d.turns) && d.turns.every(validTurn) &&
  validControls(d.controls);
const validSend = (d) => isObj(d) && typeof d.reply === "string" && typeof d.state === "string" && validInspector(d.inspector);
const validMeta = (d) => isObj(d) && Array.isArray(d.consent_scripts) && d.consent_scripts.every((n) => typeof n === "string");
const validClaimChange = (d) => isObj(d) && d.ok === true && typeof d.updated_at === "string";
const validTurnId = (v) => Number.isSafeInteger(v) && v > 0;
const validRate = (d) => isObj(d) && d.ok === true;
const validOutbox = (d) => Array.isArray(d) && d.every(isObj);

// ---- Operations -------------------------------------------------------------------------------------------------
// begin() supersedes any running operation except a send (a send is never abandoned); it returns false then.
function begin(tok) {
  if (app.op && app.op.kind === "send") return false;
  if (app.op) app.op.ctl.abort();
  app.op = tok;
  renderControlsState();
  return true;
}
function current(tok) { return app.op === tok; }
function finish(tok) { if (current(tok)) { app.op = null; renderControlsState(); } }
function isEnded() { return app.state !== null && ENDED_LINE.has(app.state); }

// Every operation runs inside this one wrapper: it takes the token and releases it on every exit, throws included.
async function run(kind, fn) {
  const tok = { kind, ctl: new AbortController() };
  try {
    if (!begin(tok)) return undefined;
    return await fn(tok);
  } catch (e) {
    if (current(tok)) { if (kind === "open") noConversation(UNEXPECTED_LINE); else showBanner(UNEXPECTED_LINE); }
    return undefined;
  } finally {
    finish(tok);
  }
}

// Send, the input and Apply are enabled only with an open conversation, not ended, and no operation running.
function renderControlsState() {
  const busy = app.op !== null;
  const off = busy || isEnded() || !app.cid;
  els.send.disabled = off;
  els.input.disabled = off;
  els.apply.disabled = off;
  els.reloadScripts.disabled = busy;
  if (adminEls) adminEls.button.disabled = off;
  els.form.setAttribute("aria-busy", busy ? "true" : "false");
  for (const b of els.transcript.querySelectorAll("button[data-rating]")) b.disabled = busy;
}

// ---- Transcript -------------------------------------------------------------------------------------------------
function addBubble(who, text, extra) {
  const b = el("div", "msg " + who + (extra ? " " + extra : ""), text);
  els.transcript.appendChild(b);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return b;
}

// Thumbs under one agent reply: the rating goes to the server with an optional comment, and the latest rating wins.
// A turn with no id of its own (the server did not give one) gets no thumbs.
function addThumbs(bubble, turn) {
  if (!turn || !validTurnId(turn.turn_id)) return;
  const bar = el("div", "thumbs");
  const note = el("span", "muted", turn.rating === 1 ? RATED_UP_LINE : turn.rating === -1 ? RATED_DOWN_LINE : "");
  note.setAttribute("role", "status");
  const comment = el("input", "thumbs-comment");
  comment.type = "text";
  comment.maxLength = 1000;
  comment.setAttribute("aria-label", "Optional comment on this reply");
  comment.placeholder = "Optional comment";
  for (const [value, label, shown] of [[1, "Helpful reply", "\u{1F44D}"], [-1, "Not helpful reply", "\u{1F44E}"]]) {
    const b = el("button", "plain", shown);
    b.type = "button";
    b.setAttribute("data-rating", String(value));
    b.setAttribute("aria-label", label);
    b.addEventListener("click", () => rateTurn(turn, value, comment.value, note));
    bar.appendChild(b);
  }
  bar.appendChild(comment);
  bar.appendChild(note);
  bubble.appendChild(bar);
}

async function rateTurn(turn, rating, comment, note) {
  if (app.op) return;
  await run("rate", async (tok) => {
    const body = { rating };
    const text = comment.trim();
    if (text) body.comment = text;
    const r = await api("POST", "/api/turns/" + encodeURIComponent(String(turn.turn_id)) + "/feedback", body, tok.ctl.signal);
    if (!current(tok)) return;
    if (ok(r) && validRate(r.data)) {
      turn.rating = rating;
      note.textContent = rating === 1 ? RATED_UP_LINE : RATED_DOWN_LINE;
    } else {
      note.textContent = ok(r) ? RATE_FAILED_LINE : errorText(r);
    }
  });
}

// A rebuild (restore or new conversation) is silent: the live region is marked busy while it is filled.
function rebuildTranscript() {
  els.transcript.setAttribute("aria-busy", "true");
  els.transcript.textContent = "";
  for (const t of app.turns) {
    if (t.message !== null) addBubble("caller", t.message);
    if (t.reply !== null) addThumbs(addBubble("agent", t.reply), t);
  }
  els.transcript.setAttribute("aria-busy", "false");
}

// The status element stays in the page; only its text (and the button's hidden flag) change, so it is announced.
function applyState(state) {
  app.state = typeof state === "string" ? state : null;
  const ended = isEnded();
  els.endedText.textContent = ended ? ENDED_LINE.get(app.state) : "";
  els.ended.classList.toggle("active", ended);
  els.newBtn.hidden = !ended;
  renderControlsState();
}

// There is either an open conversation or a plain statement that there is none, and why.
function noConversation(message) {
  keepBoxAsDraft(app.cid);
  if (app.carry !== null) els.input.value = app.carry;
  app.cid = null; app.state = null; app.turns = []; app.selected = null;
  rebuildTranscript(); renderPicker(); renderBody();
  applyState(null);
  els.none.hidden = false;
  showBanner(message);
}

// ---- Address hash -----------------------------------------------------------------------------------------------
// Only a UUID is ever accepted from the address; anything else is discarded, so no odd segment reaches a URL.
function hashParts() {
  const raw = location.hash.replace(/^#/, "");
  return { raw, id: UUID.test(raw) ? raw : "" };
}
function convPath(id) { return "/api/conversations/" + encodeURIComponent(id); }
function setHash(id, push) {
  const url = id ? "#" + id : location.pathname + location.search;
  if (push) history.pushState(null, "", url); else history.replaceState(null, "", url);
}

// ---- Conversation lifecycle (each takes the token of the operation it runs under) ---------------------------------
function showControls(c) {
  if (c && typeof c === "object") {
    const s = typeof c.consent_script === "string" ? c.consent_script : "";
    const d = typeof c.today === "string" ? c.today : "";
    if (s && ![...els.script.options].some((o) => o.value === s)) {
      const o = el("option", "", s); o.value = s; els.script.appendChild(o);
    }
    els.script.value = s;
    els.today.value = /^\d{4}-\d{2}-\d{2}$/.test(d) ? d : "";
    els.note.textContent = "Current settings: consent script " + (s || "unknown") + ", date " + (d || "unknown");
  } else {
    els.script.value = "";
    els.today.value = "";
    els.note.textContent = "Current settings: unknown";
  }
}

// Unsent or unconfirmed text belongs to one conversation: app.drafts maps a conversation id to
// {before (its turn count when the text was sent or left), text, sent, ended}, in memory only (never in browser
// storage). A sent draft is dropped once the server shows the turn recorded.
function keepBoxAsDraft(cid) {
  if (!cid) return;
  const text = els.input.value;
  const d = app.drafts.get(cid);
  if (!text.trim()) app.drafts.delete(cid); // the caller cleared it: it does not come back
  else if (d) d.text = text;
  else app.drafts.set(cid, { before: app.turns.length, text, sent: false, ended: false });
}

// A conversation the server no longer has: its text (in the box, if it is open) is carried to the next new one.
function orphan(id) {
  if (app.cid === id) { keepBoxAsDraft(id); els.input.value = ""; }
  const d = app.drafts.get(id);
  if (d) { app.carry = d.text; app.drafts.delete(id); }
}

// The waiting line names only drafts that can still be sent: another conversation's, not ended.
function showWaiting(id) {
  const waiting = [...app.drafts].some(([k, d]) => k !== id && !d.ended);
  els.draftNote.textContent = waiting ? WAITING_LINE : "";
}

// Returns true when the loaded conversation shows its sent draft was recorded.
function settleDraft(id, turnCount, changed, ended) {
  if (changed) els.input.value = ""; // another conversation's box starts empty
  const d = app.drafts.get(id);
  let recorded = false;
  if (d && d.sent && turnCount > d.before) { app.drafts.delete(id); els.input.value = ""; recorded = true; }
  else if (d) { els.input.value = d.text; d.ended = ended; }
  showWaiting(id);
  return recorded;
}

function opened(id, state, turns, controls, pushHash, keepBanner) {
  const changed = app.cid !== id;
  if (changed) keepBoxAsDraft(app.cid);
  app.cid = id;
  app.turns = turns;
  app.selected = turns.length ? turns.length - 1 : null;
  els.none.hidden = true;
  const recorded = settleDraft(id, turns.length, changed, typeof state === "string" && ENDED_LINE.has(state));
  if (recorded) showBanner(RECEIVED_LINE);
  else if (!keepBanner) { if (app.metaError) showBanner(app.metaError); else clearBanner(); }
  if (pushHash !== null) setHash(id, pushHash); // null: leave the address alone (it may already name another conversation)
  rebuildTranscript(); renderPicker(); renderBody();
  showControls(controls);
  applyState(state);
  refreshOutbox();
  return recorded;
}

async function startNew(tok, pushHash, keepBanner) {
  const r = await api("POST", "/api/conversations", controlsBody(), tok.ctl.signal);
  if (!current(tok)) return false;
  if (!ok(r) || !validCreate(r.data)) {
    noConversation(ok(r) ? UNEXPECTED_LINE : errorText(r));
    return false;
  }
  const id = r.data.conversation_id;
  opened(id, r.data.state, [], r.data.controls, pushHash, keepBanner);
  if (app.carry !== null) {
    const text = app.carry;
    app.carry = null;
    app.drafts.set(id, { before: 0, text, sent: false, ended: false });
    els.input.value = text;
    showWaiting(id);
  }
  return true;
}

async function restore(tok, id) {
  const r = await api("GET", convPath(id), undefined, tok.ctl.signal);
  if (!current(tok)) return false;
  if (r.status === 404) { orphan(id); return startNew(tok, false, false); }
  if (!ok(r) || !validGet(r.data)) {
    noConversation(ok(r) ? UNEXPECTED_LINE : errorText(r));
    return false;
  }
  opened(id, r.data.state, r.data.turns, r.data.controls, null, false);
  return true;
}

// Open whatever the address says: a UUID is restored; anything else is discarded and a new conversation starts.
// An empty or odd address while a conversation is open keeps that conversation.
async function openFromAddress(tok) {
  if (!app.metaOk) await loadMeta(tok);
  if (!current(tok)) return false;
  const { raw, id } = hashParts();
  if (id) return restore(tok, id);
  if (app.cid) { setHash(app.cid, false); return restore(tok, app.cid); }
  if (raw) setHash(null, false);
  return startNew(tok, false, false);
}

async function runOpen() {
  const ok = await run("open", (tok) => openFromAddress(tok));
  if (ok && !els.input.disabled) els.input.focus();
  return ok;
}

// Back / forward / a typed hash: the page follows the address. During a send the change waits until it settles.
async function followAddress() {
  if (app.op && app.op.kind === "send") { app.addressDirty = true; return; }
  const { id } = hashParts();
  if (!app.op && app.cid && (!id || id === app.cid)) { if (!id) setHash(app.cid, false); return; }
  await runOpen();
}

function showMetaState() {
  els.metaProblem.hidden = app.metaOk;
  els.metaText.textContent = app.metaOk ? "" : "The consent script list is missing: " + app.metaError;
}

async function loadMeta(tok) {
  const r = await api("GET", "/api/meta", undefined, tok.ctl.signal);
  if (!current(tok)) return;
  if (!ok(r) || !validMeta(r.data)) {
    app.metaError = ok(r) ? UNEXPECTED_LINE : errorText(r);
    showMetaState();
    return;
  }
  app.metaOk = true;
  app.metaError = "";
  while (els.script.options.length > 1) els.script.remove(1);
  for (const name of r.data.consent_scripts) {
    const o = el("option", "", name);
    o.value = name;
    els.script.appendChild(o);
  }
  els.loadReport.textContent = "";
  renderValue(els.loadReport, r.data.load_report, 0, "No load problems reported.");
  showMetaState();
  if (r.data.admin === true) showAdminControl();
}

// ---- Demo control: change a claim of the verified customer (built only when the server says it is on) ------------
let adminEls = null;
function showAdminControl() {
  if (adminEls) return;
  const box = el("fieldset", "controls");
  box.appendChild(el("legend", "", "Change a claim (demo only)"));
  const field = (label, type) => {
    const id = "admin-" + label.toLowerCase().replace(/[^a-z]+/g, "-");
    const lab = el("label", "", label);
    lab.htmlFor = id;
    const input = type === "textarea" ? el("textarea") : el("input");
    if (type !== "textarea") input.type = type;
    input.id = id;
    box.appendChild(lab);
    box.appendChild(input);
    return input;
  };
  const claim = field("Claim number", "text");
  const status = field("New status", "text");
  const deadline = field("New appeal deadline", "date");
  const documents = field("Documents needed (one per line)", "textarea");
  const button = el("button", "", "Change claim");
  button.type = "button";
  const note = el("p", "muted", "");
  note.setAttribute("role", "status");
  box.appendChild(button);
  box.appendChild(note);
  els.picker.parentNode.insertBefore(box, els.picker);
  adminEls = { button, note };
  button.addEventListener("click", async () => {
    if (app.op || isEnded() || !app.cid) return;
    const body = {};
    if (status.value.trim()) body.status = status.value.trim();
    if (deadline.value) body.appeal_deadline = deadline.value;
    const names = documents.value.split("\n").map((n) => n.trim()).filter((n) => n);
    if (names.length) body.documents_needed = names;
    const number = claim.value.trim();
    if (!number) { note.textContent = "Enter a claim number."; return; }
    if (!Object.keys(body).length) { note.textContent = "Enter at least one change."; return; }
    await run("claim", async (tok) => {
      note.textContent = "";
      const r = await api("PATCH", "/api/admin/claims/" + encodeURIComponent(app.cid) + "/" + encodeURIComponent(number), body, tok.ctl.signal);
      if (!current(tok)) return;
      if (!ok(r)) note.textContent = errorText(r);
      else if (!validClaimChange(r.data)) note.textContent = UNEXPECTED_LINE;
      else note.textContent = "Claim changed. The next reply will use it.";
    });
  });
  renderControlsState();
}

function controlsBody() {
  const body = {};
  if (els.script.value) body.consent_script = els.script.value;
  if (els.today.value) body.today = els.today.value;
  return body;
}

async function applyControls() {
  if (app.op || isEnded() || !app.cid) return;
  await run("controls", async (tok) => {
    clearBanner();
    const sent = controlsBody();
    const r = await api("PATCH", convPath(app.cid) + "/controls", sent, tok.ctl.signal);
    if (!current(tok)) return;
    if (!ok(r)) showBanner(errorText(r));
    else if (isObj(r.data) && !absent(r.data.controls) && !validControls(r.data.controls)) showBanner(UNEXPECTED_LINE);
    else showControls(isObj(r.data) && r.data.controls ? r.data.controls : sent);
  });
}

async function reloadScripts() {
  if (app.op) return;
  await run("meta", async (tok) => { await loadMeta(tok); });
}

// ---- Sending: one message in flight; the button and Enter share this one path ------------------------------------
// The caller's text is never discarded until the server has confirmed the turn was recorded (see settleDraft).
async function sendMessage() {
  if (app.op || isEnded() || !app.cid) return;
  const text = els.input.value.trim();
  if (!text) return;
  await run("send", async (tok) => {
    clearBanner();
    els.input.value = "";
    const cid = app.cid, before = app.turns.length;
    app.drafts.set(cid, { before, text, sent: true, ended: false });
    showWaiting(cid);
    const mine = addBubble("caller", text);
    const waiting = addBubble("agent", "Thinking...", "pending");
    waiting.setAttribute("aria-hidden", "true");
    try {
      const r = await api("POST", convPath(cid) + "/messages", { text });
      // A send holds the token until it settles: nothing can supersede it (an address change waits).
      if (ok(r) && validSend(r.data)) {
        waiting.remove();
        app.drafts.delete(cid);
        app.turns.push({ turn_no: before + 1, message: text, reply: r.data.reply, state_after: r.data.state, inspector: r.data.inspector,
          turn_id: validTurnId(r.data.turn_id) ? r.data.turn_id : null });
        app.selected = app.turns.length - 1;
        addThumbs(addBubble("agent", r.data.reply), app.turns[app.selected]);
        renderPicker(); renderBody();
        applyState(r.data.state);
        refreshOutbox();
      } else if (isCleanRefusal(r)) {
        // The server says nothing was recorded: show why and keep the text so it can be sent again.
        waiting.remove(); mine.remove();
        app.drafts.delete(cid); // nothing was recorded: the text is back in the box, unsent
        els.input.value = text;
        showBanner(errorText(r));
        if (r.status === 404) {
          orphan(cid); // that conversation is gone: the text moves to the new one
          await startNew(tok, false, true);
        }
      } else {
        // Unknown outcome (5xx, timeout, unreadable reply, no network): ask the server what it holds.
        waiting.remove(); mine.remove();
        els.input.value = text;
        const msg = errorText(r);
        const g = await api("GET", convPath(cid));
        if (ok(g) && validGet(g.data)) {
          if (!opened(cid, g.data.state, g.data.turns, g.data.controls, null, true)) showBanner(msg);
        } else if (g.status === 404) {
          orphan(cid);
          await startNew(tok, false, true);
          showBanner(msg);
        } else {
          noConversation(msg); // the text stays in the (disabled) box and in the draft
        }
      }
    } catch (e) {
      waiting.remove(); mine.remove();
      els.input.value = text;
      throw e;
    }
  });
  if (isEnded()) els.newBtn.focus(); else if (!els.input.disabled) els.input.focus();
  if (app.addressDirty) { app.addressDirty = false; await followAddress(); }
}

// ---- Inspector ---------------------------------------------------------------------------------------------------
function renderValue(parent, value, depth, emptyText) {
  if (value === null || value === undefined) { parent.appendChild(el("span", "leaf empty", emptyText || "none")); return; }
  if (typeof value !== "object") { parent.appendChild(el("span", "leaf", String(value))); return; }
  if (depth >= MAX_DEPTH) { parent.appendChild(el("span", "leaf", "(nested too deep to show)")); return; }
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i + 1), v]) : Object.entries(value);
  if (entries.length === 0) { parent.appendChild(el("span", "leaf empty", emptyText || "empty")); return; }
  for (const [k, v] of entries) {
    const row = el("div", "kv");
    row.appendChild(el("span", "k", k + ": "));
    renderValue(row, v, depth + 1);
    parent.appendChild(row);
  }
}

function tagFor(key, value) {
  const who = key === "reply_source" ? (REPLY_DECIDER.get(value) || "code") : DECIDER.get(key);
  return who ? TAG.get(who) : null;
}

function renderPicker() {
  els.picker.textContent = "";
  app.turns.forEach((t, i) => {
    const b = el("button", "", "Turn " + (i + 1));
    b.type = "button";
    b.setAttribute("aria-pressed", i === app.selected ? "true" : "false");
    b.addEventListener("click", () => selectTurn(i));
    els.picker.appendChild(b);
  });
}

// Selecting a turn changes aria-pressed in place, so focus stays on the button.
function selectTurn(i) {
  app.selected = i;
  [...els.picker.children].forEach((b, j) => b.setAttribute("aria-pressed", j === i ? "true" : "false"));
  renderBody();
}

function renderBody() {
  els.body.textContent = "";
  if (!app.turns.length) {
    els.body.appendChild(el("p", "empty", "No turns yet. Send a message to see what was decided."));
    return;
  }
  const turn = app.turns[app.selected];
  const payload = turn && turn.inspector;
  if (!payload || typeof payload !== "object") {
    els.body.appendChild(el("p", "empty", "No inspector data for this turn."));
    return;
  }
  const keys = ORDER.filter((k) => Object.hasOwn(payload, k)).concat(Object.keys(payload).filter((k) => !ORDER.includes(k)));
  for (const key of keys) {
    const section = el("div", "section");
    const h = el("h3", "", key);
    const tag = tagFor(key, payload[key]);
    if (tag) h.appendChild(el("span", tag[0], tag[1]));
    section.appendChild(h);
    renderValue(section, payload[key], 0);
    els.body.appendChild(section);
  }
}

async function refreshOutbox() {
  const cid = app.cid;
  if (!cid) return;
  const seq = ++app.outboxSeq; // an older refresh can never overwrite a newer one
  const r = await api("GET", convPath(cid) + "/outbox");
  if (seq !== app.outboxSeq || app.cid !== cid) return;
  els.outbox.textContent = "";
  if (ok(r) && validOutbox(r.data)) renderValue(els.outbox, r.data, 0, "Nothing sent.");
  else els.outbox.appendChild(el("span", "leaf empty", "Outbox not available."));
}

// ---- Wiring ------------------------------------------------------------------------------------------------------
els.form.addEventListener("submit", (ev) => { ev.preventDefault(); sendMessage(); });
els.apply.addEventListener("click", applyControls);
els.reloadScripts.addEventListener("click", reloadScripts);
els.newBtn.addEventListener("click", async () => {
  if (app.op) return;
  const ok = await run("open", (tok) => startNew(tok, true, false));
  if (ok) els.input.focus();
});
els.retry.addEventListener("click", runOpen);
window.addEventListener("hashchange", followAddress); // one listener only: popstate would double every Back
renderControlsState();
runOpen();
