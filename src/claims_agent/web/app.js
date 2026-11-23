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
const HOSTED_PANEL_LINE = "The operator view is off on this public demo: it would show how each caller's details were checked.";
const KEY_NEEDED_LINE = "Enter a model key to start.";
const KEY_FORM_LINE = "That is not a model key this page can use. Check it, then enter it again.";
const KEY_SHAPE = /^[!-~]{1,512}$/; // visible ASCII, as the server accepts it
const MAX_DEPTH = 8;
const REQUEST_TIMEOUT_MS = 150000; // above the server's worst case of about two minutes
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Who decided each part of the payload. Fixed maps: API text never picks a tag or a class name.
const TAG = new Map([
  ["code", ["tag tag-code", "Code"]], ["fast", ["tag tag-fast", "Fast model"]],
  ["text", ["tag tag-text", "Text model"]], ["large", ["tag tag-large", "Larger model"]],
]);
// reply_source: "live" is worded by the text model, "rewrite" by the larger model, "prepared" by the text model ahead
// of time (and checked again before it is sent), the rest is code.
const REPLY_DECIDER = new Map([["live", "text"], ["rewrite", "large"], ["prepared", "text"]]);

const $ = (id) => document.getElementById(id);
const els = {
  banner: $("banner"), needsScript: $("needs-script"), transcript: $("transcript"), ended: $("ended"),
  endedText: $("ended-text"), newBtn: $("new-conversation"), none: $("no-conversation"), retry: $("retry"),
  form: $("composer"), input: $("message"), send: $("send"), script: $("consent-script"), today: $("today"),
  apply: $("apply-controls"), note: $("controls-note"), picker: $("turn-picker"), body: $("inspector-body"),
  loadReport: $("load-report"), outbox: $("outbox"), draftNote: $("draft-note"),
  metaProblem: $("meta-problem"), metaText: $("meta-text"), reloadScripts: $("reload-scripts"),
  panel: $("operator-view"), startOver: $("start-over"), layout: $("layout"), widen: $("widen"),
  callers: $("sample-callers"), callerList: $("sample-callers-list"),
};
els.needsScript.hidden = true; // the script loaded

// One owner for "what the page is doing": app.op is the token of the one running operation, or null.
// Only the operation that holds the current token may touch the busy state, the open conversation, the transcript,
// the banner or the controls. A superseded operation does nothing at all when it returns.
const app = { cid: null, state: null, turns: [], selected: null, op: null, addressDirty: false, metaOk: false, metaError: "", drafts: new Map(), carry: null, outboxSeq: 0, formFilter: "all" };

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
// `extra` adds request headers (only the message request passes one: the hosted model key). Every path is this page's own.
async function api(method, path, body, outer, extra) {
  const ctl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctl.abort(); }, REQUEST_TIMEOUT_MS);
  if (outer) outer.addEventListener("abort", () => ctl.abort());
  try {
    const res = await fetch(path, {
      method, signal: ctl.signal,
      headers: Object.assign(body === undefined ? {} : { "Content-Type": "application/json" }, extra || {}),
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
  els.startOver.disabled = busy;
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
// A turn with no id of its own (the server did not give one) gets no thumbs. A rating is sent for the conversation
// the turn was shown in.
function addThumbs(bubble, turn) {
  if (!turn || !validTurnId(turn.turn_id) || !app.cid) return;
  const cid = app.cid;
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
    b.addEventListener("click", () => rateTurn(cid, turn, value, comment.value, note));
    bar.appendChild(b);
  }
  bar.appendChild(comment);
  bar.appendChild(note);
  bubble.appendChild(bar);
}

async function rateTurn(cid, turn, rating, comment, note) {
  if (app.op) return;
  await run("rate", async (tok) => {
    const body = { rating };
    const text = comment.trim();
    if (text) body.comment = text;
    const r = await api("POST", convPath(cid) + "/turns/" + encodeURIComponent(String(turn.turn_id)) + "/feedback", body, tok.ctl.signal);
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
  if (r.data.hosted === true) { showKeyBox(); hidePanel(); }
}

// Hosted: the server sends no operator data, so the panel is replaced by one fixed line.
function hidePanel() {
  if (els.panel.dataset.off === "true") return;
  for (const child of els.panel.children) child.hidden = true;
  els.panel.appendChild(el("p", "muted", HOSTED_PANEL_LINE));
  els.panel.dataset.off = "true";
}

// ---- Hosted mode: the visitor's own model key (built only when the server says it is hosted) ----------------------
// The key is kept in this one variable while the page is open, never in browser storage, a cookie or the address,
// so a reload asks for it again. It is sent only to this page's own server, only on the message request.
let modelKey = "";
let keyEls = null;
function showKeyBox() {
  if (keyEls) return;
  const box = el("div", "key-box");
  const label = el("label", "", "Your model key");
  label.htmlFor = "model-key";
  const input = el("input");
  input.type = "password";
  input.id = "model-key";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.maxLength = 512;
  const note = el("p", "muted", "Used only for your messages while this page is open. It is not saved; after a reload you enter it again.");
  input.addEventListener("input", () => { modelKey = input.value.trim(); });
  box.appendChild(label);
  box.appendChild(input);
  box.appendChild(note);
  els.transcript.parentNode.insertBefore(box, els.transcript);
  keyEls = { input };
}
// Hosted: the header for the message request, or null (and the reason shown) when there is no usable key.
function keyHeader() {
  if (!keyEls) return {};
  if (!modelKey || !KEY_SHAPE.test(modelKey)) {
    showBanner(modelKey ? KEY_FORM_LINE : KEY_NEEDED_LINE);
    return null;
  }
  return { "X-Model-Key": modelKey };
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
  const outcome = await run("send", async (tok) => {
    const keyed = keyHeader();
    if (keyed === null) return "no_key"; // nothing was sent: the text stays in the box
    clearBanner();
    els.input.value = "";
    const cid = app.cid, before = app.turns.length;
    app.drafts.set(cid, { before, text, sent: true, ended: false });
    showWaiting(cid);
    const mine = addBubble("caller", text);
    const waiting = addBubble("agent", "Thinking...", "pending");
    waiting.setAttribute("aria-hidden", "true");
    try {
      const r = await api("POST", convPath(cid) + "/messages", { text }, undefined, keyed);
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
        if (r.data.error === "model_key_needed") showKeyBox(); // the server is hosted even if the page missed it
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
  if (outcome === "no_key") keyEls.input.focus();
  else if (isEnded()) els.newBtn.focus(); else if (!els.input.disabled) els.input.focus();
  if (app.addressDirty) { app.addressDirty = false; await followAddress(); }
}

// The header's "New conversation": always available, one operation at a time. The old conversation stays open
// (and its unsent text stays its own) until the server has made the new one; the new id goes into the address
// so Back returns to the old one.
async function startOver() {
  if (app.op) return;
  const made = await run("open", async (tok) => {
    const r = await api("POST", "/api/conversations", controlsBody(), tok.ctl.signal);
    if (!current(tok)) return false;
    if (!ok(r) || !validCreate(r.data)) {
      showBanner(ok(r) ? UNEXPECTED_LINE : errorText(r));
      return false;
    }
    opened(r.data.conversation_id, r.data.state, [], r.data.controls, true, false);
    els.outbox.textContent = "";
    return true;
  });
  if (made) els.input.focus(); else els.startOver.focus();
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

// The inspector reads one turn's record. Nothing in a record is trusted: every value is type-checked where it is
// read, and a record from before a field existed shows that field as not recorded.
const arr = (v) => (Array.isArray(v) ? v : []);
const obj = (v) => (isObj(v) ? v : {});
const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v) => (typeof v === "string" ? v : "");
const words = (name) => String(name).replace(/[_:]+/g, " ");
const prob = (p) => (num(p) === null ? "no answer" : num(p).toFixed(2));
const filled = (v) => !(v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0));

// Plain words for the names a record uses. A name missing here is shown as itself, with underscores as spaces.
const JOB_WORDS = new Map([
  ["form", "Fill the form"], ["extract", "Pull out stated values"], ["split", "Pull out values and split the questions"],
  ["word", "Word the reply"], ["rewrite", "Rewrite the reply after it failed the check"],
  ["settle", "Settle an unsure reading"],
]);
const JOB_ROLE = new Map([["form", "fast"], ["extract", "text"], ["split", "text"], ["word", "text"], ["rewrite", "large"], ["settle", "large"]]);
const FILLER_WORDS = new Map([
  ["decisions", "Decisions API (the fast model)"], ["structured", "Structured Outputs fallback (the text model)"],
]);
const FREEDOM_WORDS = new Map([
  ["claim_facts", "claim facts may reach a model"], ["settle_unsure", "the larger model may settle an unsure reading"],
  ["prepared_replies", "replies may be prepared ahead"],
]);
const STATED_WORDS = new Map([
  ["policyholder_name", "Policyholder's name"], ["caller_name", "Caller's own name"], ["relationship", "Relationship"],
  ["date_of_birth_as_stated", "Date of birth, as typed"], ["date_of_birth", "Date of birth, read as"],
  ["phone", "Phone"], ["email", "Email"], ["id_digits", "ID digits"], ["policy_number", "Policy number"],
  ["claim_number", "Claim number"], ["question_parts", "Questions the message was split into"],
]);
const KIND_WORDS = new Map([["name", "name"], ["dob", "date of birth"], ["phone", "phone"], ["email", "email"],
  ["id", "ID digits"], ["policy", "policy number"]]);
const FLAG_WORDS = new Map([
  ["format_failed", "failed its format check"],
  ["cross_check_failed", "cross-check failed: the form and the extractor disagree on whether a detail was given"],
  ["cross_check", "cross-check failed: the form and the extractor disagree on whether a detail was given"],
  ["partial", "treated as only part of a detail"], ["corrected_next_turn", "the caller corrected it on a later turn"],
  ["message_cut", "the message was longer than a model request may carry, so the models saw only its start"],
]);
const MARK_CLASS = new Map([["confident", "mark mark-ok"], ["unsure", "mark mark-unsure"], ["below", "mark mark-below"],
  ["no_answer", "mark mark-none"]]);
const MARK_WORDS = new Map([["confident", "confident"], ["unsure", "unsure"], ["below", "below the line"],
  ["no_answer", "no answer"]]);
const STEP_WORDS = new Map([["same_answer", "both likely readings give the same answer"],
  ["narrowed", "readings the claim rules out were dropped and the fast model asked again"],
  ["larger", "the larger model was asked"], ["ask_caller", "the caller was asked"]]);
const SOURCE_WORDS = new Map([
  ["live", "Worded by the text model from the facts below, and passed the check."],
  ["rewrite", "The first wording failed the check; the larger model rewrote it, and the rewrite passed."],
  ["template", "The model's wording failed the check, so each part's fixed text was sent."],
  ["fixed", "Fixed text: no model worded this reply."],
  ["last_resort", "Every text failed the check, so the last-resort line offering a person was sent."],
  ["prepared", "Prepared ahead by the text model, and checked again before it was sent."],
]);
const RULE_WORDS = new Map([
  ["verify_id", "Identity step ran"], ["await_consent", "Consent step ran"], ["resolve_intent", "Worked out which claim and need"],
  ["process_case", "Answered from the claim record"], ["post_process", "Wrap-up step ran"],
  ["bind_policyholder", "Asked the database to verify the caller"], ["request_consent", "Asked the database for the policyholder's consent"],
  ["check_consent", "Checked the consent answer"], ["refusal", "The caller refused a detail"],
  ["every_message", "A rule for every message decided the turn"], ["handover", "Handed over to a person"],
  ["stall_limit", "Nothing moved for too many turns: a person was offered"],
  ["frustrated_again", "An upset caller was still not helped: a person was offered"],
  ["offer_dropped", "A yes/no offer was dropped for an offer of a person"], ["question_dropped", "A second question was dropped from the reply"],
  ["question_parts", "The message was split into its questions"], ["more_questions_later", "Some questions were left for the next message"],
  ["nothing_said", "No step said anything, so the agent asked what is needed"], ["reopened", "A finished conversation was reopened"],
  ["after_the_end", "The conversation stays finished"], ["freedom", "Claim facts were kept from the model in this state"],
  ["larger_model", "The larger model settled nothing"], ["opening", "Opened the claim with its first answer"],
  ["answer", "Answered from the claim record"], ["not_on_file", "That answer is not on file"],
  ["followup_offered", "Offered a follow-up"], ["followup_accepted", "The caller took the follow-up"],
  ["followup_declined", "The caller declined the follow-up"], ["alternatives", "Gave alternatives for a document"],
  ["alternatives_exhausted", "No alternatives left for a document"], ["calculation", "Worked out a figure in the database"],
  ["prepared_reply", "Used a reply prepared ahead"], ["unsure_answer", "Not sure which answer was wanted"],
  ["unsure_need", "Not sure what the caller needs"], ["listed_claim", "The caller picked a listed claim"],
  ["claim_number", "Found the claim by its number"], ["offer_email", "Offered the summary email"],
  ["email_sent", "Stored the summary email"], ["email_promised", "Promised the summary email at the end"],
]);
const EVENT_WORDS = new Map([["escalation", "Passed on"], ["below_threshold", "Below its line"], ["mood_change", "Mood changed"],
  ["consent_check", "Consent checked"], ["resolved_need", "Need settled"], ["followup_choice", "Follow-up chosen"],
  ["wrap_up", "Wrap-up"], ["dropped", "Events not kept (over the limit)"]]);
// Who answered a form question (its "by") -> the role in the record's models, and its name in words.
const BY_ROLE = new Map([["fast", ["fast", "fast model"]], ["structured", ["text", "text model, Structured Outputs"]],
  ["larger", ["larger", "larger model"]]]);
const FILTERS = [["all", "All"], ["answered", "Answered: yes or a real pick"], ["unsure", "Unsure"]];

function tagEl(who) {
  const t = TAG.get(who);
  return t ? el("span", t[0], t[1]) : null;
}

// One section of the inspector: a heading with who decided it, an optional one-line note, then its body.
function section(title, who, note) {
  const sec = el("section", "section");
  const h = el("h3", "", title);
  const tag = who ? tagEl(who) : null;
  if (tag) h.appendChild(tag);
  sec.appendChild(h);
  if (note) sec.appendChild(el("p", "note", note));
  els.body.appendChild(sec);
  return sec;
}

function line(parent, label, value, cls) {
  const row = el("div", "line" + (cls ? " " + cls : ""));
  row.appendChild(el("span", "k", label));
  row.appendChild(el("span", "v", value));
  parent.appendChild(row);
  return row;
}

// The role of a model call: by the model's name when the record names the roles' models, else by its job.
function roleOf(call, models) {
  const model = str(call.model);
  for (const [role, who] of [["larger", "large"], ["text", "text"], ["fast", "fast"]]) {
    const name = str(obj(models)[role]);
    if (name && model.startsWith(name)) return who;
  }
  return JOB_ROLE.get(str(call.job)) || "code";
}

function turnLabel(t, i) {
  const p = obj(t.inspector);
  const before = str(p.state_before), after = str(p.state_after) || str(t.state_after);
  return String(i + 1) + (before || after ? " · " + (before ? before + " → " : "") + after : "");
}

function renderPicker() {
  els.picker.textContent = "";
  app.turns.forEach((t, i) => {
    const b = el("button", "", turnLabel(t, i));
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
  const p = turn && turn.inspector;
  if (!isObj(p)) {
    els.body.appendChild(el("p", "empty", "No inspector data for this turn."));
    return;
  }
  showSummary(p);
  showCalls(p);
  showExtracted(p);
  showForm(p);
  showDecided(p);
  showReply(p);
  showMemory(p);
  const raw = el("details", "section raw");
  raw.appendChild(el("summary", "", "Raw record"));
  renderValue(raw, p, 0);
  els.body.appendChild(raw);
}

// a. What happened, in one look.
function showSummary(p) {
  const sec = section("Summary", "code");
  const head = el("p", "states");
  head.appendChild(el("strong", "", str(p.state_before) || "?"));
  head.appendChild(el("span", "arrow", " → "));
  head.appendChild(el("strong", "", str(p.state_after) || "?"));
  sec.appendChild(head);
  const path = arr(p.path).filter((s) => typeof s === "string");
  if (path.length) line(sec, "Steps ran in", path.join(" → "));
  const f = obj(p.freedom);
  if (str(f.level)) {
    const sw = Object.entries(f).filter(([k]) => k !== "level")
      .map(([k, v]) => (FREEDOM_WORDS.get(k) || words(k)) + ": " + (v === true ? "yes" : v === false ? "no" : String(v)));
    line(sec, "Freedom", str(f.level) + (sw.length ? " (" + sw.join("; ") + ")" : ""));
  } else {
    line(sec, "Freedom", "no state step ran");
  }
  const models = obj(p.models);
  const filler = str(p.filler);
  const fillerModel = filler === "structured" ? str(models.text) : filler === "decisions" ? str(models.fast) : "";
  line(sec, "Form filler", (FILLER_WORDS.get(filler) || filler || "unknown") + (fillerModel ? ", " + fillerModel : ""));
  line(sec, "Larger model", largerUse(p));
}

function largerUse(p) {
  const name = str(obj(p.models).larger);
  if (!isObj(p.usage)) {
    return p.reply_source === "rewrite" ? "used to rewrite the reply (model calls not recorded)" : "model calls not recorded for this turn";
  }
  const jobs = arr(p.usage.calls).filter((c) => isObj(c) && roleOf(c, p.models) === "large").map((c) => str(c.job));
  if (!jobs.length) return "not used this turn";
  const counted = new Map();
  for (const j of jobs) counted.set(j, (counted.get(j) || 0) + 1);
  return "used" + (name ? " (" + name + ")" : "") + " to " +
    [...counted].map(([j, n]) => (JOB_WORDS.get(j) || words(j)).toLowerCase() + (n > 1 ? " (" + n + " calls)" : "")).join("; ");
}

// b. Every call code made, models and database together, in the order they started.
function showCalls(p) {
  const sec = section("Calls made this turn", null,
    "The models here never call tools. Code calls each model and the database itself, in the order listed.");
  const calls = isObj(p.usage) ? arr(p.usage.calls).filter(isObj) : null;
  const acts = isObj(p.db_actions) ? arr(p.db_actions.actions).filter(isObj) : null;
  const items = [];
  for (const c of calls || []) items.push({ at: num(c.started) || 0, kind: "model", c });
  for (const a of acts || []) items.push({ at: num(a.started) || 0, kind: "db", c: a });
  items.sort((x, y) => x.at - y.at);
  const list = el("ol", "calls");
  for (const it of items) {
    const li = el("li", "call");
    const c = it.c;
    const secs = num(c.ended) !== null && num(c.started) !== null ? Math.max(0, c.ended - c.started) : null;
    if (it.kind === "model") {
      li.appendChild(tagEl(roleOf(c, p.models)));
      li.appendChild(el("span", "what", JOB_WORDS.get(str(c.job)) || words(str(c.job) || "model call")));
      li.appendChild(el("code", "", str(c.model) || "model not named"));
      li.appendChild(el("span", "muted-inline", (num(c.input_tokens) ?? "?") + " in / " + (num(c.output_tokens) ?? "?") +
        " out tokens" + (secs === null ? "" : " · " + secs.toFixed(1) + " s")));
    } else {
      li.appendChild(tagEl("code"));
      li.appendChild(el("span", "what", "Database: " + str(c.action)));
      const out = Object.entries(obj(c.outcome)).map(([k, v]) => words(k) + " " + String(v)).join(", ");
      if (out) li.appendChild(el("span", "outcome", out));
      if (secs !== null) li.appendChild(el("span", "muted-inline", Math.round(secs * 1000) + " ms"));
    }
    list.appendChild(li);
  }
  if (items.length) sec.appendChild(list);
  if (calls === null) sec.appendChild(el("p", "empty", "Model calls: not recorded for this turn."));
  else if (!calls.length) sec.appendChild(el("p", "empty", "No model was called this turn."));
  else {
    const t = obj(p.usage.totals);
    sec.appendChild(el("p", "note", calls.length + " model call" + (calls.length === 1 ? "" : "s") + ": " +
      (num(t.input_tokens) ?? "?") + " tokens in, " + (num(t.output_tokens) ?? "?") + " out."));
  }
  if (acts === null) sec.appendChild(el("p", "empty", "Database actions: not recorded for this turn."));
  else if (num(p.db_actions.dropped)) sec.appendChild(el("p", "note", p.db_actions.dropped + " more database actions were not listed (over the limit)."));
}

// c. What the text model pulled out of the message, with what code made of each value.
function showExtracted(p) {
  const sec = section("Extracted from this message", "text");
  const stated = Object.entries(obj(p.stated)).filter(([, v]) => filled(v));
  for (const [k, v] of stated) line(sec, STATED_WORDS.get(k) || words(k), Array.isArray(v) ? v.map(String).join(" | ") : String(v));
  if (!stated.length) sec.appendChild(el("p", "empty", "Nothing was pulled out (the extractor runs on every message before verification; after it, only when a message quotes a claim number, asks several things, or a representative may mean someone else)."));
  const checks = arr(p.events).filter((e) => isObj(e) && e.kind === "extraction" && isObj(e.body));
  if (checks.length) {
    const ul = el("ul", "flags");
    for (const e of checks) {
      const flags = arr(e.body.flags).map((f) => FLAG_WORDS.get(f) || words(f));
      ul.appendChild(el("li", flags.length ? "flagged" : "", (KIND_WORDS.get(str(e.body.kind)) || words(str(e.body.kind))) + ": " +
        (flags.length ? flags.join("; ") : "passed its checks")));
    }
    sec.appendChild(ul);
  }
  const refused = arr(p.fired).filter((f) => isObj(f) && f.rule === "refusal").flatMap((f) => arr(f.kinds));
  if (refused.length) line(sec, "Refused this turn", refused.map((k) => KIND_WORDS.get(k) || words(k)).join(", "), "flagged");
  for (const f of arr(p.flags)) if (typeof f === "string") line(sec, "Flag", (FLAG_WORDS.get(f) || words(f)) + " (" + f + ")", "flagged");
}

// d. Every question the fast model was shown, grouped by topic, with its answer and how sure it was.
function showForm(p) {
  const sec = section("The form: what the fast model was shown and answered", "fast");
  const fa = p.form_asked;
  if (!isObj(fa)) {
    sec.appendChild(el("p", "empty", "Not recorded for this turn (no form was filled, or the turn is older than this record)."));
    return;
  }
  const th = obj(fa.thresholds);
  sec.appendChild(el("p", "note", "Lines used: a yes/no reads yes at " + prob(th.yes) + " or above and is unsure between " +
    prob(th.unsure_low) + " and " + prob(th.unsure_high) + "; a pick counts at " + prob(th.pick_confident) + " or above."));
  const bar = el("div", "filter");
  bar.setAttribute("role", "group");
  bar.setAttribute("aria-label", "Show questions");
  sec.appendChild(bar);
  const rows = [];
  for (const r of arr(fa.readings).filter(isObj)) rows.push(...showReading(sec, r, p.models));
  const counts = new Map([["all", rows.length], ["answered", rows.filter((x) => x.answered).length],
    ["unsure", rows.filter((x) => x.unsure).length]]);
  for (const [key, label] of FILTERS) {
    const b = el("button", "plain", label + " (" + counts.get(key) + ")");
    b.type = "button";
    b.setAttribute("aria-pressed", app.formFilter === key ? "true" : "false");
    b.addEventListener("click", () => { app.formFilter = key; applyFilter(bar, rows); });
    b.dataset.filter = key;
    bar.appendChild(b);
  }
  applyFilter(bar, rows);
}

function applyFilter(bar, rows) {
  for (const b of bar.children) b.setAttribute("aria-pressed", b.dataset.filter === app.formFilter ? "true" : "false");
  for (const x of rows) x.row.hidden = !(app.formFilter === "all" || (app.formFilter === "answered" ? x.answered : x.unsure));
  for (const g of new Set(rows.map((x) => x.group))) g.hidden = ![...g.querySelectorAll(".q")].some((q) => !q.hidden);
}

function mostCommon(values) {
  const n = new Map();
  for (const v of values) n.set(v, (n.get(v) || 0) + 1);
  return [...n].sort((x, y) => y[1] - x[1]).map(([v]) => v)[0] || "";
}

function showReading(sec, r, models) {
  const box = el("div", "reading");
  const title = num(r.part) === null ? "Whole message" : "Part " + r.part + ": “" + str(r.message) + "”";
  box.appendChild(el("h4", "", title));
  const qs = arr(r.questions).filter(isObj);
  box.appendChild(el("p", "note", qs.length + " questions in state " + str(r.state) + (r.verified === true ? ", caller verified" :
    ", caller not verified") + (r.truncated === true ? "; this list was cut to fit the record" : "") + ". What shaped them:"));
  const seen = el("ul", "seen");
  for (const s of arr(r.what_it_saw)) if (typeof s === "string") seen.appendChild(el("li", "", s));
  box.appendChild(seen);
  const usual = mostCommon(qs.map((q) => str(obj(q.answer).by)).filter(Boolean));
  const who = BY_ROLE.get(usual);
  if (who) box.appendChild(el("p", "note", "Answered by the " + who[1] + (str(obj(models)[who[0]]) ? " (" + str(obj(models)[who[0]]) + ")" : "") +
    " unless a question says otherwise."));
  const groups = new Map();
  const rows = [];
  for (const q of qs) {
    const name = str(q.topic) || "Other";
    if (!groups.has(name)) {
      const g = el("details", "group");
      g.open = true;
      g.appendChild(el("summary", "", name));
      groups.set(name, { g, n: 0, yes: 0, unsure: 0 });
      box.appendChild(g);
    }
    const grp = groups.get(name);
    const x = questionRow(q, models, usual);
    grp.n += 1; if (x.answered) grp.yes += 1; if (x.unsure) grp.unsure += 1;
    x.group = grp.g;
    grp.g.appendChild(x.row);
    rows.push(x);
  }
  for (const [name, grp] of groups) {
    grp.g.firstChild.textContent = name + " — " + grp.n + " asked, " + grp.yes + " answered, " + grp.unsure + " unsure";
  }
  sec.appendChild(box);
  return rows;
}

// `usual` is who answered most of this form's questions: a row says who answered it only when that differs.
function questionRow(q, models, usual) {
  const row = el("div", "q");
  const text = str(q.text);
  const cut = text.indexOf("?");
  const head = cut > 0 && cut < 220 ? text.slice(0, cut + 1) : text.slice(0, 160) + (text.length > 160 ? "…" : "");
  const main = el("div", "q-main");
  main.appendChild(el("span", "q-text", head));
  const meta = el("div", "q-meta");
  meta.appendChild(el("code", "q-id", str(q.id)));
  main.appendChild(meta);
  row.appendChild(main);
  const ans = el("div", "q-answer");
  const a = obj(q.answer);
  const mark = str(q.mark);
  if (q.type === "pick") {
    ans.appendChild(el("strong", "", filled(q.reading) ? String(q.reading) : "no pick"));
    if (num(a.p) !== null && filled(q.reading)) ans.appendChild(el("span", "", " " + prob(a.p)));
  } else {
    ans.appendChild(el("strong", "", filled(q.reading) ? String(q.reading) : "no answer"));
    if (num(a.p) !== null) ans.appendChild(el("span", "", " p(yes) " + prob(a.p)));
  }
  ans.appendChild(el("span", MARK_CLASS.get(mark) || "mark", MARK_WORDS.get(mark) || "not read"));
  const by = str(a.by) !== usual ? BY_ROLE.get(str(a.by)) || (str(a.by) ? [null, str(a.by)] : null) : null;
  if (by) {
    const modelName = by[0] ? str(obj(models)[by[0]]) : "";
    ans.appendChild(el("span", "by", "by the " + by[1] + (modelName ? " (" + modelName + ")" : "")));
  }
  row.appendChild(ans);
  const opts = Object.entries(obj(a.options)).filter(([, v]) => num(v) !== null).sort((x, y) => y[1] - x[1]);
  if (opts.length > 1) row.appendChild(el("div", "q-options", opts.slice(0, 5).map(([k, v]) => k + " " + prob(v)).join(" · ") + (opts.length > 5 ? " · …" : "")));
  if (isObj(q.settled)) {
    const s = q.settled;
    const steps = arr(s.steps).map((k) => STEP_WORDS.get(k) || words(k)).join(", then ");
    const larger = arr(s.steps).includes("larger");
    const end = filled(s.value) ? "settled as " + String(s.value) + (larger ? " by the larger model" : " by code") :
      "not settled" + (filled(s.larger) ? " (larger model: " + words(s.larger) + ")" : "");
    const settledLine = el("div", "q-settled", "Below its line: " + steps + "; " + end + ".");
    if (larger) settledLine.prepend(tagEl("large"));
    row.appendChild(settledLine);
  }
  const more = el("details", "q-more");
  more.appendChild(el("summary", "", "Words sent"));
  more.appendChild(el("p", "", text));
  const choices = arr(q.options).filter(isObj);
  if (choices.length) {
    const ul = el("ul", "");
    for (const c of choices) ul.appendChild(el("li", "", str(c.value) + (str(c.text) && str(c.text) !== words(str(c.value)) ? " — " + str(c.text) : "")));
    more.appendChild(ul);
  }
  meta.appendChild(more);
  const answered = q.answered === true;
  const unsure = mark === "unsure" || mark === "below" || mark === "no_answer";
  return { row, answered, unsure, group: null };
}

// e. What code decided: the rules in the order they fired, then the turn's events and any hand-over note.
function showDecided(p) {
  const sec = section("What code decided", "code");
  const fired = arr(p.fired).filter(isObj);
  const ol = el("ol", "rules");
  for (const f of fired) {
    const li = el("li", "");
    if (typeof f.rule === "string") {
      li.appendChild(el("span", "", RULE_WORDS.get(f.rule) || words(f.rule)));
      li.appendChild(el("code", "", f.rule));
    } else if (Array.isArray(f.unsure_steps)) {
      li.appendChild(el("span", "", "An unsure reading: " + f.unsure_steps.map((k) => STEP_WORDS.get(k) || words(k)).join(", then ")));
    }
    const rest = Object.entries(f).filter(([k]) => k !== "rule" && k !== "unsure_steps");
    if (rest.length) li.appendChild(el("span", "detail", rest.map(([k, v]) => words(k) + ": " + (isObj(v) || Array.isArray(v) ? JSON.stringify(v) : String(v))).join("; ")));
    ol.appendChild(li);
  }
  if (fired.length) sec.appendChild(ol); else sec.appendChild(el("p", "empty", "No rule recorded anything this turn."));
  const events = arr(p.events).filter((e) => isObj(e) && e.kind !== "extraction");
  if (events.length) {
    sec.appendChild(el("h4", "", "Events recorded"));
    const ul = el("ul", "events");
    for (const e of events) {
      const b = obj(e.body);
      const bits = ["field", "why", "went_to", "before", "after", "result", "need", "claim", "count"].filter((k) => filled(b[k]))
        .map((k) => words(k) + ": " + String(b[k]));
      ul.appendChild(el("li", "", (EVENT_WORDS.get(str(e.kind)) || words(str(e.kind))) + (bits.length ? " — " + bits.join("; ") : "")));
    }
    sec.appendChild(ul);
  }
  if (isObj(p.handover_note)) {
    sec.appendChild(el("h4", "", "Hand-over note"));
    renderValue(sec, p.handover_note, 0);
  }
}

// f. The reply: what it was allowed to say, where its text came from, and the check.
function showReply(p) {
  const sec = section("The reply", REPLY_DECIDER.get(p.reply_source) || "code");
  sec.appendChild(el("p", "", SOURCE_WORDS.get(p.reply_source) || "Source: " + str(p.reply_source)));
  const parts = arr(p.parts).filter(isObj);
  for (const part of parts) {
    const box = el("div", "part");
    box.appendChild(el("code", "", str(part.instruction)));
    const facts = arr(part.facts).filter(isObj);
    if (facts.length) {
      const ul = el("ul", "facts");
      for (const f of facts) ul.appendChild(el("li", "", str(f.text) + (str(f.claim) ? " (" + f.claim + ")" : "")));
      box.appendChild(ul);
    } else {
      box.appendChild(el("p", "empty", "no facts"));
    }
    if (str(part.plain)) box.appendChild(el("p", "plain", "Fixed text: " + part.plain));
    if (isObj(part.pending)) box.appendChild(el("p", "plain", "Asks: " + str(part.pending.label)));
    sec.appendChild(box);
  }
  arr(p.checks).filter(isObj).forEach((c, i) => {
    const bad = ["failures", "leak", "invented", "other"].flatMap((k) => arr(c[k]).map((v) => words(k) + ": " + String(v)));
    line(sec, "Check " + (i + 1), (c.passed === true ? "passed" : "failed") + (bad.length ? " — " + bad.join("; ") : ""),
      c.passed === true ? "" : "flagged");
  });
}

// g. What the agent holds after this turn. Which typed details matched which customer is never shown here.
function showMemory(p) {
  const sec = section("What the agent remembers after this turn", "code");
  const m = obj(p.memory), id = obj(m.identity), focus = obj(m.focus), c = obj(m.counters), counts = obj(p.counts);
  const held = Object.entries(obj(id.held)).filter(([, v]) => isObj(v));
  line(sec, "Identity details held", held.length ? held.map(([k, v]) => (KIND_WORDS.get(k) || words(k)) + " " + str(v.as_stated)).join("; ") : "none");
  line(sec, "Caller", [str(id.caller_role) && words(id.caller_role), str(id.caller_name), str(id.relationship)].filter(Boolean).join(", ") || "not stated");
  line(sec, "Details refused", arr(id.refused).map((k) => KIND_WORDS.get(k) || words(k)).join(", ") || "none");
  line(sec, "Claims in focus", arr(focus.claim_ids).join(", ") || "none");
  if (arr(focus.last_list).length) line(sec, "Last list shown", arr(focus.last_list).join(", "));
  if (str(focus.need)) line(sec, "Need", words(focus.need));
  line(sec, "Pending question", isObj(m.pending) ? str(m.pending.label) + (m.pending.yes_no === true ? " (yes/no)" : "") : "none");
  line(sec, "Counters", "turn " + (num(c.turn_no) ?? "?") + "; stalled " + (num(counts.stalled) ?? "?") + "; wrong values " +
    (num(counts.wrong_values) ?? "?") + "; consent checks " + (num(counts.consent_checks) ?? "?"));
  if (num(p.claims_visible) !== null) line(sec, "Claims the database lets this conversation see", String(p.claims_visible));
  const prepared = arr(p.prepared).filter(isObj);
  if (prepared.length) line(sec, "Replies prepared ahead", prepared.map((x) => str(x.claim) + " " + words(str(x.answer))).join("; "));
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

// ---- Sample callers ----------------------------------------------------------------------------------------------
// The server lists invented sample callers from a file next to the data (never from the database); a folder of other
// data lists none and the drawer stays hidden. A message button only fills the message box: nothing is sent.
const validCaller = (c) => isObj(c) && typeof c.name === "string" && typeof c.good_for === "string" &&
  Array.isArray(c.claims) && Array.isArray(c.opening_messages) &&
  c.opening_messages.every((m) => isObj(m) && typeof m.label === "string" && typeof m.text === "string");
const validCallers = (d) => isObj(d) && Array.isArray(d.callers) && d.callers.every(validCaller);

function callerDetails(c) {
  const dl = el("dl");
  const rows = [["Policy number", c.policy_number], ["Date of birth", c.date_of_birth],
    ["ID, last four", c.id_last4], ["Phone", c.phone], ["Email", c.email]];
  if (Array.isArray(c.name_aliases) && c.name_aliases.length) rows.push(["Also accepted", c.name_aliases.join(", ")]);
  for (const r of Array.isArray(c.representatives) ? c.representatives : []) {
    rows.push(["Representative", String(r.name) + " (" + String(r.relationship) + ")"]);
  }
  for (const [k, v] of rows) { dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", v)); }
  return dl;
}

function callerClaims(c) {
  const ul = el("ul");
  if (!c.claims.length) ul.appendChild(el("li", "", "No claims on file."));
  for (const k of c.claims) {
    let text = String(k.case_id) + ": " + String(k.case_type) + ", " + String(k.status) + ", created " + String(k.created);
    if (k.appeal_deadline) text += ", appeal deadline " + String(k.appeal_deadline) + " (" + String(k.appeal_deadline_note) + ")";
    if (Array.isArray(k.documents_needed) && k.documents_needed.length) text += ", needs " + k.documents_needed.join(" and ");
    ul.appendChild(el("li", "", text));
  }
  return ul;
}

function renderCallers(callers) {
  els.callerList.textContent = "";
  for (const c of callers) {
    const box = el("section", "caller");
    box.appendChild(el("h3", "", c.name));
    box.appendChild(el("p", "", c.good_for));
    box.appendChild(callerDetails(c));
    box.appendChild(callerClaims(c));
    for (const m of c.opening_messages) {
      const b = el("button", "use");
      b.type = "button";
      b.appendChild(el("span", "", "Put in message box: " + m.label));
      b.appendChild(document.createTextNode(m.text));
      b.addEventListener("click", () => {
        els.input.value = m.text;
        els.input.dispatchEvent(new Event("input", { bubbles: true }));
        els.input.focus();
      });
      box.appendChild(b);
    }
    els.callerList.appendChild(box);
  }
  els.callers.hidden = callers.length === 0;
}

async function loadSampleCallers() {
  const r = await api("GET", "/api/sample-callers");
  if (ok(r) && validCallers(r.data)) renderCallers(r.data.callers);
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
els.startOver.addEventListener("click", startOver);
// The operator panel can take most of the window; the chat keeps a narrow column beside it.
els.widen.addEventListener("click", () => {
  const wide = els.layout.classList.toggle("panel-wide");
  els.widen.setAttribute("aria-pressed", wide ? "true" : "false");
  els.widen.textContent = wide ? "Narrow panel" : "Widen panel";
});
window.addEventListener("hashchange", followAddress); // one listener only: popstate would double every Back
renderControlsState();
runOpen();
loadSampleCallers();
