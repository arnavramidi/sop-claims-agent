"""The check code runs on every reply before it is sent: the last line of defence against an invented or leaked
detail. It reads text a model wrote, so every ambiguous case is read the strict way: a false failure costs one
rewrite, a false pass would send an invented detail to a caller.

Numbers are compared by what they ARE, not as bare digits. One extractor turns a text into typed mentions
(claim number, money, date, duration, percentage, spelled-out number, relative date, bare number); it runs over
the facts and over the reply, and every mention in the reply must equal a mention of the same type in the facts.

Strict readings chosen:
- text is normalised first (width forms, accents, case, invisible characters, non-ASCII digits, dash variants);
- a date is one unit: month and day (and year, if stated) must be those of ONE fact date; a two-digit year never
  matches; slash dates are read month-first or day-first, either may match;
- "may" counts as a month only in a date; a lone "one" is ordinary speech and is not a number;
- weekday names, "tomorrow", "next week" and spelled-out numbers fail unless the same words are in the facts;
- the phone pattern runs on what is left after the typed mentions are taken out;
- a web address is any host-like shape (labels joined by dots ending in two or more letters, an IP, "www"),
  after dot look-alikes and the spelled forms "(dot)", " dot ", "-dot-" are folded to ".".

- number words are numbers: "thirteen days" is the duration 13 days, "the first of March" is a date; only vague
  quantities ("a few", "several") and relative dates ("today", "next week", weekdays) must appear as the same words;
- a bare number (no currency, unit or month) may equal a bare number, a money value, a duration's number or the
  day of a fact date, never digits that only occur inside a claim number; a four-digit year must be a fact date's year.

- a vague word's unit is not compared (the fact "a few days" lets "a few weeks" pass: a documented limit);
- "first" is a date only after "the" or a month name, or before "of" plus a month; elsewhere it is speech;
- a word ordinal followed by a comma is a sequence only when it opens a sentence ("First, send..."; not "by the
  second, if you can");
- an ordinal ("13th", "thirteenth", "second") may only equal the DAY of a fact date, never a duration, amount or
  count. A word ordinal used for sequence is not a date and is ignored: one opening a sentence and followed by a
  comma ("First, send..."), or followed by a noun like step, time, thing, question ("the first step");
- durations are only day(s), business day(s), week(s), month(s), year(s); "a minute", "an hour", "one second"
  are conversation, not claim quantities; "a few"/"several"/"a couple" count only before those units or
  "documents"/"claims";
- "today" is filler and allowed (a deadline that is today shows as zero days left in the facts); "tonight",
  "tomorrow", "yesterday", weekdays, "next/this/last" plus a unit, "end of month" and "mid-March" must be in the facts.

Known limits: a bare ending word with no dot and no spelled "dot" ("example com") is not read as an address.
A wary status word used as a claim's state is found by sentence patterns (a claim-like subject, a form of "to be"
within eight words; or "have accepted your claim"); patterns cannot cover every phrasing. Look-alike letters from other alphabets (a Cyrillic "e" in "example.com") and number words in
languages other than English are not covered by patterns; digits, "@", and dot shapes are caught in any language.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal

from dataclasses import dataclass

from claims_agent.reply.brief import ReplyBrief

_LEAK = ("web_address", "phone_number", "email_address", "claim_detail_before_verification")
_INVENTED = ("number_not_in_facts", "month_not_in_facts", "status_word")
_OTHER = ("too_long",)


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    failures: tuple[str, ...]

    @property
    def leak(self) -> tuple[str, ...]:
        return tuple(f for f in self.failures if f in _LEAK)

    @property
    def invented(self) -> tuple[str, ...]:
        return tuple(f for f in self.failures if f in _INVENTED)

    @property
    def other(self) -> tuple[str, ...]:
        """Failures that belong to neither group (`too_long`)."""
        return tuple(f for f in self.failures if f in _OTHER)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = unicodedata.normalize("NFD", t)
    out = []
    for c in t:
        cat = unicodedata.category(c)
        if cat in ("Cf", "Mn"):
            continue
        if cat == "Nd":
            c = str(unicodedata.decimal(c))
        elif cat == "Pd" or c == "−":
            c = "-"
        elif c in "\u2019\u2018\u02bc":
            c = "'"
        elif ord(c) > 127 and _is_dot(c):
            c = "."
        out.append(c)
    return "".join(out).casefold()


def _is_dot(c: str) -> bool:
    name = unicodedata.name(c, "")
    return any(k in name for k in ("FULL STOP", "MIDDLE DOT", "HYPHENATION POINT"))


# ---- web addresses, emails ----

_HOST = re.compile(r"(?<![a-z0-9-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?![a-z0-9-])")
_IP = re.compile(r"(?<!\d)\d{1,3}(?:\.\d{1,3}){3}(?!\d)")
_WEB_WORDS = re.compile(r"https?:?//|://|\bwww\b|\bhttps?\b|\blocalhost\b")


def _fold_address(t: str) -> str:
    # only spaces and tabs are ever joined, never a line break
    t = re.sub(r"[ \t]*[\[({<][ \t]*\.[ \t]*[\])}>][ \t]*", ".", t)
    t = re.sub(r"[ \t]*[\[({<][ \t]*dot[ \t]*[\])}>][ \t]*", ".", t)
    t = re.sub(r"-dot-", ".", t)
    t = re.sub(r"[ \t]+dot[ \t]+", ".", t)
    t = re.sub(r"(?<=\w)[ \t]+\.[ \t]+(?=\w)", ".", t)
    t = re.sub(r"[ \t]*[\[({<][ \t]*at[ \t]*[\])}>][ \t]*", "@", t)
    t = re.sub(r"(?<=[a-z0-9._-])[ \t]+at[ \t]+(?=[a-z0-9-]+\.[a-z]{2,})", "@", t)
    return t


def _has_web_address(t: str) -> bool:
    t = _fold_address(t)
    squashed = re.sub(r"[ \t\[\](){}]+", "", t)
    return bool(_HOST.search(t) or _IP.search(t) or _WEB_WORDS.search(t) or re.search(r"https?:?//|www\.", squashed))


def _has_email(t: str) -> bool:
    return "@" in _fold_address(t)


# ---- phone numbers ----

_PHONE = re.compile(r"(?<!\d)\+?\(?\d[\d\s().\-/_]{4,}\d(?!\d)")
_DIGIT_WORDS = set("zero oh one two three four five six seven eight nine".split())
_WORD = re.compile(r"[a-z]+")


def _digit_word_phone(t: str) -> bool:
    run, last_end = 0, None
    for m in _WORD.finditer(t):
        if m.group() in _DIGIT_WORDS and (last_end is None or re.fullmatch(r"[\s,.\-()]*", t[last_end:m.start()])):
            run += 1
            if run >= 7:
                return True
        else:
            run = 1 if m.group() in _DIGIT_WORDS else 0
        last_end = m.end()
    return False


def _digits_phone(rest: str) -> bool:
    return any(sum(c.isdigit() for c in m.group()) >= 7 for m in _PHONE.finditer(rest))


# ---- typed mentions ----

_MONTHS: dict[str, int] = {}
for _i, _names in enumerate([
    "january jan janvier januar enero", "february feb fevrier februar febrero", "march mar mars marz maerz marzo",
    "april apr avril abril", "mai mayo", "june jun juin juni junio", "july jul juillet juli julio",
    "august aug aout agosto", "september sep sept septembre septiembre setiembre",
    "october oct octobre oktober octubre", "november nov novembre noviembre", "december dec decembre dezember diciembre",
], start=1):
    for _n in _names.split():
        _MONTHS[_n] = _i
_MON = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_MON_D = "(?:" + "|".join(sorted([*_MONTHS, "may"], key=len, reverse=True)) + ")"
_ORD = r"(?:st|nd|rd|th)"
_YEAR = r"((?:19|20)\d{2})(?!\d)"
_AMOUNT = r"\d+(?:,\d{3})*(?:\.\d+)?"

_RE_CLAIM = re.compile(r"(?<![a-z0-9])([a-z]{1,4})-?(\d{3,})(?!\d)")
_RE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_RE_NUMDATE = re.compile(r"(?<![\d/.-])(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})(?!\d)")
_RE_SHORTDATE = re.compile(r"(?<![\d/.-])(\d{1,2})/(\d{1,2})(?![\d/])")
_RE_MON_DAY = re.compile(rf"\b({_MON_D})\.?\s+(\d{{1,2}}){_ORD}?(?!\d)(?:,?\s+{_YEAR})?")
_RE_DAY_MON = re.compile(
    rf"(?<!\d)(\d{{1,2}}){_ORD}?\s+(?:de\s+|of\s+(?:the\s+)?)?({_MON_D})\b\.?(?:,?\s+(?:de\s+)?{_YEAR})?")
_RE_MON_YEAR = re.compile(rf"\b({_MON})\.?,?\s+{_YEAR}")
_RE_MONTH = re.compile(rf"\b({_MON})\b")
_RE_MONEY = re.compile(
    rf"[$€£]\s?({_AMOUNT})|({_AMOUNT})\s?(?:dollars?|usd|euros?|eur|pounds?|gbp|bucks)\b"
    rf"|(?<![\d.,])(\d+(?:,\d{{3}})*\.\d{{2}})(?!\d)")
_RE_DUR = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*((?:business\s+)?(?:day|week|month|year))s?\b")
_RE_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)")
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_RE_MID = re.compile(rf"\bmid[- ]?{_MON_D}\b")
_ABBR = "mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
_RE_REL = re.compile(
    rf"\b(?:next|last|this|every|each)\s+(?:week|month|year|weekend|{_WEEKDAYS})\b|\b(?:{_WEEKDAYS})s?\b"
    rf"|\b(?:{_ABBR})\b\.?|\b(?:tomorrow|yesterday|tonight|fortnight|fortnightly|biweekly)\b"
    r"|\bend of (?:the )?(?:week|month|year)\b|\bcouple of (?:days|weeks|months|years)\b"
    r"|\b(?:a (?:few|couple)(?: of)?|several|a number of)(?=\s+(?:business\s+)?(?:days?|weeks?|months?|years?|documents?|claims?)\b)")
_RE_BARE = re.compile(rf"({_AMOUNT})(?:{_ORD}(?![a-z]))?")

_NUMWORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
             "million billion third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
             "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth "
             "fiftieth sixtieth seventieth eightieth ninetieth hundredth thousandth dozen".split())
_VALUE = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_VALUE.update({w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"})
_VALUE.update({"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9})
_ORDINAL = {"first": "one", "second": "two", "third": "three", "fourth": "four", "fifth": "five", "sixth": "six",
            "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten", "eleventh": "eleven",
            "twelfth": "twelve", "thirteenth": "thirteen", "fourteenth": "fourteen", "fifteenth": "fifteen",
            "sixteenth": "sixteen", "seventeenth": "seventeen", "eighteenth": "eighteen", "nineteenth": "nineteen",
            "twentieth": "twenty", "thirtieth": "thirty", "fortieth": "forty", "fiftieth": "fifty",
            "sixtieth": "sixty", "seventieth": "seventy", "eightieth": "eighty", "ninetieth": "ninety",
            "hundredth": "hundred", "thousandth": "thousand"}
_FOLLOWS_NUMBER_USE = re.compile(rf"\s+(?:of\s+(?:the\s+)?)?(?:{_MON_D}\b|(?:business\s+)?(?:day|week|month|year)s?\b)")
_OPENS_SENTENCE = re.compile(r"(?:\A|[.!?;:\n])\s*\Z")
_SEQUENCE_NOUN = re.compile(r"\s+(?:step|time|thing|question|one|place|document|claim|option|reason|part|point|attempt|call|visit|half|stage|line)s?\b")


def _dec(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def _valid(m: int, d: int) -> bool:
    return 1 <= m <= 12 and 1 <= d <= 31


def _year(s: str | None):
    if s is None:
        return None
    return int(s) if len(s) == 4 else "bad"  # a two-digit year never matches


def _mon(w: str) -> int:
    return 5 if w == "may" else _MONTHS[w]


def _take(rest: str, pattern: re.Pattern, make) -> tuple[str, list]:
    found: list = []

    def sub(m: re.Match) -> str:
        r = make(m)
        if r is None:
            return m.group()
        found.append(r)
        return " " * len(m.group())

    return pattern.sub(sub, rest), found


def _kind(w: str) -> str:
    w = _ORDINAL.get(w, w)
    if w == "dozen":
        return "dozen"
    if w in ("hundred", "thousand", "million", "billion"):
        return "big"
    v = _VALUE[w]
    return "unit" if v < 10 else "teen" if v < 20 else "tens"


def _part_value(words: list[str]) -> int:
    total = current = 0
    for w in words:
        w = _ORDINAL.get(w, w)
        if w == "dozen":
            current = (current or 1) * 12
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in ("thousand", "million", "billion"):
            total += (current or 1) * _VALUE[w]
            current = 0
        else:
            current += _VALUE[w]
    return total + current


def _digitize(t: str) -> str:
    """Number words become digits ("thirteen" -> "13", "nineteenth" -> "19th"), so they meet the same typed rules."""
    runs: list[list[re.Match]] = []
    prev_end = None
    for m in _WORD.finditer(t):
        if m.group() in _NUMWORDS or m.group() in _ORDINAL:
            joined = prev_end is not None and re.fullmatch(r"[\s-]+(?:and[\s-]+)?", t[prev_end:m.start()])
            if joined and runs:
                runs[-1].append(m)
            else:
                runs.append([m])
            prev_end = m.end()
        else:
            prev_end = None
    pieces, pos = [], 0
    for run in runs:
        parts: list[list[re.Match]] = []
        for m in run:
            if m.group() == "second" and ((parts and _kind(parts[-1][-1].group()) in ("unit", "teen")) or re.search(r"\b(?:a|an)\s+\Z", t[:m.start()])):
                continue  # "one second", "a second": time, not an ordinal
            k = _kind(m.group())
            pk = _kind(parts[-1][-1].group()) if parts else None
            new = pk is not None and ((k == "unit" and pk in ("unit", "teen")) or (k == "teen" and pk in ("tens", "unit", "teen"))
                                      or (k == "tens" and pk in ("tens", "unit", "teen")))
            if parts and not new:
                parts[-1].append(m)
            else:
                parts.append([m])
        for part in parts:
            words = [m.group() for m in part]
            is_ordinal = any(w in _ORDINAL for w in words)
            before, end = t[:part[0].start()], part[-1].end()
            if is_ordinal and (_SEQUENCE_NOUN.match(t, end)
                               or (re.match(r"\s*,", t[end:]) and _OPENS_SENTENCE.search(before))):
                continue  # "First, ..." opening a sentence, "the first step": a sequence, not a date
            if words == ["first"] and not (re.search(r"\bthe\s+\Z", before) or re.search(rf"\b{_MON_D}\s+\Z", before)
                                           or re.match(rf"\s+of\s+(?:the\s+)?{_MON_D}\b", t[end:])):
                continue  # "First of all", "let me first check": ordinary speech
            if words == ["one"] and not _FOLLOWS_NUMBER_USE.match(t, part[-1].end()):
                continue
            suffix = "th" if is_ordinal else ""
            pieces.append(t[pos:part[0].start()])
            pieces.append(str(_part_value(words)) + suffix)
            pos = part[-1].end()
    pieces.append(t[pos:])
    t = "".join(pieces)
    return re.sub(r"\b(?:a|an)\s+(?=(?:business\s+)?(?:day|week|month|year)s?\b)", "1 ", t)


def _mentions(t: str, claims: Sequence[str], caller_digits: Mapping[str, str]) -> tuple[list, str]:
    """Typed mentions of t, and what is left of t after they are taken out."""
    out: list = []
    rest = _digitize(t)
    rest, f = _take(rest, _RE_MID, lambda m: ("rel", _rel_key(m.group())))
    out += f
    for key in claims:  # known claim numbers, also written with spaces: "cl 2048"
        letters, digits = re.sub(r"\d", "", key), re.sub(r"\D", "", key)
        if letters and digits:
            rest, f = _take(rest, re.compile(rf"(?<![a-z0-9]){re.escape(letters)}[\s-]*{digits}(?!\d)"), lambda m, k=key: ("claim", k))
            out += f
    rest, f = _take(rest, _RE_CLAIM, lambda m: ("claim", m.group(1) + m.group(2)))
    out += f
    rest, f = _take(rest, _RE_ISO, lambda m: ("date", [(int(m.group(2)), int(m.group(3)), int(m.group(1)))])
                    if _valid(int(m.group(2)), int(m.group(3))) else None)
    out += f

    def numeric(m):
        a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3)) if m.lastindex >= 3 else None
        return ("date", [(mo, d, y) for mo, d in ((a, b), (b, a)) if _valid(mo, d)])

    rest, f = _take(rest, _RE_NUMDATE, numeric)
    out += f
    rest, f = _take(rest, _RE_SHORTDATE, numeric)
    out += f
    rest, f = _take(rest, _RE_MON_DAY, lambda m: ("date", [(_mon(m.group(1)), int(m.group(2)), _year(m.group(3)))]))
    out += f
    rest, f = _take(rest, _RE_DAY_MON, lambda m: ("date", [(_mon(m.group(2)), int(m.group(1)), _year(m.group(3)))]))
    out += f
    rest, f = _take(rest, _RE_MON_YEAR, lambda m: ("date", [(_mon(m.group(1)), None, int(m.group(2)))]))
    out += f
    rest, f = _take(rest, _RE_MONTH, lambda m: ("month", _mon(m.group(1))))
    out += f
    rest, f = _take(rest, _RE_MONEY, lambda m: ("money", _dec(next(g for g in m.groups() if g))))
    out += f
    rest, f = _take(rest, _RE_DUR, lambda m: ("dur", Decimal(m.group(1)), m.group(2)))
    out += f
    rest, f = _take(rest, _RE_PCT, lambda m: ("pct", Decimal(m.group(1))))
    out += f
    rest, f = _take(rest, _RE_REL, lambda m: ("rel", _rel_key(m.group())))
    out += f
    phone_view = rest
    bare: list = []

    def bare_num(m):
        digits = m.group(1)
        if digits in caller_digits:
            return ("claim", caller_digits[digits])
        if m.group(0) != digits:
            return ("ord", _dec(digits))
        if re.fullmatch(r"(?:19|20)\d{2}", digits):
            return ("year", int(digits))
        return ("num", _dec(digits))

    _, bare = _take(rest, _RE_BARE, bare_num)
    return out + bare, phone_view


def _rel_key(phrase: str) -> str:
    return " ".join(w for w in re.sub(r"[-.]", " ", phrase).split() if w != "the")


def _claim_key(s: str) -> str:
    return "".join(c for c in _norm(s) if c.isalnum())


def _status_regexes(forms: Sequence[str]) -> list[re.Pattern]:
    return [re.compile(r"(?<![a-z])" + r"\s+".join(re.escape(w) for w in f.split()) + r"(?![a-z])") for f in forms]


_SUBJECT = r"(?:\b(?:claims?|cases?|it|requests?|appeals?|applications?)\b|\b[a-z]{1,4}-?\d{3,}\b)"
_BE = r"(?:'s\b|\W+(?:is|was|are|were|be|been|being|has|have|had)\b)"
_STATE_BEFORE = re.compile(_SUBJECT + r"(?:\W+[\w'-]+){0,8}?" + _BE + r"(?:\W+[\w'-]+){0,3}\W*\Z")
_ACTIVE_AFTER = re.compile(r"\A(?:\W+[\w'-]+){0,2}?\W+(?:your|the|this)\W+(?:claims?|cases?|appeals?|requests?|applications?)\b")
_FIRST_PERSON_ING = re.compile(r"(?:i'm|we're|i am|we are)\W+\Z")
_BOUNDARY = re.compile(r"[.!?;](?:\s|\Z)|\n")


def _sentence_before(t: str, pos: int) -> str:
    last = None
    for last in _BOUNDARY.finditer(t, 0, pos):
        pass
    return t[(last.end() if last else 0):pos]


def _status_failure(t: str, fact_text: str, state_text: str, wary: Sequence[str], forms_map: Mapping[str, Sequence[str]],
                    data_statuses: Sequence[str]) -> bool:
    statuses = {_norm(s).strip() for s in data_statuses}
    for word in wary:
        w = _norm(word).strip()
        forms = [_norm(f).strip() for f in forms_map.get(word, forms_map.get(w, [])) or [w]]
        if w not in forms:
            forms.append(w)
        for form, pat in zip(forms, _status_regexes(forms)):
            for m in pat.finditer(t):
                if statuses & set(forms):
                    continue
                before = _sentence_before(t, m.start())
                state = bool(_STATE_BEFORE.search(before))
                if not state and _ACTIVE_AFTER.search(t[m.end():m.end() + 60]):
                    state = not (form.endswith("ing") and _FIRST_PERSON_ING.search(before))
                if not pat.search(state_text if state else fact_text):
                    return True
    return False


# ---- the check ----

def check_reply(reply: str, brief: ReplyBrief, *, wary_status_words: Sequence[str], data_statuses: Sequence[str],
                status_forms: Mapping[str, Sequence[str]] | None = None, max_length: int = 1500) -> CheckResult:
    if len(reply) > max_length:
        return CheckResult(False, ("too_long",))
    t = _norm(reply)
    facts = brief.facts()
    fact_text = _norm("\n".join(f"{f.text}\n{f.claim or ''}" for f in facts))
    state_text = _norm("\n".join(f.text for f in facts if f.kind in ("status", "consent")))
    failed: set[str] = set()

    if _has_web_address(t):
        failed.add("web_address")
    if _has_email(t):
        failed.add("email_address")

    caller_keys = [_claim_key(n) for n in brief.caller_claim_numbers]
    if not brief.verified:
        for number in brief.caller_claim_numbers:
            chars = [re.escape(c) for c in _norm(number) if c.isalnum()]
            if chars and re.search(r"[\W_]*".join(chars), t):
                failed.add("claim_detail_before_verification")

    fact_claims = [_claim_key(f.claim) for f in facts if f.claim]
    known = [k for k in {*fact_claims, *caller_keys} if k]
    caller_digits = {re.sub(r"\D", "", k): k for k in caller_keys if len(re.sub(r"\D", "", k)) >= 3}
    fm, _ = _mentions(fact_text, known, {})
    fm_claims = {m[1] for m in fm if m[0] == "claim"} | set(fact_claims)
    fact_dates = [c for m in fm if m[0] == "date" for c in m[1]]
    fact_months = {c[0] for c in fact_dates} | {m[1] for m in fm if m[0] == "month"}
    fact_years = {c[2] for c in fact_dates if isinstance(c[2], int)}
    fact_other = {m for m in fm if m[0] in ("money", "dur", "pct", "rel")}
    fact_years |= {m[1] for m in fm if m[0] == "year"}
    fact_days = {Decimal(c[1]) for c in fact_dates if c[1] is not None}
    fact_bare = ({m[1] for m in fm if m[0] == "num"} | {m[1] for m in fm if m[0] == "money"}
                 | {m[1] for m in fm if m[0] == "dur"} | fact_days)

    rm, rest = _mentions(t, known, caller_digits)
    if _digit_word_phone(t) or _digits_phone(rest):
        failed.add("phone_number")
    for m in rm:
        kind = m[0]
        if kind == "claim":
            ok = m[1] in fm_claims
        elif kind == "date":
            ok = any(any(fd[0] == c[0] and (c[1] is None or fd[1] == c[1]) and (c[2] is None or fd[2] == c[2])
                         for fd in fact_dates) for c in m[1])
            if not ok:
                months = {c[0] for c in m[1]}
                failed.add("number_not_in_facts" if months & fact_months else "month_not_in_facts")
                if not months:
                    failed.add("number_not_in_facts")
            continue
        elif kind == "month":
            if m[1] not in fact_months:
                failed.add("month_not_in_facts")
            continue
        elif kind == "num":
            ok = m[1] in fact_bare
        elif kind == "ord":
            ok = m[1] in fact_days
        elif kind == "year":
            ok = m[1] in fact_years
        else:
            ok = m in fact_other
        if not ok:
            failed.add("number_not_in_facts")

    if _status_failure(t, fact_text, state_text, wary_status_words, status_forms or {}, data_statuses):
        failed.add("status_word")

    ordered = tuple(f for f in (*_LEAK, *_INVENTED) if f in failed)
    return CheckResult(not ordered, ordered)
