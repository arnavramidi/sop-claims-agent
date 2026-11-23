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

Polarity: a reply may not negate what the facts state, nor state what the facts about a claim only negate.
- the guarded words are the claim statuses the data holds, the wary status words (with their forms), and the groups
  of `polarity_forms` in settings (needed/required/missing, deadline/due, paid, covered, received, passed, eligible);
  the words of one group count as one, so "no documents are needed" lets "nothing is required" through;
- a negator is not, no, never, nothing, none, neither, nor, without, cannot, nobody, nowhere, "no longer", "no one",
  or any "n't" (with or without the apostrophe); "un"/"non" before a guarded word ("unpaid") negates it too;
- a negator reaches a guarded word at most four words after it in the same clause ("without" only one: it heads a
  phrase), or one before it when the negator follows a verb like "is", "has", "can" ("the deadline is not on file"); punctuation and words such as and, but,
  because, so, if, until end a clause, "or" does not; "can not you ...?" is a question and negates nothing;
- a clause opened by "if", "unless" or "whether" states nothing and is skipped; a question never drives the reverse
  direction ("is there anything else you need?");
- in a reply a status counts only where it is used as a claim's state (the status-word patterns), so "I can't open a
  new claim" is not about the status "open"; in a fact every use counts;
- a negated guarded word passes only when a fact negates a word of its group; a guarded word said without a negation
  fails when every fact about a claim that uses its group negates it ("not paid yet" -> "it was paid"); a note that
  names no claim ("not a guaranteed payment") does not drive that reverse direction;
- when the reply's sentence names a claim, only that claim's facts and the facts naming no claim count.

Known limits: a bare ending word with no dot and no spelled "dot" ("example com") is not read as an address.
A wary status word used as a claim's state is found by sentence patterns (a claim-like subject, a form of "to be"
within eight words; or "have accepted your claim"); patterns cannot cover every phrasing. Look-alike letters from other alphabets (a Cyrillic "e" in "example.com") and number words in
languages other than English are not covered by patterns; digits, "@", and dot shapes are caught in any language.
Polarity covers only the guarded words: a denial reason's own words, a document's name or any other word can be
negated or un-negated freely ("the review file did include the report" passes); a fact that negates a word for one
claim lets a sentence that names no claim negate it for another; negation further than four words away, in another
clause (a line break ends one), by meaning ("you can skip the office note"), or in another language is not seen; "no need to worry" fails.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal

from dataclasses import dataclass

from claims_agent.reply.brief import ReplyBrief

_LEAK = ("web_address", "phone_number", "email_address", "claim_detail_before_verification")
_INVENTED = ("number_not_in_facts", "month_not_in_facts", "status_word", "polarity")
_OTHER = ("too_long", "required_mention_missing", "forbidden_mention")


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    failures: tuple[str, ...]

    @property
    def leak(self) -> tuple[str, ...]:
        return tuple(f for f in self.failures if _label(f) in _LEAK)

    @property
    def invented(self) -> tuple[str, ...]:
        return tuple(f for f in self.failures if _label(f) in _INVENTED)

    @property
    def other(self) -> tuple[str, ...]:
        """Failures that belong to neither group (`too_long`)."""
        return tuple(f for f in self.failures if _label(f) in _OTHER)


def _label(failure: str) -> str:
    """The rule a failure belongs to: a polarity failure carries its reason after the label ("polarity: ...")."""
    return failure.split(":", 1)[0]


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
                status_forms: Mapping[str, Sequence[str]] | None = None, max_length: int = 1500,
                polarity_forms: Mapping[str, Sequence[str]] | None = None) -> CheckResult:
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
        if _states_a_status(t, data_statuses, status_forms or {}):
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

    failed |= required_mentions(t, brief)
    ordered = tuple(f for f in (*_LEAK, *_INVENTED, *_OTHER) if f in failed)
    ordered += tuple(_polarity_failures(t, facts, data_statuses, wary_status_words, status_forms or {}, polarity_forms or {}))
    return CheckResult(not ordered, ordered)


def _states_a_status(t: str, data_statuses: Sequence[str], forms_map: Mapping[str, Sequence[str]]) -> bool:
    """Before verification no claim's state may be stated, not even one the caller named ("I'm sorry your claim was
    denied" confirms it). A status the data uses, or one of its forms, counts when the sentence uses it as a claim's
    state, by the same sentence patterns as the wary words; "a denied claim" or "claim details" do not."""
    for status in data_statuses:
        s = _norm(status).strip()
        if not s:
            continue
        forms = [_norm(f).strip() for f in forms_map.get(status, forms_map.get(s, [])) or [s]]
        if s not in forms:
            forms.append(s)
        for form, pat in zip(forms, _status_regexes(forms)):
            for m in pat.finditer(t):
                before = _sentence_before(t, m.start())
                if _STATE_BEFORE.search(before):
                    return True
                if _ACTIVE_AFTER.search(t[m.end():m.end() + 60]) and not (
                        form.endswith("ing") and _FIRST_PERSON_ING.search(before)):
                    return True
    return False


# ---- polarity: a reply may not negate what the facts state, nor state what the facts about a claim only negate ----

_POLARITY = "polarity"
# "n't" is written out first ("don't" -> "do not", "can't" -> "can not"), so one word list covers every contraction,
# with or without the apostrophe; "needn't" becomes "need not" and so meets the word it negates.
_CONTRACTION = re.compile(r"\b(ca|wo|sha|ai|do|does|did|is|are|was|were|could|should|would|has|have|had|need|must"
                          r"|might|dare)n'?t\b")
_SPELLED = {"ca": "can", "wo": "will", "sha": "shall", "ai": "is"}
_NEGATOR = re.compile(r"\bno\s+(?:longer|one)\b|\b(?:not|no|never|nothing|none|neither|nor|without|cannot|nobody"
                      r"|nowhere)\b")
# A negation reaches only as far as its clause: punctuation, a spaced dash, or a word that opens another clause ends
# it. "or" and "nor" do not ("not denied or closed" negates both).
_CLAUSE_END = re.compile(r"[.!?;:,()\[\]{}\"\n]|\s-+\s|\s-{2,}|\b(?:and|but|because|so|until|unless|although|though"
                         r"|while|whereas|if|whether|when|once|since|before|after|then|however|which)\b")
_SUPPOSING = ("if", "unless", "whether")  # a clause these open supposes, it states nothing
_NEAR = 4  # at most this many words between a negator and the guarded word it negates
_NEAR_PHRASE = {"without": 1}  # "without" heads a phrase ("without the office note", "without being paid"), and
                               # reaches no further: "without losing track of the missing item" negates nothing here
# A word before the negator is negated only when the negator follows a verb like these ("the deadline is not on
# file", "payment has not been made"); "which requested document can not you get" asks, it negates nothing: a negator
# followed by a subject pronoun is an inverted question.
_AUXILIARY = frozenset("am is are was were be been being has have had do does did will would shall should can could "
                       "may might must need".split())
_PRONOUN = frozenset("i you we they he she it".split())
_TOKEN = re.compile(r"[a-z0-9']+")
_PLAIN_WORDS = re.compile(r"[a-z][a-z _-]{0,29}")


@dataclass(frozen=True)
class _Said:
    group: tuple[str, str]   # ("word", group name) or ("status", status)
    negated: bool
    phrase: str              # the words as they stand, built only from the negator lists and the guarded forms
    pos: int


def _expand(t: str) -> str:
    return _CONTRACTION.sub(lambda m: _SPELLED.get(m.group(1), m.group(1)) + " not", t)


def _clauses(t: str) -> list[tuple[int, int, str]]:
    """Each clause, with the word that opened it ("" after punctuation or at the start)."""
    out, start, opener = [], 0, ""
    for m in _CLAUSE_END.finditer(t):
        out.append((start, m.start(), opener))
        start, opener = m.end(), m.group().strip()
    out.append((start, len(t), opener))
    return [(a, b, o) for a, b, o in out if b > a]


def _word_index(spans: list[tuple[int, int]], pos: int) -> int:
    return sum(1 for a, _ in spans if a < pos)


def _guarded_patterns(polarity_forms: Mapping[str, Sequence[str]], data_statuses: Sequence[str],
                      wary: Sequence[str], status_forms: Mapping[str, Sequence[str]]):
    """(group, form, pattern, named, is_status) for every guarded form. A form after "un" or "non" ("unpaid",
    "non-covered", "unnecessary") is a negated use of it."""
    out = []

    def add(group, forms, named, is_status):
        for f in dict.fromkeys(_norm(x).strip() for x in forms):
            if f:
                body = r"\s+".join(re.escape(w) for w in f.split())
                out.append((group, f, re.compile(r"(?<![a-z0-9'])(un|non-?)?" + body + r"(?![a-z])"), named, is_status))

    for name, forms in polarity_forms.items():
        add(("word", _norm(name).strip()), forms, True, False)
    for status in data_statuses:
        s = _norm(status).strip()
        if s:
            forms = [*(status_forms.get(status) or status_forms.get(s) or []), s, s.replace("_", " ")]
            add(("status", s), forms, bool(_PLAIN_WORDS.fullmatch(s)), True)
    for word in wary:
        w = _norm(word).strip()
        add(("status", w), [*(status_forms.get(word) or status_forms.get(w) or []), w], True, True)
    return out


def _said(t: str, patterns, *, as_state_only: bool) -> list[_Said]:
    """Every guarded word in t, with whether its clause negates it. In a reply a status counts only where it is used
    as a claim's state (the status-word patterns); in a fact every use counts."""
    out: list[_Said] = []
    for a, b, opener in _clauses(t):
        if opener in _SUPPOSING:
            continue  # "if you need anything else", "if the report isn't needed": nothing is stated
        clause = t[a:b]
        spans = [m.span() for m in _TOKEN.finditer(clause)]
        words = [clause[x:y] for x, y in spans]
        negs = []
        for m in _NEGATOR.finditer(clause):
            i, j = _word_index(spans, m.start()), _word_index(spans, m.end()) - 1
            before = words[i - 1] if i > 0 else ""
            aux = before in _AUXILIARY or before.endswith(("'s", "'re", "'ve", "'m"))  # "it's not", "they're not"
            if m.group() == "not" and before in _AUXILIARY and j + 1 < len(words) and words[j + 1] in _PRONOUN:
                continue  # "can not you get ...?", "is not it ...?": a question
            n = " ".join(m.group().split())
            negs.append((i, j, n, aux and n not in _NEAR_PHRASE))
        for group, form, pat, named, is_status in patterns:
            for m in pat.finditer(clause):
                if is_status and as_state_only:
                    before = _sentence_before(t, a + m.start())
                    after = t[a + m.end():a + m.end() + 60]
                    if not (_STATE_BEFORE.search(before) or (_ACTIVE_AFTER.search(after) and not (
                            form.endswith("ing") and _FIRST_PERSON_ING.search(before)))):
                        continue
                first, last = _word_index(spans, m.start()), _word_index(spans, m.end()) - 1
                shown = form if named else "a claim status"
                near = [(i, j, n) for i, j, n, aux in negs if (j < first and first - j - 1 <= _NEAR_PHRASE.get(n, _NEAR))
                        or (aux and i > last and i - last - 1 <= _NEAR)]
                if m.group(1):
                    out.append(_Said(group, True, f"{m.group(1)}{form}" if named else shown, a + m.start()))
                elif near:
                    i, j, n = min(near, key=lambda x: abs(x[0] - first))
                    gap = " ... " if (j < first and first - j > 1) or (i > last and i - last > 1) else " "
                    phrase = f"{n}{gap}{shown}" if j < first else f"{shown}{gap}{n}"
                    out.append(_Said(group, True, phrase, a + m.start()))
                else:
                    out.append(_Said(group, False, shown, a + m.start()))
    return out


def _claims_named(sentence: str, keys: Sequence[str]) -> set[str]:
    named = set()
    for key in keys:
        letters, digits = re.sub(r"\d", "", key), re.sub(r"\D", "", key)
        if digits and re.search(rf"(?<![a-z0-9]){re.escape(letters)}[\s-]*{digits}(?!\d)", sentence):
            named.add(key)
    return named


def _sentences(t: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in _BOUNDARY.finditer(t):
        out.append((start, m.end()))
        start = m.end()
    out.append((start, len(t)))
    return out


def _polarity_failures(t: str, facts: Sequence, data_statuses: Sequence[str], wary: Sequence[str],
                       status_forms: Mapping[str, Sequence[str]], polarity_forms: Mapping[str, Sequence[str]]) -> list[str]:
    """A guarded word the reply negates must be negated by a fact (any negator: "no documents are needed" lets the
    reply say "you don't need anything"). A guarded word the reply states without a negation fails when every fact
    about a claim that uses it negates it ("not paid yet" -> "it was paid"). When the reply's sentence names a claim,
    only that claim's facts and the facts that name no claim count."""
    patterns = _guarded_patterns(polarity_forms, data_statuses, wary, status_forms)
    if not patterns:
        return []
    fact_said = []
    for f in facts:
        key = _claim_key(f.claim) if f.claim else None
        for s in _said(_expand(_norm(f.text)), patterns, as_state_only=False):
            fact_said.append((s, key))
    keys = sorted({k for _, k in fact_said if k} | {_claim_key(f.claim) for f in facts if f.claim})
    t = _expand(t)
    out: list[str] = []
    for a, b in _sentences(t):
        named = _claims_named(t[a:b], keys)
        here = [(s, k) for s, k in fact_said if not named or k is None or k in named]
        for s in _said(t[a:b], patterns, as_state_only=True):
            same = [(f, k) for f, k in here if f.group == s.group]
            if s.negated:
                if any(f.negated for f, _ in same):
                    continue
                why = "the facts state it without a negation" if same else "the facts do not state it negated"
                out.append(f'{_POLARITY}: the reply says "{s.phrase}", but {why}')
            elif not t[a:b].rstrip().endswith("?"):  # a question states nothing: "anything else you need?"
                about_claim = [f for f, k in same if k is not None]
                if about_claim and all(f.negated for f in about_claim) and not any(
                        not f.negated for f, k in same if k is None):
                    out.append(f'{_POLARITY}: the reply says "{s.phrase}" without a negation, but the facts state it '
                               f'only negated')
    return list(dict.fromkeys(out))


# ---- what a reply part says the reply must, or must not, mention ----

def _mentioned(t: str, wording: str) -> bool:
    """`wording` appears in the normalised reply `t` as whole words (any spacing between them)."""
    words = _norm(wording).split()
    return bool(words) and re.search(r"(?<![a-z0-9])" + r"\s+".join(re.escape(w) for w in words) + r"(?![a-z0-9])",
                                     t) is not None


def required_mentions(t: str, brief: ReplyBrief) -> set[str]:
    """The parts' own requirements, over the whole normalised reply: every `must_mention` entry needs one of its
    alternatives, and no `must_not_mention` wording may appear. Kept apart from the checks above: these are what a
    part promises the caller (every option still open, the rule), not what a reply may never contain."""
    failed: set[str] = set()
    for part in brief.parts:
        if any(not any(_mentioned(t, alt) for alt in entry) for entry in part.must_mention):
            failed.add("required_mention_missing")
        if any(_mentioned(t, word) for word in part.must_not_mention):
            failed.add("forbidden_mention")
    return failed
