"""The sample callers: who a person trying the agent can pretend to be.

`build(folder, today)` reads the data files of a folder and returns, for every policyholder, the details that verify
them, their claims, any listed representative, what the caller is good for testing and ready-to-send opening
messages. Everything factual comes from the data files; only the one-line purposes and the shape of the opening
messages are written here, keyed by party id (a policyholder with no entry here makes `build` fail, so the list can
never silently miss someone).

The result is committed as `sample_callers.json` in the data folder, together with a digest of the data files it was
built from. The server lists callers from that file only (never from the database, whose account cannot read the
customers table), and only while the digest still matches the folder's files, so a folder of other data, or a
changed one, shows nothing. `scripts/sample_callers.py` writes the file and `docs/sample-callers.md`.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

FILE = "sample_callers.json"
SOURCES = ("policyholders.json", "claims.json", "representatives.json")  # what the digest covers
NOTE = ("Invented sample data for trying the agent. Every person, number and address here is made up. "
        "The page shows this list only while the data folder is the bundled sample data it was generated from.")

ID_WORDS = {"ssn_last4": "social security number", "national_id_last4": "national ID number"}

# Editorial, by party id: what the caller is good for testing, and the opening messages. A message names which
# details it gives (the template's blanks are filled from the data) so the three details always match the data.
_VERIFY = "My name is {name}, my date of birth is {dob_words} and the last four digits of my {id_word} are {id_last4}."
_PHONE_EMAIL = "I'm {name}. My phone number is {phone_dashed} and my email is {email}."
EDITORIAL: dict[str, dict[str, Any]] = {
    "P9": {
        "good_for": "The main walk-through: a denied healthcare claim that needs two documents, plus closed and open "
                    "claims of other types. A listed representative (her son) can ask on her behalf.",
        "messages": [("Name, birth date and ID", _VERIFY + " What is the status of my denied claim?"),
                     ("Her son, as representative",
                      "Hello, I'm {rep_name}, {name}'s {rep_relationship}. Her date of birth is {dob_words} and the "
                      "last four digits of her {id_word} are {id_last4}. May I ask about her denied claim?")],
    },
    "P7": {
        "good_for": "A caller with no claims on file: after verifying, the agent has nothing to look up.",
        "messages": [("Name, birth date and ID", _VERIFY + " Do I have any open claims?")],
    },
    "P12": {
        "good_for": "A national ID instead of a social security number, and one denied claim missing one document.",
        "messages": [("Name, birth date and ID", _VERIFY + " Why was my claim denied?")],
    },
    "P13": {
        "good_for": "Accepted name, phone and email aliases: the agent also accepts the alternate spellings below.",
        "messages": [("Name, birth date and ID", _VERIFY + " What do I still need to send for my claim?"),
                     ("Using the alternate name and email",
                      "Hi, I'm {alias_name}. My date of birth is {dob_words} and my email is {alias_email}. "
                      "Can you look up my claim?")],
    },
    "P14": {
        "good_for": "Two denied claims of the same type (healthcare): the agent has to ask which one. One appeal "
                    "deadline is tomorrow.",
        "messages": [("Name, birth date and ID", _VERIFY + " Can I still appeal my denied claim?"),
                     ("Name, phone and email", _PHONE_EMAIL + " My healthcare claim was denied. What do I need?")],
    },
    "P15": {
        "good_for": "An appeal deadline that has already passed, so the agent must not say days are left. Also a "
                    "listed spouse as representative, and a closed dental claim.",
        "messages": [("Name, birth date and ID", _VERIFY + " How long do I have to appeal my denied auto claim?"),
                     ("Her spouse, as representative",
                      "Hello, I'm {rep_name}, {name}'s {rep_relationship}. Her date of birth is {dob_words} and the "
                      "last four digits of her {id_word} are {id_last4}. May I ask about her auto claim?")],
    },
    "P16": {
        "good_for": "A name with accents (the database compares names ignoring accents, so \"Jose Garcia\" works). "
                    "One open claim with a document still outstanding and no denial, and a closed dental claim.",
        "messages": [("Name written with accents", _VERIFY + " What do I still need to send for my open claim?"),
                     ("Name without accents",
                      "My name is {plain_name}, my date of birth is {dob_words} and the last four digits of my "
                      "{id_word} are {id_last4}. What is the status of my claim?")],
    },
    "P17": {
        "good_for": "Shares a full name with another caller (Sam Rivera, P18): the name alone is not enough, and "
                    "three details must single out one person. The appeal deadline is today.",
        "messages": [("Name, birth date and ID", _VERIFY + " Is my denied claim still open to appeal?"),
                     ("Name and one other detail only (not enough)",
                      "Hi, I'm {name} and my phone number is {phone_dashed}. What is the status of my claim?")],
    },
    "P18": {
        "good_for": "The other Sam Rivera: only the name is the same as P17, so a different birth date, ID, phone "
                    "or email picks this person out. One closed dental claim.",
        "messages": [("Name, birth date and ID", _VERIFY + " Do I have a claim on file?")],
    },
}


def digest(folder: Path) -> str:
    """A digest of the data files the list is built from (their names and exact bytes)."""
    h = hashlib.sha256()
    for name in SOURCES:
        path = Path(folder) / name
        h.update(name.encode() + b"\0")
        h.update(path.read_bytes() if path.is_file() else b"")
        h.update(b"\0")
    return h.hexdigest()


def _read(folder: Path, name: str) -> list[dict]:
    return json.loads((Path(folder) / name).read_text(encoding="utf-8"))


def _dashed(phone: str) -> str:
    d = phone[-10:]
    return f"{d[:3]}-{d[3:6]}-{d[6:]}"


def _words(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _plain(name: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c))


def _deadline_note(deadline: str | None, today: date) -> str | None:
    if deadline is None:
        return None
    left = (date.fromisoformat(deadline) - today).days
    if left < 0:
        return f"passed {-left} day{'s' if left != -1 else ''} before the agent's date"
    if left == 0:
        return "is the agent's date: the last day"
    return f"{left} day{'s' if left != 1 else ''} after the agent's date"


def _claim(c: dict, today: date) -> dict:
    out: dict[str, Any] = {"case_id": c["case_id"], "case_type": c.get("case_type"), "status": c["status"],
                           "created": c.get("created_at"), "summary": c.get("summary"),
                           "appeal_deadline": c.get("appeal_deadline"),
                           "appeal_deadline_note": _deadline_note(c.get("appeal_deadline"), today),
                           "documents_needed": list(c.get("documents_needed") or [])}
    for key in ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee"):
        out[key] = c.get(key)
    return out


def build(folder: Path, today: date) -> dict:
    """The sample-callers document for the data in `folder`, as the date `today` sees it."""
    people, claims, reps = (_read(folder, n) for n in SOURCES)
    callers = []
    for p in people:
        pid = p["party_id"]
        if pid not in EDITORIAL:
            raise ValueError(f"policyholder {pid} has no entry in EDITORIAL")
        mine = [r for r in reps if r["buyer_party_id"] == pid]
        facts = {"name": p["name"], "plain_name": _plain(p["name"]), "dob_words": _words(p["dob"]),
                 "id_word": ID_WORDS[p["id_type"]], "id_last4": p["id_last4"], "phone_dashed": _dashed(p["phone"]),
                 "email": p["email"], "policy_number": p["policy_number"],
                 "alias_name": (p.get("name_aliases") or [""])[0], "alias_email": (p.get("email_aliases") or [""])[0],
                 "rep_name": mine[0]["rep_name"] if mine else "", "rep_relationship": mine[0]["relationship"] if mine else ""}
        callers.append({
            "party_id": pid, "name": p["name"], "name_aliases": list(p.get("name_aliases") or []),
            "policy_number": p["policy_number"], "date_of_birth": p["dob"], "id_type": p["id_type"],
            "id_last4": p["id_last4"], "phone": p["phone"], "phone_aliases": list(p.get("phone_aliases") or []),
            "email": p["email"], "email_aliases": list(p.get("email_aliases") or []),
            "claims": [_claim(c, today) for c in claims if c["party_id"] == pid],
            "representatives": [{"name": r["rep_name"], "relationship": r["relationship"]} for r in mine],
            "good_for": EDITORIAL[pid]["good_for"],
            "opening_messages": [{"label": label, "text": template.format(**facts)}
                                 for label, template in EDITORIAL[pid]["messages"]],
        })
    return {"note": NOTE, "agent_today": today.isoformat(), "digest": digest(folder), "callers": callers}


def to_json(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def served(folder: Path) -> list[dict]:
    """The callers the server may list for `folder`: those in its sample_callers.json, provided the file exists, reads
    as a document, and carries the digest of the folder's own data files. Anything else (no file, a file for other
    data, a damaged file) gives an empty list."""
    try:
        doc = json.loads((Path(folder) / FILE).read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("digest") != digest(folder) or not isinstance(doc.get("callers"), list):
            return []
        return doc["callers"]
    except (OSError, ValueError):
        return []


def to_markdown(doc: dict) -> str:
    lines = ["# Sample callers", "",
             "Who to pretend to be when you try the agent. " + doc["note"], "",
             f"The agent treats **{doc['agent_today']}** as today, so the deadlines below are counted from that date. "
             "To use one: open the page, expand \"Sample callers\", press a message button (it fills the message box, "
             "it does not send), then send it. Or paste a message yourself.", "",
             "This file is generated by `scripts/sample_callers.py` from the files in `data/starter/`; do not edit it.", ""]
    for c in doc["callers"]:
        lines += [f"## {c['name']} ({c['party_id']})", "", f"**Good for:** {c['good_for']}", "",
                  "| Detail | Value |", "| --- | --- |",
                  f"| Name | {c['name']} |"]
        if c["name_aliases"]:
            lines.append(f"| Also accepted as | {', '.join(c['name_aliases'])} |")
        lines += [f"| Policy number | {c['policy_number']} |", f"| Date of birth | {c['date_of_birth']} |",
                  f"| ID ({ID_WORDS[c['id_type']]}, last four) | {c['id_last4']} |", f"| Phone | {c['phone']} |"]
        if c["phone_aliases"]:
            lines.append(f"| Other phone | {', '.join(c['phone_aliases'])} |")
        lines.append(f"| Email | {c['email']} |")
        if c["email_aliases"]:
            lines.append(f"| Other email | {', '.join(c['email_aliases'])} |")
        for r in c["representatives"]:
            lines.append(f"| Listed representative | {r['name']} ({r['relationship']}) |")
        lines += ["", "Claims:", ""]
        if not c["claims"]:
            lines.append("- none on file")
        for k in c["claims"]:
            bits = [f"{k['case_id']}: {k['case_type']}, {k['status']}, created {k['created']}"]
            if k["appeal_deadline"]:
                bits.append(f"appeal deadline {k['appeal_deadline']} ({k['appeal_deadline_note']})")
            if k["documents_needed"]:
                bits.append("documents needed: " + ", ".join(k["documents_needed"]))
            bits.append(f"expected reimbursement {k['expected_reimbursement_amount']}, allowed maximum "
                        f"{k['allowed_max_amount']}, net pay {k['net_pay']}, net fee {k['net_fee']}")
            lines.append("- " + "; ".join(bits) + f". {k['summary']}")
        lines += ["", "Opening messages:", ""]
        lines += [f"- {m['label']}: \"{m['text']}\"" for m in c["opening_messages"]]
        lines.append("")
    return "\n".join(lines)
