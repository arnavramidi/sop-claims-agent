"""Masked forms of what a caller typed, for anything that is kept or shown: memory, hand-over notes.

Only the masked form is kept beside the normalised key a comparison uses, so nothing that compares depends on it, and
nothing kept shows a whole ID number or email address.
"""

import re

from claims_agent.inspect import mask_ids

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def mask_email(email: str) -> str:
    """An email address as its first letter and its domain: "m***@email.com"."""
    local, _, domain = email.strip().rpartition("@")
    return f"{local[:1]}***@{domain}" if local else "***"


def mask_details(text: str) -> str:
    """Text with every email address masked and every run of nine or more digits cut to its last four."""
    return mask_ids(_EMAIL.sub(lambda m: mask_email(m.group()), text))
