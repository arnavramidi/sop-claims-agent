"""Every setting and threshold the files hold is read by the app: a key nothing reads would look like a switch that
does nothing."""

import re
import tomllib
from collections.abc import Iterator

from pydantic import BaseModel

from claims_agent.settings import ROOT, Settings, Thresholds

SOURCE = ROOT / "src" / "claims_agent"


def _app_source() -> str:
    """The app's own code, without the module that declares the settings."""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SOURCE.rglob("*.py"))
                     if p != SOURCE / "settings.py")


def _model(annotation) -> type[BaseModel] | None:
    """The settings model a field holds, directly or as the values of a mapping."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    args = getattr(annotation, "__args__", ())
    return next((a for a in args if isinstance(a, type) and issubclass(a, BaseModel)), None)


def _fields(model: type[BaseModel], data: dict, prefix: str = "") -> Iterator[tuple[str, str]]:
    """(dotted path, attribute name) of every field the file sets, walking into nested settings models; the keys of a
    mapping (state names, need names, follow-up topics) are data, so only the fields of its values are walked."""
    for name, info in model.model_fields.items():
        if name not in data:
            continue
        yield f"{prefix}{name}", name
        inner = _model(info.annotation)
        if inner is None:
            continue
        value = data[name]
        if inner is info.annotation:
            yield from _fields(inner, value, f"{prefix}{name}.")
        else:
            for key, item in value.items():
                yield from _fields(inner, item, f"{prefix}{name}.{key}.")


def _unread(model: type[BaseModel], path) -> list[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    source = _app_source()
    return sorted({dotted for dotted, name in _fields(model, data)
                   if not re.search(rf"\.{re.escape(name)}\b", source)})


def test_every_key_in_settings_toml_is_read_by_the_app():
    assert _unread(Settings, ROOT / "settings.toml") == []


def test_every_threshold_is_read_by_the_app():
    assert _unread(Thresholds, ROOT / "thresholds.toml") == []


def test_the_limits_the_rules_count_against_are_settings():
    data = tomllib.loads((ROOT / "settings.toml").read_text(encoding="utf-8"))["limits"]
    for name in ("unclear_email_replies", "upset_turns_before_offer", "questions_per_message"):
        assert isinstance(data.get(name), int) and data[name] > 0, name
