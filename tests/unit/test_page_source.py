import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "src" / "claims_agent" / "web"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def all_web_source() -> str:
    return "\n".join(read(f"src/claims_agent/web/{n}") for n in ("index.html", "app.js", "styles.css"))


def test_the_page_never_writes_html_from_text(): assert "innerHTML" not in read("src/claims_agent/web/app.js") and "insertAdjacentHTML" not in read("src/claims_agent/web/app.js")
def test_the_conversation_id_lives_in_the_address_hash(): assert "location.hash" in read("src/claims_agent/web/app.js")
def test_the_inspector_is_labelled_as_the_operators_view(): assert "Operator view" in read("src/claims_agent/web/index.html")


def test_no_other_way_to_run_or_write_markup_from_text():
    for name in ("app.js", "index.html"):
        src = read(f"src/claims_agent/web/{name}")
        for banned in ("outerHTML", "document.write", "eval(", "new Function", "javascript:", "setAttribute(\"on", "setAttribute('on"):
            assert banned not in src, f"{banned} in {name}"
    assert not re.search(r"\son[a-z]+\s*=", read("src/claims_agent/web/index.html"), re.I)


def test_nothing_is_loaded_from_another_origin():
    assert not re.search(r"(https?:)?//[a-z0-9.-]+\.[a-z]{2,}", all_web_source(), re.I)
    assert "@import" not in read("src/claims_agent/web/styles.css")


def test_api_text_never_goes_into_href_src_style_or_class():
    js = read("src/claims_agent/web/app.js")
    for banned in (".href", ".src", ".style.", "setAttribute(\"style", "setAttribute(\"href", "setAttribute(\"src"):
        assert banned not in js, banned
    assert "encodeURIComponent" in js


def js() -> str:
    return read("src/claims_agent/web/app.js")


def test_no_other_markup_parsers_or_code_runners_in_the_script():
    src = js()
    for banned in ("DOMParser", "createContextualFragment", "setHTMLUnsafe", "parseHTMLUnsafe", "importScripts", "srcdoc"):
        assert banned not in src, banned
    assert not re.search(r"\bFunction\s*\(", src)
    assert not re.search(r"set(Timeout|Interval)\(\s*[\"'`]", src)


def builds_property_name_from_strings(src: str) -> bool:
    """True for `obj["a" + x]`, `obj[x + "a"]` and template-literal names: a name built inside brackets that
    directly follow an identifier, `)` or `]`. An array literal after `;` or an operator is not matched."""
    return bool(re.search(r"[\w)\]]\[\s*\(*\s*(?:[\"'`][^\"'`]*[\"'`]\s*\)*\s*\.concat\(|[\w.$]+\s*\)*\s*\.concat\(\s*[\"'`]|[\"'`][^\]\"'`]*[\"'`]\s*\+|[^\]]*\+\s*[\"'`][^\]\"'`]*[\"'`]\s*\]|`[^`]*\$\{)", src))


def test_the_bracket_check_catches_built_names_and_not_array_literals():
    assert builds_property_name_from_strings('obj["a" + key]')
    assert builds_property_name_from_strings("obj[key + 'x']")
    assert builds_property_name_from_strings("data[`row_${n}`]")
    assert not builds_property_name_from_strings('const t = count + " turns";\n[...els].forEach(f);')
    assert not builds_property_name_from_strings('x = a + "b"; [1, 2]')
    assert builds_property_name_from_strings('obj[("a" + k)]')
    assert builds_property_name_from_strings('obj["a".concat(k)]')
    assert builds_property_name_from_strings('obj[("a").concat(k)]')
    assert builds_property_name_from_strings('obj[k.concat("a")]')
    assert not builds_property_name_from_strings("ORDER.filter(f).concat(Object.keys(p))")
    assert not builds_property_name_from_strings("payload[key]")


def test_no_property_names_built_from_strings_and_no_unsafe_attributes():
    src = js()
    assert not builds_property_name_from_strings(src)
    assert not re.search(r"setAttribute\(\s*[\"'](style|href|src|on|action|formaction)", src, re.I)
    assert not re.search(r"setAttribute\(\s*(?![\"'])", src), "setAttribute takes a string literal name"
    assert not re.search(r"\bclassName\s*=", src)
    assert not re.search(r"\blocation(\.\w+)?\s*=[^=]", src)
    for banned in ("location.assign", "location.replace", "window.open"):
        assert banned not in src, banned
    assert not re.search(r"\.(style|action)\s*=[^=]", src)


def test_the_form_cannot_put_text_in_a_url_if_the_script_fails():
    html = read("src/claims_agent/web/index.html")
    form = re.search(r"<form[^>]*>", html).group(0)
    assert 'method="post"' in form and "action=" in form
    assert not re.search(r"<input[^>]*\sname=", html)
    assert "<noscript>" in html and "needs its script" in html


def test_a_hash_is_accepted_only_when_shaped_like_a_uuid():
    assert "UUID.test" in js()


def test_each_function_and_constant_is_declared_once():
    src = js()
    names = re.findall(r"^(?:async\s+)?(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=)", src, re.M)
    declared = [a or b for a, b in names]
    assert len(declared) == len(set(declared)), sorted(n for n in set(declared) if declared.count(n) > 1)


def test_unsent_text_is_kept_per_conversation_in_memory_only():
    src = js()
    assert "drafts: new Map()" in src and "app.draft " not in src and "app.draft=" not in src
    for banned in ("localStorage", "sessionStorage", "indexedDB", "document.cookie"):
        assert banned not in src, banned
