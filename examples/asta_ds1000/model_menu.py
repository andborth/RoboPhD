"""The solver-model menu: which `model_registry` handles a run may use.

`--model-menu GPT_5_4` (or a comma-separated list) narrows a run's menu to a
subset of the nine registry handles. It can only narrow — a name that is not
one of the nine is rejected, so the flag can never introduce a model. The
restriction is enforced in three places, each independent of the others:

  1. `model_registry` exports only the menu's handles (it reads
     MODEL_MENU_ENV at import), so importing anything else fails.
  2. The evaluator scores an example 0 when its recorded model usage names
     a model outside the menu — whatever route the call took.
  3. main.py rewrites background.md's handle table and code examples to the
     menu, and the default seed's handle to an on-menu one, so evolution is
     never shown a handle it cannot use.

This module is dependency-free (no Inspect import, no provider keys) so
main.py can parse and validate a menu before anything constructs a model.
`model_registry.py` deliberately does NOT import it: the submit scripts copy
the registry into self-contained submission directories, so the registry
keeps its own copy of the handle list and env-var name. The unit tests pin
the two copies together.

This file is kept identical in asta_ds1000 and asta_paper_finder.
"""

import ast
import os
import re

# The nine agent-facing handles, in background.md's table order. Archive-only
# registry names (superseded-model aliases, replay handles) are deliberately
# absent: they are not on the menu and can never be selected.
MENU = (
    "GPT_5_4_MINI",
    "GPT_5_4",
    "GPT_5_5",
    "CLAUDE_HAIKU_4_5",
    "CLAUDE_SONNET_4_6",
    "CLAUDE_OPUS_4_8",
    "GEMINI_3_1_FLASH_LITE",
    "GEMINI_3_5_FLASH",
    "GEMINI_3_1_PRO_PREVIEW",
)

# Read by model_registry at import. The evaluator sets (or clears) it before
# its first registry import, and its subprocess workers inherit it.
MODEL_MENU_ENV = "ROBOPHD_MODEL_MENU"

# Longest names first so the alternation never stops at a prefix; the word
# boundaries already keep GPT_5_4 from matching inside GPT_5_4_MINI ("_" is a
# word character), the ordering is belt-and-braces.
_HANDLE_RE = re.compile(
    r"\b(" + "|".join(sorted(MENU, key=len, reverse=True)) + r")\b"
)
# A handle-table row in background.md: | `HANDLE` | ...
_TABLE_ROW_RE = re.compile(r"^\| `([A-Z][A-Z0-9_]*)` \|")
_FENCE_RE = re.compile(r"(```python\n)(.*?)(```)", re.S)
_IMPORT_LINE_RE = re.compile(r"^from model_registry import [^\n]*$", re.M)
# A stanza whose comments name a provider family documents family-specific
# behaviour, so it must not be re-pointed at another family's handle.
_PROVIDER_WORDS_RE = re.compile(r"\b(OpenAI|Anthropic|Gemini|Google)\b")
_HANDLE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def family(handle: str) -> str:
    """Provider family of a handle name: "GPT", "CLAUDE" or "GEMINI"."""
    return handle.split("_", 1)[0]


def validate_menu(names) -> tuple[str, ...] | None:
    """Normalize a menu to an ordered, de-duplicated tuple (None = full menu).

    Raises ValueError for an empty menu or any name not among the nine — the
    menu can only narrow the registry, never extend it.
    """
    if names is None:
        return None
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    if not seen:
        raise ValueError("model menu is empty; name at least one handle")
    unknown = [n for n in seen if n not in MENU]
    if unknown:
        raise ValueError(
            f"not on the model menu: {', '.join(unknown)}. "
            f"Valid handles: {', '.join(MENU)}"
        )
    return tuple(seen)


def parse_model_menu(spec: str | None) -> tuple[str, ...] | None:
    """Parse a --model-menu value ("GPT_5_4" or "GPT_5_4,GPT_5_4_MINI")."""
    if spec is None:
        return None
    return validate_menu([s.strip() for s in spec.split(",") if s.strip()])


def menu_from_env() -> tuple[str, ...] | None:
    raw = os.environ.get(MODEL_MENU_ENV, "")
    return parse_model_menu(raw) if raw.strip() else None


def registry_names_used(source: str) -> set[str]:
    """Handle-like names an agent takes from model_registry.

    Covers `from model_registry import X` and `model_registry.X` attribute
    access (including `import model_registry as mr; mr.X`). Raises
    SyntaxError for unparseable source.
    """
    tree = ast.parse(source)
    aliases = {"model_registry"}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "model_registry":
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "model_registry":
                    aliases.add(a.asname or a.name)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            names.add(node.attr)
    return {n for n in names if _HANDLE_NAME_RE.match(n)}


def off_menu_handles(source: str, menu) -> list[str]:
    """Registry handles the source uses that the menu does not offer."""
    return sorted(n for n in registry_names_used(source) if n not in menu)


def _replacement(handle: str, menu) -> str | None:
    """Same-family menu handle for an off-menu one, else None."""
    return next((m for m in menu if family(m) == family(handle)), None)


def _dedupe_import_line(text: str) -> str:
    def fix(m: re.Match) -> str:
        head, _, tail = m.group(0).partition(" import ")
        out: list[str] = []
        for n in (s.strip() for s in tail.split(",")):
            if n and n not in out:
                out.append(n)
        return f"{head} import {', '.join(out)}"
    return _IMPORT_LINE_RE.sub(fix, text)


def restrict_seed_source(source: str, menu) -> str:
    """Re-point a seed agent's off-menu handles at menu handles.

    Same family where the menu has one, else the menu's first handle; the
    import line is de-duplicated afterwards. Only names the source actually
    takes from model_registry are touched.
    """
    for h in off_menu_handles(source, menu):
        source = re.sub(rf"\b{h}\b", _replacement(h, menu) or menu[0], source)
    return _dedupe_import_line(source)


def _restrict_example(body: str, menu) -> str:
    """Rewrite one python code example to use only menu handles.

    The example is split into blank-line-separated stanzas. A stanza that
    calls an off-menu handle is re-pointed at a same-family menu handle; if
    the menu has none, it is re-pointed at the menu's first handle unless its
    comments describe family-specific behaviour, in which case it is dropped.
    Stanzas made identical by the rewrite are dropped, and the import line
    lists exactly the handles the surviving stanzas use.
    """
    trailing = body[len(body.rstrip("\n")):]
    stanzas = body.rstrip("\n").split("\n\n")
    kept: list[str] = []
    for st in stanzas:
        if _IMPORT_LINE_RE.search(st):
            kept.append(st)
            continue
        new = st
        drop = False
        for h in sorted(set(_HANDLE_RE.findall(st)) - set(menu)):
            rep = _replacement(h, menu)
            if rep is None:
                if _PROVIDER_WORDS_RE.search(st):
                    drop = True
                    break
                rep = menu[0]
            new = re.sub(rf"\b{h}\b", rep, new)
        if not drop and new not in kept:
            kept.append(new)
    used: list[str] = []
    for st in kept:
        if _IMPORT_LINE_RE.search(st):
            continue
        for h in _HANDLE_RE.findall(st):
            if h not in used:
                used.append(h)
    if not used:
        # Every call stanza was family-specific to an absent family. Keep the
        # first one re-pointed at the menu's first handle rather than ship an
        # example with no call in it.
        first = next(
            st for st in stanzas
            if not _IMPORT_LINE_RE.search(st) and _HANDLE_RE.search(st)
        )
        insert_at = next(
            (i + 1 for i, st in enumerate(kept) if _IMPORT_LINE_RE.search(st)), 0
        )
        kept.insert(insert_at, _HANDLE_RE.sub(menu[0], first))
        used = [menu[0]]
    # Keep the original import order for the handles that survive, so a menu
    # covering the whole example leaves it byte-identical; substituted
    # handles follow in order of use.
    original = [
        n.strip()
        for n in _IMPORT_LINE_RE.search(body).group(0).partition(" import ")[2].split(",")
    ]
    names = [n for n in original if n in used] + [n for n in used if n not in original]
    out = "\n\n".join(kept)
    out = _IMPORT_LINE_RE.sub(f"from model_registry import {', '.join(names)}", out)
    return out + trailing


def restrict_background(text: str, menu) -> str:
    """Narrow background.md's model section to the menu.

    Drops handle-table rows for off-menu handles and rewrites every python
    example that imports from model_registry (see _restrict_example). Only
    called when a menu is set: an unrestricted run's background.md is passed
    through untouched, byte for byte.
    """
    lines = [
        ln for ln in text.split("\n")
        if not ((m := _TABLE_ROW_RE.match(ln))
                and m.group(1) in MENU and m.group(1) not in menu)
    ]
    text = "\n".join(lines)

    def fix(m: re.Match) -> str:
        if not _IMPORT_LINE_RE.search(m.group(2)):
            return m.group(0)
        return m.group(1) + _restrict_example(m.group(2), menu) + m.group(3)

    return _FENCE_RE.sub(fix, text)


# ---------------------------------------------------------------------------
# Run-level resolution (main.py)
# ---------------------------------------------------------------------------

def _canonical(menu) -> tuple[str, ...] | None:
    """All nine handles IS the unrestricted menu; represent it as None."""
    if menu is None or set(menu) == set(MENU):
        return None
    return tuple(menu)


def stored_menu_value(menu) -> list[str]:
    """What a run persists in its checkpoint: always the explicit list (the
    nine for an unrestricted run), so a stored value is never mistaken for a
    missing one."""
    return list(menu) if menu else list(MENU)


def resolve_run_menu(cli_spec: str | None, store: dict, *, resume: bool,
                     eval_only: bool, key: str = "model_menu"):
    """Resolve a run's menu from --model-menu and the checkpoint's store.

    Fresh run: the flag, or None (all nine).
    Resume with a stored menu: the stored menu; a flag that disagrees exits —
      changing the menu mid-run would silently change what evolution may use.
    Resume without one (the run predates --model-menu, so it ran on all
      nine): a training resume keeps all nine and rejects a restricting flag;
      an --eval-only resume takes the flag (engines that persist no task
      config, GEPA and Autoresearch, rely on this).
    """
    try:
        cli = _canonical(parse_model_menu(cli_spec))
        if not resume:
            return cli
        if key in store:
            raw = store[key]
            stored = _canonical(validate_menu(raw) if raw else None)
            if cli_spec is not None and cli != stored:
                raise SystemExit(
                    f"--model-menu cannot be changed on --resume; the menu is "
                    f"locked for the lifetime of the run (stored: "
                    f"{', '.join(stored) if stored else 'all nine handles'})."
                )
            return stored
        if eval_only:
            return cli
        if cli is not None:
            raise SystemExit(
                "--model-menu cannot restrict a resumed run that predates the "
                "flag: it ran with all nine handles, and narrowing the menu "
                "mid-run would change what evolution may use. Start a new run."
            )
        return None
    except ValueError as e:
        raise SystemExit(f"--model-menu: {e}")


def _agent_sources(agent) -> dict[str, str]:
    """{relative path: source} for an agent given as a dir or an artifact dict."""
    if isinstance(agent, dict):
        return {k: v for k, v in agent.items() if k.endswith(".py")}
    return {
        str(p.relative_to(agent)): p.read_text()
        for p in sorted(agent.rglob("*.py")) if "__pycache__" not in p.parts
    }


def check_seed_agents(seed_agents: dict, menu) -> None:
    """Exit if any seed agent uses a handle the menu does not offer.

    Seeds from prior runs are kept verbatim (they are someone's evolved
    agent), so instead of rewriting one, refuse it: under the menu every one
    of its examples would score 0.
    """
    bad = []
    for name, agent in seed_agents.items():
        for rel, src in _agent_sources(agent).items():
            off = off_menu_handles(src, menu)
            if off:
                bad.append(f"{name}/{rel}: {', '.join(off)}")
    if bad:
        raise SystemExit(
            f"--model-menu {','.join(menu)}: seed agent(s) use handles not on "
            f"the menu:\n  " + "\n  ".join(bad)
        )
