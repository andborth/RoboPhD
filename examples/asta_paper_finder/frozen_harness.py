#!/usr/bin/env python3
"""Config-only evolution: freeze an agent's code, evolve only its settings.

`main.py --frozen-harness AGENT_DIR` takes an existing agent (typically a
prior run's winner) and makes its source immutable for the run. The evolvable
artifact is no longer `agent.py` but a JSON file, `harness_config.json`,
holding exactly the settings this module finds at the harness's module level:

  models     every `NAME = <registry handle>` and `NAME = (<handle>, ...)`
             assignment, each with an optional reasoning effort
  constants  every `NAME = <number or bool>` assignment (a constant
             arithmetic expression such as `21 * 60` counts as a number)

Strings (prompts, field lists), every literal inside a function body, and all
code are frozen. The split is derived from the source by `ast`, never chosen
by hand, so the same harness always yields the same config schema.

The restriction is enforced rather than described:

  1. A candidate holds no code. `parse_config` accepts only a JSON document
     with the harness's own key set and each value's own type; anything else
     scores 0 on every example with the reason in `error.md`.
  2. The evaluator never runs candidate-supplied source. `Harness.render`
     writes each config value over the right-hand side of its assignment in
     the frozen source and hands the result to the ordinary evaluator.
  3. Rendering is checked by round trip: re-deriving the config from the
     rendered source must give back the config that was rendered.

Rendering the harness's own config returns the harness source byte for byte,
so the run's seed IS the frozen agent, not an approximation of it.

A slot's reasoning effort is rendered as `with_effort(HANDLE, "high")`, a
`model_registry` function returning the same model with a different default
effort; the harness code that calls the slot is untouched.

Stdlib-only on purpose: main.py ships this file to evolution sessions as a
session tool (`validate` / `render` below), where it runs without the
evaluator's dependencies.
"""

import ast
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

from model_menu import MENU, family

CONFIG_FILENAME = "harness_config.json"
AGENT_FILENAME = "agent.py"
# The frozen source as evolution sessions see it, next to this file.
HARNESS_SESSION_FILENAME = "harness_agent.py"
RUNTIME_KEY = "frozen_harness"
EFFORT_FN = "with_effort"

# Per-call reasoning options by provider family: background.md's handle
# table, "Reasoning options" column. A unit test pins the two together.
EFFORTS = {
    "GPT": ("low", "medium", "high"),
    "CLAUDE": ("low", "medium", "high"),
    "GEMINI": ("low", "high"),
}


class HarnessError(ValueError):
    """The source cannot serve as a frozen harness."""


class ConfigError(ValueError):
    """A candidate config is not valid for the harness."""


# ---------------------------------------------------------------------------
# Reading settings out of source
# ---------------------------------------------------------------------------

def _number(node: ast.expr):
    """Value of a constant numeric expression, else None.

    Literals, unary minus/plus and + - * / // over those — enough for
    `21 * 60`. bool is kept distinct from int throughout.
    """
    if isinstance(node, ast.Constant):
        v = node.value
        return v if isinstance(v, (bool, int, float)) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _number(node.operand)
        if v is None or isinstance(v, bool):
            return None
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _number(node.left), _number(node.right)
        if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
            return None
        try:
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.FloorDiv):
                return a // b
        except ZeroDivisionError:
            return None
    return None


def _slot(node: ast.expr):
    """{"model", "effort"} for `HANDLE` or `with_effort(HANDLE, "x")`, else None."""
    if isinstance(node, ast.Name) and node.id in MENU:
        return {"model": node.id, "effort": None}
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == EFFORT_FN
            and len(node.args) == 2 and not node.keywords
            and isinstance(node.args[0], ast.Name) and node.args[0].id in MENU
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)):
        return {"model": node.args[0].id, "effort": node.args[1].value}
    return None


def _classify(node: ast.expr):
    """("model" | "models" | "constant", value) for an evolvable right-hand
    side, else None."""
    slot = _slot(node)
    if slot is not None:
        return "model", slot
    if isinstance(node, ast.Tuple):
        slots = [_slot(e) for e in node.elts]
        # An empty tuple is a model list only where the config says so; in
        # raw source it is indistinguishable from any other empty tuple.
        if slots and all(s is not None for s in slots):
            return "models", slots
        return None
    num = _number(node)
    if num is not None:
        return "constant", num
    return None


def _same(a, b) -> bool:
    """Equal in value AND type (True != 1, 1 != 1.0)."""
    return type(a) is type(b) and a == b


class Harness:
    """A frozen agent source and the settings that may be re-rendered in it."""

    def __init__(self, source: str):
        self.source = source
        self.sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise HarnessError(f"harness source does not parse: {e}") from e

        self._nodes: dict[str, ast.expr] = {}
        self._kinds: dict[str, str] = {}
        self._values: dict[str, object] = {}
        seen: set[str] = set()
        duplicated: set[str] = set()
        for stmt in tree.body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                continue
            name = stmt.targets[0].id
            if name in seen:
                duplicated.add(name)
            seen.add(name)
            found = _classify(stmt.value)
            if found is None:
                continue
            self._kinds[name], self._values[name] = found
            self._nodes[name] = stmt.value
        # A name bound twice at module level has no single right-hand side to
        # re-render; freeze it rather than guess which binding is live.
        for name in duplicated:
            for d in (self._kinds, self._values, self._nodes):
                d.pop(name, None)

        imports = [
            s for s in tree.body
            if isinstance(s, ast.ImportFrom) and s.module == "model_registry"
        ]
        if len(imports) != 1:
            raise HarnessError(
                f"expected exactly one module-level `from model_registry "
                f"import ...` in the harness, found {len(imports)}"
            )
        self._import = imports[0]
        if any(a.asname for a in self._import.names):
            raise HarnessError(
                "the harness imports a model_registry name under an alias; "
                "its model slots cannot be told apart from other names"
            )
        self._imported = [a.name for a in self._import.names]

        slot_nodes = {id(n) for v in self._nodes.values() for n in ast.walk(v)}
        # Handles the frozen code names directly (outside any slot) must stay
        # imported whatever the config says.
        self._pinned = {
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id in MENU and id(n) not in slot_nodes
        }
        if not any(k != "constant" for k in self._kinds.values()):
            raise HarnessError("the harness has no module-level model slot")
        if EFFORT_FN not in self._imported and any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == EFFORT_FN
            or isinstance(n, ast.Name) and n.id == EFFORT_FN
            for n in ast.walk(tree)
        ):
            raise HarnessError(
                f"the harness defines or uses its own `{EFFORT_FN}`, which "
                f"would shadow the registry function slot efforts render to"
            )

    # -- the config this harness defines ------------------------------------

    def config(self) -> dict:
        """The harness's own settings, in source order."""
        return {
            "models": {n: self._values[n] for n, k in self._kinds.items()
                       if k != "constant"},
            "constants": {n: self._values[n] for n, k in self._kinds.items()
                          if k == "constant"},
        }

    def seed_config_text(self) -> str:
        return json.dumps(self.config(), indent=2) + "\n"

    # -- validation ----------------------------------------------------------

    def parse_config(self, text: str, menu=None) -> dict:
        """Parse and validate a candidate config; raise ConfigError listing
        every problem found. Returns the config in canonical form (harness
        key order, float-typed constants as floats)."""
        allowed = tuple(menu) if menu else MENU

        def no_dupes(pairs):
            keys = [k for k, _ in pairs]
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            if dupes:
                raise ConfigError(f"duplicate key(s): {', '.join(dupes)}")
            return dict(pairs)

        def no_constants(token):
            raise ConfigError(f"{token} is not a valid constant value")

        try:
            raw = json.loads(text, object_pairs_hook=no_dupes,
                             parse_constant=no_constants)
        except json.JSONDecodeError as e:
            raise ConfigError(f"not valid JSON: {e}") from e
        if not isinstance(raw, dict) or set(raw) != {"models", "constants"}:
            raise ConfigError(
                'the config must be a JSON object with exactly the keys '
                '"models" and "constants"'
            )
        problems: list[str] = []
        own = self.config()

        def check_keys(section: str):
            got = raw[section]
            if not isinstance(got, dict):
                problems.append(f'"{section}" must be an object')
                return {}
            missing = [k for k in own[section] if k not in got]
            extra = [k for k in got if k not in own[section]]
            if missing:
                problems.append(f'{section}: missing {", ".join(missing)}')
            if extra:
                problems.append(
                    f'{section}: {", ".join(extra)} not a setting of this '
                    f'harness (settings cannot be added)'
                )
            return got

        def check_slot(where: str, slot):
            if not isinstance(slot, dict) or set(slot) != {"model", "effort"}:
                problems.append(
                    f'{where}: expected {{"model": ..., "effort": ...}}'
                )
                return None
            model, effort = slot["model"], slot["effort"]
            if not isinstance(model, str) or model not in allowed:
                problems.append(
                    f'{where}: model {model!r} is not on the menu '
                    f'({", ".join(allowed)})'
                )
                return None
            options = EFFORTS[family(model)]
            if effort is not None and effort not in options:
                problems.append(
                    f'{where}: effort {effort!r} is not available on {model} '
                    f'(null for its default, or one of: {", ".join(options)})'
                )
                return None
            return {"model": model, "effort": effort}

        out = {"models": {}, "constants": {}}
        got_models = check_keys("models")
        for name, ours in own["models"].items():
            if name not in got_models:
                continue
            value = got_models[name]
            if self._kinds[name] == "model":
                out["models"][name] = check_slot(name, value)
            elif not isinstance(value, list):
                problems.append(f"{name}: expected a list of model entries")
            else:
                out["models"][name] = [
                    check_slot(f"{name}[{i}]", s) for i, s in enumerate(value)
                ]
        got_constants = check_keys("constants")
        for name, ours in own["constants"].items():
            if name not in got_constants:
                continue
            value = got_constants[name]
            if isinstance(ours, bool):
                ok = isinstance(value, bool)
                want = "true or false"
            elif isinstance(ours, int):
                ok = isinstance(value, int) and not isinstance(value, bool)
                want = "an integer"
            else:
                ok = (isinstance(value, (int, float))
                      and not isinstance(value, bool) and math.isfinite(value))
                want = "a number"
                if ok:
                    value = float(value)
            if not ok:
                problems.append(f"{name}: expected {want}, got {value!r}")
            else:
                out["constants"][name] = value
        if problems:
            raise ConfigError("\n".join(problems))
        return out

    # -- rendering -----------------------------------------------------------

    @staticmethod
    def _slot_src(slot: dict) -> str:
        if slot["effort"] is None:
            return slot["model"]
        return f'{EFFORT_FN}({slot["model"]}, {json.dumps(slot["effort"])})'

    def _value_src(self, name: str, value) -> str:
        kind = self._kinds[name]
        if kind == "model":
            return self._slot_src(value)
        if kind == "models":
            parts = [self._slot_src(s) for s in value]
            if len(parts) == 1:
                return f"({parts[0]},)"
            return "(" + ", ".join(parts) + ")"
        return repr(value)

    def render(self, config: dict) -> str:
        """The harness source with `config` written into it.

        `config` must come from `parse_config` (or `config()`). A setting
        left at the harness's own value is not rewritten, so rendering the
        harness's own config returns its source unchanged.
        """
        values = {**config["models"], **config["constants"]}
        edits = []  # (node, replacement source)
        for name, value in values.items():
            ours = self._values[name]
            unchanged = (_same(ours, value) if self._kinds[name] == "constant"
                         else ours == value)
            if not unchanged:
                edits.append((self._nodes[name], self._value_src(name, value)))

        used = set(self._pinned)
        needs_effort = False
        for value in config["models"].values():
            for slot in (value if isinstance(value, list) else [value]):
                used.add(slot["model"])
                needs_effort = needs_effort or slot["effort"] is not None
        # Registry names that are not handles (helpers the harness imported)
        # are kept as they are; the handle list is exactly what is used.
        names = [n for n in self._imported if n not in MENU or n in used]
        names += [h for h in MENU if h in used and h not in names]
        if needs_effort and EFFORT_FN not in names:
            names.append(EFFORT_FN)
        if set(names) != set(self._imported):
            edits.append(
                (self._import, f"from model_registry import {', '.join(names)}")
            )

        out = self._splice(edits)
        # Round trip: what we rendered must read back as what was asked for.
        # Compared as JSON text so that 1 / 1.0 / True stay distinct.
        if (json.dumps(self._read_back(out), sort_keys=True)
                != json.dumps(values, sort_keys=True)):
            raise HarnessError(
                "rendered source does not read back as the config that was "
                "rendered; refusing to evaluate it"
            )
        return out

    def _read_back(self, rendered: str) -> dict:
        """This harness's settings as they stand in `rendered`.

        Read with this harness's own kinds, because an emptied model list
        renders as `()`, which raw source cannot tell from any other tuple.
        """
        found = {}
        for stmt in ast.parse(rendered).body:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id in self._kinds):
                continue
            name = stmt.targets[0].id
            if (self._kinds[name] == "models"
                    and isinstance(stmt.value, ast.Tuple) and not stmt.value.elts):
                found[name] = []
                continue
            got = _classify(stmt.value)
            found[name] = got[1] if got and got[0] == self._kinds[name] else None
        return found

    def _splice(self, edits) -> str:
        if not edits:
            return self.source
        lines = self.source.split("\n")
        # ast column offsets count UTF-8 bytes; work on encoded lines.
        raw = [ln.encode("utf-8") for ln in lines]
        for node, text in sorted(
            edits, key=lambda e: (e[0].lineno, e[0].col_offset), reverse=True
        ):
            first, last = node.lineno - 1, node.end_lineno - 1
            merged = (raw[first][:node.col_offset] + text.encode("utf-8")
                      + raw[last][node.end_col_offset:])
            raw[first:last + 1] = [merged]
        return "\n".join(ln.decode("utf-8") for ln in raw)


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------

class FrozenHarnessEvaluator:
    """Scores a `{harness_config.json: ...}` candidate by rendering it into
    the frozen harness and delegating to the ordinary evaluator.

    Everything the wrapped evaluator exposes (`aggregate`, cost totals, ...)
    is reachable through this object unchanged.
    """

    def __init__(self, inner, harness: Harness, menu=None):
        self._inner = inner
        self._harness = harness
        self._menu = tuple(menu) if menu else None
        self._render = lru_cache(maxsize=64)(self._render_uncached)

    def _render_uncached(self, text: str) -> str:
        return self._harness.render(self._harness.parse_config(text, self._menu))

    def __call__(self, candidate: dict, example, *, problem_dir=None):
        text = candidate.get(CONFIG_FILENAME, "")
        if not text:
            return 0.0, {"error.md": f"candidate missing {CONFIG_FILENAME}"}
        try:
            agent = self._render(text)
        except ConfigError as e:
            return 0.0, {
                "error.md": f"{CONFIG_FILENAME} is not a valid config for the "
                            f"frozen harness; the agent was not run.\n\n{e}"
            }
        return self._inner({AGENT_FILENAME: agent}, example, problem_dir=problem_dir)

    def with_overrides(self, **overrides):
        return FrozenHarnessEvaluator(
            self._inner.with_overrides(**overrides), self._harness, self._menu
        )

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Run-level resolution (main.py)
# ---------------------------------------------------------------------------

def load_harness(agent_dir) -> Harness:
    """Harness from an agent directory (or a path to the source file)."""
    path = Path(agent_dir)
    if path.is_dir():
        path = path / AGENT_FILENAME
    if not path.is_file():
        raise SystemExit(f"--frozen-harness: no agent source at {path}")
    try:
        return Harness(path.read_text())
    except HarnessError as e:
        raise SystemExit(f"--frozen-harness {agent_dir}: {e}")


def resolve_run_harness(cli_path, store: dict, *, resume_dir=None):
    """Resolve a run's frozen harness from --frozen-harness and the checkpoint.

    Returns (harness, stored_value) or (None, None).

    Fresh run: the flag's agent, or none.
    Resume of a frozen run: the run's own copy of the harness (shipped to its
      session_tools at start), falling back to the stored path; either must
      hash to the stored sha256. A flag naming a different harness exits —
      the frozen code is the run's identity.
    Resume of an ordinary run: a flag exits; freezing mid-run would change
      what every existing agent in the pool is.
    """
    stored = store.get(RUNTIME_KEY) if store else None
    if resume_dir is None:
        if cli_path is None:
            return None, None
        harness = load_harness(cli_path)
        return harness, {"path": str(cli_path), "sha256": harness.sha256}
    if not stored:
        if cli_path is not None:
            raise SystemExit(
                "--frozen-harness cannot be added on --resume: the run was "
                "started as ordinary agent evolution. Start a new run."
            )
        return None, None
    if cli_path is not None and load_harness(cli_path).sha256 != stored["sha256"]:
        raise SystemExit(
            f"--frozen-harness cannot be changed on --resume; the run is "
            f"frozen on {stored['path']} (sha256 {stored['sha256'][:12]})."
        )
    own_copy = Path(resume_dir) / "session_tools" / HARNESS_SESSION_FILENAME
    for candidate in (own_copy, Path(stored["path"]), cli_path):
        if candidate is None:
            continue
        candidate = Path(candidate)
        if candidate.is_dir():
            candidate = candidate / AGENT_FILENAME
        if candidate.is_file():
            harness = Harness(candidate.read_text())
            if harness.sha256 == stored["sha256"]:
                return harness, stored
    raise SystemExit(
        f"Cannot resume: the run's frozen harness (sha256 "
        f"{stored['sha256'][:12]}) was found neither at {own_copy} nor at "
        f"{stored['path']}. Pass --frozen-harness pointing at the original "
        f"agent."
    )


def materialize_session_files(harness: Harness, cache_root) -> list[str]:
    """Write the harness under a content-addressed directory and return the
    session_tools entries for a frozen run: the frozen source and this
    module (whose CLI validates and renders configs), plus the menu module
    this one imports."""
    out = Path(cache_root) / harness.sha256[:16]
    out.mkdir(parents=True, exist_ok=True)
    target = out / HARNESS_SESSION_FILENAME
    if not target.is_file() or target.read_text() != harness.source:
        target.write_text(harness.source)
    here = Path(__file__).resolve().parent
    return [str(target), str(here / "frozen_harness.py"), str(here / "model_menu.py")]


# ---------------------------------------------------------------------------
# CLI (session tool, and the way to turn a winning config into an agent.py)
# ---------------------------------------------------------------------------

def _cli(argv) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Validate a harness config, or render it into the frozen "
                    "harness to see the agent it produces."
    )
    p.add_argument("command", choices=["validate", "render", "seed"])
    p.add_argument("config", nargs="?",
                   help=f"path to a {CONFIG_FILENAME} (not needed for `seed`)")
    p.add_argument("--harness", default=None,
                   help="agent directory or source file; default: "
                        f"{HARNESS_SESSION_FILENAME} next to this script")
    p.add_argument("--menu", default=None,
                   help="comma-separated handles the run's menu is narrowed to")
    args = p.parse_args(argv)

    source = (Path(args.harness) if args.harness
              else Path(__file__).resolve().parent / HARNESS_SESSION_FILENAME)
    if source.is_dir():
        source = source / AGENT_FILENAME
    harness = Harness(source.read_text())
    if args.command == "seed":
        sys.stdout.write(harness.seed_config_text())
        return 0
    if not args.config:
        p.error(f"{args.command} needs a config path")
    menu = [s.strip() for s in args.menu.split(",") if s.strip()] if args.menu else None
    try:
        config = harness.parse_config(Path(args.config).read_text(), menu)
    except ConfigError as e:
        print(f"INVALID\n{e}", file=sys.stderr)
        return 1
    if args.command == "validate":
        print("OK")
    else:
        sys.stdout.write(harness.render(config))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
