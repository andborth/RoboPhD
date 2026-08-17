"""Every declared dependency must appear in an attribution file.

Third-party notices rot the moment a requirement changes: someone adds a
package, the table doesn't follow, and the file quietly starts describing a
dependency set the repo no longer has. This pins the one thing a test can
actually check — that each declared package is *named* somewhere in the
attribution file covering it.

Deliberately NOT checked: whether the license recorded for a package is
correct. Nothing here can verify that, and a test that appeared to would be
worse than none. Licenses are confirmed by hand against
``importlib.metadata`` when the tables are written.

Layout this enforces:
  requirements.txt, requirements-gepa.txt        -> NOTICE.md
  examples/<domain>/requirements.txt             -> examples/<domain>/THIRD_PARTY.md
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"

sys.path.insert(0, str(REPO_ROOT))


def _declared_packages(requirements: Path) -> list[str]:
    """Package names from a requirements file, ignoring comments and -r includes."""
    names = []
    for raw in requirements.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # Strip version specifiers, extras, environment markers.
        name = re.split(r"[<>=!~\[;]", line)[0].strip()
        if name:
            names.append(name)
    return names


def _example_dirs() -> list[Path]:
    return sorted(d for d in EXAMPLES.iterdir() if (d / "main.py").is_file())


# -- core ---------------------------------------------------------------


@pytest.mark.parametrize("requirements", ["requirements.txt", "requirements-gepa.txt"])
def test_core_packages_are_in_notice(requirements):
    notice = (REPO_ROOT / "NOTICE.md").read_text().lower()
    missing = [
        p for p in _declared_packages(REPO_ROOT / requirements)
        if p.lower() not in notice
    ]
    assert not missing, (
        f"{requirements} declares {missing}, absent from NOTICE.md. Add a row "
        f"to the dependency table (license confirmed by hand, not by this test)."
    )


def test_notice_does_not_list_packages_core_no_longer_declares():
    """The mirror failure: a package moved to an example but left in the core
    table, so NOTICE.md overstates what core pulls in. Checked only for names
    that once lived in core and now live in an example."""
    notice = (REPO_ROOT / "NOTICE.md").read_text()
    core = {p.lower() for f in ("requirements.txt", "requirements-gepa.txt")
            for p in _declared_packages(REPO_ROOT / f)}

    # Section the core dependency tables, stopping before per-example prose.
    core_section = notice.split("## Services and tools")[0]
    rows = re.findall(r"^\| `([A-Za-z0-9_.-]+)` \|", core_section, re.MULTILINE)

    stale = [r for r in rows if r.lower() not in core]
    assert not stale, (
        f"NOTICE.md's core tables list {stale}, which core no longer declares. "
        f"If they moved to an example, move the attribution too."
    )


# -- examples -----------------------------------------------------------


@pytest.mark.parametrize(
    "example", _example_dirs(), ids=lambda d: d.name
)
def test_every_example_has_an_attribution_file(example):
    """Including examples with no extra dependencies — 'this example needs
    nothing beyond core, and makes no LLM calls' is useful information, and
    its absence is indistinguishable from an oversight."""
    assert (example / "THIRD_PARTY.md").is_file(), (
        f"{example.name} has no THIRD_PARTY.md. Every example needs one, even "
        f"when it declares no packages of its own."
    )


@pytest.mark.parametrize(
    "example",
    [d for d in _example_dirs() if (d / "requirements.txt").is_file()],
    ids=lambda d: d.name,
)
def test_example_packages_are_in_its_attribution_file(example):
    attribution = (example / "THIRD_PARTY.md").read_text().lower()
    missing = [
        p for p in _declared_packages(example / "requirements.txt")
        if p.lower() not in attribution
    ]
    assert not missing, (
        f"{example.name}/requirements.txt declares {missing}, absent from its "
        f"THIRD_PARTY.md."
    )


def test_attribution_files_point_back_at_the_core_notice():
    """A reader who opens an example's file first must be able to find the
    core one; otherwise the split hides half the picture."""
    offenders = [
        d.name for d in _example_dirs()
        if "NOTICE.md" not in (d / "THIRD_PARTY.md").read_text()
    ]
    assert not offenders, (
        f"THIRD_PARTY.md in {offenders} does not reference the root NOTICE.md"
    )


# -- vendored code ------------------------------------------------------


def test_the_vendored_subtree_keeps_its_licenses():
    """The only third-party source in the repo. Its license texts are the one
    hard obligation here, so their presence is worth a test rather than a
    convention."""
    vendored = EXAMPLES / "cant_be_late" / "utils"
    assert (vendored / "README.md").is_file(), "vendored subtree lost its README"

    licenses = sorted(p.name for p in (vendored / "LICENSES").glob("LICENSE.*"))
    assert licenses == ["LICENSE.ADRS-Apache-2.0", "LICENSE.gepa-MIT"], (
        f"vendored subtree's license texts changed: {licenses}. Both layers of "
        f"the provenance chain must stay present — see the subtree README."
    )
