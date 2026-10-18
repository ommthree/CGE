#!/usr/bin/env python
"""Generate or verify a resolver-complete, hash-pinned dependency lock (`requirements.lock`).

    python scripts/gen_lock.py          # (re)compile the lock from pyproject + constraints.txt
    python scripts/gen_lock.py --verify # DETERMINISTIC CI gate (no re-resolve; see below)
    python scripts/gen_lock.py --check  # DEVELOPER: re-resolve & compare before a deliberate bump

This produces a DETERMINISTIC lock, not an ambient environment snapshot (review P2 2026-09-05, which
correctly flagged that ``pip freeze`` is neither hash-pinned nor resolver-complete). It shells out
to a SINGLE PINNED resolver, ``uv==0.9.6`` (review P3 2026-09-06 — one pinned resolver, not "any
uv or pip-compile", so regeneration is deterministic across environments), to resolve the FULL
closure of the project's extras under ``constraints.txt`` and emit every package with hashes.
A consumer then gets a byte-for-byte install with:

    pip install --require-hashes -r requirements.lock

The lock is resolved ``--universal`` (platform markers, e.g. watchdog on non-Darwin).

The committed lock is the FROZEN resolution (review-14 P2#5). CI gates on ``--verify``, a
DETERMINISTIC check that the lock exists, is hash-pinned, and covers every DIRECT dependency
declared in pyproject + extras — it does NOT re-resolve transitives against today's package index
(that re-resolve, ``--check``, drifts on every run as new transitives publish, which forced a string
of "transitive drift" commits). CI then APPLICATION-tests the exact lock: ``pip install
--require-hashes`` + ``pip check``, install the project ``--no-deps``, and run the full suite +
validation IN the locked venv. Regenerate DELIBERATELY with the bare command (and verify with
``--check``) whenever ``constraints.txt`` or the ``pyproject`` dependency set changes; commit the
result. Transitive-only refreshes are a deliberate maintenance choice, not an every-CI-run event.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_LOCK = _ROOT / "requirements.lock"
# Relative to the repo root (we run the resolver from there) so the generated header is portable:
# an absolute path would embed the machine's home dir and make the lock differ between dev and CI.
_CONSTRAINTS = "constraints.txt"
# The full install target the deployment/CI uses: all extras.
_EXTRAS = "cge,dev,data,gui"

# A stable, path-free header we substitute for the resolver's auto-generated one (which embeds the
# absolute invocation, so it would differ between machines and defeat --check).
_HEADER = "\n".join(
    [
        "# Resolver-complete, hash-pinned dependency lock. Regenerate with",
        "# `python scripts/gen_lock.py` (uv or pip-tools) after bumping constraints.txt or the",
        "# pyproject dependency set. Byte-for-byte install:",
        "#     pip install --require-hashes -r requirements.lock",
        f"# Resolved for the project's extras [{_EXTRAS}] under constraints.txt (review P2).",
        "",
        "",
    ]
)


def _normalise(text: str) -> str:
    """Drop the resolver's volatile auto-generated header (it embeds the absolute invocation path)
    and prepend a stable, path-free header, so the lock is identical on any machine given the same
    inputs — a prerequisite for the --check drift gate to be meaningful."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    return _HEADER + body.lstrip("\n") + "\n"


# The ONE pinned resolver used to (re)generate the lock (review P3 2026-09-06). Requiring a single
# pinned uv — not "any uv or pip-compile" — makes regeneration deterministic across environments: uv
# and pip-tools do not produce identical output, so allowing either would let the committed lock
# depend on which tool a developer happened to have. CI installs exactly this version.
_UV_VERSION = "0.9.6"


def _resolve() -> str:
    """Resolve the full hash-pinned closure and return the NORMALISED lock text, using the single
    pinned uv resolver (``uv==0.9.6``). Runs from the repo root so the constraint path is relative
    and the output is machine-independent."""
    if not shutil.which("uv"):
        raise SystemExit(
            f"`uv` is not available — install the pinned resolver (`pip install uv=={_UV_VERSION}`)"
            " to (re)generate requirements.lock. Regeneration is pinned to one resolver so the "
            "lock is deterministic across environments (review P3 2026-09-06)."
        )
    installed = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
    if _UV_VERSION not in installed.stdout:
        print(
            f"WARNING: uv version is {installed.stdout.strip()!r}, not the pinned {_UV_VERSION}; "
            "the generated lock may differ from CI's. Install the pinned version for a match."
        )
    extras: list[str] = []
    for e in _EXTRAS.split(","):
        extras += ["--extra", e]
    out = _LOCK.with_suffix(".lock.tmp")
    cmd = [
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        *extras,
        "--constraint",
        _CONSTRAINTS,
        "--generate-hashes",
        "--universal",  # broadly-installable lock (markers for platform-conditional deps)
        "--output-file",
        out.name,
    ]
    try:
        subprocess.run(cmd, cwd=_ROOT, check=True)
        return _normalise(out.read_text())
    finally:
        out.unlink(missing_ok=True)


_NAME_RE = __import__("re").compile(r"^\s*([A-Za-z0-9._-]+)")


def _canon(name: str) -> str:
    # PEP 503 canonical form: lowercase, runs of -/_/. collapsed to a single hyphen.
    return __import__("re").sub(r"[-_.]+", "-", name.strip().lower())


def _declared_top_level() -> set[str]:
    """The project's DIRECT dependencies (base + the locked extras) from pyproject — the names whose
    presence in the lock we verify deterministically. Transitives are NOT enumerated here."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib

    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    proj = data.get("project", {})
    reqs: list[str] = list(proj.get("dependencies", []))
    optional = proj.get("optional-dependencies", {})
    for extra in _EXTRAS.split(","):
        reqs += optional.get(extra, [])
    names: set[str] = set()
    for r in reqs:
        # Strip any environment marker / extras / version spec; keep the distribution name.
        head = r.split(";", 1)[0].split("[", 1)[0]
        m = _NAME_RE.match(head)
        if m:
            names.add(_canon(m.group(1)))
    return names


def _locked_names() -> set[str]:
    names: set[str] = set()
    for ln in _LOCK.read_text().splitlines():
        if ln.startswith("#") or not ln.strip() or ln.lstrip().startswith(("--", "    ")):
            continue
        m = _NAME_RE.match(ln)
        if m and "==" in ln:
            names.add(_canon(m.group(1)))
    return names


def _verify_frozen() -> int:
    """DETERMINISTIC check (review-14 P2#5): the committed lock is the FROZEN resolution, so verify
    it WITHOUT re-resolving against today's package index (which drifts every run as new transitives
    publish — the six prior "transitive drift" commits). We assert the lock exists, is hash-pinned,
    and covers every DIRECT dependency declared in pyproject + the locked extras; a missing one
    means a real input change that needs a deliberate regenerate. The full installability proof is
    CI's
    `pip install --require-hashes` + `pip check` + running the suite IN the locked venv."""
    if not _LOCK.exists():
        print("requirements.lock is missing — run `python scripts/gen_lock.py` and commit it.")
        return 1
    text = _LOCK.read_text()
    if "--hash=" not in text:
        print("requirements.lock is not hash-pinned — regenerate with `scripts/gen_lock.py`.")
        return 1
    locked = _locked_names()
    missing = sorted(_declared_top_level() - locked)
    if missing:
        print(
            "requirements.lock is MISSING declared direct dependenc(ies) "
            f"{missing} — a pyproject/extras input changed. Deliberately regenerate with "
            "`python scripts/gen_lock.py` and commit."
        )
        return 1
    print(f"requirements.lock is frozen & complete ({len(locked)} pinned, hash-enforced).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="re-resolve and compare (DEVELOPER use before a deliberate bump); non-zero on drift",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="DETERMINISTIC CI gate: lock exists, hash-pinned, covers declared direct deps (no "
        "re-resolve against today's index)",
    )
    args = ap.parse_args()

    if args.verify:
        return _verify_frozen()

    fresh = _resolve()
    if not args.check:
        _LOCK.write_text(fresh)
        print(f"Wrote {_LOCK}.")
        return 0

    # --check: compare the committed lock against a fresh resolve (both normalised). This
    # RE-RESOLVES against the current index, so it drifts as new transitives publish — it is a
    # DEVELOPER tool for a deliberate refresh, NOT the CI gate (see --verify, review-14 P2#5).
    if not _LOCK.exists():
        print("requirements.lock is missing — run `python scripts/gen_lock.py` and commit it.")
        return 1
    if _LOCK.read_text() != fresh:
        print(
            "requirements.lock differs from a fresh resolve (newer transitives available).\n"
            "If pyproject/constraints changed, regenerate with `python scripts/gen_lock.py`."
        )
        return 1
    print("requirements.lock matches a fresh resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
