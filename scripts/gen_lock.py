#!/usr/bin/env python
"""Generate or verify a resolver-complete, hash-pinned dependency lock (`requirements.lock`).

    python scripts/gen_lock.py          # (re)compile the lock from pyproject + constraints.txt
    python scripts/gen_lock.py --check  # exit non-zero if the committed lock is stale/uncommitted

This produces a DETERMINISTIC lock, not an ambient environment snapshot (review P2 2026-09-05, which
correctly flagged that ``pip freeze`` is neither hash-pinned nor resolver-complete). It shells out
to a SINGLE PINNED resolver, ``uv==0.9.6`` (review P3 2026-09-06 — one pinned resolver, not "any
uv or pip-compile", so regeneration is deterministic across environments), to resolve the FULL
closure of the project's extras under ``constraints.txt`` and emit every package with hashes.
A consumer then gets a byte-for-byte install with:

    pip install --require-hashes -r requirements.lock

The lock is resolved ``--universal`` (platform markers, e.g. watchdog on non-Darwin). CI regenerates
it on a clean runner, fails if the committed file drifts, and then proves a FRESH venv installs from
the lock with ``--require-hashes`` followed by ``pip check`` — so "reproducible" is tested
end-to-end, not asserted. Regenerate (drop ``--check``) whenever ``constraints.txt`` or the
``pyproject`` dependency set changes, and commit the result.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed lock matches a fresh resolve (CI); non-zero on drift",
    )
    args = ap.parse_args()

    fresh = _resolve()
    if not args.check:
        _LOCK.write_text(fresh)
        print(f"Wrote {_LOCK}.")
        return 0

    # --check: compare the committed lock against a fresh resolve (both normalised).
    if not _LOCK.exists():
        print("requirements.lock is missing — run `python scripts/gen_lock.py` and commit it.")
        return 1
    if _LOCK.read_text() != fresh:
        print(
            "requirements.lock is OUT OF DATE vs a fresh resolve of pyproject + constraints.txt.\n"
            "Regenerate with `python scripts/gen_lock.py` and commit the result."
        )
        return 1
    print("requirements.lock is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
