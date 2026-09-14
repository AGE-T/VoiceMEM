#!/usr/bin/env python3
"""PowerShell 5.1 structural lint (pure stdlib, no PowerShell needed).

WHY THIS EXISTS (v0.7.1 field report):
    The v0.7.0 release shipped ``scripts/install_m1.ps1`` with ONE missing
    closing brace. The sandbox has no Windows PowerShell, so nothing parsed
    the script before shipping - the target machine's ``powershell -File``
    rejected it with ``ParserError: Missing closing '}'`` (line 612) and the
    whole one-click bootstrap failed on the first run. This module is the
    permanent guard: it tokenizes PowerShell source well enough to catch
    brace/paren/bracket imbalance, unterminated strings, malformed
    here-strings, and PowerShell-7-only operators - the exact classes of
    defect a Linux sandbox cannot otherwise catch.

USED BY (both sides of the release gate):
    * ``tests/validation/test_feature_scripts.py`` - every repo *.ps1 must
      lint clean BEFORE a build is allowed;
    * ``scripts/build_release_sandbox.py`` - every *.ps1 inside the finished
      ZIP must lint clean (self-check; a failure deletes the zip).

Scope/limits (deliberate):
    It is NOT a full PowerShell parser. It validates STRUCTURE only:
    balanced {} [] () in code, closed strings, well-formed here-strings,
    no PS7-only code operators. Semantic correctness stays with the tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

#: Operators that PowerShell 5.1 cannot parse (PowerShell 7+ only).
PS7_ONLY_OPERATORS = ("&&", "||", "??", "?.")


def check_ps_source(text: str) -> List[str]:
    """Return a list of structural problems in PowerShell ``text``.

    An empty list means: braces/parens/brackets balanced, every string
    closed, every here-string well-formed, and no PS7-only operator in
    code (comments excluded).
    """
    problems: List[str] = []
    stack: List[tuple] = []  # (char, lineno)
    here_state = None  # None | "single" | "double"
    lineno = 1
    i = 0
    n = len(text)

    def line_end(pos: int) -> int:
        j = text.find("\n", pos)
        return n if j == -1 else j

    while i < n:
        c = text[i]
        if c == "\r":
            i += 1
            continue
        if c == "\n":
            lineno += 1
            i += 1
            continue
        line_start = text.rfind("\n", 0, i) + 1

        # ---- here-string terminator: '@ or "@ at column 0 ----
        if here_state is not None and i == line_start and c in "'\"":
            want = "'" if here_state == "single" else '"'
            if c == want and i + 1 < n and text[i + 1] == "@":
                here_state = None
                i += 2
                continue
        # ---- inside a here-string: skip the whole line ----
        if here_state is not None:
            i = line_end(i)
            continue

        # ---- block comment <# ... #> ----
        if c == "<" and i + 1 < n and text[i + 1] == "#":
            j = text.find("#>", i + 2)
            j = n if j == -1 else j + 2
            lineno += text.count("\n", i, j)
            i = j
            continue
        # ---- line comment ----
        if c == "#":
            i = line_end(i)
            continue
        # ---- here-string opener: @' / @" then only whitespace till EOL ----
        if c == "@" and i + 1 < n and text[i + 1] in "'\"":
            rest = text[i + 2:line_end(i)]
            if rest.strip() == "":
                here_state = "single" if text[i + 1] == "'" else "double"
                i = line_end(i)
                continue
        # ---- single-quoted string ('' is an escaped quote, STAYS in-string) ----
        if c == "'":
            j = i + 1
            closed = False
            while j < n:
                if text[j] == "\n":
                    break
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2  # '' escape: still inside the string
                        continue
                    closed = True
                    break
                j += 1
            if not closed:
                problems.append(f"line {lineno}: unterminated single-quoted string")
                i = line_end(i)
                continue
            i = j + 1
            continue
        # ---- double-quoted string ----
        if c == '"':
            j = i + 1
            closed = False
            while j < n:
                if text[j] == "`":
                    j += 2
                    continue
                if text[j] == '"':
                    closed = True
                    break
                if text[j] == "\n":
                    break
                j += 1
            if not closed:
                problems.append(f"line {lineno}: unterminated double-quoted string")
                i = line_end(i)
                continue
            i = j + 1
            continue
        # ---- nesting ----
        if c in "{[(":
            stack.append((c, lineno))
            i += 1
            continue
        if c in "}])":
            pairs = {"}": "{", "]": "[", ")": "("}
            if not stack:
                problems.append(f"line {lineno}: unmatched closer {c!r}")
            else:
                opener, oline = stack.pop()
                if opener != pairs[c]:
                    problems.append(
                        f"line {lineno}: mismatched {opener!r} "
                        f"(opened line {oline}) closed by {c!r}"
                    )
            i += 1
            continue
        # ---- PS7-only operators ----
        pair = text[i:i + 2]
        if pair in PS7_ONLY_OPERATORS:
            problems.append(f"line {lineno}: PS7-only operator {pair!r}")
            i += 2
            continue
        i += 1

    for opener, oline in stack:
        problems.append(
            f"line {oline}: {opener!r} opened here is never closed "
            f"(PowerShell ParserError: Missing closing brace/paren)"
        )
    if here_state is not None:
        problems.append("unterminated here-string (missing terminator '@ or \"@)")
    return problems


def check_ps_file(path: Path) -> List[str]:
    """Convenience wrapper: lint a .ps1 file on disk."""
    return check_ps_source(
        Path(path).read_text(encoding="utf-8-sig", errors="replace")
    )


if __name__ == "__main__":
    import sys

    failed = 0
    for arg in sys.argv[1:]:
        probs = check_ps_file(Path(arg))
        if probs:
            failed = 1
            print(f"== {arg}")
            for p in probs:
                print(f"   {p}")
        else:
            print(f"== {arg}: OK")
    sys.exit(failed)
