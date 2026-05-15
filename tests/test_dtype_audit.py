"""Source-audit regression test for dangerous dtype patterns.

Scans every ``.py`` file under the ``pgmuvi`` package for patterns that
violate the dtype policy documented in ``docs/development/dtype_policy.md``:

* ``torch.Tensor(``          — uses global default dtype (should use
                               ``torch.as_tensor`` with explicit dtype/device)
* ``dtype=torch.float32``    — hard-codes float32
* ``.float()``               — silent cast to float32

For each (file, pattern) combination the test asserts that the number of
occurrences does **not exceed** the count recorded in ``_ALLOWLIST``.  Any
(file, pattern) pair absent from the allowlist may have **zero** occurrences.

How to reduce the allowlist over time
--------------------------------------
When a violation in a file is fixed, decrement (or remove) the corresponding
entry in ``_ALLOWLIST``.  The test will then enforce the lower limit going
forward, preventing regressions.

Files excluded from the audit
-------------------------------
``test_script.py`` is excluded because it is not production code and is
already excluded from the Ruff linter.
"""

import pathlib
import re
import unittest

# ---------------------------------------------------------------------------
# Patterns to audit
# ---------------------------------------------------------------------------

_PATTERNS = [
    "torch.Tensor(",
    "dtype=torch.float32",
    ".float()",
]

# ---------------------------------------------------------------------------
# Known existing violations
# (filename without directory, pattern) -> maximum allowed count
#
# Reduce these numbers in later PRs as violations are fixed.
# ---------------------------------------------------------------------------

_ALLOWLIST: dict[tuple[str, str], int] = {
    # --- lightcurve.py ---
    # torch.Tensor(: 5 occurrences (lines 2296, 2297, 2302, 9982, 10499)
    ("lightcurve.py", "torch.Tensor("): 5,
    # dtype=torch.float32: 5 occurrences (1 in docstring + 4 in code)
    ("lightcurve.py", "dtype=torch.float32"): 5,
    # .float(): 3 occurrences in _acf_data return statement
    ("lightcurve.py", ".float()"): 3,
    # --- synthetic.py ---
    # dtype=torch.float32: 12 occurrences across all four generators
    ("synthetic.py", "dtype=torch.float32"): 12,
}

# Files to skip entirely (not production code)
_EXCLUDED_FILES = {"test_script.py"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _count_pattern(text: str, pattern: str) -> int:
    """Return the number of non-overlapping occurrences of *pattern* in *text*.

    Uses ``re.escape`` so that ``pattern`` is treated as a literal string.
    """
    return len(re.findall(re.escape(pattern), text))


def _package_py_files() -> list[pathlib.Path]:
    """Return all .py files under the pgmuvi package, excluding known scripts."""
    import pgmuvi

    pkg_root = pathlib.Path(pgmuvi.__file__).parent
    return [
        p
        for p in pkg_root.rglob("*.py")
        if p.name not in _EXCLUDED_FILES
    ]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestDtypeAudit(unittest.TestCase):
    """Each (file, pattern) count must not exceed the allowlist maximum."""

    def _check(self, filepath: pathlib.Path, pattern: str) -> None:
        text = filepath.read_text(encoding="utf-8")
        count = _count_pattern(text, pattern)
        filename = filepath.name
        allowed = _ALLOWLIST.get((filename, pattern), 0)
        self.assertLessEqual(
            count,
            allowed,
            msg=(
                f"Dtype audit FAILED: '{pattern}' found {count} time(s) in "
                f"'{filename}' but allowlist permits at most {allowed}.\n"
                f"  File: {filepath}\n"
                f"  Fix the violation(s) and reduce the allowlist entry."
            ),
        )

    def test_no_new_torch_tensor_calls(self):
        """No new bare ``torch.Tensor(`` calls beyond the allowlist."""
        for path in _package_py_files():
            with self.subTest(file=path.name):
                self._check(path, "torch.Tensor(")

    def test_no_new_dtype_float32(self):
        """No new ``dtype=torch.float32`` beyond the allowlist."""
        for path in _package_py_files():
            with self.subTest(file=path.name):
                self._check(path, "dtype=torch.float32")

    def test_no_new_dot_float(self):
        """No new ``.float()`` beyond the allowlist."""
        for path in _package_py_files():
            with self.subTest(file=path.name):
                self._check(path, ".float()")


if __name__ == "__main__":
    unittest.main()
