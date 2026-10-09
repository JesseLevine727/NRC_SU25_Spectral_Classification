"""Required process support must be installed by a clean package installation."""

import ast
import re
from pathlib import Path


def test_owned_compiler_process_dependency_is_declared():
    # The project uses a literal list of requirement strings. This check also
    # works on supported Python 3.10, which has no stdlib tomllib.
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    project = text.split("[project]\n", 1)[1].split("\n[", 1)[0]
    match = re.search(r"^dependencies\s*=\s*(\[[^\]]*\])", project, re.MULTILINE)
    assert match is not None
    requirements = ast.literal_eval(match.group(1))
    names = {re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0] for requirement in requirements}
    assert "psutil" in names, "Owned figure compiler requires psutil in core dependencies"
