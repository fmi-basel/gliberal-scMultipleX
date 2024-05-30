from pathlib import Path

import pytest
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "resources"

def pytest_runtest_logstart(nodeid, location):
    warnings.showwarning = warn_with_traceback

