"""Test configuration.

Redirects the data directory into a temporary location for the whole session.
Without this, importing config binds STORAGE.db_path to the real data/ folder
and a test run would write into the operator's run history.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="capex-tests-")
os.environ.setdefault("CAPEX_DATA_DIR", _TMP)
