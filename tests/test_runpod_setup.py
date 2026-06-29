from __future__ import annotations

from enum import Enum
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_RUNPOD_PATH = PROJECT_ROOT / "scripts" / "shell" / "setup_runpod.sh"


class SetupToken(str, Enum):
    diagnostics_repo = "https://github.com/KaleabTessera/probing-dec-pomdps.git"
    diagnostics_install = 'pip install -e "$PROBING_DIR"'
    diagnostics_import = "import dec_pomdp_diagnostics"


def test_runpod_setup_installs_and_verifies_diagnostics_backend() -> None:
    script = SETUP_RUNPOD_PATH.read_text(encoding="utf-8")

    assert SetupToken.diagnostics_repo.value in script
    assert SetupToken.diagnostics_install.value in script
    assert SetupToken.diagnostics_import.value in script
