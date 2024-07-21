from pathlib import Path
import pytest

ROOT_PATH = Path(__file__).absolute().parent.parent


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return ROOT_PATH / "examples"
