import pytest
# from src.module import *


@pytest.fixture
def fixture_example() -> int:
    return 42


def test_dataset(fixture_example):
    assert fixture_example == 42
