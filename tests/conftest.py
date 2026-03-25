"""Shared pytest configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (integration tests that train models)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip unless --run-slow is passed)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_records():
    """Simple flat dicts with no nesting."""
    return [
        {"id": 1, "name": "Alice", "age": 30, "score": 9.5},
        {"id": 2, "name": "Bob", "age": 25, "score": 7.0},
        {"id": 3, "name": "Carol", "age": 35, "score": 8.2},
    ]


@pytest.fixture
def nested_records():
    """Nested dicts with arrays (including empty arrays) and sub-objects."""
    return [
        {
            "id": 1,
            "user": {"name": "Alice", "active": True},
            "tags": ["python", "ml"],
            "scores": [9.5, 8.0],
        },
        {
            "id": 2,
            "user": {"name": "Bob", "active": False},
            "tags": [],  # empty array — the critical regression case
            "scores": [7.0],
        },
        {
            "id": 3,
            "user": {"name": "Carol", "active": True},
            "tags": ["data"],
            "scores": [],  # another empty array
        },
    ]


@pytest.fixture
def mixed_type_records():
    """Records where some fields have mixed types (int/str/None)."""
    return [
        {"value": 42, "label": "A", "flag": True, "extra": None},
        {"value": "high", "label": "B", "flag": False, "extra": 3.14},
        {"value": None, "label": "A", "flag": True, "extra": None},
        {"value": 7, "label": "C", "flag": False},
    ]
