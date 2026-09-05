"""
tests/test_tools.py

Unit tests for tools.py, mocking MassiveClient so tests run without a
real API key or network access.

Run with: pytest
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402


def test_get_market_status_calls_client():
    with patch.object(tools.client, "market_status", return_value={"market": "open"}) as mock:
        result = tools.get_market_status()
    mock.assert_called_once()
    assert result == {"market": "open"}


def test_get_stock_snapshot_uppercases_and_calls_client():
    with patch.object(tools.client, "ticker_snapshot", return_value={"ticker": "AAPL"}) as mock:
        tools.get_stock_snapshot("aapl")
    mock.assert_called_once_with("aapl")


def test_call_tool_success():
    with patch.object(tools.client, "market_status", return_value={"market": "extended-hours"}):
        output = tools.call_tool("get_market_status", {})
    assert json.loads(output) == {"market": "extended-hours"}


def test_call_tool_unknown_tool_returns_error_json():
    output = tools.call_tool("not_a_real_tool", {})
    parsed = json.loads(output)
    assert "error" in parsed


def test_call_tool_catches_exceptions():
    with patch.object(tools.client, "ticker_snapshot", side_effect=RuntimeError("boom")):
        output = tools.call_tool("get_stock_snapshot", {"ticker": "AAPL"})
    parsed = json.loads(output)
    assert "boom" in parsed["error"]


def test_all_schemas_have_matching_registered_functions():
    schema_names = {s["name"] for s in tools.TOOL_SCHEMAS}
    registry_names = set(tools.TOOL_REGISTRY.keys())
    assert schema_names == registry_names
