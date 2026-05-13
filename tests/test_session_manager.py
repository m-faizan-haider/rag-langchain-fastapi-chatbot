# tests/test_session_manager.py
"""Tests for the conversation session manager."""
import pytest
from Backend.session_manager import (
    create_session,
    get_history,
    append_turn,
    delete_session,
    format_history_for_prompt,
)


def test_create_session_returns_uuid():
    sid = create_session()
    assert isinstance(sid, str)
    assert len(sid) == 36  # UUID format


def test_new_session_has_empty_history():
    sid = create_session()
    assert get_history(sid) == []


def test_append_and_retrieve():
    sid = create_session()
    append_turn(sid, "What is AI?", "AI is artificial intelligence.")
    history = get_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[0]["content"] == "What is AI?"


def test_history_is_trimmed():
    sid = create_session()
    # Append more than max turns (6 turns = 12 messages)
    for i in range(10):
        append_turn(sid, f"Question {i}", f"Answer {i}")
    history = get_history(sid)
    assert len(history) <= 12   # _MAX_HISTORY * 2


def test_delete_session():
    sid = create_session()
    append_turn(sid, "hello", "world")
    delete_session(sid)
    assert get_history(sid) == []


def test_format_history_prompt():
    sid = create_session()
    append_turn(sid, "What is machine learning?", "It is a subset of AI.")
    history = get_history(sid)
    prompt = format_history_for_prompt(history)
    assert "CONVERSATION HISTORY" in prompt
    assert "machine learning" in prompt


def test_format_empty_history():
    result = format_history_for_prompt([])
    assert result == ""
