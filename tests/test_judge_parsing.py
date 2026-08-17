import pytest

from ragmeter.judge.parsing import ParseError, extract_json


def test_plain_json():
    assert extract_json('{"score": 4}') == {"score": 4}


def test_fenced_json():
    assert extract_json('```json\n{"score": 4}\n```') == {"score": 4}


def test_fenced_without_language_tag():
    assert extract_json('```\n{"score": 4}\n```') == {"score": 4}


def test_prose_before_and_after():
    text = 'Here is my assessment:\n{"score": 4}\nHope that helps!'
    assert extract_json(text) == {"score": 4}


def test_nested_objects_survive():
    text = 'blah {"claims": [{"claim": "x", "supported": true}]} blah'
    assert extract_json(text) == {"claims": [{"claim": "x", "supported": True}]}


def test_leading_whitespace_and_newlines():
    assert extract_json('\n\n  {"score": 1}  \n') == {"score": 1}


def test_no_json_at_all_raises():
    with pytest.raises(ParseError, match="no JSON object"):
        extract_json("I refuse to answer.")


def test_malformed_json_raises():
    with pytest.raises(ParseError, match="not valid JSON"):
        extract_json('{"score": }')


def test_empty_string_raises():
    with pytest.raises(ParseError, match="no JSON object"):
        extract_json("")


def test_top_level_array_is_rejected():
    # Every judge prompt asks for an object. An array means the model ignored
    # the shape, and guessing which key it meant is worse than failing.
    with pytest.raises(ParseError, match="no JSON object"):
        extract_json("[1, 2, 3]")
