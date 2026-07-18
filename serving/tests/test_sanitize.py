from app.sanitize import sanitize     # pure function; importing it loads no model

def test_strips_control_chars_and_collapses_whitespace():
    assert sanitize("a\x00b\t\tc  d") == "a b c d"

def test_length_cap():
    assert len(sanitize("x" * 10000)) == 5000