from hate_speech_adv.grouping import (
    GROUP_VERSION,
    STRICT_VERSION,
    group_normalize,
    strict_normalize,
    template_group_hash,
)


def test_strict_normalization_is_representation_only():
    assert strict_normalize("  A&amp;B\tＣ  ") == "a&b c"


def test_template_normalization_collapses_retweet_handle_and_url():
    left = "!!! RT @Alice: Read HTTPS://example.com/x now"
    right = "@Bob: read https://other.example/y NOW"
    assert group_normalize(left) == "@user: read <url> now"
    assert group_normalize(left) == group_normalize(right)
    assert template_group_hash(left) == template_group_hash(right)


def test_group_hash_is_versioned_and_full_sha256():
    value = template_group_hash("hello")
    assert STRICT_VERSION == "strict_v1"
    assert GROUP_VERSION == "template_v1"
    assert len(value) == 64
    assert set(value) <= set("0123456789abcdef")

