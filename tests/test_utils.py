import pytest

from dl_lab import utils


def test_nested_getattr():
    s = "hello!"
    assert utils.nested_getattr(s, ()) is s

    assert utils.nested_getattr(s, ("upper",))() == "HELLO!"

    assert utils.nested_getattr(s, ("upper", "__name__")) == "upper"

    with pytest.raises(AttributeError):
        utils.nested_getattr(s, ("to_upper"))()
