from onto2robot import add


def test_add_basic():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-4, 7) == 3
