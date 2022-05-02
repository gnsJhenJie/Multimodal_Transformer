import math_func
import pytest


@pytest.mark.number
def test_add():
    assert math_func.add(7, 3) == 10
    assert math_func.add(7) == 9
    assert math_func.add(3) == 5


@pytest.mark.number
def test_product():
    assert math_func.product(5, 5) == 25
    assert math_func.product(5) == 10
    assert math_func.product(3) == 6


@pytest.mark.strings
def test_add_strings():
    result = math_func.add("Hello", " World")
    assert result == "Hello World"
    assert isinstance(result, str)
    assert "Heldlo" not in result
