from math import isclose

from robots_drivers.controlers.wheel_controler import DirectionToWheelCommand

FULL_LEFT = 0.0
FULL_RIGHT = 40.0
MAX_WHEEL = 5.0
FORWARD = (FULL_LEFT + FULL_RIGHT) / 2.0


def _controller() -> DirectionToWheelCommand:
    return DirectionToWheelCommand(full_left=FULL_LEFT, full_right=FULL_RIGHT, max_wheel=MAX_WHEEL)


def test_call_returns_expected_command_at_anchor_points():
    controller = _controller()

    left = controller(FULL_LEFT)
    forward = controller(FORWARD)
    right = controller(FULL_RIGHT)

    assert left.left == -MAX_WHEEL
    assert left.right == MAX_WHEEL

    assert forward.left == MAX_WHEEL
    assert forward.right == MAX_WHEEL

    assert right.left == MAX_WHEEL
    assert right.right == -MAX_WHEEL


def test_call_interpolates_left_wheel_linearly_between_full_left_and_forward():
    controller = _controller()
    quarter_left = FULL_LEFT + (FORWARD - FULL_LEFT) * 0.25

    command = controller(quarter_left)

    expected_left = (-1.0 + 2.0 * 0.25) * MAX_WHEEL
    assert isclose(command.left, expected_left)
    assert command.right == MAX_WHEEL


def test_call_interpolates_right_wheel_linearly_between_forward_and_full_right():
    controller = _controller()
    quarter_right = FORWARD + (FULL_RIGHT - FORWARD) * 0.25

    command = controller(quarter_right)

    expected_right = (1.0 - 2.0 * 0.25) * MAX_WHEEL
    assert command.left == MAX_WHEEL
    assert isclose(command.right, expected_right)


def test_call_clamps_direction_outside_range():
    controller = _controller()

    below = controller(FULL_LEFT - 100.0)
    above = controller(FULL_RIGHT + 100.0)

    assert below.left == -MAX_WHEEL
    assert below.right == MAX_WHEEL
    assert above.left == MAX_WHEEL
    assert above.right == -MAX_WHEEL
