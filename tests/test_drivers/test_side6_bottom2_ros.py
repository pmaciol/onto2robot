from robots_drivers.side6_bottom2_ros import SideBottom2Twist

MAX_FORWARD_SPEED = 1.0
MIN_FORWARD_SPEED = 0.1
MAX_ANGULAR_SPEED = 1.0
MAX_LEFT_DIRECTION = 0.0
MAX_RIGHT_DIRECTION = 40.0
FRONT_DIRECTION = MAX_LEFT_DIRECTION + (MAX_RIGHT_DIRECTION - MAX_LEFT_DIRECTION) / 2


def mock_reasoner(returning: list[float], _):
    return returning[0]  # Mocked reasoner output getting the front left sensor value as the final move direction


def _default_robot_driver():
    direction_scale: tuple[float, float] = (MAX_LEFT_DIRECTION, MAX_RIGHT_DIRECTION)
    max_forward_speed: float = MAX_FORWARD_SPEED
    min_forward_speed: float = MIN_FORWARD_SPEED
    max_angular_speed: float = MAX_ANGULAR_SPEED
    return SideBottom2Twist(
        mock_reasoner, 6, 2, direction_scale, max_forward_speed, min_forward_speed, max_angular_speed
    )


def test_forward():
    robot = _default_robot_driver()
    twist = robot.update_and_evaluate(
        side_sensors=[FRONT_DIRECTION, 0, 0, 0, 0, 0],
        bottom_sensors=[0, 0],
    )
    assert twist.linear.x == MAX_FORWARD_SPEED
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0
    assert twist.angular.x == 0.0
    assert twist.angular.y == 0.0
    assert twist.angular.z == 0.0


def test_full_left():
    robot = _default_robot_driver()
    twist = robot.update_and_evaluate(
        side_sensors=[MAX_LEFT_DIRECTION, 0, 0, 0, 0, 0],
        bottom_sensors=[0, 0],
    )
    assert twist.linear.x == MIN_FORWARD_SPEED
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0
    assert twist.angular.x == 0.0
    assert twist.angular.y == 0.0
    assert twist.angular.z == -MAX_ANGULAR_SPEED


def test_full_right():
    robot = _default_robot_driver()
    twist = robot.update_and_evaluate(
        side_sensors=[MAX_RIGHT_DIRECTION, 0, 0, 0, 0, 0],
        bottom_sensors=[0, 0],
    )
    assert twist.linear.x == MIN_FORWARD_SPEED
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0
    assert twist.angular.x == 0.0
    assert twist.angular.y == 0.0
    assert twist.angular.z == MAX_ANGULAR_SPEED
