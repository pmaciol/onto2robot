from robots_drivers.side_bottom_to_twist import SideBottom2Twist

MAX_FORWARD_SPEED = 1.0
MIN_FORWARD_SPEED = 0.1
MAX_ANGULAR_SPEED = 1.0
MAX_LEFT_DIRECTION = 0.0
MAX_RIGHT_DIRECTION = 40.0
MAX_SENSOR_VALUE = 40.0
MIN_SENSOR_VALUE = 0.0
FRONT_DIRECTION = MAX_LEFT_DIRECTION + (MAX_RIGHT_DIRECTION - MAX_LEFT_DIRECTION) / 2
MOCKED_SIDE_SENSORS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MOCKED_BOTTOM_SENSORS = [0.0, 0.0]


def mock_reasoner(constant_return_value: float):
    def return_value(side_sensors_left_to_right: list[float], bottom_sensors_left_to_right: list[float]):
        return constant_return_value  # Mocked reasoner output

    return return_value


# Mock the robot driver with a reasoner that always returns the same direction value
def _mocked_robot_driver(constant_return_value: float):
    direction_scale: tuple[float, float] = (MAX_LEFT_DIRECTION, MAX_RIGHT_DIRECTION)
    max_forward_speed: float = MAX_FORWARD_SPEED
    min_forward_speed: float = MIN_FORWARD_SPEED
    max_angular_speed: float = MAX_ANGULAR_SPEED
    return SideBottom2Twist(
        mock_reasoner(constant_return_value),
        6,
        2,
        direction_scale,
        max_forward_speed,
        min_forward_speed,
        max_angular_speed,
    )


def test_forward():
    robot = _mocked_robot_driver(FRONT_DIRECTION)
    twist = robot.update_and_evaluate(
        side_sensors_left_to_right=MOCKED_SIDE_SENSORS,
        bottom_sensors_left_to_right=MOCKED_BOTTOM_SENSORS,
    )
    assert twist.linear.x == MAX_FORWARD_SPEED
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0
    assert twist.angular.x == 0.0
    assert twist.angular.y == 0.0
    assert twist.angular.z == 0.0


def test_full_left():
    robot = _mocked_robot_driver(MAX_LEFT_DIRECTION)
    twist = robot.update_and_evaluate(
        side_sensors_left_to_right=MOCKED_SIDE_SENSORS,
        bottom_sensors_left_to_right=MOCKED_BOTTOM_SENSORS,
    )
    assert twist.linear.x == MIN_FORWARD_SPEED
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0
    assert twist.angular.x == 0.0
    assert twist.angular.y == 0.0
    assert twist.angular.z == -MAX_ANGULAR_SPEED


def test_full_right():
    robot = _mocked_robot_driver(MAX_RIGHT_DIRECTION)
    twist = robot.update_and_evaluate(
        side_sensors_left_to_right=MOCKED_SIDE_SENSORS,
        bottom_sensors_left_to_right=MOCKED_BOTTOM_SENSORS,
    )
    assert twist.linear.x == MIN_FORWARD_SPEED
    assert twist.linear.y == 0.0
    assert twist.linear.z == 0.0
    assert twist.angular.x == 0.0
    assert twist.angular.y == 0.0
    assert twist.angular.z == MAX_ANGULAR_SPEED
