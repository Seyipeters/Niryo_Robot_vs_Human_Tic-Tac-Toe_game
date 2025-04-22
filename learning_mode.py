from pyniryo2 import *

robot = NiryoRobot("10.10.10.10")

robot.arm.calibrate_auto()

robot.arm.move_joints([-0.829, -0.317, -0.566, -0.644, -0.641, -1.821])
robot.arm.set_learning_mode(enabled=False)


robot.end()