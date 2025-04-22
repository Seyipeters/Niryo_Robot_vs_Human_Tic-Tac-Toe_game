# robot_player.py

import time
from pyniryo import NiryoRobot, NiryoRobotException
from tic_tac_toe_config import CONFIG

# Pre-defined joint configurations for each board cell (indices 0-8)
BOARD_JOINTS = {
    0: [-0.22, -0.60, -0.67, -0.04, -0.31, -0.13],
    1: [-0.06, -0.58, -0.70,  0.03, -0.30, -0.12],
    2: [ 0.11, -0.59, -0.69,  0.03, -0.29, -0.13],
    3: [-0.20, -0.68, -0.48, -0.08, -0.30, -0.13],
    4: [-0.05, -0.68, -0.50, -0.05, -0.30, -0.14],
    5: [ 0.10, -0.68, -0.49, -0.04, -0.30, -0.14],
    6: [-0.16, -0.79, -0.28, -0.19, -0.29, -0.14],
    7: [-0.04, -0.78, -0.30, -0.19, -0.29, -0.14],
    8: [ 0.09, -0.78, -0.29, -0.14, -0.28, -0.14],
}

class RobotPlayer:
    def __init__(self):
        self.cfg = CONFIG
        self.robot = NiryoRobot(self.cfg["robot_ip"])

    def safe_move_joints(self, joints, description=""):
        """
        Safely move the robot to the specified joint positions, handling collisions if they occur.
        """
        try:
            self.robot.move_joints(joints)
            if description:
                print(f"Moved to {description}.")
        except NiryoRobotException as e:
            if "clear_collision_detected" in str(e):
                print("Collision detected. Clearing collision state...")
                self.robot.clear_collision_detected()
                self.robot.move_joints(joints)
                if description:
                    print(f"Moved to {description} after clearing collision.")

    def move_to_observation_pose(self):
        """
        Move the robot to the observation pose.
        """
        obs_pose = self.cfg["observation_pose"]
        self.safe_move_joints(obs_pose, "observation pose")

    def _move_to_position(self, pose, description):
        """
        Move the robot to a specific position with a description.
        """
        self.safe_move_joints(pose, description)

    def pick_up_red_marker(self):
        """
        Move to the red marker position, pick it up using the gripper, 
        and return to the observation pose.
        """
        red_pose = self.cfg["red_marker_pos"]
        self.safe_move_joints(red_pose, "red marker position")
        print("Closing gripper to pick up RED marker...")
        self.robot.close_gripper()
        time.sleep(2)

        # Return to observation pose after picking up the red marker
        self.move_to_observation_pose()

    def place_marker_in_cell(self, cell_index):
        """
        Place a marker in the specified cell index and return to the observation pose.
        """
        if cell_index not in BOARD_JOINTS:
            print(f"No joint configuration defined for cell index {cell_index}")
            return

        # Move to the target cell to place the marker
        target_joints = BOARD_JOINTS[cell_index]
        self.safe_move_joints(target_joints, f"cell {cell_index}")
        print("Opening gripper to release marker...")
        self.robot.open_gripper()
        time.sleep(1)

        # Return to observation pose after placing the marker
        self.move_to_observation_pose()

    def close(self):
        """
        Close the robot connection.
        """
        self.robot.close_connection()
        print("Robot connection closed.")
