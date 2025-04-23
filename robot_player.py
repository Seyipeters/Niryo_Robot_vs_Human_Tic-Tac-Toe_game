# robot_player.py

import time
from pyniryo import NiryoRobot, NiryoRobotException
from tic_tac_toe_config import CONFIG

# Pre-defined joint configurations for each board cell (indices 0-8)
BOARD_JOINTS = {
    0: [-0.530, -0.447, -1.092, -0.219, 0.009, 0.065],
    1: [-0.170, -0.417, -1.116,  -0.189, 0.008, 0.065],
    2: [ 0.240, -0.420, -1.108,  0.338, 0.008, 0.065],
    3: [-0.389, -0.534, -0.817, -0.034, -0.210, 0.206],
    4: [-0.124, -0.496, -0.866, -0.182, -0.210, 0.206],
    5: [ 0.144, -0.478, -0.902, -0.074, -0.158, -0.169],
    6: [-0.285, -0.647, -0.536, -0.093, -0.325, 0.184],
    7: [-0.092, -0.622, -0.576, -0.141, -0.325, 0.184],
    8: [ 0.109, -0.623, -0.576, 0.006, -0.325, 0.184],
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