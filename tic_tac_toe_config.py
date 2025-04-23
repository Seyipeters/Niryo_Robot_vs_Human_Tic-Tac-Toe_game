#tic_tac_toe_config.py

CONFIG = {
    "robot_ip": "10.10.10.10",  # Your robot's IP
    "workspace_name": "game_workspace",
    "observation_pose": [0.018, 0.610, -1.098, 0.002, -1.125, -0.031],  # Pose for looking straight down
    "sleep_joints": [0.109, 0.305, -1.127, 0.109, -0.473, -1.437],
    "red_marker_pos": [-0.888, -0.360, -0.713, -0.669, -0.554, 0.152],
    "grid_z_offset": 0.005,  # Offset for grid detection
    "global_z_offset": 0.0,  # Global offset if needed
}
