# game.py

import random
import time
import sys
import cv2
import numpy as np
from robot_player import RobotPlayer
from tictactoe_vision import BoardDetector
from tic_tac_toe_config import CONFIG
from pyniryo import NiryoRobot, NiryoRobotException
from robot_player import BOARD_JOINTS

class TicTacToe:
    def __init__(self):
        
        self.board = [' ' for _ in range(9)]

    def print_board(self):
        print("\nBoard State:")
        for i in range(3):
            row = self.board[3*i:3*i+3]
            print(' | '.join(row))
            if i < 2:
                print("-----------")

    def make_move(self, idx, symbol):
        if 0 <= idx < 9 and self.board[idx] == ' ':
            self.board[idx] = symbol
            return True
        return False

    def check_win(self):
        b = self.board
        wins = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        return next((b[a] for (a, b_idx, c) in wins if b[a] != ' ' and b[a] == b[b_idx] == b[c]), None)

    def check_tie(self):
        return ' ' not in self.board

    def available_moves(self):
        return [i for i, val in enumerate(self.board) if val == ' ']

def detect_human_move(robot, grid_cells, game):
    """
    Continuously captures images from the robot's camera,
    detects a blue marker (using an HSV range for blue),
    and returns the index of the grid cell in which the blue marker appears.
    """
    print("Waiting for human to place BLUE marker...")
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    detected_index = None
    while detected_index is None:
        img = robot.get_img_compressed()
        if img is None:
            continue
        frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for idx, cell in enumerate(grid_cells):
            (x1, y1), (x2, y2) = cell
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w / 2
                cy = y + h / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2 and game.board[idx] == ' ':
                    detected_index = idx
                    print(f"Detected BLUE marker in cell {detected_index}.")
                    return detected_index
        cv2.waitKey(50)
    return detected_index

def robot_best_move(game):
    """
    Determine the best move for the robot using the Minimax algorithm.
    """
    best_score = -float('inf')
    best_move = None
    for idx in game.available_moves():
        game.board[idx] = 'R'
        score = minimax(game, 0, False, -float('inf'), float('inf'))
        game.board[idx] = ' '  # Undo move
        if score > best_score:
            best_score = score
            best_move = idx
    return best_move

def minimax(game, depth, is_maximizing, alpha, beta):
    """
    Minimax algorithm with Alpha-Beta Pruning to determine the best move.
    :param game: The current game state.
    :param depth: The depth of the recursion (used for optimization).
    :param is_maximizing: True if the robot is making the move, False if the human is making the move.
    :param alpha: The best value that the maximizing player can guarantee.
    :param beta: The best value that the minimizing player can guarantee.
    :return: The best score for the current player.
    """
    winner = game.check_win()
    if winner == 'R':  # Robot wins
        return 10 - depth
    if winner == 'B':  # Human wins
        return depth - 10
    if game.check_tie():  # Tie
        return 0

    if is_maximizing:
        max_eval = -float('inf')
        for idx in game.available_moves():
            game.board[idx] = 'R'
            eval = minimax(game, depth + 1, False, alpha, beta)
            game.board[idx] = ' '  # Undo move
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Prune the branch
        return max_eval
    else:
        min_eval = float('inf')
        for idx in game.available_moves():
            game.board[idx] = 'B'
            eval = minimax(game, depth + 1, True, alpha, beta)
            game.board[idx] = ' '  # Undo move
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Prune the branch
        return min_eval

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

    def pick_up_red_marker(self):
        """
        Move to the red marker position, pick it up using the vacuum pump, 
        and return to the observation pose.
        """
        red_pose = self.cfg["red_marker_pos"]
        self.safe_move_joints(red_pose, "red marker position")
        print("Activating vacuum pump to pick up RED marker...")
        self.robot.pull_air_vacuum_pump()  # Activate the vacuum pump to pick up the marker
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
        print("Deactivating vacuum pump to release marker...")
        self.robot.push_air_vacuum_pump()  # Deactivate the vacuum pump to release the marker
        time.sleep(1)

        # Return to observation pose after placing the marker
        self.move_to_observation_pose()

    def close(self):
        """
        Close the robot connection.
        """
        self.robot.close_connection()
        print("Robot connection closed.")

def main():
    # Initialize robot player
    robot_player = RobotPlayer()

    # Move to observation pose immediately after starting
    print("Moving robot arm to observation pose...")
    robot_player.move_to_observation_pose()

    # Detect board grid
    detector = BoardDetector(robot_player.robot)
    print("Detecting board. View the window and press 'q' when the grid is finalized.")
    grid_cells = detector.detect_board()
    if not grid_cells or len(grid_cells) != 9:
        print("Error: Could not detect a valid 3x3 board.")
        robot_player.close()
        sys.exit(1)
    else:
        print("Board detected. Grid cells:")
        for idx, cell in enumerate(grid_cells):
            print(f"Cell {idx}: {cell}")

    # Move to observation pose after board detection
    robot_player.move_to_observation_pose()

    # Initialize game logic
    game = TicTacToe()
    game.print_board()

    # Main game loop
    while True:
        # Human turn: Wait for blue marker detection.
        print("\nHuman's turn:")
        human_idx = detect_human_move(robot_player.robot, grid_cells, game)
        if human_idx is None:
            print("No valid move detected. Exiting game.")
            break
        if not game.make_move(human_idx, 'B'):
            print(f"Cell {human_idx} is already occupied. Waiting for a valid move...")
            continue
        game.print_board()
        winner = game.check_win()
        if winner == 'B':
            print("Human (BLUE) wins!")
            break
        if game.check_tie():
            print("Game is a tie!")
            break

        # Robot turn: Use the Minimax algorithm to choose the best move.
        print("\nRobot's turn:")
        robot_idx = robot_best_move(game)
        if not game.make_move(robot_idx, 'R'):
            print("Robot attempted to place in an occupied cell. This should not happen!")
            continue
        print(f"Robot (RED) selects cell {robot_idx}.")
        robot_player.pick_up_red_marker()
        robot_player.place_marker_in_cell(robot_idx)
        robot_player.move_to_observation_pose()
        game.print_board()
        winner = game.check_win()
        if winner == 'R':
            print("Robot (RED) wins!")
            break
        if game.check_tie():
            print("Game is a tie!")
            break

    # Move to sleep position after the game is over
    print("Game over. Moving robot arm to sleep position...")
    sleep_joints = CONFIG["sleep_joints"]
    robot_player.safe_move_joints(sleep_joints, "sleep position")

    # Close robot connection
    print("Closing robot connection...")
    time.sleep(2)
    robot_player.close()

if __name__ == "__main__":
    main()