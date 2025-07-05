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
        """
        Make a move on the board if the cell is empty.
        """
        if 0 <= idx < 9 and self.board[idx] == ' ':
            self.board[idx] = symbol
            print(f"Move made: {symbol} at index {idx}")
            self.print_board()  # Debugging: Print the board state after the move
            return True
        print(f"Invalid move: Cell {idx} is already occupied.")
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
    Detect the human's move by identifying a blue marker on the board.
    """
    print("Waiting for human to place BLUE marker...")
    lower_blue = np.array([100, 150, 100])  # Adjust these values based on your environment
    upper_blue = np.array([130, 255, 255])
    detection_counts = [0] * len(grid_cells)  # Track detection confidence for each cell
    threshold = 0.2  # Minimum percentage of blue pixels required to detect a marker

    while True:
        try:
            img = robot.get_img_compressed()
            if img is None:
                print("No image captured. Retrying...")
                time.sleep(1)
                continue

            frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode image. Retrying...")
                time.sleep(1)
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            for idx, cell in enumerate(grid_cells):
                (x1, y1), (x2, y2) = cell
                cell_mask = mask[y1:y2, x1:x2]
                blue_pixel_count = cv2.countNonZero(cell_mask)
                total_pixel_count = cell_mask.size
                blue_percentage = blue_pixel_count / total_pixel_count

                print(f"Cell {idx}: Blue Pixel Percentage = {blue_percentage:.2f}")

                if blue_percentage > threshold and game.board[idx] == ' ':
                    detection_counts[idx] += 1
                    print(f"Potential move detected in cell {idx}, confirming... ({detection_counts[idx]}/3)")
                    if detection_counts[idx] >= 3:
                        print(f"Confirmed: BLUE marker detected in cell {idx}.")
                        return idx
                else:
                    detection_counts[idx] = 0

            time.sleep(1)
        except Exception as e:
            print(f"Error detecting move: {str(e)}")
            time.sleep(1)

def robot_best_move(game):
    """
    Determine the best move for the robot using the Minimax algorithm.
    """
    print("\nRobot is calculating its best move...")
    print("Current Board State:")
    for i in range(3):
        print(game.board[3 * i:3 * i + 3])

    best_score = -float('inf')
    best_move = None

    # Iterate over available moves
    for idx in game.available_moves():
        print(f"Evaluating move for cell {idx}...")
        # Simulate the robot's move
        game.board[idx] = 'R'
        score = minimax(game, 0, False, -float('inf'), float('inf'))
        game.board[idx] = ' '  # Undo the move

        # Choose the move with the highest score
        if score > best_score:
            best_score = score
            best_move = idx

    print(f"Robot's best move is cell {best_move} with score {best_score}.")
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

def print_board_state(game):
    """
    Print the current state of the board for debugging.
    """
    print("\nCurrent Board State:")
    for i in range(3):
        row = game.board[3 * i:3 * i + 3]
        print(' | '.join(row))
        if i < 2:
            print("-----------")

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