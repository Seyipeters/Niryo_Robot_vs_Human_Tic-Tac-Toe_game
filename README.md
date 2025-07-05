# Niryo Tic-Tac-Toe Robot

This project implements a physical Tic-Tac-Toe game using a Niryo robot arm, computer vision, and Python. The robot plays against a human by detecting human moves with a camera and placing its own markers on a real board using robotic manipulation.

**This project was developed as part of the fulfillment of my BEng Information Technology (IoT) at the Savonia University of Applied Sciences, Finland.**

## Features

- **Computer Vision Board Detection:** Uses OpenCV to detect the Tic-Tac-Toe board and human moves (blue markers).
- **Robot Control:** Controls a Niryo robot arm to pick and place markers (red for robot, blue for human).
- **AI Opponent:** The robot uses the Minimax algorithm with alpha-beta pruning to play optimally.
- **Interactive Gameplay:** Human and robot take turns until there is a win or tie.

## Project Structure

- `game.py` - Main script for running the game loop, robot control, and vision.
- `robot_player.py` - Robot control logic and joint configurations.
- `tictactoe_vision.py` - Board and marker detection using OpenCV.
- `tic_tac_toe_config.py` - Configuration for robot and board.
- `learning_mode.py`, `object_detection.py`, `inspecthsv.py` - Additional scripts for development and calibration.
- `niryo_one_python_api/` - Niryo robot Python API (external dependency).
- `niryo_env/` - Python virtual environment.

## Requirements

- Niryo One robot arm
- Niryo One Python API
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- pyniryo

Install dependencies with:

```sh
pip install -r requirements.txt
```

## How to Run

1. Power on the Niryo robot and connect it to your network.
2. Update the robot IP and joint configurations in `tic_tac_toe_config.py`.
3. Place the Tic-Tac-Toe board in the robot's camera view.
4. Run the main script:

```sh
python game.py
```

5. Follow the on-screen instructions. The robot will detect the board, wait for your move (place a blue marker), and then make its own move.

## Demo Video

[Watch the project demo here.](https://your-video-link-here.com)

## License

This project is for educational and demonstration purposes.
