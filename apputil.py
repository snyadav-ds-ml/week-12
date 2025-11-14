import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def update_board(current_board):
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # Count live neighbors using convolution
    neighbors = convolve2d(current_board, kernel, mode="same", boundary="wrap")

    # Apply Conway's Game of Life rules
    birth = (neighbors == 3) & (current_board == 0)        # Dead -> Alive
    survive = ((neighbors == 2) | (neighbors == 3)) & (current_board == 1)
    # Create next board state
    current_board_board = np.zeros_like(current_board)
    current_board[birth | survive] = 1

    
    updated_board = current_board

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)