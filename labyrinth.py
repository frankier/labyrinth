from gym import error

import random
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import six

## Game data

FIXED_TREASURES = 12
MOBILE_TREASURES = 12

STRAIGHT_TYPE = 0
CORNER_TYPE = 1
T_TYPE = 2

# orientation = 0 corresponds to │ └ ├
# orientation is measured in right turns clockwise
# note due to symmetry | only takes orientation = 0, 1

tile_dt = np.dtype({'names': ['path_type', 'orientation', 'treasure', 'base'],
                 'formats': [np.uint8, np.uint8, np.int8, np.int8]})


initial_board = np.ndarray((7, 7), tile_dt)
mobile_tiles = np.ndarray(34, tile_dt)


treasure = 0

# board is set up with raster coordinates
# red  _|_ yellow
# green |  blue

# set up static board
board_view = initial_board
# outside
for orientation in range(4):
    # t-junctions
    for i in range(3):
        tile = (T_TYPE, orientation, treasure, -1)
        board_view[(i // 2) * 2, (i % 2) * 2 + 2] = tile
        treasure += 1
    # corner
    base = (1 - orientation) % 4
    tile = (CORNER_TYPE, orientation, -1, base)
    board_view[0, 6] = tile
    board_view = np.rot90(board_view, 3)

mobile_tiles_idx = 0

for i in range(34):
    if i < 6:
        mobile_tiles[i] = (STRAIGHT_TYPE, 0, treasure, -1)
    elif i < 12:
        mobile_tiles[i] = (CORNER_TYPE, 0, treasure, -1)
    elif i < 25:
        mobile_tiles[i] = (STRAIGHT_TYPE, 0, -1, -1)
    else:
        mobile_tiles[i] = (CORNER_TYPE, 0, -1, -1)
    treasure += 1


NUM_TREASURES = 24
PLAYER_COLORS = ['Red', 'Green', 'Blue', 'Yellow']
PLAYER_SYMBOLS = ['R', 'G', 'B', 'Y']
PATH_SYMBOLS = [
    ("│", "─"),
    ("└", "┌", "┐", "┘"),
    ("├", "┬", "┤", "┴"),
]


## State
class LabyrinthState(object):
    '''
    Labyrinth game state. Consists of a board, a current player, player
    positions, player cards & number of revealed cards per player.
    '''
    def __init__(self, board_state, player_turn, player_positions, player_cards, player_cards_found):
        self.board_state = board_state
        self.player_turn = player_turn
        self.player_positions = player_positions
        self.player_cards = player_cards
        self.player_cards_found = player_cards_found

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            a new LabyrinthState with the new board and the player switched
        '''
        # TODO
        push, move = action
        return LabyrinthState()

    def __repr__(self):
        bits = []
        bits.append('To play: {}'.format(PLAYER_COLORS[self.player_turn]))
        bits.append(draw_board(self.board_state[0], self.player_positions))
        bits.append("")
        bits.append("Spare tile: {}".format(draw_tile(self.board_state[1])))
        bits.append("")
        player_statuses = ([], [], [], [])
        for player_idx, (player_cards, cards_found) in enumerate(zip(self.player_cards, self.player_cards_found)):
            player_bits = player_statuses[player_idx]
            player_bits.append("= {} =".format(PLAYER_COLORS[player_idx]))
            for card_idx, card in enumerate(player_cards):
                current = card_idx == cards_found
                player_bits.append("{}{}".format("*" if current else "", treasure_letter(card)))
        for row in zip(*player_statuses):
            bits.append("{: >18} {: >18} {: >18} {: >18}".format(*row))
        return "\n".join(bits)


def treasure_letter(treasure):
    return chr(97 + treasure)


def draw_tile(tile):
    s1 = PATH_SYMBOLS[tile['path_type']][tile['orientation']]
    if tile['treasure'] != -1:
        s2 = treasure_letter(tile['treasure'])
    elif tile['base'] != -1:
        s2 = PLAYER_SYMBOLS[tile['base']]
    else:
        s2 = " "
    return s1 + s2


def draw_players(coord, player_positions):
    players_nibble = 0
    for i in range(3, -1, -1):
        players_nibble <<= 1
        if player_positions[i] == coord:
            players_nibble |= 0x01
    if players_nibble:
        return hex(players_nibble)[2:]
    return " "


def draw_board(board_state, player_positions):
    rows, cols = board_state.shape
    return "\n".join(" ".join(draw_tile(board_state[x, y]) + draw_players((x, y), player_positions)
                     for x in range(cols))
           for y in range(rows))


def shuffle_rotate_tiles(tiles):
    for i in np.ndindex(tiles.shape):
        if tiles[i]['path_type'] == STRAIGHT_TYPE:
            tiles[i]['orientation'] = random.randint(0, 1)
        else:
            tiles[i]['orientation'] = random.randint(0, 3)
    return


def mk_initial_labyrinth_state():
    board = np.copy(initial_board)
    to_place = np.random.permutation(mobile_tiles)
    shuffle_rotate_tiles(to_place)
    place_idx = 0
    for x, y in np.ndindex(board.shape):
        if x % 2 == 1 or y % 2 == 1:
            board[x, y] = to_place[place_idx]
            place_idx += 1
    board_state = (board, to_place[place_idx])
    turn = 0
    player_positions = [(0, 0), (0, 6), (6, 6), (6, 0)]
    player_cards = np.split(np.random.permutation(NUM_TREASURES), 4)
    player_cards_found = [0, 0, 0, 0]
    return LabyrinthState(board_state, turn, player_positions, player_cards, player_cards_found)


# Adversary policies
def make_random_policy(np_random):
    def random_policy(curr_state, prev_state, prev_action):
        pass
        # TODO
    return random_policy


# TODO: Write a LabyrinthEnv like GoEnv (see gym/envs/board_game/go.py)


if __name__ == '__main__':
    print(repr(mk_initial_labyrinth_state()))
