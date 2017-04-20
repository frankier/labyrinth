from gym import error

import math
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

# Tile types
STRAIGHT_TYPE = 0
CORNER_TYPE = 1
T_TYPE = 2
CROSSROADS_TYPE = 3
# Tile type data
NUM_ORIENTATIONS = [2, 4, 4, 1]
PATH_SYMBOLS = [
    ("│", "─"),
    ("└", "┌", "┐", "┘"),
    ("├", "┬", "┤", "┴"),
    ("+"),
]
TILE_PASSABILITIES = [
    [(True, False, True, False)],
    [(True, True, False, False)],
    [(True, True, True, False)],
    [(True, True, True, True)],
]

# Player info
PLAYER_COLORS = ['Red', 'Green', 'Blue', 'Yellow']
PLAYER_SYMBOLS = ['R', 'G', 'B', 'Y']


# Tile datatype
tile_dt = np.dtype({'names': ['path_type', 'orientation', 'treasure', 'base'],
                    'formats': [np.uint8, np.uint8, np.int8, np.int8]})
tile_dt.__doc__ = """
The datatype for a Labyrinth tile.

path_type is one of STRAIGHT_TYPE, CORNER_TYPE, T_TYPE or CROSSROADS_TYPE

orientation = 0 corresponds to │ └ ├
orientation is measured in right turns clockwise
note due to symmetry | only takes orientation = 0, 1

treasure is a number >= 0, corresponding to a treasure if the tile contains a
treasure, -1 otherwise

base is a number, 0 <= base < 4, corresponding to the colour of the base on a
tile if the tile is a base, -1 otherwise

A board is made up of a 2d ndarray of these.
"""


def place_corners(board):
    """
    Place the 4 static corner/base tiles on a board.

    board is set up with raster coordinates
    red  _|_ yellow
    green |  blue
    """
    # set up static board
    size, size = board.shape
    board_view = board
    # outside
    for orientation in range(4):
        # corner
        base = (1 - orientation) % 4
        tile = (CORNER_TYPE, orientation, -1, base)
        board_view[0, size - 1] = tile
        board_view = np.rot90(board_view, 3)


def ssize_of_size(size):
    """
    Get the static size from a board size, eg a 7x7 board has 4x4 static tiles.
    """
    return math.ceil(size / 2)


def place_static(board, center_treasure=False):
    """
    Place all static tiles apart from the corner tiles on a board.
    """
    treasure_idx = 0
    size, size = board.shape
    ssize = ssize_of_size(size)
    if ssize % 2 == 1:
        if center_treasure:
            tile = (CROSSROADS_TYPE, 0, treasure_idx, -1)
            treasure_idx += 1
        else:
            tile = (CROSSROADS_TYPE, 0, -1, -1)
        board[ssize - 1, ssize - 1] = tile
    board_view = board
    for orientation in range(4):
        start_pos = 1
        end_pos = ssize - 1
        # t-junctions
        for layer in range(math.floor(ssize / 2)):
            for item in range(start_pos, end_pos):
                tile = (T_TYPE, orientation, treasure_idx, -1)
                board_view[layer * 2, item * 2] = tile
                treasure_idx += 1
            if layer % 2 == 1:
                start_pos += 1
            end_pos -= 1
        board_view = np.rot90(board_view, 3)
    return treasure_idx


def mk_mobile_tiles(st_t, co_t, st_nt, co_nt, treasure_idx):
    """
    Make a collection of the movable tiles on a Labyrinth board.
    """
    num_tiles = st_t + co_t + st_nt + co_nt
    mobile_tiles = np.ndarray(num_tiles, tile_dt)
    mobile_tiles_idx = 0

    for i in range(st_t):
        mobile_tiles[mobile_tiles_idx] = (STRAIGHT_TYPE, 0, treasure_idx, -1)
        mobile_tiles_idx += 1
        treasure_idx += 1

    for i in range(co_t):
        mobile_tiles[mobile_tiles_idx] = (CORNER_TYPE, 0, treasure_idx, -1)
        mobile_tiles_idx += 1
        treasure_idx += 1

    for i in range(st_nt):
        mobile_tiles[mobile_tiles_idx] = (STRAIGHT_TYPE, 0, -1, -1)
        mobile_tiles_idx += 1

    for i in range(co_nt):
        mobile_tiles[mobile_tiles_idx] = (CORNER_TYPE, 0, -1, -1)
        mobile_tiles_idx += 1

    return mobile_tiles


def is_valid_board_size(size):
    return size >= 3 and size % 2 == 1


def mk_box_contents(size=7):
    """
    Make the contents of a (variation of a) Ravensburger Labyrinth box.

    Takes a optionally a size which must be odd and >= 3. For example, passing
    in 7 will make a 7x7 version of Labyrinth configured the same as the
    original.  Passing in 3 will make a tiny 3x3 version of Labyrinth.

    Returns (board, mobile_tiles, num_treasures) where
    board is a 2d array of tile_dt (a board populated with the static tiles)
    mobile_tiles is a 1d array of tile_dt (a list of the movable tiles)
    num_treasures is the total number of treasures
    """
    assert is_valid_board_size(size)
    board = np.ndarray((size, size), tile_dt)
    ssize = ssize_of_size(size)
    place_corners(board)
    treasure_idx = place_static(board)
    m_treasures = max(treasure_idx, 4 * (ssize - 1))
    hm_treasures  = math.floor(m_treasures / 2)
    mobile_tiles = size * size - ssize * ssize + 1
    nt_tiles = mobile_tiles - m_treasures
    st_nt = math.floor(nt_tiles * 3 / 5)
    co_nt = nt_tiles - st_nt
    mobile_tiles = mk_mobile_tiles(hm_treasures, hm_treasures, st_nt, co_nt, treasure_idx)
    return (board, mobile_tiles, treasure_idx + m_treasures)


def get_tile_passability(tile):
    """
    Get the passability of a tile.

    Takes tile_dt
    Returns passability (north, east, south, west)
    """
    passability = TILE_PASSABILITIES[tile['path_type']]
    return np.roll(passability, tile['orientation'])


def get_board_reachability(board, position):
    # Takes board, position
    # Returns boolean array of all squares reachable on board from position
    # TODO
    pass


## State
class LabyrinthState(object):
    '''
    Labyrinth game state. Consists of a board, a current player, player
    positions, player cards & number of revealed cards per player.
    '''
    def __init__(self, board_state, player_turn, players, num_treasures):
        self.board_state = board_state
        self.player_turn = player_turn
        self.players = players
        self.num_treasures = num_treasures

    @property
    def long_treasures(self):
        return self.num_treasures > 26

    @property
    def player_positions(self):
        return [pos for (pos, cards, found) in self.players]

    @property
    def num_players(self):
        return len(self.players)

    @property
    def is_terminal(self):
        # TODO: Return whether state is terminal or not
        pass

    @property
    def winner(self):
        # TODO: Return the winning player (if the state is terminal)
        pass

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            a new LabyrinthState with the new board and the player switched
        '''
        type = action['type']
        if type != 'move':
            # TODO: Increment player turn and return
            return LabyrinthState() # TODO
        # TODO: Change board according to move and then return state with new board and new player_turn
        (push_side, push_row) = action['push']
        (move_to_x, move_to_y) = action['move']
        return LabyrinthState() # TODO

    def observe(self, player):
        # TODO: Should return player x's observation of the state here rather
        # than the whole state (ie strictly we should delete information the
        # agent can't see)
        return self

    def __repr__(self):
        bits = []
        bits.append('To play: {}'.format(PLAYER_COLORS[self.player_turn]))
        bits.append(draw_board(self.board_state[0], self.player_positions, long_treasures=self.long_treasures))
        bits.append("")
        bits.append("Spare tile: {}".format(draw_tile(self.board_state[1], long_treasures=self.long_treasures)))
        bits.append("")
        player_statuses = []
        for i in range(self.num_players):
            player_statuses.append([])
        for player_idx, (pos, cards, found) in enumerate(self.players):
            player_bits = player_statuses[player_idx]
            player_bits.append("= {} =".format(PLAYER_COLORS[player_idx]))
            for card_idx, card in enumerate(cards):
                current = card_idx == found
                player_bits.append("{}{}".format("*" if current else "", treasure_letter(card, long_treasures=self.long_treasures)))
        for row in zip(*player_statuses):
            bits.append(("{: >18} " * self.num_players).format(*row))
        return "\n".join(bits)


def treasure_letter(treasure, long_treasures=False):
    if long_treasures:
        return treasure_letter(treasure // 26) + treasure_letter(treasure % 26)
    else:
        return chr(97 + treasure)


def draw_tile(tile, long_treasures=False):
    s1 = PATH_SYMBOLS[tile['path_type']][tile['orientation']]
    pad = " " if long_treasures else ""
    if tile['treasure'] != -1:
        s2 = treasure_letter(tile['treasure'], long_treasures=long_treasures)
    elif tile['base'] != -1:
        s2 = PLAYER_SYMBOLS[tile['base']] + pad
    else:
        s2 = " " + pad
    return s1 + s2


def draw_players(coord, player_positions):
    players_nibble = 0
    for i in range(len(player_positions) - 1, -1, -1):
        players_nibble <<= 1
        if player_positions[i] == coord:
            players_nibble |= 0x01
    if players_nibble:
        return hex(players_nibble)[2:]
    return " "


def draw_board(board_state, player_positions, long_treasures=False):
    rows, cols = board_state.shape
    return "\n".join(
        " ".join(
            draw_tile(board_state[x, y],
                      long_treasures=long_treasures) +
            draw_players((x, y), player_positions)
            for x in range(cols))
        for y in range(rows))


def shuffle_rotate_tiles(np_random, tiles):
    for i in np.ndindex(tiles.shape):
        tiles[i]['orientation'] = np_random.randint(
            0, NUM_ORIENTATIONS[tiles[i]['path_type']] - 1)


def mk_initial_labyrinth_state(np_random, board, mobile_tiles, num_treasures, num_players=4):
    """
    Return the starting Labyrinth state by placing all but one of mobile_tiles
    on a board, dealing the treasure cards to the players and placing their
    pieces on their bases.
    """
    assert 1 <= num_players <= 4
    board = np.copy(board)
    size, size = board.shape
    far = size - 1
    to_place = np_random.permutation(mobile_tiles)
    shuffle_rotate_tiles(to_place)
    place_idx = 0
    for x, y in np.ndindex(board.shape):
        if x % 2 == 1 or y % 2 == 1:
            board[x, y] = to_place[place_idx]
            place_idx += 1
    board_state = (board, to_place[place_idx])
    turn = 0
    player_positions = []
    for pos in [0, 3, 1, 2][:num_players]:
        player_positions.append(((pos // 2) * far, (pos % 2) * far))
    player_cards = np.split(
        np_random.permutation(num_treasures), 4)[:num_players]
    player_cards_found = [0] * num_players
    players = list(zip(player_positions, player_cards, player_cards_found))
    return LabyrinthState(board_state, turn, players, num_treasures)


# Adversary policies
def make_random_policy(np_random):
    def random_policy(curr_state):
        pass
        # TODO: Make a policy which makes random moves
    return random_policy


class LabyrinthEnv(gym.Env):
    def __init__(self, board_size, opponents, illegal_mode_mode):
        assert is_valid_board_size(size)
        self.board_size = board_size

        for opponent in opponents:
            assert opponent in ['random']
        self.opponent = opponents

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        # Properly initialised by _reset
        self.state = None

    def _seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)
        return [self.seed]

    def _reset(self):
        board, mobile_tiles, num_treasures = mk_box_contents(self.board_size)

        self.state = mk_initial_labyrinth_state(
            self.np_random, board, mobile_tiles, num_treasures,
            num_players=len(self.opponent_policies) + 1)

        # Possible TODO: Currently agent always moves first - would have to do
        # first opponent moves and check for termination here otherwise

        return self.state.observe(0)

    def _close(self):
        # Cleanup everything initialised in _reset
        self.state = None

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile

    def _step(self, action):
        assert self.state.player_turn == 0

        # If already terminal, then don't do anything
        if self.done:
            return self.state.observe(0), 0., True, {'state': self.state}

        # If resigned, then we're done
        if action['type'] == 'resign':
            self.done = True
            return self.state.observe(0), -1., True, {'state': self.state}

        assert action['type'] == 'move'

        # Play
        prev_state = self.state
        try:
            self.state = self.state.act(action)
        except IllegalMove:
            if self.illegal_move_mode == 'raise':
                six.reraise(*sys.exc_info())
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state.observe(0), -1., True, {'state': self.state}

        # Opponent play
        for i, opponent in enumerate(self.opponents):
            if not self.state.is_terminal:
                self.state, opponent_resigned = self._exec_opponent_play(self.state)
                # After opponent play, we should be back to the original color
                assert self.state.color == self.player_color

                # If the opponent resigns, then the agent wins
                # TODO: Not true, should instead make player only win when all
                # opponents resign
                if opponent_resigned:
                    self.done = True
                    return self.state.board.encode(), 1., True, {'state': self.state}

        # Reward: if nonterminal, then the reward is 0
        if not self.state.board.is_terminal:
            self.done = False
            return self.state.observe(0), 0., False, {'state': self.state}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal
        self.done = True
        return self.state.observe(0), 1 if self.state.board.winner == 0 else -1, True, {'state': self.state}

    def _exec_opponent_play(self, curr_state):
        assert curr_state.turn != 0
        opponent_action = self.opponent_policies[curr_state.turn - 1](curr_state)
        opponent_resigned = opponent_action['type'] == 'resign'
        return curr_state.act(opponent_action), opponent_resigned

    @property
    def _state(self):
        return self.state

    def _reset_opponents(self, board):
        self.opponent_policies = []
        for opponent in self.opponents:
            if opponent == 'random':
                self.opponent_policies.append(make_random_policy(self.np_random))

# Just do registration here for now - might be better somewhere else. Note this
# must be after LabyrinthEnv since otherwise we will have import problems. This
# is method may be hacky/fragile so should be fixed at some point.

from gym.envs.registration import register

for board_size in [3, 5, 7]:
    register(
        id='Labyrinth-{0}x{0}'.format(board_size),
        entry_point='labyrinth:LabyrinthEnv',
        kwargs={
            'opponent': 'random',
            'illegal_move_mode': 'lose',
            'board_size': board_size,
        },
    )


if __name__ == '__main__':
    # This is just a test to show board generation is working
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    num_players = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(repr(mk_initial_labyrinth_state(*mk_box_contents(size), num_players=num_players)))
