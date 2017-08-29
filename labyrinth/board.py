import numpy as np
import math

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
    (True, False, True, False),
    (True, True, False, False),
    (True, True, True, False),
    (True, True, True, True),
]
VALID_PLAYERS = [1, 2, 4]

# Player info
PLAYER_COLORS = ['Red', 'Green', 'Blue', 'Yellow']
PLAYER_SYMBOLS = ['R', 'G', 'B', 'Y']


# Tile datatype
tile_dt = np.dtype({'names': ['path_type', 'orientation', 'treasure', 'base'],
                    'formats': [np.uint8, np.uint8, np.int8, np.int8]})

"""
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

## Board generation

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


def place_static(board, players=4):
    """
    Place all static tiles apart from the corner tiles on a board.
    """
    assert is_valid_players(players)
    treasure_idx = 0
    size, size = board.shape
    ssize = static_size(size)
    if ssize % 2 == 1:
        tile = (CROSSROADS_TYPE, 0, -1, -1)
        board[ssize - 1, ssize - 1] = tile
    board_view = board
    treasure_sides = range(0, 4, 4 // players)
    for orientation in range(4):
        start_pos = 1
        end_pos = ssize - 1
        # t-junctions
        for layer in range(math.floor(ssize / 2)):
            for item in range(start_pos, end_pos):
                if orientation in treasure_sides:
                    tile = (T_TYPE, orientation, treasure_idx, -1)
                    treasure_idx += 1
                else:
                    tile = (T_TYPE, orientation, -1, -1)
                board_view[layer * 2, item * 2] = tile
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


def get_mobile_tile_composition(size, static_treasures, players=4):
    assert is_valid_players(players)
    ssize = static_size(size)
    m_treasures = max(static_treasures, players * (ssize - 1))
    mobile_tiles = size * size - ssize * ssize + 1
    nt_tiles = mobile_tiles - m_treasures
    st_nt = math.floor(nt_tiles * 3 / 5)
    co_nt = nt_tiles - st_nt
    return {
        'st_t': math.ceil(m_treasures / 2),
        'co_t': math.floor(m_treasures / 2),
        'st_nt': st_nt,
        'co_nt': co_nt,
    }


def mk_box_contents(size=7, players=4):
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
    assert is_valid_players(players)
    board = np.ndarray((size, size), tile_dt)
    place_corners(board)
    treasure_idx = place_static(board, players=players)
    tile_composition = get_mobile_tile_composition(size, treasure_idx,
                                                   players=players)
    mobile_tiles = mk_mobile_tiles(treasure_idx=treasure_idx,
                                   **tile_composition)
    m_treasures = tile_composition['st_t'] + tile_composition['co_t']
    return (board, mobile_tiles, treasure_idx + m_treasures)


## Utilities

def reset_orientation(tile):
    tile['orientation'] = 0


def static_size(size):
    """
    Get the static size from a board size, eg a 7x7 board has 4x4 static tiles.
    """
    return math.ceil(size / 2)


def num_lanes(size):
    """
    Get the static size from a board size, eg a 7x7 board has 3x3 lanes.
    """
    return math.floor(size / 2)


def do_push(board_state, push):
    """
    Takes a (board, spare) -> (new_board, new_spare)
    by performing (side, lane, orientation
    side 0, 1, 2, 3 = n, e, s, w
    lane is from the left corner, if we rotate so our push side is on top
    orientation is 0, 1, 2, 3 = n, e, s, w
    """
    (push_side, push_lane, orientation) = push
    push_row = 2 * push_lane + 1
    new_board = np.copy(board_state[0])
    spare_tile = np.copy(board_state[1])
    board = np.rot90(new_board, 4 - push_side)
    new_orientation = orientation % NUM_ORIENTATIONS[spare_tile['path_type']]
    spare_tile['orientation'] = new_orientation
    a = np.append(board[push_row, :], spare_tile)
    a = np.roll(a, 1)
    new_spare_tile = a[-1]
    reset_orientation(new_spare_tile)
    a = a[:-1]
    board[push_row, :] = a
    return (new_board, new_spare_tile)


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
    reachability = np.zeros(board.shape, dtype=bool)
    reachability[position[0]][position[1]] = True
    return get_reach_aux(reachability, board)


def get_neighbour_coords(position):
    return ((position[0],     position[1] - 1),
            (position[0] + 1, position[1]),
            (position[0],     position[1] + 1),
            (position[0] - 1, position[1]))


def can_pass(direction, tile_from, tile_to):
    return get_tile_passability(tile_to)[(direction + 2) % 4] and \
            get_tile_passability(tile_from)[direction]


def get_reach_aux(reachability, board):
    # Iterate through all cells and find all that are known to be reachable.
    previous = reachability
    # Traverse over all tiles
    for x in range(0, len(reachability)):
        for y in range(0, len(reachability[0])):
            # If that tile is reachable
            if reachability[x][y]:
                neighbours = get_neighbour_coords((x, y))
                # then traverse over all neighbours. z = 0 means north neighbour. z = 1 means east neighbour etc. 
                for z in range(4):
                    neighbour = neighbours[z]
                    # The neighbour might not actually exist. test if it's inside the board.
                    if not is_inside_board(neighbour, board):
                        continue
                    passing = can_pass(z, board[x][y], board[neighbour])
                    if passing:
                        reachability[neighbour[0]][neighbour[1]] = True
    # If there was a change repeat. Otherwise it's done
    change = not np.array_equal(previous, reachability)
    if change:
        return get_reach_aux(reachability, board)
    else:
        return reachability


def is_inside_board(position, board):
    return position[0]>=0 and position[1]>=0 and position[0]<len(board) and position[1]<len(board)


def is_valid_board_size(size):
    return size >= 3 and size % 2 == 1

def is_valid_players(players):
    return players in VALID_PLAYERS


def valid_pushes(size):
    return ((push_side, push_lane, orientation)
            for push_lane in range(num_lanes(size))
            for push_side in range(4)
            for orientation in range(4))


def valid_moves(board_state, current_position, push):
    (possible_board, spare_tile) = do_push(board_state, push)
    reachability = get_board_reachability(possible_board, current_position)
    return np.transpose(np.nonzero(reachability))


def get_possible_actions(board_state, current_position):
    actions = []
    size, size = board_state[0].shape
    pushes = valid_pushes(size)
    for push in pushes:
        moves = valid_moves(board_state, current_position, push)
        for move in moves:
            actions.append((push, tuple(move)))
    return actions
