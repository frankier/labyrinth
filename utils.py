import numpy as np


def static_size(size):
    """
    Get the static size from a board size, eg a 7x7 board has 4x4 static tiles.
    """
    return math.ceil(size / 2)


def num_lanes(width):
    """
    Get the static size from a board size, eg a 7x7 board has 3x3 lanes.
    """
    return math.floor(size / 2)


def do_push(board_state, push):
    (push_side, push_lane) = push
    push_row = 2 * push_lane + 1
    new_board = np.copy(board_state[0])
    board = np.rot90(new_board, push_side)
    a=np.append(board[:,push_row],board_state[1])
    a = np.roll(a,1)
    new_spare_tile = a[-1]
    a=a[:-1]
    board[:,push_row] = a
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
    reachability = np.zeros(board.shape,dtype=bool)
    reachability[position[0]][position[1]]=True
    return get_reach_aux(reachability,board);


def get_neighbour_coords(position):
    return ((position[0]-1,position[1]),
        (position[0],position[1]+1),
        (position[0]+1,position[1]),
        (position[0],position[1]-1))


def can_pass(direction, tile_from, tile_to):
    return get_tile_passability(tile_to)[(direction+2)%4] and get_tile_passability(tile_from)[direction]


def get_reach_aux(reachability, board):
    # Iterate through all cells and find all that are known to be reachable.
    previous = reachability
    # Traverse over all tiles
    for x in range(0,len(reachability)):
        for y in range(0,len(reachability[0])):
            # If that tile is reachable
            if (reachability[x][y]==True):
                neighbours = get_neighbour_coords((x,y))
				# then traverse over all neighbours. z = 0 means north neighbour. z = 1 means east neighbour etc. 
                for z in range(0,3):
                    neighbour = neighbours[z]
					# The neighbour might not actually exist. test if it's inside the board.
                    if is_inside_board(neighbour,board):
                        if (can_pass(z,board[x][y],board[neighbour[0]][neighbour[1]])):
                            reachability[neighbours[z][0]][neighbours[z][1]]=True
	# If there was a change repeat. Otherwise it's done
    change = not np.array_equal(previous,reachability)
    if (change):
        return get_reach_aux(reachability,board)
    else:
        return reachability


def is_inside_board(position, board):
    return position[0]>=0 and position[1]>=0 and position[0]<len(board) and position[1]<len(board)
