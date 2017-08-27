# -*- coding: utf-8 -*-

from scipy.misc import comb

from labyrinth.board import (CORNER_TYPE, NUM_ORIENTATIONS, STRAIGHT_TYPE,
                             get_mobile_tile_composition, static_size)


def get_board_states(players=4, size=7):
    # Mobile tiles state
    ssize = static_size(size)
    if ssize % 2 == 1:
        static_treasures = ssize * ssize - 5
    else:
        static_treasures = ssize * ssize - 4
    static_treasures = static_treasures * players // 4
    tile_composition = get_mobile_tile_composition(
        size, static_treasures, players=players)
    total_treasures = (static_treasures + tile_composition['st_t'] +
                       tile_composition['co_t'])
    num_mobile_tiles = sum(tile_composition.values())
    tile_ids = []
    for i in range(tile_composition['st_t']):
        tile_ids.append({
            'orientations': NUM_ORIENTATIONS[STRAIGHT_TYPE],
            'number': 1,
        })
    tile_ids.append({
        'orientations': NUM_ORIENTATIONS[STRAIGHT_TYPE],
        'number': tile_composition['st_nt'],
    })
    for i in range(tile_composition['co_t']):
        tile_ids.append({
            'orientations': NUM_ORIENTATIONS[CORNER_TYPE],
            'number': 1,
        })
    tile_ids.append({
        'orientations': NUM_ORIENTATIONS[CORNER_TYPE],
        'number': tile_composition['co_nt'],
    })
    mobile_tile_states = 0
    for i, extra_tile_id in enumerate(tile_ids):
        visible_tile_ids = tile_ids.copy()
        del visible_tile_ids[i]
        remaining_mobile_spaces = num_mobile_tiles
        placements = 1
        total_orientations = 1
        for visible_tile_id in visible_tile_ids:
            number = visible_tile_id['number']
            orientations = visible_tile_id['number']
            placements *= comb(remaining_mobile_spaces, number)
            total_orientations *= orientations
            remaining_mobile_spaces -= number
        mobile_tile_states += placements * total_orientations
    # Card collection state
    card_collection_states = (total_treasures / players + 1) ** players
    # Player position state
    player_position_states = (size * size) ** players
    # Total states
    total_states = \
        mobile_tile_states * card_collection_states * player_position_states
    print("{:.2e} {:.2e} {:.2e} {:.2e}".format(
        mobile_tile_states, card_collection_states,
        player_position_states, total_states))
    return total_states


if __name__ == '__main__':
    for players in [1, 2, 4]:
        for length in [3, 5, 7, 9]:
            print("Players {} Length {}".format(players, length))
            get_board_states(players, length)
