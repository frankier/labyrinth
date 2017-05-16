from gym.utils import seeding
from labyrinth.board import mk_box_contents
from labyrinth.env import mk_initial_labyrinth_state


np_random, seed = seeding.np_random(0)
board, mobile_tiles, num_treasures = mk_box_contents(7)

state = mk_initial_labyrinth_state(
    np_random, board, mobile_tiles, num_treasures,
    num_players=4)

state.act((0, 0), (1, 0))
