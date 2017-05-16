import sys
import numpy as np
from labyrinth.board import mk_box_contents
from labyrinth.env import mk_initial_labyrinth_state


# This is just a test to show board generation is working
size = int(sys.argv[1]) if len(sys.argv) > 1 else 7
num_players = int(sys.argv[2]) if len(sys.argv) > 2 else 4
print(repr(mk_initial_labyrinth_state(np.random, *mk_box_contents(size),
                                      num_players=num_players)))
