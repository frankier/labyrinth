from gym.envs.registration import register

# Just do registration here for now - might be better somewhere else. Note this
# must be after LabyrinthEnv since otherwise we will have import problems. This
# is method may be hacky/fragile so should be fixed at some point.


for board_size in [3, 5, 7]:
    for treasure_reward in [0, 1]:
        register(
            id='Labyrinth{0}x{0}-tr{1}-v0'.format(board_size, treasure_reward),
            entry_point='labyrinth.env:LabyrinthEnv',
            kwargs={
                'opponents': [],
                'illegal_move_mode': 'lose',
                'board_size': board_size,
                'treasure_reward': treasure_reward,
            },
        )
