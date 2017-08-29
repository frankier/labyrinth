# -*- coding: utf-8 -*-
import sys

import numpy as np
import six
from six import StringIO

import gym
from gym import spaces
from gym.utils import seeding

from .board import (NUM_ORIENTATIONS, PATH_SYMBOLS, PLAYER_COLORS,
                    PLAYER_SYMBOLS, do_push, get_board_reachability,
                    get_possible_actions, is_valid_board_size, mk_box_contents,
                    num_lanes, reset_orientation)


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
        return any(self.player_has_won(player) for player in range(self.num_players))

    @property
    def winner(self):
        # TODO: Return the winning player (if the state is terminal)
        for player in range(self.num_players):
            if self.player_has_won(player):
                return player

    def player_has_won(self, player):
        (pos, cards, found) = self.players[player]
        return found == len(cards) and self.board_state[0][pos]['base'] == player

    def current_position(self):
        return self.player_positions[self.player_turn]

    def next_turn(self):
        return (self.player_turn + 1) % self.num_players

    def act_push(self, push):
        # Do push
        positions = [pos for (pos, _cards, _found) in self.players]
        new_board_state, new_positions = do_push(
                self.board_state, positions, push)

        # Make positions moves resulting from push
        new_players = [(new_pos, cards, found)
                       for ((old_pos, cards, found), new_pos)
                       in zip(self.players, new_positions)]

        return new_board_state, new_players

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            a new LabyrinthState with the new board and the player switched
        '''
        (push, move) = action

        new_board_state, new_players = self.act_push(push)

        # Check move is legal
        current_position = new_players[self.player_turn][0]
        reachability = get_board_reachability(
                new_board_state[0], current_position)
        if not reachability[move]:
            raise IllegalMove()

        # Update if treasure is found
        (_pos, cards, found) = self.players[self.player_turn]
        new_found = found
        if found < len(cards) and \
                cards[found] == new_board_state[0][move]['treasure']:
            new_found += 1

        # Make player move and update player treasures
        new_players[self.player_turn] = (move, cards, new_found)

        return LabyrinthState(
            new_board_state,
            self.next_turn(),
            new_players,
            self.num_treasures), new_found != found

    def observe(self, player):
        # TODO: Should return player x's observation of the state here rather
        # than the whole state (ie strictly we should delete information the
        # agent can't see)
        return self

    def item(self):
        def extract_tuple(dictish, keys):
            tuple(dictish[k] for k in keys)
        immutable_board = tuple(
            extract_tuple(cell, ['path_type', 'treasure', 'orientation'])
            for cell in self.board_state[0].flatten())
        spare_tile = extract_tuple(self.board_state[1],
                                   ('path_type', 'treasure'))
        immutable_state = (immutable_board, spare_tile, tuple(self.players))
        return immutable_state

    def get_possible_actions(self):
        """
        Convience method
        """
        return get_possible_actions(self.board_state, self.current_position())

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
                current = card_idx < found
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


def draw_board(board_state, player_positions=(), long_treasures=False):
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
    shuffle_rotate_tiles(np_random, to_place)
    place_idx = 0
    for x, y in np.ndindex(board.shape):
        if x % 2 == 1 or y % 2 == 1:
            board[x, y] = to_place[place_idx]
            place_idx += 1
    spare_tile = to_place[place_idx]
    reset_orientation(spare_tile)
    board_state = (board, spare_tile)
    turn = 0
    player_positions = []
    for pos in [0, 3, 1, 2][:num_players]:
        player_positions.append(((pos // 2) * far, (pos % 2) * far))
    player_cards = (
        tuple(cards) for cards in np.split(
            np_random.permutation(num_treasures), num_players))
    player_cards_found = [0] * num_players

    players = list(zip(player_positions, player_cards, player_cards_found))

    return LabyrinthState(board_state, turn, players, num_treasures)

# Adversary policies
def make_random_policy(np_random):
    pass


class IllegalMove(Exception):
    pass


class LabyrinthEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}
    def __init__(self, board_size, opponents, illegal_move_mode,
                 treasure_reward=0, win_reward=1,
                 illegal_reward=-1, loss_reward=-1,
                 opp_treasure_reward=0):
        assert is_valid_board_size(board_size)
        self.board_size = board_size

        for opponent in opponents:
            assert opponent in ['random']
        self.opponents = opponents

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        self.treasure_reward = treasure_reward
        self.win_reward = win_reward
        self.illegal_reward = illegal_reward
        self.loss_reward = loss_reward
        self.opp_treasure_reward = opp_treasure_reward

        # Properly initialised by _reset
        self.state = None

        push_space = spaces.Tuple((spaces.Discrete(4),
                                   spaces.Discrete(num_lanes(board_size)),
                                   spaces.Discrete(4)))
        move_space = spaces.Tuple((spaces.Discrete(board_size),
                                   spaces.Discrete(board_size)))

        self.action_space = spaces.Tuple((push_space, move_space))

    def _seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)
        return [self.seed]

    @property
    def num_players(self):
        return 1 + len(self.opponents)

    def _reset(self):
        board, mobile_tiles, num_treasures = \
                mk_box_contents(self.board_size, players=self.num_players)

        self.state = mk_initial_labyrinth_state(
            self.np_random, board, mobile_tiles, num_treasures,
            num_players=self.num_players)

        self.done = False

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
        #if action['type'] == 'resign':
            #self.done = True
            #return self.state.observe(0), -1., True, {'state': self.state}

        #assert action['type'] == 'move'

        # Play
        try:
            self.state, got_treasure = self.state.act(action)
        except IllegalMove:
            if self.illegal_move_mode == 'raise':
                six.reraise(*sys.exc_info())
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state.observe(0), -1., True, {'state': self.state}

        # Opponent play
        opp_treasures = 0
        for i, opponent in enumerate(self.opponents):
            if not self.state.is_terminal:
                (self.state, opp_got_treasure), opponent_resigned = self._exec_opponent_play(self.state)
                if opp_got_treasure:
                    opp_treasures += 1
                # After opponent play, we should be back to the original color
                assert self.state.color == self.player_color

                # If the opponent resigns, then the agent wins
                # TODO: Not true, should instead make player only win when all
                # opponents resign
                if opponent_resigned:
                    self.done = True
                    return self.state.observe(0), self.win_reward, True, {'state': self.state}

        # Reward: if nonterminal, then the reward is 0
        if not self.state.is_terminal:
            self.done = False
            reward = (got_treasure and self.treasure_reward) - opp_treasures * self.opp_treasure_reward
            return self.state.observe(0), 0., False, {'state': self.state}
        else:
            # We're in a terminal state. Reward is 1 if won, -1 if lost
            assert self.state.is_terminal
            self.done = True
            return self.state.observe(0), self.win_reward if self.state.winner == 0 else self.loss_reward, True, {'state': self.state}

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
