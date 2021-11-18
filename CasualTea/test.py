# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging

import numpy as np
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
from jass.game.const import card_strings, color_masks
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

import play_strategy
import trump_strategy
from player import Jass

runs = 100


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=runs, print_every_x_games=1)

    # Team 1
    player = Jass(trump_strategy.LogisticRegression,
                  play_strategy.DeepNeuralNetwork)

    # Team 0
    my_player = Jass(trump_strategy.DeepNeuralNetwork,
                     play_strategy.DeepNeuralNetwork)

    arena.set_players(my_player, player, my_player, player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
