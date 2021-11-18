import pickle
from functools import reduce
from pathlib import Path
from typing import List

import numpy as np
from jass.game.const import (CLUBS, DIAMONDS, HEARTS, PUSH, SPADES, A_offset,
                             Eight_offset, J_offset, K_offset, Nine_offset,
                             Q_offset, Seven_offset, Six_offset, Ten_offset,
                             card_strings, color_of_card, trump_ints,
                             trump_strings_short)
from jass.game.game_observation import GameObservation
from tensorflow import keras


class Common:
    def __init__(self):
        self.card_strings = list(card_strings)
        self.int_to_trump = trump_ints + [PUSH] * 5

        current_working_dir = Path(__file__).parent.absolute()
        self.path_models = current_working_dir / Path("trained_model")
        self.initialize_model()

    def initialize_model(self):
        pass  # virtual function

    def choose_trump(self, obs: GameObservation) -> int:
        raise NotImplementedError  # deferred function

    def load_model_pickle(self, filename):
        with open(self.path_models / filename, "rb") as file:
            self.predictor = pickle.load(file)

    def load_model_keras(self, filename):
        self.predictor = keras.models.load_model(self.path_models / filename)

    def get_interaction_data(self, hand: np.array, interaction: str) -> List[bool]:
        return [reduce(lambda a, b: a & b, [hand[self.card_strings.index(color + feature)] == 1 for feature in interaction]) for color in "DHSC"]


class WithInteractions(Common):
    def choose_trump(self, obs: GameObservation) -> int:
        [trump] = self.predictor.predict([
            [card == 1 for card in obs.hand] +
            [obs.forehand != -1] +
            self.get_interaction_data(obs.hand, "J9") +
            self.get_interaction_data(obs.hand, "AKQ")
        ])
        return trump_strings_short.index(trump[0])


class NoInteractions(Common):
    def choose_trump(self, obs: GameObservation) -> int:
        [trump] = self.predictor.predict([
            [card == 1 for card in obs.hand] +
            [obs.forehand != -1]
        ])
        return trump_strings_short.index(trump[0])


class LogisticRegression(WithInteractions):
    def initialize_model(self):
        self.load_model_pickle("trump_logistic_regression.pkl")


class GradientBoosting(NoInteractions):
    def initialize_model(self):
        self.load_model_pickle("trump_gradient_boosting.pkl")


class DeepNeuralNetwork(Common):
    def initialize_model(self):
        self.load_model_keras("trump_deep_neural_network.h5")

    def choose_trump(self, obs: GameObservation) -> int:
        card_indices = np.flatnonzero(obs.hand)
        nr_cards = []
        for color in [DIAMONDS, HEARTS, SPADES, CLUBS]:
            nr_cards = nr_cards + \
                [float(sum(color_of_card[card] == color for card in card_indices)) / 9.]
        for card_value in [A_offset, K_offset, Q_offset, J_offset, Ten_offset, Nine_offset, Eight_offset, Seven_offset, Six_offset]:
            nr_cards = nr_cards + \
                [float(sum(card % 9 == card_value for card in card_indices)) / 4.]

        prediction = np.argsort(self.predictor.predict([
            list(obs.hand.astype(float)) +
            nr_cards +
            [float(obs.forehand + 1)]
        ]))[0][::-1]

        result = [self.int_to_trump[trump] for trump in prediction]
        return next(trump for trump in result if obs.forehand == -1 or trump != PUSH)
