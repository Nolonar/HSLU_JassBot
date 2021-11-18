import datetime
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from jass.game.const import (UNE_UFE, color_of_card, lower_trump, next_player,
                             same_team)
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from tensorflow import keras


class Common:
    def __init__(self, rule: GameRule):
        self.rule = rule
        self.rng = np.random.default_rng()
        self.initialize_parameters()

    def initialize_parameters(self):
        pass  # virtual function

    def choose_card(self, obs: GameObservation) -> int:
        raise NotImplementedError  # deferred function

    def get_cards_remaining(self, obs: GameObservation) -> np.array:
        result = np.ones(shape=36, dtype=np.int32) - obs.hand
        result[obs.tricks[np.where(obs.tricks >= 0)]] = 0
        return np.flatnonzero(result)

    def get_playable_cards(self, obs) -> np.array:
        return self.rule.get_valid_cards_from_obs(obs)

    def get_playable_card_indices(self, obs) -> np.array:
        return np.flatnonzero(self.get_playable_cards(obs))

    def get_current_trick(self, obs) -> np.array:
        return obs.current_trick[np.where(obs.current_trick >= 0)]

    def group_cards(self, hand, trump) -> Tuple[List[int], List[int]]:
        regular_cards = []
        trump_cards = []
        for card in hand:
            to_append_to = trump_cards if color_of_card[card] == trump else regular_cards
            to_append_to.append(card)

        return regular_cards, trump_cards


class Rule(Common):
    def choose_card(self, obs: GameObservation) -> int:
        trick = self.get_current_trick(obs)
        hand = self.get_playable_card_indices(obs)
        regular_cards, trump_cards = self.group_cards(hand, obs.trump)

        card = self.get_strongest_card(regular_cards, trick, obs.trump)
        if card is None:
            card = self.get_strongest_card(trump_cards, trick, obs.trump)

        return card if card is not None else self.get_weakest_card(hand, obs.trump)

    def get_weakest_card(self, hand: np.array, trump: int) -> int:
        if len(hand) == 0:
            return None

        results = [hand[0]]
        weakest_value = hand[0] % 9
        is_trump_weakest = color_of_card[hand[0]] == trump
        is_value_reversed = trump == UNE_UFE
        for card in hand[1:]:
            color = color_of_card[card]
            if color == trump and not is_trump_weakest:
                continue

            value = card % 9
            if value == weakest_value:
                results.append(card)
            elif (value > weakest_value) != is_value_reversed:
                results = [card]
                weakest_value = value
                is_trump_weakest = color == trump

        return results[0] if len(results) == 1 else self.rng.choice(results)

    def get_strongest_card(self, hand: np.array, trick: np.array, trump: int) -> Optional[int]:
        return next((card for card in reversed(hand) if self.can_win(card, trick, trump)), None)

    def can_win(self, card: int, trick: np.array, trump: int) -> bool:
        if trick.shape[0] == 0:
            return True

        color = color_of_card[card]
        trumps = [c for c in trick if color_of_card[c] == trump]
        if len(trumps) > 0:
            if color != trump:
                return False

            return all(lower_trump[card, c] for c in trumps)

        trick_color = color_of_card[trick[0]]
        if color != trick_color:
            return False

        is_value_reversed = trump == UNE_UFE
        cards_to_beat = [c for c in trick if color_of_card[c] == trick_color]
        return all((c > card) != is_value_reversed for c in cards_to_beat)


class Monte_Carlo_Tree_Search(Common):
    def initialize_parameters(self):
        self.timeout_seconds = 9
        self.c = math.sqrt(2)

    def choose_card(self, obs: GameObservation) -> int:
        end_time = datetime.timedelta(seconds=self.timeout_seconds)
        start_time = datetime.datetime.utcnow()

        hand = self.get_playable_card_indices(obs)
        if len(hand) == 1:  # speedup
            return hand[0]

        cards_remaining = self.get_cards_remaining(obs)

        nr_tries = len(hand)
        tries = [1 for card in hand]
        score = [self.get_points(
            card, cards_remaining.tolist(), obs) for card in hand]
        ucb1 = [self.get_upper_confidence_bound(
            s, t, nr_tries) for s, t in zip(score, tries)]

        while datetime.datetime.utcnow() - start_time < end_time:
            i = ucb1.index(max(ucb1))
            nr_tries = nr_tries + 1
            tries[i] = tries[i] + 1
            score[i] = score[i] + self.get_points(
                hand[i], cards_remaining.tolist(), obs)
            ucb1[i] = self.get_upper_confidence_bound(
                score[i], tries[i], nr_tries)

        return hand[tries.index(max(tries))]

    def get_points(self, card_played: int, cards_remaining: List[int], obs: GameObservation) -> int:
        result = 0

        self.rng.shuffle(cards_remaining)
        obs = self.get_observation_clone(obs)
        me = obs.player

        obs.hand[card_played] = 0
        self.update_observation(obs, card_played)

        while obs.nr_tricks < 9:
            is_last = obs.nr_tricks == 8
            while obs.nr_cards_in_trick < 4:
                if obs.player == me:
                    valid_cards = self.rule.get_valid_cards_from_obs(obs)
                    card_played = self.rng.choice(np.flatnonzero(valid_cards))
                    obs.hand[card_played] = 0
                else:
                    card_played = cards_remaining.pop()

                self.update_observation(obs, card_played)

            obs.player = self.rule.calc_winner(
                obs.current_trick, obs.trick_first_player[obs.nr_tricks], obs.trump)
            if same_team[me][obs.player]:
                result = result + self.rule.calc_points(
                    obs.current_trick, is_last, obs.trump)

            obs.current_trick = np.full(shape=4, fill_value=-1, dtype=np.int32)
            obs.nr_tricks = obs.nr_tricks + 1
            obs.nr_cards_in_trick = 0
            if not is_last:
                obs.trick_first_player[obs.nr_tricks] = obs.player

        return result

    def update_observation(self, obs: GameObservation, card_played: int):
        obs.current_trick[obs.nr_cards_in_trick] = card_played
        obs.nr_cards_in_trick = obs.nr_cards_in_trick + 1
        obs.player = next_player[obs.player]

    def get_observation_clone(self, obs: GameObservation) -> GameObservation:
        result = GameObservation()
        result.player = obs.player
        result.hand = np.copy(obs.hand)
        result.trump = obs.trump
        result.current_trick = np.copy(obs.current_trick)
        result.trick_first_player = np.copy(obs.trick_first_player)
        result.nr_cards_in_trick = obs.nr_cards_in_trick
        result.nr_tricks = obs.nr_tricks
        return result

    def get_upper_confidence_bound(self, score: int, tries: int, tries_total: int):
        return score / tries + self.c * math.sqrt(math.log(tries_total) / tries)


class Monte_Carlo_Tree_Search_Fast(Monte_Carlo_Tree_Search):
    def initialize_parameters(self):
        super().initialize_parameters()
        self.timeout_seconds = 1


class DeepNeuralNetwork(Common):
    def initialize_parameters(self):
        self.load_model("play_deep_neural_network.h5")

    def load_model(self, filename):
        path = Path(__file__).parent.absolute() / Path("trained_model")
        self.predictor = keras.models.load_model(path / filename)

    def choose_card(self, obs: GameObservation) -> int:
        hand = list(obs.hand.astype(float))
        playable_hand = list(self.get_playable_cards(obs).astype(float))
        used_cards = list(self.get_used_cards(obs).astype(float))
        current_trick = list(self.get_current_trick(obs).astype(float))
        color = list(self.get_color_of_first_card(obs).astype(float))
        trump = list(self.get_trump(obs).astype(float))
        played_now = [float(obs.nr_cards_in_trick) / 3.0]
        played_total = [float(obs.nr_played_cards) / 35.0]

        prediction = np.argsort(self.predictor.predict([
            hand +
            playable_hand +
            used_cards +
            current_trick +
            color +
            trump +
            played_now +
            played_total
        ]))[0][::-1]

        playable = list(self.get_playable_card_indices(obs))
        return next(card for card in prediction if card in playable)

    def get_used_cards(self, obs: GameObservation) -> np.array:
        used_cards = obs.tricks.flat
        result = np.zeros(shape=36, dtype=np.int32)
        result[used_cards[used_cards >= 0]] = 1
        return result

    def get_current_trick(self, obs: GameObservation) -> np.array:
        result = np.zeros(shape=36, dtype=np.int32)
        result[obs.current_trick[obs.current_trick >= 0]] = 1
        return result

    def get_color_of_first_card(self, obs: GameObservation) -> np.array:
        result = np.zeros(shape=4, dtype=np.int32)
        first_card = obs.current_trick[0]
        if first_card >= 0:
            result[color_of_card[first_card]] = 1
        return result

    def get_trump(self, obs: GameObservation) -> np.array:
        result = np.zeros(shape=6, dtype=np.int32)
        result[obs.trump] = 1
        return result


class Test(Rule):
    pass  # use this to test modifications to another strategy.
