from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class Jass(Agent):
    def __init__(self, trump_strategy, play_strategy):
        self.trump_strategy = trump_strategy()
        self.play_strategy = play_strategy(RuleSchieber())

    def action_trump(self, obs: GameObservation) -> int:
        return self.trump_strategy.choose_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        return self.play_strategy.choose_card(obs)
