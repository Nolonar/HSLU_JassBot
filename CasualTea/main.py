from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.rule_schieber import RuleSchieber
from jass.service.player_service_app import PlayerServiceApp

import play_strategy
import trump_strategy
from player import Jass


def create_app():
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player("random", AgentRandomSchieber())  # random
    # The default casualT bot is the currently "best" variant of casualT
    app.add_player("casualT", Jass(
        trump_strategy.LogisticRegression,
        play_strategy.Monte_Carlo_Tree_Search
    ))
    app.add_player("casualT_Rule", Jass(
        trump_strategy.LogisticRegression,
        play_strategy.Rule
    ))
    app.add_player("casualT_MCTS", Jass(
        trump_strategy.LogisticRegression,
        play_strategy.Monte_Carlo_Tree_Search
    ))
    app.add_player("casualT_MCTS_fast", Jass(
        trump_strategy.LogisticRegression,
        play_strategy.Monte_Carlo_Tree_Search_Fast
    ))
    # The default specialT bot is the currently "best" variant of specialT
    app.add_player("specialT", Jass(
        trump_strategy.LogisticRegression,
        play_strategy.DeepNeuralNetwork
    ))
    app.add_player("specialT_LogReg", Jass(
        trump_strategy.LogisticRegression,
        play_strategy.DeepNeuralNetwork
    ))
    app.add_player("specialT_Deep", Jass(
        trump_strategy.DeepNeuralNetwork,
        play_strategy.DeepNeuralNetwork
    ))

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0")
