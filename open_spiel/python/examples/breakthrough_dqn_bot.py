# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN bot example."""

import random
import sys

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.algorithms import dqn
import pyspiel

flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_string("player1_checkpoint_dir", None, "Path to the dqn_net weigths for player 1.")
flags.DEFINE_string("player2_checkpoint_dir", None, "Path to the dqn_net weigths for player 1.")

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)

def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None

def main(argv):
  game_name = "breakthrough"
  game_configs = {"columns": 5, "rows": 5}
  game = pyspiel.load_game(game_name, game_configs)
  print(FLAGS.player1_checkpoint_dir)
  print(FLAGS.player2_checkpoint_dir)
  
  bots = [
      dqn.DQNBot(game, player_id=0, checkpoint_dir=FLAGS.player1_checkpoint_dir),
      dqn.DQNBot(game, player_id=1, checkpoint_dir=FLAGS.player2_checkpoint_dir),
  ]
  state = game.new_initial_state()
  _opt_print("Initial state:\n{}".format(state))

  history = []

  initial_actions = [state.action_to_string(
        state.current_player(), random.choice(state.legal_actions()))]
  
  for action_str in initial_actions:
    action = _get_action(state, action_str)
    if action is None:
      sys.exit("Invalid action: {}".format(action_str))
    
    history.append(action_str)

    for bot in bots:
      bot.inform_action(state, state.current_player(), action)
    state.apply_action(action)
    _opt_print("Forced action", action_str)
    _opt_print("Next state:\n{}".format(state))

  while not state.is_terminal():
    current_player = state.current_player()
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      _opt_print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Sampled action: ", action_str)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      # Decision node: sample action for the single current player
      bot = bots[current_player]
      action = bot.step(state)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Player {} sampled action: {}".format(current_player,
                                                       action_str))
    
    for i, bot in enumerate(bots):
      if i != current_player:
        bot.inform_action(state, current_player, action)
    history.append(action_str)
    state.apply_action(action)

    _opt_print("Next state:\n{}".format(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))
  


if __name__ == "__main__":
  app.run(main)

