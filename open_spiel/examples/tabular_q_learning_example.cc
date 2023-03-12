// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <vector>
#include <fstream>  // For ifstream/ofstream.

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/tabular_q_learning.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/bots/human/human_bot.h"
#include "open_spiel/policy.h"

using open_spiel::Action;
using open_spiel::Game;
using open_spiel::Player;
using open_spiel::State;

Action GetOptimalAction(
    absl::flat_hash_map<std::pair<std::string, Action>, double> q_values,
    const std::unique_ptr<State>& state) {
  std::vector<Action> legal_actions = state->LegalActions();
  Action optimal_action = open_spiel::kInvalidAction;

  double value = -1;
  for (const Action& action : legal_actions) {
    double q_val = q_values[{state->ToString(), action}];
    if (q_val >= value) {
      value = q_val;
      optimal_action = action;
    }
  }
  return optimal_action;
}

void SolveTicTacToe() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("tic_tac_toe");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  // std::vector<double> rewards;
  int iter = 1000; // 1000000
  while (iter-- > 0) {
    tabular_q_learning_solver.RunIteration();

    if (iter % 10 == 0) {
    // if (true) {
      // rewards.push_back(tabular_q_learning_solver.GetLastReward());
      const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
        tabular_q_learning_solver.GetQValueTable();
      const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values_info_state =
        tabular_q_learning_solver.GetQValueTableInfoState();

      // Explored states
      std::string state;
      Action action;
      std::vector<std::string> exploredStates;
      for (auto q_table_cell: q_values_info_state) {
        std::pair<std::string, Action> table_key = q_table_cell.first;
        state = table_key.first;
        action = table_key.second;
        exploredStates.push_back(state);
        // std::cout << state << " State: " << state << std::endl;
      }

      // Populate state-action table
      std::string table_state;
      Action table_action;
      std::unordered_map<std::string, Action> action_map;
      for (std::string explored_state: exploredStates) {
        table_action = tabular_q_learning_solver.GetBestActionFromInfoState(explored_state);
        action_map[explored_state] = table_action;
      }
      open_spiel::TabularPolicy dummy_policy_;
      dummy_policy_ = open_spiel::GetUniformPolicy(*game);
      open_spiel::TabularPolicy tabular_policy_ = open_spiel::TabularPolicy(dummy_policy_, action_map);

      // Print action-probs
      std::unordered_map<std::string, open_spiel::ActionsAndProbs> policy_table = tabular_policy_.PolicyTable();
      // std::cout << policy_table.size() << std::endl;
      for (auto policy_state: policy_table) {
        std::string state = policy_state.first;
        open_spiel::ActionsAndProbs act_probs = policy_state.second;
        // std::cout << act_probs.size() << std::endl;
        for (auto act_prob: act_probs) {
          // std::cout << state << " Action: " << act_prob.first << " Prob: " << act_prob.second << std::endl;
        }
      }

      double nash_conv = open_spiel::algorithms::NashConv(*game, tabular_policy_);
      std::cout << nash_conv << std::endl;
    }

  }
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();

  tabular_q_learning_solver.storeQTableCSVFile();

  std::unique_ptr<open_spiel::Bot> human_bot = std::make_unique<open_spiel::HumanBot>();
  open_spiel::Player current_player;

  std::unique_ptr<State> state = game->NewInitialState();
  while (!state->IsTerminal()) {
    current_player = state->CurrentPlayer();
    if (current_player == 0) {
      state->ApplyAction(human_bot->Step(*state));
    } else if (current_player == 1) {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
    } else {
      continue;
    }
    std::cout << "Player " << current_player << std::endl;
    std::cout << "Next state:\n" << state->ToString() << std::endl;
  }
}

void SolveCatch() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("catch");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int training_iter = 100000;
  while (training_iter-- > 0) {
    tabular_q_learning_solver.RunIteration();
  }
  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();

  int eval_iter = 1000;
  int total_reward = 0;
  while (eval_iter-- > 0) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      Action optimal_action = GetOptimalAction(q_values, state);
      state->ApplyAction(optimal_action);
      total_reward += state->Rewards()[0];
    }
  }

  SPIEL_CHECK_GT(total_reward, 0);
}

int main(int argc, char** argv) {
  SolveTicTacToe();
  // SolveCatch();
  return 0;
}
