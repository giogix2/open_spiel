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

#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/tabular_q_learning.h"
#include "open_spiel/games/squadro.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/bots/human/human_bot.h"

using open_spiel::Action;
using open_spiel::Game;
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

void SolveSquadro() {
  std::shared_ptr<const Game> game = open_spiel::LoadGame("squadro");
  open_spiel::algorithms::TabularQLearningSolver tabular_q_learning_solver(
      game);

  int iter = 1000000;
  while (iter-- > 0) {
    tabular_q_learning_solver.RunIteration();
    if (iter % 10000 == 0) {
      std::cout << iter << std::endl;
      tabular_q_learning_solver.storeQTableCSVFile();
    }
  }
  tabular_q_learning_solver.storeQTableCSVFile();

  std::unique_ptr<open_spiel::Bot> human_bot = std::make_unique<open_spiel::HumanBot>();
  open_spiel::Player current_player;

  const absl::flat_hash_map<std::pair<std::string, Action>, double>& q_values =
      tabular_q_learning_solver.GetQValueTable();
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

int main(int argc, char** argv) {
  SolveSquadro();
  return 0;
}