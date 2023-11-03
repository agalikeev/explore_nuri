import argparse
import csv
import json
import pathlib
import time

import ecole
import numpy as np

import sys

sys.path.insert(1, '/'.join(str(pathlib.Path.cwd()).split('/')[0:-1]))
parser = argparse.ArgumentParser()
parser.add_argument(
    "-problem",
    help="Problem benchmark to process.",
    choices=["policy2_64_0", "policy2_64_1", "policy2_64_2","policy2_64_3", \
    "policy2_128_0", "policy2_128_1", "policy2_128_2","policy2_128_3", \
    "policy2_256_0", "policy2_256_1", "policy2_256_2","policy2_256_3", \
    "policy3_64_0", "policy3_64_1", "policy3_64_2","policy3_64_3", \
    "policy3_128_0", "policy3_128_1", "policy3_128_2","policy3_128_3", \
    "policy3_256_0", "policy3_256_1", "policy3_256_2","policy3_256_3", \
    ],
)
args = parser.parse_args()

from dual import (
    Policy,
    ObservationFunction,
)  # agents.dual submissions.random.

from environments import Branching as Environment  # environments
from rewards import TimeLimitDualIntegral as BoundIntegral  # rewards

inst = "/content/explore_nuri/Nuri/instances/0train/mas76.mps.gz"
print(inst)

time_limit = 15 * 60

strbr = ecole.observation.StrongBranchingScores()
policy = Policy(problem=args.problem)

env = ecole.environment.Branching(observation_function=ObservationFunction(problem=args.problem))

observation, action_set, reward, done, info = env.reset(inst)
correct_predictions = 0
total_predictions = 0
rand_accuracy = 0
total_action_set = 0
sum_acc = 0
while not done:
    policy_action = policy(action_set, observation)

    strbr_scores = strbr.extract(env.model, done)
    strbr_action = action_set[strbr_scores[action_set].argmax()]

    #if torch.allclose(policy_action, strbr_action):
    if policy_action.item() == strbr_action:
        correct_predictions += 1
    total_predictions += 1
    total_action_set += len(action_set)
    rand_accuracy =  total_predictions / total_action_set
    policty_action_id = np.where(action_set == policy_action.item())[0][0]
    policy_score = strbr_scores[action_set][policty_action_id]
    sorted_strbr_scores = sorted(strbr_scores[action_set])
    policy_score_top = np.where(sorted_strbr_scores == policy_score)[0][0]
    sum_acc = sum_acc + policy_score_top / len(action_set)
    print("======================================")
    print(f"iteration: {total_predictions}")
    #print(f"strbr_scores[action_set]: {strbr_scores[action_set]}")
    print(f"sorted sb: {sorted_strbr_scores}")
    print(f"policy_score: {policy_score}")
    #print(f"len(sorted_strbr_scores):{len(sorted_strbr_scores)}, len(action_set): {len(action_set)}, len(strbr_scores[action_set]): {len(strbr_scores[action_set])}")
    #print(action_set, type(action_set), policty_action_id)
    #print(f"policy_score_top: {policy_score_top}, len(action): {len(action_set)}")

    print(f"policy_action: {policy_action}")
    print(f"strbr_action: {strbr_action}")
    #print(f"action_set: {action_set}")
    #print(f"strbr_scores: {strbr_scores[action_set]}")
    print(f"current accuracy: {correct_predictions/total_predictions}")
    print(f"random accuracy:  {rand_accuracy}")
    print(f"top_acc: {sum_acc/total_predictions}")

    observation, action_set, reward, done, info = env.step(strbr_action)

print('accuracy of GNN', correct_predictions/total_predictions)
print('top accuracy', sum_acc/total_predictions)
