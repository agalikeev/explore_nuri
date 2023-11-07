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

inst = "/content/explore_nuri/Nuri/instances/train_test/mas76.mps.gz"
print(f"instance:{inst}")

time_limit = 15 * 60

strbr = ecole.observation.StrongBranchingScores()
policy = Policy(problem=args.problem)

env = ecole.environment.Branching(observation_function=ObservationFunction(problem=args.problem))

observation, action_set, reward, done, info = env.reset(inst)
correct_predictions = 0
total_predictions = 0
rand_accuracy = 0

sum_metric1 = 0 # policy_num / strbr_num
sum_metric2 = 0 # policy_score / strbr_score
sum_rand_metric1 = 0
sum_rand_metric2 = 0

sum_err = 0
sum_rand_err = 0
total_action_set = 0
sum_acc = 0

while not done:

    policy_action = policy(action_set, observation)
    strbr_scores = strbr.extract(env.model, done)
    strbr_action = action_set[strbr_scores[action_set].argmax()]

    if policy_action.item() == strbr_action:
        correct_predictions += 1
    total_predictions += 1
    total_action_set += len(action_set)
    rand_accuracy =  total_predictions / total_action_set

    policty_action_id = np.where(action_set == policy_action.item())[0][0]
    policy_score = strbr_scores[action_set][policty_action_id]
    sorted_strbr_scores = sorted(strbr_scores[action_set])
    policy_score_top = np.where(sorted_strbr_scores == policy_score)[0][0]
    
    sum_metric1 += policy_score_top / len(action_set)
    sum_metric2 += policy_score / sorted_strbr_scores[-1]

    sum_rand_metric1 += 0.5 * (len(action_set) + 1) / len(action_set)
    sum_rand_metric2 += sum(sorted_strbr_scores) / (len(sorted_strbr_scores) * sorted_strbr_scores[-1])

    sum_err +=(sorted_strbr_scores[-1] - policy_score)
    sum_rand_err += (sorted_strbr_scores[-1] - sum(sorted_strbr_scores) / len(sorted_strbr_scores))
    print("======================================")
    print(f"iteration: {total_predictions}")
    #print(f"strbr_scores[action_set]: {strbr_scores[action_set]}")
    print(f"sorted sb: {sorted_strbr_scores}")
    print(f"policy_score: {policy_score}")
    print(f"strbr_score: {sorted_strbr_scores[-1]}")
    print(f"policy_score_top {policy_score_top}")

    print(f"policy_action: {policy_action}")
    print(f"strbr_action: {strbr_action}")
    print(f"action_set: {action_set}")

    print(f"current rough accuracy: {correct_predictions/total_predictions}")
    print(f"random accuracy:  {rand_accuracy}")

    print(f"metric1: {sum_metric1 / total_predictions}")
    print(f"rand_metric1: {sum_rand_metric1 / total_predictions}")

    print(f"metric2: {sum_metric2 / total_predictions}")
    print(f"rand_metric2: {sum_rand_metric2 / total_predictions}")

    print(f"sum_err: {sum_err/total_predictions}")
    print(f"rand_sum_err: {sum_rand_err / total_predictions}")
    observation, action_set, reward, done, info = env.step(strbr_action)

print(f"total rough accuracy {correct_predictions/total_predictions}")
print(f"total random rought accuracy:  {rand_accuracy}")

print(f"total metric1: {sum_metric1 / total_predictions}")
print(f"total rand_metric1: {sum_rand_metric1 / total_predictions}")

print(f"total metric2: {sum_metric2 / total_predictions}")
print(f"total rand_metric2: {sum_rand_metric2 / total_predictions}")
