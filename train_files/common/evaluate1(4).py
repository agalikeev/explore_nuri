import argparse
import csv
import json
import pathlib
import time

import ecole
import numpy as np

import sys

sys.path.insert(1, str(pathlib.Path.cwd()))
parser = argparse.ArgumentParser()
parser.add_argument(
    "problem",
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

from agents.dual import (
    Policy,
    ObservationFunction,
)  # agents.dual submissions.random.
from environments import Branching as Environment  # environments
from rewards import TimeLimitDualIntegral as BoundIntegral  # rewards

instances_path = pathlib.Path(f"/content/neur/Nuri/instances/0train/")
instance_files = list(instances_path.glob("*.mps.gz"))
inst = str(instance_files[0])
print(inst)

time_limit = 15 * 60

strbr = ecole.observation.StrongBranchingScores()
policy = Policy(problem=args.problem)

env = ecole.environment.Branching(observation_function=ObservationFunction(problem=args.problem))

observation, action_set, reward, done, info = env.reset(inst)
correct_predictions = 0
total_predictions = 0

while not done:
    policy_action = policy(action_set, observation)

    strbr_scores = strbr.extract(env.model, done)
    strbr_action = action_set[strbr_scores[action_set].argmax()]

    # не понмю, чем эти actionы являются, allcloes просто для примера
    #if torch.allclose(policy_action, strbr_action):
    if policy_action == strbr_action:
        correct_predictions += 1
    total_predictions += 1
    print("======================================")
    print(f"iteration {total_predictions}")
    print(f"policy_action: {policy_action}")
    print(f"strbr_action: {strbr_action}")
    print(f"current accruracy: {correct_predictions/total_predictions}")


    observation, action_set, reward, done, info = env.step(strbr_action)

print('accuracy of GNN', correct_predictions/total_predictions)
