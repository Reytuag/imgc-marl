from itertools import combinations, product

import numpy as np

landmarks = int(input("Enter the number of landmarks: "))

all_goals = input("Should we consider all goals? (y/n) ")

all_goals = all_goals == "y"

if all_goals:
    individual_goals = np.eye(landmarks, dtype=np.uint8).tolist()
    collective_goals = np.array(list(combinations(individual_goals, 2))).sum(1).tolist()
    goal_space = individual_goals + collective_goals
else:
    individual_goals = np.eye(landmarks, dtype=np.uint8).tolist()
    goal_space = np.array(list(combinations(individual_goals, 2))).sum(1).tolist()

all_pairs = list(product(goal_space, goal_space))
compatible_pairs = [pair for pair in all_pairs if np.bitwise_or.reduce(pair).sum() <= 2]

print("Number of goals: ", len(goal_space))
print("Number of goal pairs ", len(all_pairs))
if all_goals:
    print(
        "Ratio of Collective / Individual goals ",
        len(collective_goals) / len(individual_goals),
    )
print("Probability of alignment ", 1 / len(goal_space))
print(
    "Ratio of compatible pairs / total pairs ", len(compatible_pairs) / len(all_pairs)
)
