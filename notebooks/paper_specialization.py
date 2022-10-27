
# coding: utf-8

# # Notebook to compute the specialization for all seeds of a given experiment
# 
# * Set the correct number of landmarks
# * Set the number of agents in the population 
# * Set the results dir containing all the seeds to report

# In[17]:


import json
import os

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import pandas as pd
import seaborn
import math
import pickle
# In[18]:
params = {'legend.fontsize': 6,
                  "figure.autolayout": True,
                  'font.size': 8,
                  'pdf.fonttype':42,
                  'ps.fonttype':42}
plt.rcParams.update(params)

cm = 1 / 2.54  # for converting inches to cm
fig_size = (10.48 * cm, 6 * cm)
plt.figure(figsize=fig_size)

NUMBER_OF_LANDMARKS = 6
n_agents = 2
#results_dir = os.environ["SCRATCH"]+"/elias_expe/2_agents/cooperative/modified_reward/6_landmarks/independent"
results_dirs = {"3_landmarks": "/media/elena/LaCie/aamas_2023/paper/align_3_landmarks",
              "6_landmarks": "/media/elena/LaCie/aamas_2023/paper/align_6_landmarks"}
#results_dir = "/media/elena/LaCie/aamas_2023/paper/align_6_landmarks/independent"
#results_dir = "/media/elena/LaCie/elias_expe/2_agents/all/modified_reward/6_landmarks/centralized"



# In[ ]:


independent_3 = [0.4,0.5]
independent_6 = [0.1,0.1]
centralized_3 = [0.6,0.8]
centralized_6 = [0.7,0.76]
naming_3 = [0.8,0.8]
naming_6 = [0.9, 0.7]

"""
df = pd.DataFrame([["3 landmarks", "independent", independent_3[0]],
                   ["6 landmarks", "independent", independent_6[0]],
                   ["3 landmarks", "independent", independent_3[1]],
                   ["6 landmarks", "independent", independent_6[1]],
                   ["3 landmarks", "centralized", centralized_3[0]],
                   ["6 landmarks", "centralized",centralized_6[0]],
                    ["3 landmarks", "centralized", centralized_3[1]],
                   ["6 landmarks", "centralized",centralized_6[1]],
                   ["3 landmarks", "naming", naming_3[0]],
                   ["6 landmarks", "naming", naming_6[0]],
                    ["3 landmarks", "naming", naming_3[1]],
                   ["6 landmarks", "naming", naming_6[1]]],
                 columns = ["landmarks", "method", "special"])
"""

df = pd.DataFrame(columns= ["landmarks", "method", "special"])

for landmarks, methods_dir in results_dirs.items():
    print(landmarks, methods_dir)
    NUMBER_OF_LANDMARKS = int(landmarks[0])
    individual_goals = np.eye(NUMBER_OF_LANDMARKS, dtype=np.uint8).tolist()
    collective_goals = np.array(list(combinations(individual_goals, 2))).sum(1).tolist()
    goals = ["".join(str(t) for t in g) for g in collective_goals]
    goals_index = {i: g for i, g in zip(range(len(goals)), goals)}
    agents = [f"agent_{i}" for i in range(n_agents)]
    results_dir =  [os.path.join(methods_dir, o) for o in os.listdir(methods_dir) if os.path.isdir(methods_dir + "/" + o)]
    for method in results_dir:
        print("method is", method)
        if "independent" in method:
            method_name = "independent"
        if "centralized" in method:
            method_name = "centralized"
        if "game" in method:
            method_name = "coordination game"
        if "50align" not in method:
            print("method_name is",method_name)

            specializations_during_training = []
            specializations_convergence = []
            for subdir in os.listdir(method):
                if(os.path.isdir(method +"/"+subdir)):

                    full_path = os.path.join(method, subdir, "result.json")
                    print(full_path)
                    result_dump = open(full_path, "r")
                    # parse metrics
                    for result in result_dump:
                        # we always redefine this to only consider the last set of results (last evaluation)
                        metrics = json.loads(result).get("evaluation")
                        if metrics is not None:
                            print("loading")
                            for g in goals:
                                for agent in agents:
                                    if (
                                        metrics["hist_stats"].get(f"{agent} position for {g}")
                                        is not None
                                        and len(metrics["hist_stats"].get(f"{agent} position for {g}"))
                                        > 0
                                    ):
                                        aux = pd.DataFrame(
                                            metrics["hist_stats"].get(f"{agent} position for {g}")
                                        ).value_counts()
                                        specializations_during_training.append(
                                            aux.value_counts().max() / aux.value_counts().sum()
                                        )
                        special = np.mean(specializations_during_training[-len(goals) * n_agents :])
                        print(specializations_during_training[-len(goals) * n_agents :])


                        if special != None and not math.isnan(special):
                            new_row = {"landmarks": landmarks, "method": method_name, "special": special}
                            print(new_row)
                            df = df.append(new_row, ignore_index=True)

            
        
    

# view data
print(df)







# In[ ]:


# plot grouped bar chart
g=seaborn.catplot(data=df,kind="bar",x='landmarks', hue="method",y="special")
g.despine(left=True)
g.set_axis_labels("", "Specialization")
g.legend.set_title("")
plt.savefig("special.pdf")
plt.savefig("special.png")

with open("specialization.pkl", "wb") as f:
    pickle.dump(df, f)

