

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from scipy.stats import f_oneway, tukey_hsd
import seaborn
# In[2]:


params = {'legend.fontsize': 6,
                  "figure.autolayout": True,
                  'font.size': 8,
                  'pdf.fonttype':42,
                  'ps.fonttype':42}
plt.rcParams.update(params)

cm = 1 / 2.54  # for converting inches to cm
fig_size = (10.48 * cm, 6 * cm)  # these dimensions chosen to fit in latex column
    

palette = list(seaborn.color_palette())


# * Set the directory in which all the experiments to be plotted together are stored.
# * Set the list of subdirs (experiments) you want to include in the plots

# In[3]:


all_experiments = ["/Users/eleninisioti/Desktop/workspace/playground/imgc-marl/projects/6_landmarks_reward4"]

labels_to_print = {"0align": "$0%$-align",
                   "independent": "$0%$-align",
                   "centralized": "$100%$-align",
                   "75align": "$75%$-align",
                     "25align": "$25%$-align",

                  "50align": "$50%$-align",
                  "100align": "$100%$-align",
                  "naming_game_30msg_alpha10_temp30": "GC-game"}
correct_order = ["$0%$-align", "$50%$-align", "$100%$-align","GC-game" ]


def produce_plots(all_experiments, labels_to_print, correct_order, best_reward, beta_values=[], max_step=100):
    best_reward_or=best_reward
    for experiments_dir in all_experiments:
        # ----- collect data -----
        list_of_experiments =  [o for o in os.listdir(experiments_dir) if os.path.isdir(experiments_dir + "/" + o)]
        results = {}
        for experiment in list_of_experiments:

            subdir = os.path.join(experiments_dir, experiment)
            eval_reward = pd.DataFrame()
            train_reward = pd.DataFrame()
            episode_len = pd.DataFrame()
            alignment = pd.DataFrame()
            train_x = []
            eval_x = []
            ctr=0
            for j, experiment_name in enumerate(os.listdir(subdir)):

                if(os.path.isdir(subdir+"/"+experiment_name)):

                    ctr+=1
                    if(experiment_name[:5]!="p_est" and experiment_name[:5]!="0_5di"):

                        if( True):
                            print(experiment_name)

                            r = []
                            r_t = []
                            l = []
                            x_ = []
                            a = []
                            y_ = []
                            try:
                                result_raw = open(os.path.join(subdir, experiment_name, "result.json"), "r")
                            except FileNotFoundError:
                                print("no file for ", subdir, experiment_name, "result.json")

                            for result in result_raw:
                                dump = json.loads(result)

                                #if(ctr>5):
                                    #print(dump.keys())
                                a.append(dump["custom_metrics"].get("goal_alignment_mean"))
                                y_.append(dump["timesteps_total"])
                                r_t.append(dump["episode_reward_mean"])

                                metrics = dump.get("evaluation")
                                if metrics is not None:
                                    custom = metrics.get("custom_metrics")
                                    x_.append(dump["timesteps_total"])
                                    r.append(metrics["episode_reward_mean"])
                                    l.append(metrics["episode_len_mean"])

                            eval_reward = pd.concat(
                                [eval_reward, pd.DataFrame(r)], ignore_index=True, axis=1
                            )
                            episode_len = pd.concat(
                                [episode_len, pd.DataFrame(l)], ignore_index=True, axis=1
                            )
                            alignment = pd.concat([alignment, pd.DataFrame(a)], ignore_index=True, axis=1)
                            train_reward = pd.concat(
                                [train_reward, pd.DataFrame(r_t)], ignore_index=True, axis=1
                            )


                            if len(x_) > len(eval_x):
                                eval_x = x_
                            if len(y_) > len(train_x):
                                train_x = y_
                                
            new_results = {
                "eval_reward": eval_reward,
                "train_reward": train_reward,
                "episode_len": episode_len,
                "alignment": alignment,
                "train_x": train_x,
                "eval_x": eval_x,
            }
            #new_results = new_results[new_results.train_x < max_step]
            results[experiment] = new_results

        # -------------------------------------------------------------------------------------------------------

        # ----- plot training -----
        fig, axs = plt.subplots(2, figsize=(fig_size[0], fig_size[1] *1.5), sharex=True)
        slice=max_step*10
        i = 1
        n_exp = len(results)
        label_idx = 0
        for label, result in results.items():
            color = palette[label_idx]
            label_idx += 1

            label_to_print = labels_to_print[label]
            
            if best_reward_or == 0:
                beta = beta_values[label_to_print]
                best_reward = (1/2*1/beta + 1/2)*2

            axs[0].plot(result["train_x"][:slice], [el/best_reward for el in result["train_reward"][:slice].mean(axis=1)], label=label_to_print, color=color)
            axs[0].fill_between(
                result["train_x"][:slice],
                [el/best_reward for el in result["train_reward"][:slice].mean(axis=1)] - result["train_reward"][:slice].std(axis=1),
                [el/best_reward for el in result["train_reward"][:slice].mean(axis=1)] + result["train_reward"][:slice].std(axis=1),
                alpha=0.4, color=color
            )
            if i == n_exp:
                handles, labels = axs[0].get_legend_handles_labels()
                order = []
                for el in correct_order:
                    for label_idx, actual_label in enumerate(labels):
                        if el ==actual_label:
                            print("actual label", actual_label)
                            order.append(label_idx)
                axs[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=ncols)
                #plt.grid()
                axs[0].set_ylabel("Training Reward, $R_{train}$")
                plt.xlabel("Timestep, \n $t_{train}$")

            axs[1].plot(result["train_x"][:slice], result["alignment"][:slice].mean(axis=1), label=label_to_print, color=color)
            axs[1].fill_between(
                result["train_x"][:slice],
                result["alignment"][:slice].mean(axis=1) - result["alignment"][:slice].std(axis=1),
                result["alignment"][:slice].mean(axis=1) + result["alignment"][:slice].std(axis=1),
                alpha=0.4, color=color
            )
            if i == n_exp:

                axs[1].set_ylabel("Alignment, $A$")
                plt.xlabel("Timestep,\n  $t_{train}$")

            i += 1

        plt.tight_layout()
        plt.savefig(os.path.join(experiments_dir, "training.png"), dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join(experiments_dir, "training.pdf"), dpi=300, bbox_inches='tight', transparent=True)
        plt.clf()
        
        
        # -------------------------------------------------------------------------------------------------------

        # ----- plot evaluation -----
        #best_reward=best_reward_or
        fig2, axs2 = plt.subplots(2, figsize=(fig_size[0], fig_size[1] *1.5), sharex=True)

        slice=max_step +1
        i = 1
        n_exp = len(results)
        label_idx = 0
        for label, result in results.items():
            label_to_print = labels_to_print[label]
            color = palette[label_idx]
            label_idx += 1
            
            if best_reward_or == 0:
                beta = beta_values[label_to_print]
                best_reward = (1/2*1/beta + 1/2)*2
                print("here",beta,label_to_print)

            axs2[0].plot(result["eval_x"][:slice], [el/best_reward for el in result["eval_reward"][:slice].mean(axis=1)], label=label_to_print, color=color)
            if not result["eval_reward"].empty:
                stat_tests_at = [10, 30, 70]
                points = []
                for stat_test in stat_tests_at:
                    points.append([stat_test, np.mean(result['eval_reward'].loc[stat_test])/best_reward])
                for point in points:
                    axs2[0].plot(result["eval_x"][point[0]], point[1], marker = 'o',markersize=5, color=color)
            axs2[0].fill_between(
                result["eval_x"][:slice],
                [el/best_reward for el in result["eval_reward"][:slice].mean(axis=1)] - result["eval_reward"][:slice].std(axis=1),
                [el/best_reward for el in result["eval_reward"][:slice].mean(axis=1)] + result["eval_reward"][:slice].std(axis=1),
                alpha=0.4, color=color
            )
            if i == n_exp:
                handles, labels = axs2[0].get_legend_handles_labels()
                order = []
                for el in correct_order:
                    for label_idx, actual_label in enumerate(labels):
                        if el ==actual_label:
                            order.append(label_idx)
                axs2[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=ncols)
                axs2[0].set_xlabel("Timestep, $t_{eval}$")
                axs2[0].set_ylabel("Evaluation Reward, $R_{eval}$")
                axs2[0].set_ylim(0,1.3)

            axs2[1].plot(result["eval_x"][:slice], result["episode_len"][:slice].mean(axis=1), label=label_to_print, color=color)
            if not result["episode_len"].empty:
                stat_tests_at = [10, 30, 70]
                points = []
                for stat_test in stat_tests_at:
                    points.append([stat_test, np.mean(result['episode_len'].loc[stat_test])])
                for point in points:
                    axs2[1].plot(result["eval_x"][point[0]], point[1], marker = 'o', markersize=5, color=color)
            axs2[1].fill_between(
                result["eval_x"][:slice],
                result["episode_len"][:slice].mean(axis=1) - result["episode_len"][:slice].std(axis=1),
                result["episode_len"][:slice].mean(axis=1) + result["episode_len"][:slice].std(axis=1),
                alpha=0.4, color=color
            )
            if i == n_exp:

                axs2[1].set_xlabel("Timestep, \n $t_{eval}$")
                axs2[1].set_ylabel("Episode length, $L$")
                axs2[1].set_ylim(0,500)


            i += 1

        plt.tight_layout()
        # Uncomment this line for saving the plot
        plt.savefig(os.path.join(experiments_dir, "eval.png"), dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join(experiments_dir, "eval.pdf"), dpi=300, bbox_inches='tight', transparent=True)

        return results


# produce figures for 4.1 (6 landmarks)
all_experiments = ["/Users/eleninisioti/Desktop/workspace/playground/imgc-marl/projects/6_landmarks_reward4"]

labels_to_print = {"0align": "$0\%$-align",
                   "independent": "$0\%$-align",
                   "centralized": "$100\%$-align",
                   "75align": "$75\%$-align",
                   "25align": "$25\%$-align",
                   "50align": "$50\%$-align",
                   "100align": "$100\%$-align",
                   "naming_game_30msg_alpha10_temp30": "Goal-coordination game"}
correct_order = ["$0\%$-align", "$50\%$-align", "$100\%$-align","Goal-coordination game" ]
beta = 2
best_reward = (6/21*1/beta + 15/21)*2
ncols = 1

results = produce_plots(all_experiments, labels_to_print, correct_order, best_reward, max_step=70)

def statistical_analysis(results):
    metrics  = ["eval_reward", "episode_len"]
    for metric in metrics:
        print("Metric ", metric)
        timesteps = list(range(0,71,10))
        for_significance_total = {timestep: [] for timestep in timesteps}
        names = []
        trials = 5
        beta = 2
        best_reward = (6 / 21 * 1 / beta + 15 / 21) * 2
        for key, value in results.items():
            label = labels_to_print[key]
            if label in correct_order:

                names.append(label)
                for timestep in timesteps:
                    timestep_values = value[metric].loc[timestep].tolist()

                    timestep_values = [el/best_reward for el in timestep_values]
                    for_significance_total[timestep].append(timestep_values)


        for timestep, for_significance in for_significance_total.items():
            print("Timestep ", timestep)

            F, p = f_oneway(*for_significance)
            if p < 0.05:
                print("significance", str(p))
                res = tukey_hsd(*for_significance)
                anal = str(res)
                print(anal)
                print(names)
            else:
                print("no significance")

            if timestep == 70:
                for method_idx, performance in enumerate(for_significance):
                    print(names[method_idx])
                    print(str(np.mean(performance)), str(np.std(performance)))





statistical_analysis(results)

# In[121]:


# produce figures for appendix (message size for 6 landmarks and beta=2)
all_experiments = ["/media/elena/LaCie/aamas_2023/paper/message_size"]
labels_to_print = {"naming_game_21msg_alpha10_temp30": "$M=21$",
                  "naming_game_30msg_alpha10_temp30": "$M=30$",
                  "naming_game_40msg_alpha10_temp30": "$M=40$"}
correct_order = ["$M=21$", "$M=30$", "$M=40$"]
ncols = 1
beta = 2
best_reward = (6/21*1/beta + 15/21)*2
produce_plots(all_experiments, labels_to_print, correct_order, best_reward)


# In[123]:



all_experiments = ["/media/elena/LaCie/aamas_2023/paper/multiplier/3_landmarks"]
labels_to_print = {"naming_game_reward1": "$\\beta=1$",
                   "naming_game_reward2": "$\\beta=2$",
                  "naming_game_reward4": "$\\beta=4$",
                  "naming_game_reward8": "$\\beta=8$"}
correct_order = ["$\\beta=1$", "$\\beta=2$", "$\\beta=4$", "$\\beta=8$"]
ncols = 2
beta_values = {"$\\beta=1$": 1, "$\\beta=2$": 2, "$\\beta=4$":4, "$\\beta=8$":8 }
produce_plots(all_experiments, labels_to_print, correct_order, best_reward=0, beta_values=beta_values,max_step=10)


# In[124]:


all_experiments = ["/media/elena/LaCie/aamas_2023/all/3_landmarks//3_landmarks_reward4"]
labels_to_print = {"0align": "$0\%$-align",
                   "independent": "$0\%$-align",
                   "centralized": "$100\%$-align",
                   "75align": "$75\%$-align",
                     "25align": "$25\%$-align",

                  "50align": "$50\%$-align",
                  "100align": "$100\%$-align",
                  "naming_game_30msg_alpha10_temp30": "Goal-coordination game"}
beta=4
best_reward = (1/2*1/beta + 1/2)*2

correct_order = ["$0\%$-align", "$50\%$-align", "$100\%$-align","Goal-coordination game" ]
produce_plots(all_experiments, labels_to_print, correct_order, best_reward, max_step=10)


# In[ ]:




