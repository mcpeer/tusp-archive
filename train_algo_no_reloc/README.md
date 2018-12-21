Training algorithm files

- convert_event_list.py: helper function to convert event lists
- data_retrieval_30000_14151617: the instance provider
- dqn_kbh_colfax_30000_instances_14151617: the DQN agent + main training loop
- kbh_yard_b2b_clean.py: the TUSP environment 

By running dqn_kbh_colfax_30000_instances_14151617w the training procedure can be initiated. 
Will save the model to summary/dqn_dummy/###### where ##### will be a number. 

Note: the data_retrieval_30000_14151617 loads a json file with 30000 problem 
instances from the folder ../fiddles/data_prep/...

Note2: the files in.json and out.json are a test instance, only used to develop the convert_event_list function.
 
 ---

In the summary folder the 4 models used to generate the results for the paper are stored. 

1521137844
1521137935
1521138059
1521138212

 --------

ANALYSIS
in the analysis folder, 6 ipython notebooks are present that were used to obtain the 
results present in the paper and in the no relocation part of the thesis. 

- paper_test_results_all_runs.ipynb: solves the instances using the four agents, saves data to .csv for other notebooks
- paper_greedy_results.ipynb: solves instances using greedy agent. 
- paper_test_best_run_get_solved_instances.ipynb: solves the instances using only the best agent. 

- paper_test_compute_consistent_greedy_entropies.ipynb: computers entropies for greedy results
- paper_test_compute_entropies.ipynb: computes entropies for local search and DRL agent.

- paper_test_histogram_train_types_steps.ipynb: 'visualizes' used strategy by the agent. 



Note: in this work we extracted the actions of the local search by accessing a SQL database directly,
using the python code in the folder convert_solutions_roel. This connection package has been written by Shijian Zhong. 
In the 'relocation' part we used raw output of the local search algorithm to read in the solutions. 
