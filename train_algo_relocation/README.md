Training algorithm files with relocation

- convert_event_list_relocation.py: helper function to convert event lists with relocation options
- data_retrieval_30000_14151617: the instance provider
- dqn_kbh_colfax_relocation_v2-1-expensive-reloc: the DQN agent + main training loop with relocation
- kbh_yard_b2b_relocation_expensive_reloc.py: the TUSP environment with relocation

By running dqn_kbh_colfax_relocation_v2-1-expensive-reloc the training procedure can be initiated. 
Will save the model to summary/dqn_dummy/###### where ##### will be a number. 

Note: the data_retrieval_30000_14151617 loads a json file with 30000 problem 
instances from the folder ../fiddles/data_prep/...

Note2: the files in.json and out.json are a test instance, only used to develop the convert_event_list_relocation function.
 

SUMMARY 
In the summary folder the three NN parameter sets used to obtain the results are stored 
1525336829 - 150k training eps
1525445230 - 185k training eps
1526820866 - 220k training eps


ANALYSIS
In the analysis folder, the notebooks are stored that are used 
to obtain the results. 

- get_entropy_results_150k_185k_220k.ipynb is used to obtain resuls for entropy calculations
- get_solved_new_instances-24h-relocation-14151617-expensive_relocation-150k.ipynb is used to obtain the results for 150k model
- same for the 185k and 220k jupyter notebook. 
- folder 'test_solution_instances' contains files to test the results of the Local Search. 
- other files are to support the analysis, but everything relevant is shown in notebooks. 
