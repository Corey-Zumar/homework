## Exercise 2 (Warmup)


## Exercise 3

### Section 1

### Ants
To generate returns for the ants model, simply run the following from the `hw1_soln` directory:
```sh
cd BC/ants
python eval.py Ant-v1 --model_path model.hd5 --num_rollouts 20 --mean_path norm_info.sav
```
where `model.hd5` is the included ants model. You will obtain a file called `ants_model_returns.sav` containing
the pickled returns data.

To obtain the ants model, execute
```sh
python run_trainer.py
```
This will output a new model after every training epoch. After about 20 epochs, any model that you select should perform well enough to
replicate the returns listed in the writeup. Additionally, this script will output pickled mean standardization data used by `eval.py` 
as a file called `norm_data.sav`.

To generate returns for the ants expert, simply run the following from the `hw1_soln` directory:
```sh
cd data_collection/
python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20
```
A file called `expert_returns.sav` will be generated in the `hw1_soln/data_collection/data` directory containing pickled
returns data.

### Walker
To generate returns for the walker model, simply run the following from the `hw1_soln` directory:
```sh
cd BC/walker
python eval.py Walker2d-v1 --model_path model.hd5 --num_rollouts 20 --mean_path norm_info.sav
```
where `model.hd5` is the included Walker2d model. You will obtain a file called `walker_model_returns.sav` containing
the pickled returns data.

To obtain the Walker2d model, execute
```sh
python run_trainer.py
```
This will output a new model after every training epoch. After about 20 epochs, any model that you select should perform well enough to
replicate the returns listed in the writeup. Additionally, this script will output pickled mean standardization data used by `eval.py` 
as a file called `norm_data.sav`.

To generate returns for the Walker2d expert, simply run the following from the `hw1_soln` directory:
```sh
cd data_collection/
python run_expert.py experts/Walker2d-v1.pkl Walker2d-v1 --num_rollouts 20
```
A file called `expert_returns.sav` will be generated in the `hw1_soln/data_collection/data` directory containing pickled
returns data.

### Section 2
To replicate the results included in the writeup, begin by generating expert data for training on the ants environment. To do so, invoke
the following from the `hw1_soln` directory:

```sh
cd data_collection
python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 50
```
A file called `expert_data.sav` will be generated in the `hw1_soln/data_collection/data` directory containing pickled training data.
Next, run the following from the `hw1_soln` directory:

```sh
cd BC/hyperparam_experiments
python epoch_experiments.py --data_path /path/to/generated/training/data --num_epochs 20
```
This will output a pickled list of dictionaries called `epoch_experiment_returns.sav`. The *n*th dictionary entry 
corresponds to the return data for 3 rollouts executed after *n* training epochs. To plot this data, simply execute the 
following from `hw1_soln/BC/hyperparam_experiments`:

```sh
python plot_returns.py --returns_path epoch_experiment_returns.sav
```

## Exercise 4

### Section 2
To replicate the results included in the writeup, begin by generating expert data for training on the Walker2d environment. To do so,
invoke the following from the `hw1_soln` directory:

```sh
cd data_collection
python run_expert.py experts/Walker2d-v1.pkl Walker2d-v1 --num_rollouts 50
```
A file called `expert_data.sav` will be generated in the `hw1_soln/data_collection/data` directory containing pickled training data.
Next, locate the pickled Walker2d expert. This file's path is `hw1_soln/data_collections/expert/Walker2d-v1.pkl`. Now, execute the
following from the `hw1_soln` directory:

```sh
cd dagger
python run_dagger.py --expert_policy_file /path/to/expert/policy/file --data_path /path/to/expert/data \
--num_rollouts=10 --num_dagger_iters=10
```
This will output two pickled lists of dictionary objects: `dagger_expert_returns.sav` and `dagger_walker_returns.sav`.
The first file contains expert data, while the second file contains data for a dagger run on a smaller neural net
for behavioral cloning (the structure of which can be examined in `run_dagger.py`). These are pickled lists of dictionaries.
The *n*th dictionary entry corresponds to the return data for `num_rollouts` rollouts executed after *n* iterations
of dagger.

To plot this data, simply execute the following from `hw1_soln/dagger`:
```sh
python plot_dagger.py --expert_returns_path dagger_expert_returns.sav --model_returns_path dagger_walker_returns.sav
```
