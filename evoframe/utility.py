import dill as pickle
import os
import shutil
import numpy as np

def exist_file(filename):
    return os.path.exists(filename)

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def pickle_save(obj, filename='obj.pkl'):
    directories = "/".join(filename.split("/")[:-1])
    maybe_make_dir(directories)
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def pickle_load(filename='obj.pkl'):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)

def maybe_delete_dir(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)

def clean_experiment_directory(experiment_name):
    maybe_delete_dir("experiments/{}".format(experiment_name))

def indexes_of_epoch(epoch, pop_size):
    return pop_size * (epoch - 1), pop_size * epoch

def pickle_save_pop_size(experiment_name, pop_size):
    pickle_save(pop_size, filename="experiments/{}/pop_size.pkl".format(experiment_name))

def pickle_load_pop_size(experiment_name):
    return pickle_load(filename="experiments/{}/pop_size.pkl".format(experiment_name))

def pickle_save_num_epochs(experiment_name, num_epochs):
    pickle_save(num_epochs, filename="experiments/{}/num_epochs.pkl".format(experiment_name))

def pickle_load_num_epochs(experiment_name):
    return pickle_load(filename="experiments/{}/num_epochs.pkl".format(experiment_name))

def pickle_save_model(model, experiment_name, i):
    pickle_save(model, filename="experiments/{}/models/model_{}.pkl".format(experiment_name, i))

def pickle_load_model(experiment_name, i):
    filename = "experiments/{}/models/model_{}.pkl".format(experiment_name, i)
    return pickle_load(filename=filename)

def pickle_save_models_of_epoch(models, epoch, pop_size, experiment_name):
    first_index, last_index = indexes_of_epoch(epoch, pop_size)
    for i,model in enumerate(models):
        pickle_save_model(model, experiment_name, first_index + i)

def pickle_load_models_of_epoch(experiment_name, epoch, pop_size):
    models = []
    first_index, last_index = indexes_of_epoch(epoch, pop_size)
    for i in range(first_index, last_index):
        models.append(pickle_load_model(experiment_name, i))
    return models

def pickle_save_rewards(rewards, experiment_name):
    filename = "experiments/{}/rewards.pkl".format(experiment_name)
    pickle_save(rewards, filename=filename)

def pickle_load_rewards(experiment_name):
    filename = "experiments/{}/rewards.pkl".format(experiment_name)
    if exist_file(filename):
        return pickle_load(filename=filename)
    return []

def pickle_save_operators(operators, experiment_name):
    filename = "experiments/{}/operators.pkl".format(experiment_name)
    pickle_save(operators, filename=filename)

def pickle_load_operators(experiment_name):
    filename = "experiments/{}/operators.pkl".format(experiment_name)
    if exist_file(filename):
        return pickle_load(filename=filename)
    return []

def pickle_load_best_reward_of_epoch(experiment_name, epoch, pop_size):
    first_index, last_index = indexes_of_epoch(epoch, pop_size)
    rewards = pickle_load_rewards(experiment_name)[first_index:last_index]
    return np.array(rewards).max()

def pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size):
    first_index, last_index = indexes_of_epoch(epoch, pop_size)
    rewards = pickle_load_rewards(experiment_name)[first_index:last_index]
    best_model_index = np.array(rewards).argmax() + first_index
    return pickle_load_model(experiment_name, best_model_index)
