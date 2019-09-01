from evoframe.os import pickle_load
from evoframe.recursive_dict import recursively_default_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def get_num_epochs(experiment_name):
    return pickle_load("experiments/{}/num_epochs.pkl".format(experiment_name))

def get_pop_size(experiment_name):
    return pickle_load("experiments/{}/pop_size.pkl".format(experiment_name))

def load_context(experiment_name, epochs=[1], keys=["pop_size", "num_epochs", "models", "rewards", "operators"]):
    context = recursively_default_dict()
    if "pop_size" not in keys:
        keys += "pop_size"
    if "num_epochs" in keys:
        keys += "num_epochs"
    context["num_epochs"] = pickle_load("experiments/{}/num_epochs.pkl".format(experiment_name))
    context["pop_size"] = pickle_load("experiments/{}/pop_size.pkl".format(experiment_name))
    for epoch in epochs:
        if "models" in keys:
            models = []
            for i_model in range(context["pop_size"]):
                models.append(pickle_load("experiments/{}/models/epoch_{}/model_{}.pkl".format(experiment_name, epoch, i_model)))
            context["epochs"][epoch]["models"] = models
        if "rewards" in keys:
            context["epochs"][epoch]["rewards"] = pickle_load("experiments/{}/rewards/epoch_{}.pkl".format(experiment_name, epoch))
        if "operators" in keys:
            context["epochs"][epoch]["operators"] = pickle_load("experiments/{}/operators/epoch_{}.pkl".format(experiment_name, epoch))
    return context

def get_distinct_operators(experiment_name):
    context = recursively_default_dict()
    num_epochs = load_context(experiment_name, epochs=[1], keys=["num_epochs"])["num_epochs"]
    operators = []
    for epoch in range(1, num_epochs + 1):
        operators.append(pickle_load("experiments/{}/operators/epoch_{}.pkl".format(experiment_name, epoch)))
        operators = list(set(operators))
    return operators

def plot_rewards(experiment_name):
    num_epochs = load_context(experiment_name, epochs=[1], keys=["num_epochs"])["num_epochs"]
    keys = ["pop_size", "num_epochs", "rewards", "operators"]
    context = load_context(experiment_name, epochs=list(range(1, num_epochs + 1)), keys=keys)
    pop_size = context["pop_size"]
    epochs = list(range(1, num_epochs + 1))
    # Max-Mean
    xs = epochs
    ys_max = [max(context["epochs"][epoch]["rewards"]) for epoch in epochs]
    ys_mean = [sum(context["epochs"][epoch]["rewards"])/context["pop_size"] for epoch in epochs]
    ys_category = ["max" for epoch in epochs] + ["mean" for epoch in epochs]
    df = pd.DataFrame({"epochs": xs*2, "rewards": ys_max+ys_mean, "category": ys_category})
    fig_line = px.line(df, x="epochs", y="rewards", color="category")
    # Scatter
    xs = [ep + ((np.random.rand() - 0.5) * 0.4) for ep in epochs for i in range(pop_size)] # add small noise
    ys = [r for epoch in epochs for r in context["epochs"][epoch]["rewards"]]
    operators = [op for epoch in epochs for op in context["epochs"][epoch]["operators"]]
    df = pd.DataFrame({"epochs": epochs*pop_size, "epochs_noise": xs, "rewards": ys, "operators": operators})
    fig_scatter = px.scatter(df, x="epochs_noise", y="rewards", color="operators", marginal_y="rug")
    return fig_line, fig_scatter

def get_best_model_of_epoch(experiment_name, epoch):
    context = load_context(experiment_name, epochs=[epoch], keys=["models", "rewards"])
    i = np.array(context["epochs"][epoch]["rewards"]).argmax()
    return context["epochs"][epoch]["models"][i]

def show_best_fnn_weights(experiment_name, epoch):
    context = load_context(experiment_name, epochs=[epoch], keys=["models", "rewards"])
    best_model = context["epochs"][epoch]["models"][0]
    num_cols = len(best_model.weights)
    num_rows = 2
    plt.figure(figsize=(15,10))
    for i,layer in enumerate(best_model.weights):
        index = i+1
        plt.subplot(num_rows, num_cols, index)
        plt.imshow(layer, cmap='hot', interpolation='nearest', vmin=-2, vmax=2)
        plt.colorbar()
    for i,layer in enumerate(best_model.biases):
        index = num_cols+i+1
        plt.subplot(num_rows, num_cols, index)
        plt.imshow(layer.reshape(1, layer.shape[0]), cmap='hot', interpolation='nearest', vmin=-2, vmax=2)
        plt.colorbar()
    plt.show()
