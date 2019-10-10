from evoframe.utility import *
from evoframe.context import recursively_default_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def get_distinct_operators(experiment_name):
    return list(set(pickle_load_operators(experiment_name)))

def overlap_figures(*sub_figs):
    fig = go.Figure()
    for sub_fig in sub_figs:
        fig.add_traces(sub_fig.data)
    for sub_fig in sub_figs:
        fig.layout.update(sub_fig.layout)
    return fig

def plot_rewards(experiment_name, epochs=None):
    num_epochs = pickle_load_num_epochs(experiment_name)
    if epochs == None:
        epochs = list(range(1, num_epochs + 1))
    pop_size = pickle_load_pop_size(experiment_name)
    # Max-Mean
    xs = epochs
    ys_max = []
    ys_mean = []
    for epoch in epochs:
        first_index, last_index = indexes_of_epoch(epoch, pop_size)
        ys_max.append(pickle_load_best_reward_of_epoch(experiment_name, epoch, pop_size))
        ys_mean.append(sum(pickle_load_rewards(experiment_name)[first_index:last_index]) / pop_size)
    ys_category = ["max" for epoch in epochs] + ["mean" for epoch in epochs]
    df = pd.DataFrame({"epochs": xs*2, "rewards": ys_max+ys_mean, "category": ys_category})
    fig_line = px.line(df, x="epochs", y="rewards", color="category")
    # Scatter
    xs = [i/pop_size + ep for ep in epochs for i in range(pop_size)]
    first_index, _ = indexes_of_epoch(epochs[0], pop_size)
    _, last_index = indexes_of_epoch(epochs[-1], pop_size)
    ys = pickle_load_rewards(experiment_name)[first_index:last_index]
    operators = pickle_load_operators(experiment_name)[first_index:last_index]
    df = pd.DataFrame({"epochs": epochs*pop_size, "epochs_noise": xs, "rewards": ys, "operators": operators})
    fig_scatter = px.scatter(df, x="epochs_noise", y="rewards", color="operators", marginal_y="rug")
    return overlap_figures(fig_line, fig_scatter)

def show_best_fnn_weights(experiment_name, epoch):
    pop_size = pickle_load_pop_size(experiment_name)
    best_model = pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size)
    best_reward = pickle_load_best_reward_of_epoch(experiment_name, epoch, pop_size)
    num_cols = 2
    num_rows = len(best_model.weights)
    subplot_titles = ["Layer "+str(i//2+1) if i%2==0 else "Bias "+str(i//2+1) for i in range(num_cols*num_rows)]
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subplot_titles)
    fig.update_layout(height=300*num_rows, width=800, title_text="NN layers at epoch " + str(epoch) + " with reward " + str(best_reward))

    for i,layer in enumerate(best_model.weights):
        shape = layer.shape
        data = [(row+1, col+1, layer[row][col]) for row in range(shape[0]) for col in range(shape[1])]
        columns = ["neuron_input", "neuron_output", "value"]
        df = pd.DataFrame(data=data, columns=columns)
        sub_fig = px.density_heatmap(df, x="neuron_output", y="neuron_input", z="value",
                                     histfunc="sum", color_continuous_scale="RdYlGn", range_color=(-2,2),
                                     nbinsx=shape[1], nbinsy=shape[0],
                                     range_x=(0.5, shape[1]+0.5), range_y=(0.5, shape[0]+0.5))
        row, col = i+1, 1
        sub_fig.data[0].xbingroup, sub_fig.data[0].ybingroup = "a_{}_{}".format(row, col), "b_{}_{}".format(row, col)
        fig.add_trace(sub_fig.data[0], row=row, col=col)
        axis_number = "" if (i+1)==1 else str(i*num_cols+1)
        fig.update_layout({"coloraxis": sub_fig.layout.coloraxis,
                           "xaxis{}.range".format(axis_number):sub_fig.layout.xaxis.range,
                           "yaxis{}.range".format(axis_number):sub_fig.layout.yaxis.range})
        fig.update_xaxes(title_text="output", row=row, col=col)
        fig.update_yaxes(title_text="input", row=row, col=col)
    for i,bias in enumerate(best_model.biases):
        bias = bias.reshape(1, bias.shape[0])
        shape = bias.shape
        data = [(row+1, col+1, bias[row][col]) for row in range(shape[0]) for col in range(shape[1])]
        columns = ["1", "neuron_output", "value"]
        df = pd.DataFrame(data=data, columns=columns)
        sub_fig = px.density_heatmap(df, x="neuron_output", y="1", z="value",
                                     histfunc="sum", color_continuous_scale="RdYlGn", range_color=(-2,2),
                                     nbinsx=shape[1], nbinsy=shape[0],
                                     range_x=(0.5, shape[1]+0.5), range_y=(0.5, shape[0]+0.5))
        row, col = i+1, 2
        sub_fig.data[0].xbingroup, sub_fig.data[0].ybingroup = "a_{}_{}".format(row, col), "b_{}_{}".format(row, col)
        fig.add_trace(sub_fig.data[0], row=row, col=col)
        axis_number = str((i+1)*num_cols)
        fig.update_layout({"coloraxis": sub_fig.layout.coloraxis,
                           "xaxis{}.range".format(axis_number):sub_fig.layout.xaxis.range,
                           "yaxis{}.range".format(axis_number):sub_fig.layout.yaxis.range})
        fig.update_xaxes(title_text="output", row=row, col=col)
    return fig

def plot_behavioural_differences(experiment_name, get_random_input_func, epochs=None, mode="first_best", iterations=100):
    # mode is "first_best" or "last_best"
    random_inputs = [get_random_input_func() for i in range(iterations)]
    num_epochs = pickle_load_num_epochs(experiment_name)
    pop_size = pickle_load_pop_size(experiment_name)
    if epochs == None:
        epochs = list(range(1, num_epochs + 1))
    best_models = [pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size) for epoch in epochs]
    compare_model = best_models[0]
    behavioural_differences = []
    for best_model in best_models:
        best_model_results = np.array([best_model.predict(random_input) for random_input in random_inputs])
        compare_model_results = np.array([compare_model.predict(random_input) for random_input in random_inputs])
        rmse = np.sqrt(np.sum(np.square(best_model_results - compare_model_results)))
        behavioural_differences.append(rmse)
        if mode == "last_best":
            compare_model = best_model
    xs = epochs
    ys = np.array(behavioural_differences)
    df = pd.DataFrame({"epochs": xs, "behavioural differences": ys})
    return px.line(df, x="epochs", y="behavioural differences")

def plot_params_similarity(experiment_name, epochs=None, only_best=True, iterations=300):
    num_epochs = pickle_load_num_epochs(experiment_name)
    pop_size = pickle_load_pop_size(experiment_name)
    if epochs == None:
        epochs = list(range(1, num_epochs + 1))
    if only_best:
        models = [pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size) for epoch in epochs]
    else:
        models = [pickle_load_models_of_epoch(experiment_name, epoch, pop_size) for epoch in epochs]
        models = [m for l in models for m in l] # flatten
    models_params = [[w for layer in m.weights for row in layer for w in row] + [b for layer in m.biases for b in layer]
                        for m in models]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=iterations)
    tsne_results = tsne.fit_transform(models_params)
    xs = list(zip(*tsne_results))[0]
    ys = list(zip(*tsne_results))[1]
    zs = range(1, len(xs)+1)
    if not only_best:
        zs = [z//pop_size for z in zs]
    df = pd.DataFrame({"xs": xs, "ys": ys, "zs": zs})
    return px.scatter(df, x="xs", y="ys", color="zs")

def plot_behavioural_variances_to_input(experiment_name, get_random_input_func, epochs=None, iterations=100):
    # mode is "first_best" or "last_best"
    random_inputs = [get_random_input_func() for i in range(iterations)]
    num_epochs = pickle_load_num_epochs(experiment_name)
    pop_size = pickle_load_pop_size(experiment_name)
    if epochs == None:
        epochs = list(range(1, num_epochs + 1))
    best_models = [pickle_load_best_model_of_epoch(experiment_name, epoch, pop_size) for epoch in epochs]
    behavioural_variances_to_input = []
    for best_model in best_models:
        best_model_results = np.array([best_model.predict(random_input) for random_input in random_inputs])
        results_variance = np.var(best_model_results)
        behavioural_variances_to_input.append(results_variance)
    xs = epochs
    ys = np.array(behavioural_variances_to_input)
    df = pd.DataFrame({"epochs": xs, "behavioural_variances_to_input": ys})
    return px.line(df, x="epochs", y="behavioural_variances_to_input")
