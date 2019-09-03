from evoframe.os import pickle_load
from evoframe.recursive_dict import recursively_default_dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_num_epochs(experiment_name):
    return pickle_load("experiments/{}/num_epochs.pkl".format(experiment_name))

def get_pop_size(experiment_name):
    return pickle_load("experiments/{}/pop_size.pkl".format(experiment_name))

def load_context(experiment_name, epochs=[1], keys=["models", "rewards", "operators"]):
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

def overlap_figures(*sub_figs):
    fig = go.Figure()
    for sub_fig in sub_figs:
        fig.add_traces(sub_fig.data)
    for sub_fig in sub_figs:
        fig.layout.update(sub_fig.layout)
    return fig

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
    return overlap_figures(fig_line, fig_scatter)

def get_best_model_of_epoch(experiment_name, epoch):
    context = load_context(experiment_name, epochs=[epoch], keys=["models", "rewards"])
    i = np.array(context["epochs"][epoch]["rewards"]).argmax()
    return context["epochs"][epoch]["models"][i]

def show_best_fnn_weights(experiment_name, epoch):
    context = load_context(experiment_name, epochs=[epoch], keys=["models", "rewards"])
    best_model = context["epochs"][epoch]["models"][0]
    num_cols = 2
    num_rows = len(best_model.weights)
    subplot_titles = ["Layer "+str(i//2+1) if i%2==0 else "Bias "+str(i//2+1) for i in range(num_cols*num_rows)]
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subplot_titles)
    fig.update_layout(height=300*num_rows, width=800, title_text="NN layers at epoch " + str(epoch))

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
        #axis_number = "" if (i+1)==1 else str(i+1)
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
        #axis_number = str(num_rows+i+1)
        axis_number = str((i+1)*num_cols)
        fig.update_layout({"coloraxis": sub_fig.layout.coloraxis,
                           "xaxis{}.range".format(axis_number):sub_fig.layout.xaxis.range,
                           "yaxis{}.range".format(axis_number):sub_fig.layout.yaxis.range})
        fig.update_xaxes(title_text="output", row=row, col=col)
        #fig.update_yaxes(title_text="output", row=row, col=col)
    return fig
