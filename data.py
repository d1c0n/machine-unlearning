import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
with open("grid_search_results.json", "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.json_normalize(data)

# Calculate the best score
df["best_score"] = (100 - df["accuracy_target"] + df["accuracy_non_target"]) / 2


# 1. Heatmap plot
def create_heatmap(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Heatmap of Best Scores", fontsize=16)

    for i, percentage in enumerate([5, 25, 50, 100]):
        ax = axes[i // 2, i % 2]
        data = df[df["hyperparameters.train_dataset_percentage"] == percentage]
        pivot = data.pivot(
            index="hyperparameters.num_epochs",
            columns="hyperparameters.learning_rate",
            values="best_score",
        )
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f")
        ax.set_title(f"Train Dataset Percentage: {percentage}%")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Number of Epochs")

    plt.tight_layout()
    plt.savefig("heatmap_plot.png")
    plt.close()


# 2. Train data impact plot
def create_train_data_impact_plot(df):
    plt.figure(figsize=(12, 8))

    for lr in df["hyperparameters.learning_rate"].unique():
        data = df[df["hyperparameters.learning_rate"] == lr]
        mean = data.groupby("hyperparameters.train_dataset_percentage")[
            "best_score"
        ].mean()
        std = data.groupby("hyperparameters.train_dataset_percentage")[
            "best_score"
        ].std()
        plt.errorbar(
            mean.index,
            mean.values,
            yerr=std.values,
            label=f"LR: {lr}",
            capsize=5,
            marker="o",
        )

    plt.xlabel("Train Dataset Percentage")
    plt.ylabel("Best Score")
    plt.title("Impact of Train Data Percentage on Best Score")
    plt.legend()
    plt.savefig("train_data_impact.png")
    plt.close()


# 3. Learning rate analysis plot
def create_learning_rate_analysis_plot(df):
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    fig.suptitle("Learning Rate Analysis", fontsize=16)

    for i, percentage in enumerate([5, 25, 50, 100]):
        for j, epochs in enumerate(range(1, 6)):
            ax = axes[i, j]
            data = df[
                (df["hyperparameters.train_dataset_percentage"] == percentage)
                & (df["hyperparameters.num_epochs"] == epochs)
            ]

            ax.plot(
                data["hyperparameters.learning_rate"],
                data["accuracy_target"],
                label="Target",
                marker="o",
            )
            ax.plot(
                data["hyperparameters.learning_rate"],
                data["accuracy_non_target"],
                label="Non-Target",
                marker="s",
            )

            ax.set_xscale("log")
            ax.set_title(f"{percentage}% data, {epochs} epochs")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Accuracy")
            ax.legend()

    plt.tight_layout()
    plt.savefig("learning_rate_analysis.png")
    plt.close()


# Run the functions
create_heatmap(df)
create_train_data_impact_plot(df)
create_learning_rate_analysis_plot(df)
