"""
usage:
python recommend.py --data_dir data_dir --topic topic_name --concept concept_name

For example:
python recommend.py --data_dir .\papers_processed --topic fairness_prev --concept pos_mathy

Use conda to set env
conda env export > environment.yml
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve, classification_report
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import PrecisionRecallDisplay
import seaborn as sns
import pandas as pd


class Eval:
    def __init__(self):
        pass

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=10, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10), scoring=None):
        # .1, 1.0, 10
        font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 33}
        matplotlib.rc('font', **font)
        fig, ax = plt.subplots()
        ax.set_title(title + " (Fairness Feed)")
        # if ylim is not None:
        #     ax.set_ylim(*ylim)
        ax.set_xlabel("Number of Positive Training Examples")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes, return_times=False)
        # train_sizes = train_sizes / 2
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.grid(False)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        # ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score", linewidth=5.0)
        # ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score", linewidth=5.0)
        sns.lineplot(X=train_sizes, y=train_scores_mean)
        # ax.set_xlim([5, 50])
        ax.set_ylim([0.5, 1.05])
        plt.show()

    def plot_precision_recall_curve(self, classifier, X_test, y_test):
    # def plot_precision_recall_curve(self, target, scores):
        font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 33}
        matplotlib.rc('font', **font)
        # X_test = predicted_papers['embedding'].tolist()
        # y_test = predicted_papers['concept'].tolist()

        # y_true = target.tolist() # target values, np array
        # y_scores = scores.tolist() # actual values from model
        # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # print(precision)
        # print(recall)
        # print(thresholds)

        # display = PrecisionRecallDisplay.from_estimator(
        #     classifier, X_test, y_test, name="LinearSVC"
        # )
        # _ = display.ax_.set_title("2-class Precision-Recall curve")

        plot_precision_recall_curve(classifier, X_test, y_test, name="Model", pos_label=0, linewidth=1.0)
        plt.ylim([0.5, 1.02])
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()

    def compute_classification_report(self, classifier, X_test, y_test):
        prediction = classifier.predict(X_test)
        # print(prediction, y_test)
        print(classification_report(y_test, prediction))

    def plot_learning_curve_new(self, classifier, x, y):
        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 45}
        matplotlib.rc('font', **font)

        # 0.8 -> seeds -> training subsets, test_neg fixed
        t, c = train_test_split(list(zip(x, y)), test_size=0.50, random_state=42)
        x, y = zip(*t)
        test_x, test_y = zip(*c)
        train_x = np.array(x)
        train_y = np.array(y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)

        # 25 seeds = 25 times, min files indicates the minimum amount of files in the train set
        seeds = range(0, 25)
        min_files = 6
        # sizes = np.arrange(min_files, len(train_x), step=1)
        sizes = np.arange(min_files, 18, step=1)

        test_scores_trains = pd.DataFrame(dtype=float, columns=["sizes", "seeds", "scores"])
        train_scores_trains = pd.DataFrame(dtype=float, columns=["sizes", "seeds", "scores"])

        for i, train_x_size in enumerate(sizes):
            for j, seed in enumerate(seeds):
                train_x_subset, _, train_y_subset, _ \
                    = train_test_split(train_x, train_y, train_size=train_x_size, random_state=seed)

                classifier.fit(train_x_subset, train_y_subset)

                test_scores_trains.loc[-1] = [train_x_size, seed, classifier.score(test_x, test_y)]
                test_scores_trains.index += 1
                test_scores_trains = test_scores_trains.sort_index()

                train_scores_trains.loc[-1] = [train_x_size, seed, classifier.score(train_x_subset, train_y_subset)]
                train_scores_trains.index += 1
                train_scores_trains = train_scores_trains.sort_index()

        plt.grid()
        rc = {'lines.linewidth': 5, 'lines.markersize': 10, 'lines.edgecolor': None}
        sns.set_context(rc=rc)
        sns.lineplot(data=test_scores_trains, x="sizes", y="scores", marker="o", label="Test scores", linewidth=5)# , legend=None)
        sns.lineplot(data=train_scores_trains, x="sizes", y="scores", marker="+", label="Training scores", linewidth=5)# , legend=None)

        plt.ylabel("Score (average accuracy)")
        plt.xlabel("Total Number of Training Examples")
        plt.title("TF-IDF (Fairness Feed)")
        plt.legend(loc="lower left")
        # plt.xlim(5, 18)
        plt.xticks(np.arange(6, 18, 2))
        plt.yticks(np.arange(0.6, 1.01, 0.1))
        plt.show()