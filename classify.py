"""
usage: python classify.py --data-dir [papers_processed] 
"""
# conda env export > environment.yml
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import argparse
import os
from transformers import AutoTokenizer, AutoModel #pip install transformers
from sklearn.linear_model import LogisticRegression, Perceptron #pip install sklearn
from sklearn import svm
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class classify:
    def __init__(self):
        self.chunk_size = 500
        self.lr = None
        self.svm = None
        self.perceptron = None
        self.mathy_embeddings = [[], []]
        self.non_mathy_embeddings = [[], []]
        self.data = [self.mathy_embeddings, self.non_mathy_embeddings]
        self.mathy_test = []
        self.mathy_train = []
        self.non_mathy_test = []
        self.non_mathy_train = []
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.cv_x = []
        self.cv_y = []

    def process_data(self, data_dir):
        print("processing data")
        for j, dir_path in enumerate(glob.glob(data_dir)): #mathy, non_mathy
            dir_path = os.path.join(dir_path, "*")
            for k, dirr in enumerate(glob.glob(dir_path)): #test, train
                dirr = os.path.join(dirr, "*")
                for l, pdf in enumerate(glob.glob(dirr)): #txts
                    pdf = open(pdf, "r", encoding="utf8")
                    pdf_text = pdf.read()
                    pdf.close()
                    embedding_vector = None
                    r = range(0, len(pdf_text), self.chunk_size)
                    n = len(r)
                    for i in r:#txt-content
                        inputs = self.tokenizer(pdf_text[i:i+self.chunk_size], return_tensors="pt")
                        outputs = self.model(**inputs)
                        if embedding_vector is None:
                            embedding_vector = outputs.last_hidden_state[0][0]
                        else:
                            embedding_vector += outputs.last_hidden_state[0][0]
                    embedding_vector /= n
                    self.data[j][k].append([embedding_vector.detach().numpy(), j])
        self.mathy_test = self.data[0][0]
        self.mathy_train = self.data[0][1]
        self.non_mathy_test = self.data[1][0]
        self.non_mathy_train = self.data[1][1]
        training_data = self.mathy_train + self.non_mathy_train
        random.shuffle(training_data)
        self.train_x, self.train_y = zip(*training_data)
        test_data = self.mathy_test + self.non_mathy_test
        random.shuffle(test_data)
        self.test_x, self.test_y = zip(*test_data)
        print("data processed: ")
        print("mathy_test: "+str(len(self.mathy_test)))
        print("mathy_train: "+str(len(self.mathy_train)))
        print("non_mathy_test: "+str(len(self.non_mathy_test)))
        print("non_mathy_train: "+str(len(self.non_mathy_train)))
        print()
        print("saving data: ")
        os.mkdir(".\\embeddings")
        np.save(".\\embeddings\\train_x.npy", self.train_x)
        np.save(".\\embeddings\\train_y.npy", self.train_y)
        np.save(".\\embeddings\\test_x.npy", self.test_x)
        np.save(".\\embeddings\\test_y.npy", self.test_y)
        print("data saved")
        print()

    def load_embeddings(self):
        self.train_x = np.load(".\\embeddings\\train_x.npy")
        self.train_y = np.load(".\\embeddings\\train_y.npy")
        self.test_x = np.load(".\\embeddings\\test_x.npy")
        self.test_y = np.load(".\\embeddings\\test_y.npy")

        # t, c = train_test_split(list(zip(self.train_x, self.train_y)), test_size=0.20)
        # self.train_x, self.train_y = zip(*t)
        # self.cv_x, self.cv_y = zip(*c)
        # self.train_x = np.array(self.train_x)
        # self.train_y = np.array(self.train_y)
        # self.test_x = np.array(self.test_x)
        # self.test_y = np.array(self.test_y)
        # self.cv_x = np.array(self.cv_x)
        # self.cv_y = np.array(self.cv_y)

    def train_logistic_regression_classifier(self):
        x, y = self.train_x, self.train_y
        model = LogisticRegression(max_iter=300).fit(x, y)
        score = model.score(x, y)
        print(f"LR score: {score}")
        self.lr = model

    def train_svm_classifier(self):
        x, y = self.train_x, self.train_y
        model = svm.SVC()
        model.fit(x, y)
        score = model.score(x, y)
        print(f"SVM score: {score}")
        self.svm = model

    def get_test_performance_for_model(self, model):
        cvy = self.cv_y
        # cvy = np.random.choice([0, 1], size=len(self.cv_y))
        return model.score(self.cv_x, cvy)

    # def get_precision(self):
    #     x, y = self.train_x, self.train_y
    #     predictions = self.lr.predict(x)
    #     mathy_predictions = [1*(pred==0) for pred in predictions]#1s for mathy
    #     non_mathy_predictions = [1*(pred==1) for pred in predictions]#1s for non_mathy
    #     total_mathy_predictions = np.sum(mathy_predictions)
    #     total_non_mathy_predictions = np.sum(non_mathy_predictions)
    #     mathy_template = [1*(label==0) for label in y]#1s for mathy
    #     non_mathy_template = [1*(label==1) for label in y]#1s for non_mathy
    #     total_mathy = np.sum(mathy_template)
    #     total_non_mathy = np.sum(non_mathy_template)
    #     mathy_recall_num = np.sum([1*(pred==1 and pred==mathy_template[i])
    #                                for i, pred in enumerate(mathy_predictions)])
    #     non_mathy_recall_num = np.sum([1*(pred==1 and pred==non_mathy_template[i])
    #                                    for i, pred in enumerate(non_mathy_predictions)])
    #     mathy_precision = mathy_recall_num/total_mathy_predictions
    #     non_mathy_precision = non_mathy_recall_num/total_non_mathy_predictions
    #     print("mathy_precision: " + str(mathy_precision))
    #     print("non_mathy_precision: " + str(non_mathy_precision))
    #     mathy_recall = mathy_recall_num/total_mathy
    #     non_mathy_recall = non_mathy_recall_num/total_non_mathy
    #     print("mathy recall: " +str(mathy_recall))
    #     print("non mathy recall: " +str(non_mathy_recall))

    def get_model_report(self, model):
        predictions = model.predict(self.cv_x)
        report = classification_report(self.cv_y, predictions)
        print(report)

    def train_perceptron_classifier(self):
        perceptron = Perceptron()
        perceptron.fit(self.train_x, self.train_y)
        score = perceptron.score(self.train_x, self.train_y)
        print(f"Perceptron score: {score}")
        self.perceptron = perceptron

    def learning_curve(self):
        return 0

    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
args = parser.parse_args()
c = classify()
# c.process_data(args.data_dir)
c.load_embeddings()
# c.train_logistic_regression_classifier()
# c.train_svm_classifier()
# c.train_perceptron_classifier()
# # c.get_precision()
# lr_test = c.get_test_performance_for_model(c.lr)
# svm_test = c.get_test_performance_for_model(c.svm)
# perceptron_test = c.get_test_performance_for_model(c.perceptron)
# print()
# print()
# print(f"lr performance: {lr_test}")
# print(f"svm performance: {svm_test}")
# print(f"perceptron performance: {perceptron_test}")
# print()
# c.get_model_report(c.lr)
# c.get_model_report(c.svm)
# c.get_model_report(c.perceptron)
estimator = LogisticRegression()
estimator = svm.SVC()
# estimator = Perceptron()
title = "Learning curve"
plot = c.plot_learning_curve(estimator, title, c.train_x, c.train_y)
# print(c.train_y)
plot.show()

# conda env export > environment.yml
# saving models
# # store embeddings as .npy np.save np.load
# we can change the way we compute embeddings for the doc
# # cv for checking on unseen not test
# # (1) acc vs number of training examples/learning curve
# # ...(2)  w/ confidence intervals for lr, svm, perceptron
# more concepts -
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/chi2008-cueflik.pdfx