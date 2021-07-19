"""
usage: python classify.py --data-dir [papers_processed] 
"""
# conda env export > environment.yml
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import argparse
import os
from transformers import AutoTokenizer, AutoModel #pip install transformers
from sklearn.linear_model import LogisticRegression, Perceptron #pip install sklearn
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as s
# random.seed(42)

class classify:
    def __init__(self):
        self.chunk_size = 500
        self.model = None
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

    def process_data(self, data_dir, concept):
        print("processing data")
        jj=0
        for j, dir_path in enumerate(glob.glob(data_dir)): #mathy, non_mathy
            if concept in dir_path:
                print(dir_path)
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
                        self.data[jj][k].append([embedding_vector.detach().numpy(), jj])
                jj+=1
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
        os.mkdir(".\\embeddings"+concept)
        np.save(".\\embeddings"+concept+"\\train_x.npy", self.train_x)
        np.save(".\\embeddings"+concept+"\\train_y.npy", self.train_y)
        np.save(".\\embeddings"+concept+"\\test_x.npy", self.test_x)
        np.save(".\\embeddings"+concept+"\\test_y.npy", self.test_y)
        print("data saved")
        print()

    def load_embeddings(self, concept):
        self.train_x = np.load(".\\embeddings"+concept+"\\train_x.npy")
        self.train_y = np.load(".\\embeddings"+concept+"\\train_y.npy")
        self.test_x = np.load(".\\embeddings"+concept+"\\test_x.npy")
        self.test_y = np.load(".\\embeddings"+concept+"\\test_y.npy")

        # t, c = train_test_split(list(zip(self.train_x, self.train_y)), test_size=0.20)
        # self.train_x, self.train_y = zip(*t)
        # self.cv_x, self.cv_y = zip(*c)
        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        self.test_x = np.array(self.test_x)
        self.test_y = np.array(self.test_y)
        # self.cv_x = np.array(self.cv_x)
        # self.cv_y = np.array(self.cv_y)
        # print(len(self.train_x))
        # print(self.train_x[0])

    def train(self, model):
        x, y = self.train_x, self.train_y
        model.fit(x, y)
        score = model.score(x, y)
        print(f"Model score: {score}")
        self.model = model

    def get_test_performance_for_model(self, model):
        #average over different subsets
        cvy = self.cv_y
        # cvy = np.random.choice([0, 1], size=len(self.cv_y))
        return model.score(self.cv_x, cvy)

    def get_model_report(self, model):
        predictions = model.predict(self.test_x)
        report = classification_report(self.test_y, predictions)
        print(report)

    def learning_curve_new(self, x, y, model, output, concept):
        # 0.8 -> seeds -> training subsets, test fixed
        t, c = train_test_split(list(zip(x, y)), test_size=0.50, random_state=42)
        x, y = zip(*t)
        test_x, test_y = zip(*c)
        train_x = np.array(x)
        train_y = np.array(y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        # print(len(test_x))
        # print(len(train_x))

        seeds = range(0, 15)
        # print(seeds)
        min_files = 6
        sizes = np.arange(min_files, len(train_x), step=1)

        train_scores = np.zeros((len(sizes)))
        train_scores_stds = np.zeros((len(sizes)))
        test_scores_stds = np.zeros((len(sizes)))
        test_scores = np.zeros((len(sizes)))

        for i, train_x_size in enumerate(sizes):
            train_score_train = np.zeros(len(seeds))
            test_score_train = np.zeros(len(seeds))
            for j, seed in enumerate(seeds):
                # print(seed)
                train_x_subset, _, train_y_subset, _\
                    = train_test_split(train_x, train_y, train_size=train_x_size, random_state=seed)
                new_model = None
                if model == "lr":
                    new_model = LogisticRegression(random_state=seed)
                elif model == "svm":
                    new_model = svm.SVC(kernel="linear", random_state=seed)
                elif model == "perceptron":
                    # new_model = Perceptron(random_state=seed)
                    new_model = MLPClassifier(random_state=seed)

                new_model.fit(train_x_subset, train_y_subset)
                train_score_train[j] = new_model.score(train_x_subset, train_y_subset)
                test_score_train[j] = new_model.score(test_x, test_y)

            # print(model.class_weight)
            # print(new_model.score(test_x, test_y))

            # train_score /= len(seeds)
            # test_score /= len(seeds)
            # train_scores[i] = train_score_train.mean()
            # test_scores[i] = test_score_train.mean()
            train_scores_stds[i] = train_score_train.std()
            test_scores_stds[i] = test_score_train.std()
            train_scores[i] = train_score_train.mean()
            test_scores[i] = test_score_train.mean()
            # train_scores[i] = train_score_train
            # test_scores[i] = test_score_train
        # print(test_scores)
        # data_train = [[sizes[i], list(train_scores[i][:])] for i in range(len(sizes))]
        # df_tr = pd.DataFrame(data_train, columns=["x", "y"])
        # data_test = [[sizes[i], list(test_scores[i][:])] for i in range(len(sizes))]
        # df_te = pd.DataFrame(data_test, columns=["x", "y"])
        # plt.grid()
        #
        # df_train = pd.DataFrame(dict(sizes=sizes, train_scores=train_scores))
        # df_test = pd.DataFrame(dict(sizes=sizes, train_scores=test_scores))
        # s.relplot(x="sizes", y="train_scores", kind="line", data=df_train)
        # s.relplot(x="sizes", y="test_scores", kind="line", data=df_test)

        # s.lineplot(x=sizes, y=train_scores, ci=95)
        # s.lineplot(x=sizes, y=test_scores, ci=95)
        # s.lineplot(x="x", y="y", data=df_tr, ci=95)
        # s.lineplot(x="x", y="y", data=df_te, ci=95)
        # #err_kws=matplotlib.axes.Axes.fill_between()
        # # plt.savefig(output)

        # train_scores_mean = train_scores.mean()
        # test_scores_mean = test_scores.mean()
        # train_scores_std = 1.96*(train_scores.std()/train_scores_mean)
        # test_scores_std = 1.96*(test_scores.std()/test_scores_mean)

        # have to replace with seaborn
        plt.grid()
        plt.fill_between(sizes, train_scores-((train_scores_stds*1.96)),
                         train_scores+((train_scores_stds*1.96)), color="green", alpha=0.1)
        plt.fill_between(sizes, test_scores-((test_scores_stds*1.96)),
                         test_scores+((test_scores_stds*1.96)), color="red", alpha=0.1)
        print(test_scores+((test_scores_stds*1.96)))
        print(((test_scores_stds*1)/np.sqrt(len(seeds))))
        print(test_scores-((test_scores_stds*1)/np.sqrt(len(seeds))))
        plt.plot(sizes, train_scores, color="g", label="training scores")
        plt.plot(sizes, test_scores, color="r", label="test scores")
        plt.ylabel("Score (Average accuracy)")
        plt.xlabel("Number of Training Examples")
        plt.title(concept+" concept")
        plt.legend(loc="best")
        # print(len(train_scores))
        plt.xlim(5, sizes[len(sizes)-1])
        plt.ylim(0.5, 1.01)
        # # plt.show()
        plt.savefig(output)
        return train_scores, test_scores, sizes


def main():
    parser = argparse.ArgumentParser()
    # add comments/help so the users know
    # what to send in as command line args
    parser.add_argument("--data_dir")
    parser.add_argument("--model")
    parser.add_argument("--report", required=False, action="store_true")
    parser.add_argument("--learning_curve", required=False, action="store_true")
    parser.add_argument("--learning_curve_output", required=False)
    parser.add_argument("--concept")
    parser.add_argument("--process_data", required=False, action="store_true")
    args = parser.parse_args()
    c = classify()
    # c.process_data(args.data_dir, args.concept)
    model = args.model
    if args.learning_curve and not args.learning_curve_output:
        print("You need to specify an output file for the learning curve")
        exit()
    if args.report:
        c.load_embeddings(args.concept)
        c.train(model)
        model_test = c.get_test_performance_for_model(model)
        print(f"model performance: {model_test}")
        print()
        c.get_model_report(model)
    if args.learning_curve:
        c.load_embeddings(args.concept)
        c.learning_curve_new(c.train_x, c.train_y, model, args.learning_curve_output, args.concept)
    if args.process_data:
        c.process_data(args.data_dir, args.concept)


main()


# conda env export > environment.yml
# saving models
# different ways to compute embeddings for the doc
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/chi2008-cueflik.pdf

# learning_curve - confidence intervals
# collect papers for {guidelines, qualitative, sentiment} concepts
# # define concept, model from terminal, everything from terminal

# new concepts
# documentation for args
# ranking model paper - see whittlesearch
# confidence intervals, error bars
# fix the (x, y) label ranges, title
# output file for lc
# sentiment -
# guidelines, principles, ben shneiderman,

# control problem, limeade figure,
# explanation vocab incomplete, learning concept
# refining explanatory vocab


# keyword based concepts
# standardize the x(stepsize=2), y(0.5 to 1) axes
# seaborn


# results -> concepts -> plots