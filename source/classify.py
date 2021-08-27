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
import pandas as pd
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

    def process_data(self, data_dir_without_star, concept):
        data_dir = os.path.join(data_dir_without_star, "*")
        jj=0
        for j, dir_path in enumerate(glob.glob(data_dir)): #mathy, non_mathy
            print(dir_path)
            if concept in dir_path:
                print("processing data")
                print(dir_path)
                dir_path = os.path.join(dir_path, "*")
                for k, dirr in enumerate(glob.glob(dir_path)): #test, train
                    if dirr != "embeddings_"+concept:
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
                                    if embedding_vector is None:
                                        print("true")
                                else:
                                    embedding_vector += outputs.last_hidden_state[0][0]
                            print(n)
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
        print("saving data...")
        concept_path = os.path.join(data_dir_without_star, concept)
        os.mkdir(os.path.join(concept_path, "embeddings_"+concept))
        embeddings_path = os.path.join(concept_path, "embeddings_"+concept)
        np.save(os.path.join(embeddings_path, "train_x.npy"), self.train_x)
        np.save(os.path.join(embeddings_path, "train_y.npy"), self.train_y)
        np.save(os.path.join(embeddings_path, "test_x.npy"), self.test_x)
        np.save(os.path.join(embeddings_path, "test_y.npy"), self.test_y)
        print("data saved")
        print()

    def load_embeddings(self, data_dir, concept):
        concept_path = os.path.join(data_dir, concept)
        embeddings_path = os.path.join(concept_path, "embeddings_"+concept)
        self.train_x = np.load(os.path.join(embeddings_path, "train_x.npy"))
        self.train_y = np.load(os.path.join(embeddings_path, "train_y.npy"))
        self.test_x = np.load(os.path.join(embeddings_path, "test_x.npy"))
        self.test_y = np.load(os.path.join(embeddings_path, "test_y.npy"))

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

        seeds = range(0, 25)
        min_files = 6
        # sizes = np.arange(min_files, len(train_x), step=1)
        sizes = np.arange(min_files, 18, step=1)

        test_scores_trains = pd.DataFrame(dtype=float, columns=["sizes", "seeds", "scores"])
        train_scores_trains = pd.DataFrame(dtype=float, columns=["sizes", "seeds", "scores"])

        for i, train_x_size in enumerate(sizes):
            for j, seed in enumerate(seeds):
                train_x_subset, _, train_y_subset, _\
                    = train_test_split(train_x, train_y, train_size=train_x_size, random_state=seed)
                new_model = None
                if model == "lr":
                    new_model = LogisticRegression(random_state=seed)
                elif model == "svm":
                    new_model = svm.SVC(kernel="linear", random_state=seed)
                elif model == "perceptron":
                    new_model = MLPClassifier(random_state=seed)

                new_model.fit(train_x_subset, train_y_subset)

                test_scores_trains.loc[-1] = [train_x_size, seed, new_model.score(test_x, test_y)]
                test_scores_trains.index += 1
                test_scores_trains=test_scores_trains.sort_index()

                train_scores_trains.loc[-1] = [train_x_size, seed, new_model.score(train_x_subset, train_y_subset)]
                train_scores_trains.index += 1
                train_scores_trains = train_scores_trains.sort_index()

        # print(test_scores_trains[1])
        plt.grid()
        s.lineplot(data=test_scores_trains, x="sizes", y="scores", marker="o", label="Test scores")
        s.lineplot(data=train_scores_trains, x="sizes", y="scores", marker="x", label="Training scores")

        plt.ylabel("Score (average accuracy)")
        plt.xlabel("Number of Training Examples")
        plt.title("[framework] learning the " + concept.capitalize() + " concept")
        plt.legend(loc="lower left")
        plt.xlim(5, 18)
        plt.xticks(np.arange(5, 20, 5))
        plt.yticks(np.arange(0.8, 1.01, 0.1))
        plt.savefig(output)

        return test_scores_trains, train_scores_trains, sizes
# research qs
# voc refinement algorithm
# look at limeade

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
        c.load_embeddings(args.concept, args.concept)
        c.train(model)
        model_test = c.get_test_performance_for_model(model)
        print(f"model performance: {model_test}")
        print()
        c.get_model_report(model)
    if args.learning_curve:
        c.load_embeddings(args.data_dir, args.concept)
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

# range, legend, white space