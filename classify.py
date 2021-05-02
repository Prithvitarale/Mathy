"""
usage: python classify.py --data-dir [papers_processed] 
"""
import glob
import argparse
import os
from transformers import AutoTokenizer, AutoModel #pip install transformers
from sklearn.linear_model import LogisticRegression #pip install sklearn
from sklearn import svm
import random
class classify:
    def __init__(self):
        self.chunk_size = 750
        self.mathy_embeddings = [[], []]
        self.non_mathy_embeddings = [[], []]
        self.data = [self.mathy_embeddings, self.non_mathy_embeddings]
        self.mathy_test = []
        self.mathy_train = []
        self.non_mathy_test = []
        self.non_mathy_train = []
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

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
                    # embedding_vector.requires_grad=False
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
        print("data processed")

    # def get_training_data(self): # change such that you return the same dataset on multiple calls
    #     training_data = self.mathy_train + self.non_mathy_train
    #     random.shuffle(training_data)
    #     x, y = zip(*training_data)
    #     return x, y

    def train_logistic_regression_classifier(self):
        print("LR:\n")
        # x, y = self.get_training_data()
        x, y = self.train_x, self.train_y
        model = LogisticRegression(max_iter=300).fit(x, y)
        score = model.score(x, y)
        predictions = model.predict(x)
        print(predictions)
        print(y)
        print(score)
        print()

    def train_svm_classifier(self):
        print()
        print("SVM:\n")
        # x, y = self.get_training_data()
        x, y = self.train_x, self.train_y
        model = svm.SVC()
        model.fit(x, y)
        score = model.score(x, y)
        print(score)
        predictions = model.predict(x)
        print(predictions)
        print(y)

    # def train_perceptron(self):

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
args = parser.parse_args()
c = classify()
c.process_data(args.data_dir)
c.train_logistic_regression_classifier()
# conda env export > environment.yml