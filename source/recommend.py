"""
usage:
python recommend.py --data_dir data_dir --topic topic_name --concept concept_name

For example:
python recommend.py --data_dir .\papers_processed --topic fairness_prev --concept pos_mathy

Use conda to set env
conda env export > environment.yml
"""

import argparse
import datetime
import glob
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_extraction
from numpy import argmax

from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel


class Recommend:
    def __init__(self):
        self.loaded_model = None
        self.sciBERT_clf = svm.SVC(kernel="linear", random_state=42, probability=True)
        self.linreg_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")  # max_features=1000)
        self.reg = LinearRegression()

        self.df = None
        self.new_features = []
        self.X = None
        self.soft_labels = None
        self.predicted_papers = None
        self.predicted_embeddings = []

    def preprocess_papers(self, data_dir, topic):
        self.feed = topic
        # TODO: read score from file
        # TODO:
        #  1)scibert_model_1
        #  2)student_model
        #  3)apply_concept_and_generate_scores
        #  4)feed_scores_into_scibert_model_2
        #  5)train_scibert_model_2
        #  6)generate_explanation (take in last embedding value for concept explanation)

        mathy_score = [0.3008878669759426, 0.049875217243276704, 0.016754113113269353, 0.9958595153048831,
                       0.18428238548903275,
                       0.5021615823778064, 0.9790406730977911, 0.43257541324351767, 0.09145916066508097,
                       0.02341281881988244,
                       0.0757575222370398, 0.02027662463540003, 0.06808040853933639, 0.006247025076268953,
                       0.5279636982251246,
                       0.016766144724200904, 0.7518219282607441, 0.005606567217301017, 0.9574499992348058,
                       0.03295812844664214,
                       0.01903241837486147, 0.15850932135429407, 0.022035249352840514, 0.015328284685196536,
                       0.9872855594045657,
                       0.02281800423881286, 0.12231561168626659, 0.019749345439958765, 0.06273737218816211,
                       0.5302031356524342,
                       0.030691789880110232, 0.9297243653441648, 0.03120787264413516, 0.01959898449150721,
                       0.03471897708609528,
                       0.014950145358503697, 0.022349459270554095, 0.06571839328385287, 0.01959898449150721,
                       0.9941759648262924,
                       0.9434318303478683, 0.9092559448815372, 0.99972353971549, 0.9992327525337733, 0.9999324717739108,
                       0.996506296102025, 0.9996724240689266, 0.6641299858417924, 0.9929176553009722,
                       0.9998918502265071,
                       0.7465797290992018, 0.9999183892073007, 0.978306462081893, 0.9718953278142037, 0.995451723182586,
                       0.818861637252138, 0.7489932445015364, 0.9710019944345138, 0.9614486306835023,
                       0.8617884517534146,
                       0.9983379450395622, 0.932650541534706, 0.9633652666413612, 0.9998431969103789,
                       0.9911621536469162,
                       0.8047296262243525, 0.862254519436307, 0.7094489987046599, 0.9990725134197465,
                       0.9999424101950751,
                       0.974938935759433, 0.9856059547465605, 0.9371427904468581, 0.9980274249223061,
                       0.9954498764999694,
                       0.9965523636542679, 0.9999369905878129, 0.008802090925366324, 0.018280263211970738,
                       0.056332398077600154,
                       0.009358646111296443]

        topic_embedding_file = os.path.join('..\\embeddings', topic + 'test_neg.json')
        if not os.path.isfile(topic_embedding_file):
            self.df = pd.DataFrame(columns=['target', 'text', 'embedding'])

            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

            count = 0
            chunk_size = 500
            data_folder = data_dir + '\\*' + topic
            print(f'loading the data from data folder: {data_folder}')
            for i, topic_folder in enumerate(glob.glob(data_folder)):
                for j, sub_folder in enumerate(glob.glob(os.path.join(topic_folder, '*'))):
                    for k, data_file in enumerate(glob.glob(os.path.join(sub_folder, '*'))):
                        # try:
                        print(f'processing {data_file}...')
                        pdf = open(data_file, "r", encoding="utf8")
                        pdf_text = pdf.read()
                        pdf.close()

                        r = range(0, len(pdf_text), chunk_size)
                        n = len(r)
                        embedding_vector = None
                        for m in r:  # txt-content
                            inputs = tokenizer(pdf_text[m:m + chunk_size], return_tensors="pt")
                            outputs = model(**inputs)
                            if embedding_vector is None:
                                embedding_vector = outputs.last_hidden_state[0][0]
                            else:
                                embedding_vector += outputs.last_hidden_state[0][0]
                        embedding_vector /= n

                        if 'non' in topic_folder:
                            row = {'target': 0, 'text': pdf_text,
                                   "embedding": embedding_vector.detach().numpy()}  # non topic papers = 0
                        else:
                            row = {'target': 1, 'text': pdf_text,
                                   "embedding": embedding_vector.detach().numpy()}  # topic papers = 1
                        # row["embedding"].append(embedding_vector_mathy)
                        self.df = self.df.append(row, ignore_index=True)  # fairness_prev = 1
                    # except:
                    # why is paper failing? (not parsed correctly, ) dump all the files that have an error somewhere and look at them
                    # print("Error creating embeddings.")
                    count += count
            print(f'dumping the data to {topic_embedding_file} for caching...')
            self.df.to_json(topic_embedding_file, orient='records')
            self.df = shuffle(self.df)
        else:
            print(f'loading the data from data folder: {topic_embedding_file}')
            self.df = pd.read_json(topic_embedding_file)
            # self.df = shuffle(self.df)
            # self.df.reset_index(inplace=True, drop=True)

            # self.df['pos_mathy'] = mathy_score

        features = self.df['embedding']
        for feature in features:
            arr = np.array(feature)
            self.new_features.append(arr)
        print('Loaded Features')

    def scibert_classifier(self):
        print('Training Teacher Model...')
        labels = self.df['target']
        train_features, test_features, train_labels, test_labels = train_test_split(self.new_features, labels)
        train_labels = train_labels.tolist()
        self.sciBERT_clf.fit(train_features, train_labels)

        # self.plot_learning_curve(svm.SVC(), "SciBERT", self.new_features, labels, axes=None, ylim=None, cv=None, n_jobs=None, graph=0, scoring='f1')
        probabilities = self.sciBERT_clf.predict_proba(self.new_features)
        soft = []
        for probability in probabilities:
            soft.append(probability[1])  # take second value (is this probability of 0 or 1? assumed 1 for now
        self.soft_labels = pd.Series(soft)

        print('Teacher Model Trained')

    def scibert_classifier_2(self):  # train_next_scibert_model
        print('Training Teacher Model...')
        for i in range(self.df['embedding'].size):  # rows?
            test = self.df['embedding'].iloc[i]
            test.append(self.df['pos_mathy'].iloc[i])

        self.new_features = []
        for feature in self.df['embedding']:
            arr = np.array(feature)
            self.new_features.append(arr)

        y = self.df['target']
        x_train, x_test, y_train, y_test = train_test_split(self.new_features, y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)
        print("Teacher model trained")

    def linear_regression(self):
        print('Training Student Model...')
        self.X = self.linreg_vectorizer.fit_transform(self.df['text'])
        y = self.soft_labels
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=0.33, random_state=42)
        self.reg.fit(X_train, y_train)
        print('Student Model Trained')
        # self.plot_learning_curve(LinearRegression(), "Linear Regression", self.X, y, axes=None, ylim=None, cv=None, n_jobs=None, graph=1, scoring='neg_mean_squared_error')

    def get_explanation(self, feed, index):
        print(f'Generating Explanations...')
        # prediction = self.sciBERT_clf.predict_proba(self.new_features)[index, 1]
        print(f'Probability for paper ${index} prediction')
        coef = self.reg.coef_
        exp = (self.X.toarray() * coef)[index]

        # test_neg multiplication
        df = pd.DataFrame(exp / exp.sum(), index=self.linreg_vectorizer.get_feature_names(), columns=["TF-IDF score"])
        df = df.sort_values('TF-IDF score', ascending=False)
        print(f'explanation for ${feed} paper  ${str(index)}')
        print(df.head(10))
        print(self.df['text'][index][0:150])
        print((self.df['embedding'].iloc(index))[-1])
        print("Finished Generating Explanations")

    def get_predictions(self, topic, concept):
        predictions = self.sciBERT_clf.predict(self.new_features)
        ind = np.where(predictions == 1)

        # add target pos_mathy score to papers
        pos = ind[0].tolist()
        selected_ind = [False for _ in range(self.df.shape[0])]
        for i in range(0, self.df.shape[0]):
            if i in pos:
                selected_ind[i] = True
        self.predicted_papers = self.df.iloc[selected_ind]

        # self.predicted_papers = self.df.iloc[lambda x: x.index <= pos[len(pos) - 1]]
        # print(self.predicted_papers)
        # fairness_target = [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        # xai_target = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

        if concept == 'pos_mathy':
            if topic == 'fairness_prev':
                # 0 is positive class
                target = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                          0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
            if topic == 'xai':
                target = [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
        if concept == 'pos_qualitative':
            if topic == 'fairness_prev':
                # target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
                # target = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]

                target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
            if topic == 'xai':
                # target = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
                # target = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]

                target = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        self.predicted_papers['concept'] = target  # how to refactor this?

        # self.predicted_papers = shuffle(self.predicted_papers)
        # self.predicted_papers.reset_index(inplace=True, drop=True)

        # print fairness_prev papers
        df_predicted = self.df['text'][ind[0]]
        df_predicted.reset_index(inplace=True, drop=True)
        print(f'Fairness Papers:\n {df_predicted}')
        print(f'length: {len(ind[0])}')

        # get fairness_prev embeddings
        # for i in ind[0]:
        #     self.predicted_embeddings.append(np.array(self.df['embedding'][i]))

        for i in range(self.df['embedding'].size):
            self.predicted_embeddings.append(np.array(self.df['embedding'][i]))

    def learn_concept(self, topic, concept):
        print('Learning Concept...')
        # predictions = self.loaded_model.predict(self.predicted_embeddings)
        self.loaded_model = pickle.load(open(concept + '_model.pkl', 'rb'))

        soft_predictions = self.loaded_model.predict_proba(
            self.predicted_embeddings)  # all embeddings rather than predicted embeddings
        # soft_predictions = self.loaded_model.predict_proba(self.df['embedding'])

        mathy_score_list = []
        for soft_prediction in soft_predictions:
            mathy_score_list.append(soft_prediction[0])

        # mathy_papers = pd.DataFrame(columns=['target', 'text', 'embedding'])

        # self.mathy_score = np.array(self.mathy_score_list)
        print(f'${concept} score')
        print(mathy_score_list)

        # # print pos_mathy papers
        # # ind = np.where(self.mathy_score < 0.899028) # 0 or 1
        # ind = np.where(self.mathy_score < 0.5)
        # print(concept + ' ' + topic + ' Papers:')
        # print(self.df_predicted[ind[0]])
        # print(f'length: {len(ind[0])}')
        #
        # # print non-pos_mathy papers
        # # ind2 = np.where(self.mathy_score > 0.899028) # 0 or 1
        # ind2 = np.where(self.mathy_score > 0.5)
        # print('non-' + concept + ' ' + topic + ' Papers:')
        # print(self.df_predicted[ind2[0]])
        # print(f'length: {len(ind2[0])}')
        return mathy_score_list

    def svm_classifier(self):
        print('Training TF-IDF Model...')
        label = preprocessing.LabelEncoder()
        y = label.fit_transform(self.df['target'])
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            lowercase=False)  # max_features=1000) # get vectorizer # stop words = english
        X = vectorizer.fit_transform(self.df['text'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        tfidf_clf = svm.SVC(kernel="linear", random_state=42)
        tfidf_clf.fit(X_train, y_train)

        print('TF-IDF Model Trained')

        self.plot_learning_curve(svm.SVC(), "TF-IDF", X, y, ylim=None, cv=None, n_jobs=None, graph=2,
                                 scoring='f1')
        return tfidf_clf

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None, n_jobs=None,
                            train_sizes=np.linspace(.2, 1.0, 5), graph=None, scoring=None):
        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)
        fig, ax = plt.subplots()
        ax.set_title(title + " (Fairness Feed)")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Number of Positive Training Examples")
        if graph == 0 or graph == 2:
            ax.set_ylabel("Score (F1)")
        else:
            ax.set_ylabel("Score (Mean Squared Error)")
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                                                                train_sizes=train_sizes, return_times=False)
        # https: // scikit - learn.org / stable / modules / model_evaluation.html
        train_sizes = train_sizes / 2
        # https: // scikit - learn.org / stable / modules / generated / sklearn.model_selection.learning_curve.html
        if scoring == 'neg_mean_squared_error':
            train_scores_mean = -1 * np.mean(train_scores, axis=1)
            train_scores_std = -1 * np.std(train_scores, axis=1)
            test_scores_mean = -1 * np.mean(test_scores, axis=1)
            test_scores_std = -1 * np.std(test_scores, axis=1)
        else:
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
        ax.grid(False)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score", linewidth=5.0)
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score", linewidth=5.0)
        ax.set_xlim([5, 25])
        ax.set_ylim([0, 1.1])
        if graph == 2:
            ax.legend(loc="lower right")
        now = datetime.datetime.now()
        # plt.savefig('/Users/Cindy Su/source/Mathy/learning_curves/learning_curve_' + now.strftime("%Y_%m_%d_%H_%M_%S_") + str(graph))
        if graph == 2:
            plt.show()

    def plot_precision_recall(self):
        # precision, probabilities, targets
        print("Plotting precision recall curve")
        print(self.predicted_papers)
        lr_probs = self.loaded_model.predict_proba(self.predicted_papers['embedding'].tolist())
        lr_probs = lr_probs[:, 0]
        # predict class values
        yhat = self.loaded_model.predict(self.predicted_papers['embedding'].tolist())
        precision, recall, thresholds = precision_recall_curve(self.predicted_papers['concept'].tolist(), lr_probs,
                                                               pos_label=0)
        lr_f1, lr_auc = f1_score(self.predicted_papers['concept'].tolist(), yhat), auc(recall, precision)
        # summarize scores
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        fscore = (2 * precision * recall) / (precision + recall)
        ix = argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

        no_skill = len(self.predicted_papers['concept'][self.predicted_papers['concept'] == 0]) / len(
            self.predicted_papers['concept'])
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Logistic')
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        print("Plotted precision recall curve")

    def plot_precision_recall_2(self, topic, concept):
        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)

        X_test = self.predicted_papers['embedding'].tolist()
        y_test = self.predicted_papers['concept'].tolist()

        # pos_mathy/(non-pos_mathy + pos_mathy)
        mathy = 0
        non_mathy = 0
        for y in y_test:
            if y == 0:
                mathy += 1
            else:
                non_mathy += 1
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

        plot_precision_recall_curve(self.loaded_model, X_test, y_test, name="Mathy Model", pos_label=0,
                                    linewidth=5.0)  # pipeline, X_test, y_test
        # plt.plot([0, 1], [pos_mathy/(neg_mathy + pos_mathy), pos_mathy/(neg_mathy + pos_mathy)], c='k')

        # plt.set_xlim([5, 25])
        plt.ylim([0, 1.02])

        plt.title("Fairness/Mathy Precision-Recall Curve")
        # plt.title(topic + "/" + concept + " Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        # plt.title("XAI_pos/Mathy Precision-Recall Curve")

        # plot_roc_curve(self.loaded_model, X_test, y_test, pos_label=0) # use when both classes are balanced
        # plt.plot([0,1], [0,1], c='k') # 0.5 baseline
        # plt.title(topic + "/" + concept + " ROC Curve")

        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--topic")
    parser.add_argument("--learning_curve", required=False, action="store_true")
    parser.add_argument("--learning_curve_output", required=False)
    parser.add_argument("--concept")
    args = parser.parse_args()
    r = Recommend()
    r.preprocess_papers(args.data_dir, args.topic)
    r.scibert_classifier()
    r.linear_regression()
    r.get_predictions(args.topic, args.concept)
    concept_scores = r.learn_concept(args.topic, args.concept)
    r.scibert_classifier_2()
    # r.scibert_classifier_2(concept_scores)
    r.plot_precision_recall_2(args.topic, args.concept)
    # r.get_explanation(1)

    # TODO: read score from file
    # TODO: 1)scibert_model_1 2)student_model 3)apply_concept_and_generate_scores 4)feed_scores_into_scibert_model_2
    #  5)train_scibert_model_2 6)generate_explanation (take in last embedding value for concept explanation)


main()
