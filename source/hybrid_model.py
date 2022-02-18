"""
usage:
python recommend.py --data_dir data_dir --topic topic_name --concept concept_name

For example:
python recommend.py --data_dir .\papers_processed --topic fairness_prev --concept pos_mathy

Use conda to set env
conda env export > environment.yml
"""

#TODO:
# get feed of 300 random arxiv papers
# train the two concepts on the test set
# check top 10 recommended papers at 0, 250, 1000 weights
# BASICALLY have no target for y --> only use weights to differentiate
# 1. generate embeddings, but with no labels in df (method for this?) - done
# 2. learn concept with train() + either feed, then try on testset
# 3. ur done

#TODO:
# get dataset papers embeddings
# train pipeline on papers
# see plots (whether they indicate good results or not)

import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel
from eval import Eval
from classify import classify
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from tqdm import tqdm
from time import sleep
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


class RecommendationDataset:
    def __init__(self):
        self.loaded_model = None
        self.df = None
        self.vocabulary = None
        self.predicted_papers = None
        self.predicted_embeddings = []

    def preprocess_papers(self, data_dir, feed):
        self.feed = feed
        topic_embedding_file = os.path.join('..\\embeddings', feed + '.json')
        # topic_embedding_file = os.path.join('..', 'embeddings', 'dataset.json')
        if not os.path.isfile(topic_embedding_file):
            # self.df = pd.DataFrame(columns=['target', 'text', 'embedding'])
            self.df = pd.DataFrame(columns=['text', 'embedding'])

            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

            count = 0
            chunk_size = 500
            data_folder = data_dir + os.path.sep + 'papers_processed' + os.path.sep + '*' + feed + '*'
            # data_folder = os.path.sep(data_dir, 'papers_processed', '*' + feed)
            print(f'loading the data from data folder: {data_folder}')
            for i, topic_folder in enumerate(glob.glob(data_folder)): #TODO: should work with __pos and __neg
                for j, sub_folder in enumerate(glob.glob(os.path.join(topic_folder, '*'))):
                    for k, data_file in enumerate(glob.glob(os.path.join(sub_folder, '*'))):
                        try:
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

                            if 'neg' in topic_folder:
                                row = {'target': 0, 'text': pdf_text,
                                       "embedding": embedding_vector.detach().numpy()}  # non topic papers = 0
                            else:
                                row = {'target': 1, 'text': pdf_text,
                                       "embedding": embedding_vector.detach().numpy()}  # topic papers = 1

                            # row = {'text': pdf_text, 'embedding': embedding_vector.detach().numpy()}
                            self.df = self.df.append(row, ignore_index=True)  # fairness_prev = 1
                        except:
                            print("error")
                        count += count
            print(f'dumping the data to {topic_embedding_file} for caching...')
            self.df.to_json(topic_embedding_file, orient='records')
        else:
            # print(f'loading the data from data folder: {topic_embedding_file}')
            self.df = pd.read_json(topic_embedding_file)

            #self.df = shuffle(self.df)
            #self.df.reset_index(inplace=True, drop=True)

            # testing (adds noise)
            # print(f"shape before merging xai: {self.df.shape}")
            # xai_df = pd.read_json(r"C:\Users\Cindy Su\source\Mathy\embeddings\XAI_pos.json")
            # target_1_xai_df = xai_df[xai_df['target'] == 1].head(5) # gets papers from the xai introduces_dataset_papers_pos
            # target_1_xai_df = target_1_xai_df.assign(target=1)
            # target_0_xai_df = xai_df[xai_df['target'] == 0].head(5)
            # target_0_xai_df = target_0_xai_df.assign(target=1)
            # self.df = pd.concat([self.df, target_1_xai_df, target_0_xai_df])
            # # self.df = pd.concat([self.df, target_1_xai_df])
            # print(f"shape after merging xai: {self.df.shape}")

    def get_opaque_features(self):
        """returns a numpy matrix of dimension number of examples x number of opaque features"""
        return np.stack(self.df['embedding'])

    def get_text(self):
        return np.stack(self.df['text'])

    def get_text_by_index(self, index):
        return self.df['text'][index][0:200]

    def get_target(self):
        """returns an array of the feed targets"""
        return np.stack(self.df['target'])

    """embeddings?"""
    # def load_embeddings(self):
    #     self.X = self.get_opaque_features()
    #     self.Y = self.get_target()
    #
    #     self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=0.25)
    #
    #     self.train_x = np.array(self.train_x)
    #     self.train_y = np.array(self.train_y)
    #     self.test_x = np.array(self.test_x)
    #     self.test_y = np.array(self.test_y)
    #
    # def train(self):
    #     x, y = self.train_x, self.train_y
    #     model = LogisticRegression()
    #     model.fit(x, y)
    #     score = model.score(x, y)
    #     print(f"Model score: {score}")
    #     self.model = model
    #     pickle.dump(self.model, open('qualitative_model_using_hybrid_model_code.pkl', 'wb'))
    #     print('model saved')


class HybridModel:
    def __init__(self, dataset, update_alg, seed=46, model_type="logistic"):
        # opaque features, concept, explanatory vocabulary, train data, test data
        self.loaded_model = None
        self.new_features = []
        self.dataset = dataset
        self.concept_list = []
        self.update_alg = update_alg
        self.model_type = model_type
        self.explanatory_model = None
        self.seed = seed

    def get_concept_features(self, concept, opaque_features):
        """returns numpy matrix of dimensions number of examples x number of concept features"""
        concept_features = concept.concept_model.predict_proba(opaque_features)
        return concept_features

    def get_clf_class(self):
        if(self.model_type=='svm'):
            return svm.SVC(kernel='linear', random_state=self.seed, probability=True)
        elif(self.model_type=='dummy'):
            return DummyClassifier(strategy='uniform', random_state=self.seed)
        elif(self.model_type=='logistic'):
            return LogisticRegression(random_state=self.seed)# , class_weight='balanced')
        elif(self.model_type=='lasso'):
            return linear_model.Lasso(alpha=0.1)
        # TODO: add other model types

    # def get_tfidf_features(self, opaque_features): # used for testing
    #     self.linreg_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    #     # get tfidf vectorized features
    #     X = self.linreg_vectorizer.fit_transform(self.introduces_dataset_papers_pos.get_text())
    #
    #     new_features = opaque_features
    #     for concept in self.concept_list:
    #         concept_features = self.get_concept_features(concept, opaque_features)
    #         new_features = np.column_stack([new_features, concept_features[:,0].reshape(-1, 1)])
    #         # upadte tfidf vectorized features here
    #     return new_features
    #     return X

    def train(self):
        self.X = self.dataset.get_opaque_features()
        # self.X = self.get_features(opaque_features)
        self.Y = self.dataset.get_target()
        self.docs = self.dataset.get_text()
        self.X_train, self.X_test, self.y_train, self.y_test, self.docs_train, self.docs_test = \
            train_test_split(self.X, self.Y, self.docs, test_size=0.33, random_state=self.seed)
        self.clf_model = self.get_clf_class()
        self.clf_model.fit(self.X_train, self.y_train)

    def get_model(self):
        return self.clf_model

    def refine_and_train(self, new_concept, weight=1.0):
        self.refine_vocabulary(new_concept)
        self.limeade_update(new_concept, weight)
        self.fit_global_explainer()

    def refine_vocabulary(self, new_concept):
        """appends new concept to self.concept_list"""
        self.concept_list.append(new_concept)

    def add_pseudo(self, new_concept, weight):
        """update algorithm 1"""
        # x_pseudo = new_concept.get_representative_X_2()
        x_pseudo = new_concept.get_representative_X()
        y_pseudo = 1
        X_train_new = np.vstack((self.X_train, x_pseudo))
        Y_train_new = np.append(self.y_train, [y_pseudo])
        sample_weight = np.ones(X_train_new.shape[0])
        sample_weight[-1] = weight
        sample_weight = sample_weight/np.sum(sample_weight)
        return X_train_new, Y_train_new, sample_weight

    def add_pseudo_balanced(self, new_concept, weight):
        """update algorithm 2"""
        # x_pseudo = new_concept.get_representative_X_2()
        x_pseudo = new_concept.get_representative_X()
        y_pseudo = 1
        X_train_new = np.vstack((self.X_train, x_pseudo))
        Y_train_new = np.append(self.y_train, [y_pseudo]) #TODO: define y_train and make everything the same value
        sample_weight = np.ones(X_train_new.shape[0])

        num_examples = self.X_train.shape[0]
        norm_neg_examples = num_examples
        norm_pos_examples = num_examples + 2 * weight
        for i, weight in enumerate(sample_weight):
            if Y_train_new[i]==0:
                sample_weight[i] = sample_weight[i]/norm_neg_examples
            else:
                sample_weight[i] = sample_weight[i]/norm_pos_examples
        sample_weight = sample_weight * norm_pos_examples
        return X_train_new, Y_train_new, sample_weight

    def adjust_weights(self, new_concept, weight):
        """update algorithm 3"""
        concept_score = new_concept.concept_model.predict_proba(self.X_train)[:,1]
        X_train_new = self.X_train
        Y_train_new = self.y_train
        # sample_weight = np.ones(X_train_new.shape[0])
        sample_weight = 1 + weight * concept_score
        return X_train_new, Y_train_new, sample_weight

    def test(self, new_concept, weight):
        """update algorithm 4"""
        # x_pseudo = new_concept.get_representative_X_2()
        x_pseudo = new_concept.get_representative_X()
        y_pseudo = 0
        X_train_new = np.vstack((self.X_train, x_pseudo))
        Y_train_new = np.append(self.y_train, [y_pseudo])
        sample_weight = np.ones(X_train_new.shape[0])
        sample_weight[-1] = weight
        sample_weight = sample_weight/np.sum(sample_weight)
        return X_train_new, Y_train_new, sample_weight

    def limeade_update(self, new_concept, weight):
        """appends pseudo-example to training data and updates the model"""

        if(self.update_alg=='alg1'):
            X_train_new, Y_train_new, sample_weight = self.add_pseudo(new_concept, weight)
        elif(self.update_alg=='alg2'):
            X_train_new, Y_train_new, sample_weight = self.add_pseudo_balanced(new_concept, weight)
        # TODO: add algorithm
        elif(self.update_alg=='alg3'):
            X_train_new, Y_train_new, sample_weight = self.adjust_weights(new_concept, weight)
        elif(self.update_alg=='alg4'):
            X_train_new, Y_train_new, sample_weight = self.test(new_concept, weight)

        self.clf_model.fit(X_train_new, Y_train_new, sample_weight=sample_weight)

    def fit_global_explainer(self):
        # self.explanatory_model = LogisticRegression(penalty="l1", solver="saga")
        # self.explanatory_model = linear_model.Lasso(alpha=0.01)
        self.explanatory_model = linear_model.Ridge(alpha=1)

        training_docs = self.docs_train
        X_opaque = self.X_train

        self.linreg_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.linreg_vectorizer.fit(training_docs)
        self.exp_vocab = self.linreg_vectorizer.get_feature_names()

        for concept in self.concept_list:
            self.exp_vocab.append(concept.get_concept_name()) #TODO:

        X_explanation = self.get_explanation_features(training_docs, X_opaque)

        # scaler = MaxAbsScaler()
        scaler = StandardScaler()
        X_explanation = scaler.fit_transform(X_explanation)

        Y_opaque_pos = self.clf_model.predict_proba(X_opaque)[:,1] # this prediction seems right
        # Y_opaque_pos = self.clf_model.predict(X_opaque)
        self.explanatory_model.fit(X_explanation, Y_opaque_pos)

    def get_explanation_features(self, docs, X_opaque): # returns tfidf features with concept scores added
        X_tfidf = self.linreg_vectorizer.transform(docs).toarray()
        features_list = [X_tfidf]

        for concept in self.concept_list:
            concept_features = self.get_concept_features(concept, X_opaque)[:,1] #TODO:
            concept_features = np.expand_dims(concept_features, axis=1)
            features_list.append(concept_features)
        X_explanation = np.concatenate(features_list, axis=1)
        return X_explanation

    def get_global_explanation(self, explanation_budget=10):
        coef = self.explanatory_model.coef_

        print(f"coef: {np.sum(coef)}")
        print(f"coef.shape: {coef.shape}")
        print(len(self.exp_vocab))

        importance = coef

        # importance = coef * (coef > 0)
        # importance = abs(importance)/np.sum(abs(importance))

        importance = importance + abs(np.amin(importance))
        importance = importance/np.sum(importance)
        print(f'sum: {np.sum(importance)}')
        explanation = {self.exp_vocab[i]:importance[i] for i in range(importance.shape[0])}

        relative_importance = (coef - np.amin(coef)) / (np.amax(coef) - np.amin(coef))
        relative_explanation = {self.exp_vocab[i]:relative_importance[i] for i in range(relative_importance.shape[0])}

        top_features_idxs = (-importance).argsort()[:explanation_budget]
        top_feature_values = importance[top_features_idxs]
        top_feature_names = np.array(self.exp_vocab)[top_features_idxs]
        print(top_feature_names, top_feature_values)
        return explanation, relative_explanation

    def get_top_10_predictions(self):
        file = os.path.join('..', 'embeddings', 'arxiv.json')
        df = pd.read_json(file)
        arxiv_embeddings = np.stack(df['embedding'])

        predictions = self.clf_model.predict_proba(arxiv_embeddings)[:,1]
        df['predictions'] = predictions
        df = df.sort_values('predictions', ascending=False)
        top_10_papers = df['text'].head(10)
        for paper in top_10_papers:
            ind = paper.find('arXiv:')
            print(paper[ind:ind+15])

    def get_prediction(self, X_opaque):
        """predict whether a paper is aligned with the concept"""
        return np.array(self.clf_model.predict_proba(X_opaque))

    def get_features(self, opaque_features):
        """returns numpy matrix of dimensions number of examples x number of concept + opaque features"""
        new_features = opaque_features
        for concept in self.concept_list:
            concept_features = self.get_concept_features(concept, opaque_features)
            new_features = np.column_stack([new_features, concept_features[:,0].reshape(-1, 1)])
        return new_features

    def get_local_explanation(self, X_opaque, raw_text, explanation_budget=10):
        """provide an explanation for why the paper was recommended in terms of the prediction"""
        # if self.explanatory_model==None:
        #     self.fit_global_explainer() # should also fit global explainer when a new concept is introduced
        self.fit_global_explainer()
        coef = self.explanatory_model.coef_
        X_explanation = self.get_explanation_features(raw_text, X_opaque)
        # print(X_explanation.shape, coef.shape)

        importance = X_explanation * coef
        importance = importance/np.sum(importance)
        # print(f'importance shape: {importance.shape}')
        # top_features_idxs = np.argpartition(importance, -explanation_budget)[-explanation_budget:]
        top_features_idxs = (-importance).argsort(axis=1)[:,:explanation_budget]
        importance_list = []
        feature_name_list = []
        # print("vocabulary length: " + len(self.exp_vocab))

        for i in range(top_features_idxs.shape[0]): # see if this can be done without a forloop
            importance_list.append(importance[i, top_features_idxs[i,:]])
            feature_name_list.append([self.exp_vocab[idx] for idx in top_features_idxs[i]])
            # feature_name_list.append(self.exp_vocab[top_features_idxs[i,:]])
        top_feature_values = np.stack(importance_list, axis=0)
        top_feature_names = np.stack(feature_name_list, axis=0)

        # top_feature_names = np.tile(self.exp_vocab, (raw_text.size, 1))[top_features_idxs]
        # print(top_features_idxs.shape, top_feature_values.shape, top_feature_names.shape)
        for i in range(raw_text.size):
            print(f'paper id = {i}')
            print(f'predicted class = {self.clf_model.predict(X_opaque)[i]}')
            print(top_feature_names[i])
            print(top_feature_values[i])


        # for i in range(raw_text.size):
        #     print(X_explanation.shape, coef.shape)
        #     exp = (X_explanation * coef)[i]
        #     df = pd.DataFrame(exp/exp.sum(), index=self.exp_vocab, columns=["Feature Importance"])
        #     df = df.sort_values('Feature Importance', ascending=False)
        # return df.head(num_features)


    # def refine(self, concept):
    #     loaded_model = pickle.load(open(concept + '_model.pkl', 'rb'))
    #     concept_features = []
    #     if concept not in self.concept_list:
    #         soft_predictions = loaded_model.predict_proba(self.predicted_embeddings)
    #         for soft_prediction in soft_predictions:
    #             concept_features.append(soft_prediction[0])
    #     self.df[concept] = concept_features
    #     for i in range(self.df['embedding'].size):
    #         self.df['embedding'].iloc[i].append(self.df[concept].iloc[i])
    #     y = self.df['target']
    #     x_train, x_test, y_train, y_test = train_test_split(self.df['embedding'], y)
    #     clf = LogisticRegression(max_iter=1000)
    #     clf.fit(x_train, y_train)  # TODO: why is there an error here?
    #     self.concept_list.append(concept)

        # args: concept
        # reasoning not in the explanatory vocabulary --> changes explanatory vocabulary so it can later be used by explanation methods
        # similar example in train already implemented... is this the same thing? should I move the code in train here instead?

    def plot_concept_weight(self, data_df, topic, concept_name, alg, max_weight, seed_num):
        font = {'family': 'DejaVu Sans',
                'weight': 'normal',
                'size': 16}
        plt.rc('font', **font)
        rc = {'lines.linewidth': 5, 'lines.markersize': 10, 'lines.edgecolor': None}
        sns.set_context(rc=rc)

        plt.figure(1, figsize=(8.55, 5.5))
        plt.grid()
        sns.lineplot(data=data_df, x="concept weight", y="importance score", linewidth=5)
        # plt.title(concept_name.capitalize() + " (" + topic.capitalize() + " Feed), " + alg + ", " + seed_num
        #           + " seeds")
        plt.title(concept_name.capitalize() + " (" + topic.capitalize() + " Feed)")
        plt.xlabel("Concept Weight")
        plt.ylabel("Concept Importance")
        # plt.xticks(np.arange(0, 1000, 200))
        # plt.xlim([600, 800])
        name = concept_name + "_" + topic + "_importance_" + alg + "_" + max_weight + "_" + seed_num
        # plt.savefig(f"..\\experiment2_final_plots\\{name}")
        plt.savefig(os.path.join('..', "experiment2_dataset_concept", name))

        plt.figure(2, figsize=(8.55, 5.5))
        plt.grid()
        sns.lineplot(data=data_df, x="concept weight", y="accuracy", linewidth=5)
        plt.title(concept_name.capitalize() + " (" + topic.capitalize() + " Feed)")
        plt.xlabel("Concept Weight")
        plt.ylabel("Model Accuracy on Test Set")
        # plt.xticks(np.arange(0, 1000, 200))
        # plt.yticks(np.arange(0.4, 1.01, 0.2))
        plt.ylim([0.3, 1])
        name = concept_name + "_" + topic + "_accuracy_" + alg + "_" + max_weight + "_" + seed_num
        # plt.savefig(f"..\\experiment2_final_plots\\{name}")
        plt.savefig(os.path.join('..', "experiment2_dataset_concept", name))

        plt.figure(3, figsize=(8.55, 5.5))
        plt.grid()
        sns.lineplot(data=data_df, x="concept weight", y="feature rank", linewidth=5)
        plt.title(concept_name.capitalize() + " (" + topic.capitalize() + " Feed)")
        plt.xlabel("Concept Weight")
        plt.ylabel("Feature Rank")
        # plt.xlim([600, 800])
        # plt.ylim([17000, 17800])
        name = concept_name + "_" + topic + "_rank_" + alg + "_" + max_weight + "_" + seed_num
        # plt.savefig(f"..\\experiment2_final_plots\\{name}")
        plt.savefig(os.path.join('..', "experiment2_dataset_concept", name))

        plt.figure(4, figsize=(8.55, 5.5))
        plt.grid()
        sns.lineplot(data=data_df, x="concept weight", y="relative importance", linewidth=5)
        plt.title(concept_name.capitalize() + " (" + topic.capitalize() + " Feed)")
        plt.xlabel("Concept Weight")
        plt.ylabel("Relative Importance of Concept")
        # plt.xlim([600, 800])
        name = concept_name + "_" + topic + "_relative_importance_" + alg + "_" + max_weight + "_" + seed_num
        # plt.savefig(f"..\\experiment2_final_plots\\{name}")
        plt.savefig(os.path.join('..', "experiment2_dataset_concept", name))

        plt.show()


class HumanConcept:
    def __init__(self, concept_name, checkpoint_path):
        self.concept_name = concept_name
        self.checkpoint_path = checkpoint_path
        self.concept_model = pickle.load(open(f'.\\{checkpoint_path}', 'rb'))

    def get_concept_embedding(self):
        return self.concept_model.coef_

    # def get_concept_features(self):
    #     # no longer used
    #     concept_features = self.concept_classifier.get_features(r"C:\Users\Cindy Su\source\{self.concept_name()}\papers_processed", self.concept_name)
    #     return np.stack(concept_features['text']), np.stack(concept_features['target'])

    def get_concept_name(self):
        return self.concept_name

    def get_representative_X(self):
        # TODO: check data
        # train_x = np.load(os.path.join('embeddings', self.get_concept_name(), 'train_x.npy'))
        # train_y = np.load(os.path.join('embeddings', self.get_concept_name(), 'train_y.npy'))
        # test_x = np.load(os.path.join('embeddings', self.get_concept_name(), 'test_x.npy'))
        # test_y = np.load(os.path.join('embeddings', self.get_concept_name(), 'test_y.npy'))

        train_x = np.load(os.path.join('embeddings' + self.get_concept_name(), 'train_x.npy'))
        train_y = np.load(os.path.join('embeddings' + self.get_concept_name(), 'train_y.npy'))
        test_x = np.load(os.path.join('embeddings' + self.get_concept_name(), 'test_x.npy'))
        test_y = np.load(os.path.join('embeddings' + self.get_concept_name(), 'test_y.npy'))

        X = np.concatenate((train_x, test_x))
        Y = np.concatenate((train_y, test_y))
        df = pd.DataFrame(columns=['embedding', 'target'])
        df['embedding'] = pd.Series(X.tolist())
        df['target'] = pd.Series(Y)
        pos_embeddings = df.query('target==1') #TODO:
        pos_embeddings = pos_embeddings['embedding'].to_list()
        pos_embeddings = np.array(pos_embeddings)
        return np.mean(pos_embeddings, axis=0)

    def get_representative_X_2(self):
        topic_embedding_file = os.path.join('..', 'embeddings', 'dataset.json')
        df = pd.read_json(topic_embedding_file)

        pos_embeddings = df.query('target==1')  # TODO:
        pos_embeddings = pos_embeddings['embedding'].to_list()
        pos_embeddings = np.array(pos_embeddings)
        return np.mean(pos_embeddings, axis=0)


def main():
    if os.name == "posix":
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--feed")
    parser.add_argument("--model_type")
    parser.add_argument("--weight", type=int)
    args = parser.parse_args()

    r = RecommendationDataset()
    r.preprocess_papers(args.data_dir, args.feed)

    # r.load_embeddings()
    r.train()

    e = Eval()

    h = HybridModel(r, args.model_type)
    h.train()

    test_docs = r.get_text()[:3]
    X_opaque_test = r.get_opaque_features()[:3,:]

    # __________________
    print(r.get_target()[:3])
    h.explain(X_opaque_test, test_docs)

    explanation = h.explain(X_opaque_test, test_docs)
    print(explanation)
    e.compute_classification_report(h.get_model(), h.X_test, h.y_test)
    e.plot_precision_recall_curve(h.get_model(), h.X_test, h.y_test) # h.evaluate()

    e.plot_learning_curve_new(h.get_model(), h.X_train.toarray(), h.y_train)
    # # e.plot_learning_curve_new(h.get_model(), h.X_train, h.y_train)
    # # e.plot_learning_curve_new(h.get_model(), h.X.toarray(), h.Y)
    # # e.plot_learning_curve_new(h.get_model(), h.X, h.Y)
    #
    # #
    # # test_docs = []
    # # # ind = 25
    # # test_docs.append(r.get_opaque_features()) # [ind])
    # # test_predictions = h.get_prediction(test_docs) # test_examples is list or single example
    #
    #
    # # explanation = h.explain(r.get_opaque_features(), r.get_text()) # can add num features
    # # print(explanation)
    # __________________________

    # c = classify()
    # concept = "pos_mathy"
    # concept = HumanConcept(concept, f"{concept}_model.pkl", c)
    # print(f'weight = {args.weight}')
    #
    # h.refine_and_train(concept, weight=args.weight)
    # # h.get_local_explanation(X_opaque_test, test_docs)
    # explanation = h.get_global_explanation()
    # print(f"concept importance: {explanation['concept: ' + concept.get_concept_name()]}")
    #
    # e.compute_classification_report(h.clf_model, h.X_test, h.y_test)

def experiment2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--feed")
    parser.add_argument("--concept")
    parser.add_argument("--model_type")
    parser.add_argument("--algorithm")
    args = parser.parse_args()

    r = RecommendationDataset()
    r.preprocess_papers(args.data_dir, args.feed)
    print("Papers preprocessed.")
    e = Eval()

    # concept = "pos_mathy"
    # c = classify()
    # concept = HumanConcept(concept, f"{concept}_model.pkl", c)

    # weights = np.concatenate((np.linspace(0, 20, 20), np.linspace(20, 50, 10)))
    # weights = np.array([0, 1, 2, 5, 10, 100, 1000])

    # seed = 21
    # for weight in weights:
    #     h = HybridModel(r, seed, args.model_type)
    #     h.train()
    #     h.refine_and_train(concept, weight=weight)
    #     explanation = h.get_global_explanation()
    #
    #     # add something if concept_importance = 0?
    #     concept_importance = explanation[concept.get_concept_name()]
    #     importance_scores = np.append(importance_scores, [concept_importance])
    #
    #     accuracy = h.clf_model.score(h.X_test, h.y_test)
    #     accuracy_scores = np.append(accuracy_scores, [accuracy])
    #
    #     # values = list(explanation.values())
    #     # values.sort(reverse=True)
    #     # rank = values.index([concept_importance]) + 1
    #
    #     sorted_features = sorted(explanation.keys(), key=lambda x:-explanation[x])
    #     rank = sorted_features.index(concept.get_concept_name())
    #     rank_scores = np.append(rank_scores, rank)
    #
    #     print(f'importance: {concept_importance}', f'accuracy: {accuracy}', f'rank: {rank}')

    # weights = np.concatenate((np.linspace(0, 600, 50), np.linspace(600, 800, 200)))

    # weights = np.linspace(0, 3000, 600)
    weights = np.linspace(0, 100, 20)
    max_weight = np.amax(weights)
    seeds = range(0, 1)

    data = {'concept weight':[], 'seed':[],'importance score':[],'accuracy':[], 'feature rank':[], 'relative importance':[]}

    concept_path = f'{args.concept}_model.pkl'
    concept = HumanConcept(args.concept, concept_path)

    for i, weight in enumerate(weights):
        for j, seed in enumerate(seeds):
            h = HybridModel(r, args.algorithm, seed, args.model_type)
            h.train()
            h.refine_and_train(concept, weight=weight)
            explanation, relative_explanation = h.get_global_explanation()

            sorted_features = sorted(explanation.keys(), key=lambda x:-explanation[x])
            rank = sorted_features.index(concept.get_concept_name())

            data['concept weight'].append(weight)
            data['seed'].append(seed)
            data['importance score'].append(explanation[concept.get_concept_name()])
            data['accuracy'].append(h.clf_model.score(h.X_test, h.y_test))
            data['feature rank'].append(rank)
            data['relative importance'].append(relative_explanation[concept.get_concept_name()])

    data_df = pd.DataFrame(data)

    # USE THIS CODE TO STORE DATA_DF
    # data_df.to_csv(os.path.join('..', 'experiment2_dataset_concept', 'test.csv'))

    # loaded_data_df = pd.read_csv('../experiment2_data/mathy_fairness_concept_weight_data_50_seeds_alg_3.csv')
    # loaded_data_df = pd.read_csv('../experiment2_data/ridge_' + args.concept + '_' + args.feed + '.csv')
    # loaded_data_df = pd.read_csv(os.path.join('..', 'experiment2_dataset_concept', 'test.csv'))

    h = HybridModel(r, args.model_type)
    # h.plot_concept_weight(loaded_data_df, args.feed, concept.get_concept_name(), args.algorithm, str(int(max_weight)), str(len(seeds)))
    h.plot_concept_weight(data_df, args.feed, concept.get_concept_name(), args.algorithm, str(int(max_weight)), str(len(seeds)))

def experiment3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--feed")
    parser.add_argument("--concept")
    parser.add_argument("--model_type")
    parser.add_argument("--algorithm")
    args = parser.parse_args()

    r = RecommendationDataset()
    r.preprocess_papers(args.data_dir, args.feed)

    concept_path = f'new_{args.concept}_model.pkl'
    concept = HumanConcept(args.concept, concept_path)

    weight = 1000
    h = HybridModel(r, args.algorithm, 46, args.model_type)
    h.train()
    h.refine_and_train(concept, weight=weight)
    h.get_top_10_predictions()


def preprocess():
    if os.name == "posix":
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--feed")
    args = parser.parse_args()

    r = RecommendationDataset()
    r.preprocess_papers(args.data_dir, args.feed)


main()
# experiment2()
# experiment3()
# preprocess()