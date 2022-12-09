import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class SentimentAnalysisModel(object):

    SENTIMENT_DICT = {-1 : 'Negative', 1 : 'Positive'}

    def __init__(self, model_sav_file, data_file):

        # load model from SAV file
        self.loaded_model = pickle.load(open(model_sav_file, 'rb'))

        # train_data
        self.Data = pd.read_csv(data_file)
        # shuffle dataset
        self.Data = self.Data.sample(frac=1)
        # extracting 'Content'
        self.X_data = self.Data['Content']


    def make_vector(self, sentence):

        # create the transform
        vectorizer = CountVectorizer()
        # tokenize and build vocab
        vectorizer.fit(self.X_data)
        # encode document
        vector = vectorizer.transform(sentence)
        # converting dataframe into numpy array
        sentence_vector = vector.toarray()
        # return final vector
        return sentence_vector


    def predict_sentiment(self, vec_sentence):
        # predict the sentiment

        self.preds = self.loaded_model.predict(vec_sentence)
        return SentimentAnalysisModel.SENTIMENT_DICT[self.preds[0]]
