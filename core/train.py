import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

# for neural network
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

from util.consolelog import Console

# ploting
from keras.utils.vis_utils import plot_model


class MachineLearning():
    def __init__(self):
        self.df = ''
        self.vectorizer = ''
        self.posts = ''
        self.model = ''

    def train(self, path_dataset, path_model,encoded):
        df = pd.read_csv(path_dataset,encoding=encoded)
        vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
        posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
        transformed_posts=pd.DataFrame(posts)
        df=pd.concat([df,transformed_posts],axis=1)
        print(df.isnull().any())
        X=df[df.columns[2:]]
        y=df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # using simple neural network
        # note: - berapa bobotnya, layernya, berapa banyak bobot setiap layer.
        #       - berapa banyak data training, data testing.
        #       - nn jenis apa.
        # buat diagram bebas. (gambaran umumnya)
        input_dim = X_train.shape[1]
        model = Sequential()
        model.add(layers.Dense(20, input_dim=input_dim, activation="relu"))
        model.add(layers.Dense(10, activation='tanh'))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
        model.summary()
        classifier_nn = model.fit(X_train, y_train, epochs=10, verbose=True, validation_data=(X_test, y_test), batch_size=15)
        model.save(path_model)

        loss_values = classifier_nn.history['loss']
        epochs = range(1, len(loss_values)+1)
        # plt.plot(epochs, loss_values, label='Training Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()

        # plt.show()

        # o = model.predict(X_test)
        # print(o[0])
        print("saved trained data to {}".format(path_model))

    def visalizedong(self, model):
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def preps_predict(self, path_dataset, path_model, encoded):
        self.df = pd.read_csv(path_dataset,encoding=encoded)
        self.vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
        self.posts = self.vectorizer.fit_transform(self.df['Sentence'].values.astype('U')).toarray()
        self.model = load_model(path_model)

    def nn_predictions(self, payload):
        x_predict = [payload] # ini data yang bakal di deteksi
        x_predict = self.vectorizer.transform(x_predict)
        # print(x_predict)
    
        x = self.model.predict(x_predict)
        Console.info("accuracy: {} ".format(x[0][0]))
        # print(round(x[0][0])) 
        return round(x[0][0])


# welcome()
# sqli = utf-16
# xss  = utf-8
# x = MachineLearning()
# x.train("./dataset/sqli.csv", "./model/sqli.h5","utf-16")
# x.preps_predict( "./dataset/sqli.csv", "./model/sqli.h5","utf-16")
# x.nn_predictions("/oke.php?id=1")