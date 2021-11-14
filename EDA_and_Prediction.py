# import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
# from matplotlib import rcParams
import seaborn as sns
# from textblob import TextBlob
# from plotly import tools
# import plotly.graph_objs as go
# from plotly.offline import iplot

# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords


# import spacy
# from spacy import displacy

# import string
# import re
# import bs4 as BeautifulSoup
# import fasttext

# import re
# import itertools
# from collections import Counter

# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

# from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")

#data read
df = pd.read_csv("F:/Work/Data Science/Bike Purchase Prediction/Data/bike_buyers_clean.csv")
print(df.info())
print(df.head())


sns.histplot(x="Purchased Bike", data=df, hue="Age")
plt.show()
# #data clean
# df.dropna(inplace=True)
# print(df.info())

print("-------------------------------")
print(df["Purchased Bike"].value_counts())
print("-------------------------------")


#plot data
ax=sns.countplot(x="Purchased Bike", data=df, hue='Gender', hue_order=['Male', 'Female'])

#Setting labels and font size
ax.set(xlabel='Purchased Bike', ylabel='Count of No/Yes',title="Count of People who Purchased the Bike and who Did't")
ax.xaxis.get_label().set_fontsize(10)
ax.yaxis.get_label().set_fontsize(10)

plt.show()

ax=sns.countplot(x="Purchased Bike", data=df, hue='Region', hue_order=['North America', 'Europe', 'Pacific'])

#Setting labels and font size
ax.set(xlabel='Purchased Bike', ylabel='Count of No/Yes',title="Count of People who Purchased the Bike and who Did't")
ax.xaxis.get_label().set_fontsize(10)
ax.yaxis.get_label().set_fontsize(10)

plt.show()

ax=sns.countplot(x="Purchased Bike", data=df, hue='Occupation', hue_order=['Skilled Manual', 'Professional', 'Clerical', 'Management', 'Manual'])

#Setting labels and font size
ax.set(xlabel='Purchased Bike', ylabel='Count of No/Yes',title="Count of People who Purchased the Bike and who Did't")
ax.xaxis.get_label().set_fontsize(10)
ax.yaxis.get_label().set_fontsize(10)

plt.show()


#normalising distribution of labels
df = pd.concat([df[df['Purchased Bike'] == "No"].head(481), df[df['Purchased Bike'] == "Yes"]])


print("-------------------------------")
print(df["Purchased Bike"].value_counts())
print("-------------------------------")


#plot data

ax=sns.countplot(x="Purchased Bike", data=df, hue='Occupation', hue_order=['Skilled Manual', 'Professional', 'Clerical', 'Management', 'Manual'])

#Setting labels and font size
ax.set(xlabel='Purchased Bike', ylabel='Count of No/Yes',title="Count of People who Purchased the Bike and who Did't")
ax.xaxis.get_label().set_fontsize(10)
ax.yaxis.get_label().set_fontsize(10)

plt.show()

# ax1=sns.countplot(x="EUR Value", data=df)

# #Setting labels and font size
# ax1.set(xlabel='Output', ylabel='Count of 0/1',title='Count of 0 and 1 news')
# ax1.xaxis.get_label().set_fontsize(15)
# ax1.yaxis.get_label().set_fontsize(15)

# plt.show()



# #Extracting the features from the news
# df['polarity'] = df['News Title'].map(lambda text: TextBlob(text).sentiment.polarity)
# df['review_len'] = df['News Title'].astype(str).apply(len)
# df['word_count'] = df['News Title'].apply(lambda x: len(str(x).split()))

# #Plotting the distribution of the extracted feature
# plt.figure(figsize = (20, 5))
# plt.style.use('seaborn-white')
# plt.subplot(131)
# sns.distplot(df['polarity'])
# fig = plt.gcf()
# plt.subplot(132)
# sns.distplot(df['review_len'])
# fig = plt.gcf()
# plt.subplot(133)
# sns.distplot(df['word_count'])
# fig = plt.gcf()

# plt.show()

# # For the sake of simplicity and just being neat a seperate library is imported, holding the custom transformers

# from custom_transformers import CharCounter
# from custom_transformers import CaseCounter
# from custom_transformers import StopWordCounter
# from custom_transformers import WordPronCounter
# from custom_transformers import WordNounCounter
# from custom_transformers import WordAdjCounter

# stop_words = stopwords.words("english")

# url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# def tokenize(text):

#     '''
#     INPUT: String to tokenise, detect and replace URLs
#     OUTPUT: List of tokenised string items
#     '''

#     # Remove punctuations and numbers
#     text = re.sub('[^a-zA-Z]', ' ', text)

#     # Single character removal
#     text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

#     # Removing multiple spaces
#     text = re.sub(r'\s+', ' ', text)

#     text = [w for w in text.split() if not w in stop_words]

#     # Join list to string
#     text = " ".join(text)

#     # Replace URLs if any
#     detected_urls = re.findall(url_regex, text)
#     for url in detected_urls:
#         text = text.replace(url, "urlplaceholder")

#     # Setup tokens and lemmatize
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     # Create tokens and lemmatize
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens
# def model_pipeline(model):
    
#     '''
#     INPUT: None
#     OUTPUT: pipeline object used to .fit X_train and y_train datasets
#     '''
    
#     pipeline = Pipeline([
#         ('features', FeatureUnion([
#             ('text_pipeline', Pipeline([
#                 ('vect', CountVectorizer(tokenizer=tokenize)),
#                 ('tfidf', TfidfTransformer())
#             ])),
#             ('char_counter', CharCounter()),
#             ('case_counter', CaseCounter()),
#             ('stop_counter', StopWordCounter()),
#             ('pro_counter', WordPronCounter()),
#             ('noun_counter', WordNounCounter()),
#             ('adj_counter', WordAdjCounter())
#         ])),
#         ('clf', model)
#         #  ('clf', RandomForestClassifier())
#         #  ('clf', DecisionTreeClassifier())
#     ])

#     return pipeline


# def display_results(y_test, y_pred, k):
    
#     '''
#     INPUT: y_test, y_pred dfs
#     OUTPUT: print average accuracy score
#     '''
    
#     labels = np.unique(y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
#     accuracy = (y_pred == y_test).mean()

#     print(f"Accuracy: {accuracy}-- model name: {k}")
#     return accuracy

# X = df['News Title']
# # Y = df['USD Value']
# Y = df['EUR Value']

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# # model_pred = {}
# # for i in range(2):
# #     model = model_pipeline()
# #     model.fit(X_train, y_train)

# #     y_pred = model.predict(X_test)

# #     d = display_results(y_test, y_pred)
# #     model_pred[i]=d

# # a = pd.DataFrame(model_pred)
# # a.to_csv("result.csv")
# from sklearn import metrics
# import matplotlib.pyplot as plt 
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
    
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     thresh = cm.max() / 2.
#     for i in range (cm.shape[0]):
#         for j in range (cm.shape[1]):
#             plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()




# model_name = [LogisticRegression(),DecisionTreeClassifier(), RandomForestClassifier() ]
# model_names = ["logisticregression", "decisiontreeclassifier", "randomforestclassifier" ]
# dic ={}
# for i,k in zip(model_name, model_names):
#     model_pred = []
#     for j in range(1):
#         model = model_pipeline(i)
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)

#         d = display_results(y_test, y_pred, k)
#         model_pred.append(d)

#         cm = metrics.confusion_matrix(y_test, y_pred)
#         plot_confusion_matrix(cm, classes=['Fake','True'])

#     dic[k] = model_pred

# df3 = pd.DataFrame(dic)

# df3.to_csv("123csv.csv")


# #Statistical Analysis
# clfs = []
# clfs.append(LogisticRegression(solver='liblinear'))
# clfs.append(RandomForestClassifier())
# clfs.append(DecisionTreeClassifier())

# classifier_name = []
# mean_value = []
# std_value = []

# for classifier in clfs:
#     model.set_params(clf = classifier)
#     scores = cross_validate(model, X_train, y_train)
#     print('---------------------------------')
#     print(str(classifier))
#     print('-----------------------------------')
    
#     for key, values in scores.items():
        
#         classifier_name.append(classifier)
#         mean_value.append(values.mean())
#         std_value.append(values.std())
        
#         print(key,' mean ', values.mean())
#         print(key,' std ', values.std())
