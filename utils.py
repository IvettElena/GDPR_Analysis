import requests
import numpy as np
import pandas as pd
import re
from pathlib import Path
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from pandas import Grouper
from matplotlib import pyplot
import seaborn as sns
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def load_data():
    """
        transform JSON data from website into a dataframe


        Input : JSON data from https://www.enforcementtracker.com/
        Output : dataframe

    """
    url = 'https://www.enforcementtracker.com/data.json?_=1614203798435'
    json_data = requests.get(url).json()
    json_data = json_data['data']
    data_shape = np.shape(json_data)
    df = pd.DataFrame(json_data)

    country = df.iloc[:, 2]
    fine = df.iloc[:, 5]
    article = df.iloc[:, 8]

    fine_list = []
    article_list = []
    countries_list = []

    for row in range(len(df)):

        # keep only the number of the quoted article
        article_list.append(re.split('\,', article[row]))

        # keep only the fines that contain digits
        fine_list.append(re.sub('\D', '', fine[row]))

        # isolate the name of the country
        append = False
        for line in re.split('\/>', country[row]):
            if (append):
                countries_list.append(line)
                append = False
            else:
                append = True

    df = pd.DataFrame({"Id": df.iloc[:, 1],
                       'Country': countries_list,
                       'Date_of_decision': df.iloc[:, 4],
                       'Fine': fine_list,
                       'Controller_Processor': df.iloc[:, 6],
                       'Quoted_Article': article_list,
                       'Type': df.iloc[:, 9],
                       'Source': df.iloc[:, 11],
                       'Authority': df.iloc[:, 3],
                       'Sector': df.iloc[:, 7],
                       'Summary': df.iloc[:, 10]})

    df = unify_terms(df, 'Controller_Processor', 'Unknown', 'Unknown Company')
    df = unify_terms(df, 'Sector', 'Unknown', 'Not assigned')

    df.Id = df.Id.astype('category')
    df.Country = df.Country.astype('category')
    df.Date_of_decision = pd.to_datetime(df.Date_of_decision.replace('Unknown', ''))
    df.Fine = pd.to_numeric(df.Fine, downcast='integer')
    df.Controller_Processor = df.Controller_Processor.astype('category')
    df.Quoted_Article = df.Quoted_Article.replace('str', np.nan)  # Unknown
    df.Type = df.Type.astype('category')
    df.Source = df.Source.astype('category')
    df.Authority = df.Authority.astype('category')
    df.Sector = df.Sector.astype('category')
    df.Summary = df.Summary.astype('str')

    return df


def unify_terms(df, column, x, y):
    """
     x and y are terms with the same meaning: y<-x
    """
    for i in np.where((df[column] == x) | (df[column] == y)):
        df[column].loc[i] = x

    return df


def save_dataset(df, dataset_name="GDPR"):
    """
     dataset is saved into an appropriate folder
    """
    filepath = Path("Data/" + dataset_name + ".csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def merge_dataset(df, dataset_name):
    """
     dataset is merged
    """
    new_df = pd.read_csv(dataset_name)
    new_df['Id'] = df

    return new_df


def create_article_features_df(df):
    """
         Dataframe: (Id, Article)
    """
    id_list = []
    article_list = []

    for row in range(len(df)):
        article = df.iloc[row, 5]

        article = re.sub(r"\[", '', article)
        article = re.sub(r"\]", '', article)
        article = re.sub(r"\.", '', article)
        article = article.split(sep="Art")
        for i in range(len(article)):
            article_list.append(re.sub(r"\'", '', article[i]))
            id_list.append(df.iloc[row, 0])

    df = pd.DataFrame({'Id': id_list, 'Article': article_list})

    return df


def create_art_type_features_df(df):
    """
        Dataframe: (Id, Article, GDPR_Type, Other_Type,Only_Art)
                    where: GDPR_Type refers to the general category of GDPR quoted article,
                           Other_Type refers to other articles and
                           Only_Art merges other articles and only the general category of GDPR quoted article
    """
    gdpr_art = []
    other_art = []
    only_art = []
    for i in df.Article.index:
        art = unicodedata.normalize("NFKD", df.Article.iloc[i])
        if re.search('[GDPR]', art):
            other_art.append('None')
            if re.match('^\d+', art):
                gdpr_art.append(re.findall('^\d+', art)[0])
            else:
                gdpr_art.append(df.Article.iloc[i])
        else:
            gdpr_art.append('None')
            other_art.append(df.Article.iloc[i])

    df['GDPR_Type'] = gdpr_art
    df['Other_Type'] = other_art

    df['Only_Art'] = gdpr_art
    for i in df[df.GDPR_Type == 'None'].index:
        df.Only_Art.loc[i] = df.Other_Type.iloc[i]
    for i in df[df.GDPR_Type != 'None'].index:
        df.Only_Art.loc[i] = "GDPR" + df.Only_Art.iloc[i]
    return df


def categorical_representation_df(df,type):
    """
         Categorical Dataframe: (Id, [Article1],[Article2], ...)
    """
    if(type==1):
        feature_name = 'Article'
        dataset_name = "GDPR_Article_Matrix"
    elif (type==2):
        feature_name = 'Only_Art'
        dataset_name = "GDPR_Gen_Article_Matrix"
    else:
        feature_name = 'Other_Art'
        dataset_name = "GDPR_NGen_Article_Matrix"

    ids = pd.Categorical(df['Id'], categories=df['Id'].unique())
    articles = pd.Categorical(df[feature_name],
                              categories=df[feature_name].unique())
    article_representation_matrix = pd.crosstab(ids, articles)

    save_dataset(article_representation_matrix, dataset_name)

    return article_representation_matrix


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def tf_idf_representation_df(df):
    # Download stopwords list
    with open('Stop_words.txt',
              'r') as file:
        token_stop = [file.read().replace('\n', ',')]

    token_stop=token_stop[0].split(",")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.4,min_df=0.1, max_features=50, analyzer=stemmed_words, stop_words=token_stop,
                                   ngram_range=(2, 4))
    doc_vec = tfidf_vectorizer.fit_transform(df.Summary)
    tokens = tfidf_vectorizer.get_feature_names()
    tf_idf_df = pd.DataFrame(doc_vec.toarray(), columns=tokens)
    tf_idf_df['Fine'] = df.Fine
    tf_idf_df['Id'] = df.Id
    for token in tokens:
        if token in token_stop:
            del tf_idf_df[token]
    return tf_idf_df
