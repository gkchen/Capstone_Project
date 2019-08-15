import pandas as pd
import dill
import re

from sklearn import base
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split, GridSearchCV

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import plotly.offline as py
import plotly.graph_objs as go

from IPython.display import IFrame

py.init_notebook_mode(connected=True)


class TextPreProcess(base.BaseEstimator, base.TransformerMixin):
    """
    Input  : document list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text
    """

    def __init__(self, ignore):
        self.en_stop = set(stopwords.words('english'))  # English stop words list
        self.tokenizer = RegexpTokenizer(r'[a-z]+&?[a-z]+')
        self.lemmatizer = WordNetLemmatizer()
        self.replace = ignore

    def _process(self, text):
        raw = text.lower()
        for key, val in self.replace.items():
            raw = re.sub(key, val, raw)
        tokens = self.tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if i not in self.en_stop]
        lemma_tokens = [self.lemmatizer.lemmatize(i) for i in stopped_tokens]
        output = ' '.join(lemma_tokens)
        return output

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = [self._process(text) for text in X]
        return output


def get_keywords(pipe, text):
    cleaned = pipe.named_steps['clean'].transform(text)
    sparse = pipe.named_steps['tfidf'].transform(cleaned)
    keywords = pipe.named_steps['tfidf'].inverse_transform(sparse)

    return keywords


def plot_single_map(pipe, text, color, title_prefix=''):
    fig = go.Figure()
    zmax = max(pipe.predict_proba(text)[0])

    keywords = get_keywords(pipe, text)
    data = go.Choropleth(locations=pipe.named_steps['nb'].classes_,
                         z=pipe.predict_proba(text)[0],
                         zmin=0,
                         zmax=zmax,
                         locationmode='USA-states',
                         colorscale=color,
                         colorbar=dict(title='Porportion',
                                       titleside='top'
                                       )
                         )
    layout = go.Layout(geo_scope='usa', title=title_prefix + ', '.join(keywords[0]),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=[data], layout=layout)
    return fig


def plot_multi_map(pipes, text, color, title_prefix=''):
    # Create figure
    fig = go.Figure()

    keywords = get_keywords(list(pipes.values())[0], text)

    # Add traces, one for each slider step
    data = []
    steps = []
    i = 0
    zmax = max([max(pipe.predict_proba(text)[0]) for pipe in pipes.values()])
    for date, pipe in pipes.items():
        label = date.strftime('%Y-%m-%d')
        plot = go.Choropleth(locations=pipe.named_steps['nb'].classes_,
                             z=pipe.predict_proba(text)[0],
                             zmin=0,
                             zmax=zmax,
                             locationmode='USA-states',
                             colorscale=color,
                             name=label,
                             colorbar=dict(title='Proportion',
                                           titleside='top'
                                           )
                             )
        step = dict(method="restyle",
                    args=["visible", [False] * len(pipes)],
                    label=label
                    )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        data.append(plot)
        steps.append(step)
        i += 1

    sliders = [dict(active=10,
                    currentvalue={"prefix": "Week of: "},
                    steps=steps
                    )
               ]
    layout = go.Layout(geo_scope='usa', title=title_prefix+', '.join(keywords[0]), sliders=sliders,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=data, layout=layout)
    # # Make 10th trace visible
    fig.data[10].visible = True

    # py.plot(fig, filename='plots/usa_total_posts.html', auto_open=False)
    # s = py.plot(fig, include_plotlyjs=False, output_type='div')
    return fig


ignore = pd.read_feather('other_data/ignore.feather')
ignore_dict = ignore.set_index('regex').to_dict()['sub']
tpp = TextPreProcess(ignore=ignore_dict)

with open('models/tfidf/top_tfidf.pkd', 'rb') as file:
    tfv = dill.load(file)

with open('models/mnb_weighted/top_vocabulary/estimator.pkd', 'rb') as file:
    total_weighted_estimator = dill.load(file)

total_weighted_pipe = Pipeline([('clean', tpp), ('tfidf', tfv), ('nb', total_weighted_estimator)])

with open('models/mnb_unweighted/top_vocabulary/estimator.pkd', 'rb') as file:
    total_unweighted_estimator = dill.load(file)

total_unweighted_pipe = Pipeline([('clean', tpp), ('tfidf', tfv), ('nb', total_unweighted_estimator)])

with open('models/mnb_weighted/top_vocabulary/weekly_estimators.pkd', 'rb') as file:
    weekly_weighted_estimators = dill.load(file)

weekly_weighted_pipes = {key: Pipeline([('clean', tpp), ('tfidf', tfv), ('nb', val)]) for key, val in weekly_weighted_estimators.items()}

with open('models/mnb_unweighted/top_vocabulary/weekly_estimators.pkd', 'rb') as file:
    weekly_unweighted_estimators = dill.load(file)

weekly_unweighted_pipes = {key: Pipeline([('clean', tpp), ('tfidf', tfv), ('nb', val)]) for key, val in weekly_unweighted_estimators.items()}


def make_prediction(text):
    fig = plot_single_map(total_unweighted_pipe, text, 'Reds', 'Unweighted Predictions for: ')
    py.iplot(fig)
    fig = plot_single_map(total_weighted_pipe, text, 'Blues', 'Per Capita Predictions for: ')
    py.iplot(fig)

    return None


def make_weekly_prediction(text):
    fig = plot_multi_map(weekly_unweighted_pipes, text, 'Reds', 'Weekly Unweighted Predictions for: ')
    py.iplot(fig)
    fig = plot_multi_map(weekly_weighted_pipes, text, 'Blues', 'Weekly Per Capita Predictions for: ')
    py.iplot(fig)

    return None
