from collections import Counter
import re
import nltk
import numpy as np
from sklearn.pipeline import make_pipeline
import urlextract
from html import unescape
from email.message import EmailMessage
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


def html_to_plain_text(html):
    text = re.sub("<head.*?>.*?</head>", "", html, flags=re.M | re.S | re.I)
    text = re.sub("<a\s.*?>", " HYPERLINK ", text, flags=re.M | re.S | re.I)
    text = re.sub("<.*?>", "", text, flags=re.M | re.S)
    text = re.sub(r"(\s*\n)+", "\n", text, flags=re.M | re.S)
    return unescape(text)


def email_to_text(email: EmailMessage):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:  # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


"""
Let's throw in some stemming! We will use Natural Language Toolkit([NLTK](https://www.nltk.org/)):

Stemming is one of several text normalization techniques that converts raw text data into a readable format for natural 
language processing tasks.
"""
stemmer = nltk.PorterStemmer()

"""
We will also need a way to replace URLs with the word "URL". For this, we could use hard core 
[regular expressions](https://mathiasbynens.be/demo/url-regex) but we will just use the 
[urlextract library](https://github.com/lipoja/URLExtract):
"""
url_extractor = urlextract.URLExtract()

"""
We are ready to put all this together into a transformer that we will use to convert emails to word counters. Note that 
we split sentences into words using Python's split() method, which uses white spaces for word boundaries. This works 
for many written languages, but not all. For example, Chinese and Japanese scripts generally don't use spaces between 
words, and Vietnamese often uses spaces even between syllables. It's okay, in this case though, because the dataset is 
mostly in English
"""


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strip_headers=True,
        lower_case=True,
        remove_punctuation=True,
        replace_urls=True,
        replace_numbers=True,
        stemming=True,
    ):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", "NUMBER", text)
            if self.remove_punctuation:
                text = re.sub(r"\W+", " ", text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


"""
Now we have the word counts, and we need to convert them to vectors. For this, we will build another transfomer 
whose fit() method will build the vocabulary (an ordered list of the most common words) and whose transform() 
method will use the vocabulary to transform word counts to vectors. The output is a sparse matrix.
"""


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[: self.vocabulary_size]
        self.vocabulary_ = {
            word: index + 1 for index, (word, count) in enumerate(most_common)
        }
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix(
            (data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1)
        )


preprocessor_pipeline = make_pipeline(
    EmailToWordCounterTransformer(), WordCounterToVectorTransformer()
)
