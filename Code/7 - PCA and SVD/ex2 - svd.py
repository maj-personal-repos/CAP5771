from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
import numpy as np


# categories = [
#     'alt.atheism',
#     'talk.religion.misc',
#     'comp.graphics',
#     'sci.space',
# ]

categories = [
    'sci.space',
]


dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=24)

document_corpus = dataset.data[:100]

query = ["sky"]

print("%d documents" % len(dataset.data[:100]))

print("%d categories" % len(dataset.target_names))

print()

vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)\

svd_model = TruncatedSVD(n_components=50, algorithm='randomized', n_iter=10, random_state=42)

svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])

svd_matrix = svd_transformer.fit_transform(document_corpus)

print(svd_matrix.shape)

query_vector = svd_transformer.transform(query)

distance_matrix = pairwise_distances(query_vector, svd_matrix, metric='cosine', n_jobs=-1)

print(min(distance_matrix[0, :]))

best_match_index = np.where(distance_matrix[0, :] == min(distance_matrix[0, :]))

print("best match index: %d " % best_match_index[0][0])

print()

print("-- document at index %d --" % best_match_index[0][0])

print(document_corpus[best_match_index[0][0]])
