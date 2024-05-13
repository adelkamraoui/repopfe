import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from utils import*

def load_data():
    rating = pd.read_csv('tables/ratings.csv')
    movie = pd.read_csv('tables/movies.csv')
    #df = pd.merge(rating, movie, on='movieId')
    return rating, movie

def get_matrix(rating):
    mtrx_df = rating.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    mtrx_np = mtrx_df.to_numpy()
    return mtrx_df, mtrx_np

def normalize_matrix(mtrx):
    ratings_mean = np.mean(mtrx, axis = 1)
    normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
    transform_back= lambda pred : pred +  ratings_mean.reshape(-1, 1)
    return normalized_mtrx, transform_back

def apply_factorization(normalized_mtrx):
    U, sigma, Vt = svds(normalized_mtrx, k = 50)
    sigma = np.diag(sigma)
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    return all_predicted_ratings

def apply_diversity(predictions):
    recom=dataframe_to_tuple_list(predictions)
    # Calcul du score de diversité pour chaque film
    diversity_scores = []
    for i in range(len(recom)):
        genres_i = set(recom[i][2].split("|"))
        diversity_sum = 0
        for j in range(len(recom)):
            if i != j:
                genres_j = set(recom[j][2].split("|"))
                distance = len(genres_i.symmetric_difference(genres_j))
                diversity_sum += distance
        diversity_score = diversity_sum / (len(recom) - 1)
        diversity_scores.append((recom[i][0], recom[i][1], recom[i][2], diversity_score))
    return diversity_scores
