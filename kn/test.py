import math   
import uuid
import csv
import random
from collections import Counter, defaultdict
import random
import re
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
from flask import Flask, request, render_template
# Necessary libraries
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For matrix factorization
from scipy.sparse.linalg import svds
import csv 
from csv import DictWriter
from flask import Flask , render_template
import csv
import movieposters as mp
import urllib.request
from PIL import Image
from PIL import Image, ImageTk
from urllib import request
import PySimpleGUI as sg
import os
from flask import Flask, render_template, request
import requests
from sklearn.metrics.pairwise import cosine_similarity
def recommend_movies(preds_df, userId, movie, ratings_df, num_recommendations=5):
    '''Recommend top K movies to any chosen user

    Args:
    preds_df: prediction dataframe obtained from matrix factorization
    userId: chosen user
    movie: movie dataframe
    ratings_df: rating dataframe
    num_recommendations: desired number of recommendations

    Return:
    user_rated: movies that user already rated
    recommendations: final recommendations

    '''
    # Get user id, keep in mind index starts from zero
    user_row_number = userId-1 
    # Sort user's predictons
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) 
    # List movies user already rated
    user_data = ratings_df[ratings_df.userId == (userId)]
    user_rated = (user_data.merge(movie, how = 'left', left_on = 'movieId', right_on = 'movieId').
                  sort_values(['rating'], ascending=False)
                 )
    
    # f'User {userId} has already rated {user_rated.shape[0]} films.'

    recommendations = (movie[~movie['movieId'].isin(user_rated['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
               rename(columns = {user_row_number: 'Predictions'}).
               sort_values('Predictions', ascending = False).
               iloc[:num_recommendations, :-1]
                      )

    return user_rated, recommendations

"""
rating = pd.read_csv('/home/adel/Desktop/kn/ratings.csv')
movie = pd.read_csv('/home/adel/Desktop/kn/movies.csv')
df = pd.merge(rating, movie, on='movieId')


eda_rating = pd.DataFrame(df.groupby('title')['rating'].mean())
eda_rating['count of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
mtrx_df = rating.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
mtrx = mtrx_df.to_numpy()
ratings_mean = np.mean(mtrx, axis = 1)
normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
U, sigma, Vt = svds(normalized_mtrx, k = 50)
sigma = np.diag(sigma)
all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)



user_rated, recommendations = recommend_movies(preds_df, 616, movie=movie, ratings_df=rating, num_recommendations=20)
print(user_rated)
print(recommendations)
moviee=movie[0:10]
movie_data=movie
user_ratings=user_rated
"""
def mmr(user_ratings, movie_data, n_recommendations, lambda_param):
    # Create a matrix of user ratings with movieId as columns and userId as rows
    ratings_matrix = user_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Compute the pairwise similarity between the movies based on genres
    similarity_matrix = pd.DataFrame(np.zeros((len(movie_data), len(movie_data))), index=movie_data.index, columns=movie_data.index)
    for i in range(len(movie_data)):
        for j in range(len(movie_data)):
            similarity_matrix.iloc[i,j] = len(set(movie_data.iloc[i].genres.split('|')) & set(movie_data.iloc[j].genres.split('|'))) / len(set(movie_data.iloc[i].genres.split('|')) | set(movie_data.iloc[j].genres.split('|')))

    # Initialize the set of recommended items and the set of selected items
    selected_items = set()
    recommended_items = set(movie_data.index)

    # Choose the first item to recommend (highest average rating)
    avg_ratings = ratings_matrix.mean(axis=0)
    item_id = avg_ratings.idxmax()
    selected_items.add(item_id)
    recommended_items.remove(item_id)

    # Keep recommending items until reaching n_recommendations
    while len(selected_items) < n_recommendations:
        # Compute the Maximum Marginal Relevance score for each item
        scores = {}
        for item in recommended_items:
            # Similarity score between the item and the selected items
            sim = similarity_matrix.loc[selected_items, item].mean()
            # Rating score of the item
            rating = ratings_matrix.loc[:, item].mean()
            # MMR score
            mmr_score = lambda_param * rating - (1 - lambda_param) * sim
            scores[item] = mmr_score

        # Choose the item with the highest MMR score
        item_id = max(scores, key=scores.get)
        selected_items.add(item_id)
        recommended_items.remove(item_id)

    # Return the list of selected items
    return movie_data.loc[selected_items]


def suggest_diverse_movies(user_ratings, all_movies):
    # Créer un dictionnaire pour stocker les films en fonction de leur genre
    movies_by_genre = {}
    for index, row in all_movies.iterrows():
        genres = row['genres'].split('|')
        for genre in genres:
            if genre not in movies_by_genre:
                movies_by_genre[genre] = [row]
            else:
                movies_by_genre[genre].append(row)

    # Sélectionner un film aléatoire pour chaque genre
    selected_movies = []
    for genre, movies in movies_by_genre.items():
        selected_movie = None
        while True:
            random_movie = random.choice(movies)
            # Vérifier si le film a été évalué par l'utilisateur
            if (user_ratings['movieId'] == random_movie['movieId']).any():
                continue
            else:
                selected_movie = random_movie
                break
        if selected_movie is not None:
            selected_movies.append(selected_movie)

    return pd.DataFrame(selected_movies)
# print recommended movies
#recommended_movies=suggest_diverse_movies(user_rated,recommendations)
#print(recommended_movies)
api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'  # Replace with your actual TMDb API key



def remove_year(title):
    """
    Removes the year in parentheses from the given title string.

    Args:
        title (str): The title string with year in parentheses.

    Returns:
        str: The title string without year in parentheses.
    """
    return re.sub(r'\s*\(\d{4}\)', '', title)



def diversity_score(movie_list):
    genre_dict = {}
    for movie in movie_list:
        for genre in movie['genres']:
            print(genre)
            if genre in genre_dict:
                genre_dict[genre] += 1
            else:
                genre_dict[genre] = 1
    total_movies = len(movie_list)
    diversity = 0
    for genre in genre_dict:
        p = genre_dict[genre] / total_movies
        diversity -= p * math.log(p)
    return diversity

def genre_diversity_score(movie, movie_list):
    other_genres = []
    for other_movie in movie_list:
        if other_movie != movie:
            other_genres += other_movie['genres']
    other_genres = list(set(other_genres))
    new_list = []
    for genre in other_genres:
        count = 0
        for other_movie in movie_list:
            if other_movie != movie and genre in other_movie['genres']:
                count += 1
        new_list.append(count)
    return (diversity_score(new_list) - diversity_score([movie['genres']])) / (len(movie_list) - 1)

def diversify_movie_list(movie_list):
    for movie in movie_list:
        movie['genre_diversity_score'] = genre_diversity_score(movie, movie_list)
    sorted_list = sorted(movie_list, key=lambda x: x['genre_diversity_score'])
    return sorted_list


recommendationss = [
    (1, "Toy Story", "Animation|Children|Comedy"),
    (2, "Jumanji", "Adventure|Children|Fantasy"),
    (3, "Grumpier Old Men", "Comedy|Romance"),
    (4, "Waiting to Exhale", "Comedy|Drama|Romance"),
    (5, "Father of the Bride Part II", "Comedy"),
    (6, "Heat", "Action|Crime|Thriller"),
    (7, "Sabrina", "Comedy|Romance"),
    (8, "Tom and Huck", "Adventure|Children"),
    (9, "Sudden Death", "Action"),
    (10, "GoldenEye", "Action|Adventure|Thriller")
]
def tuple_list_to_dataframe(tuple_list):
    """
    Converts a list of tuples in the format (movieId, title, genres) to a
    pandas DataFrame with columns "movieId", "title", and "genres".
    """
    df = pd.DataFrame(tuple_list, columns=['movieId', 'title', 'genres','diversity_score'])
    return df
def dataframe_to_tuple_list(df):
    """
    Converts a pandas DataFrame with columns "movieId", "title", and "genres"
    to a list of tuples in the format (movieId, title, genres).
    """
    tuple_list = []
    for idx, row in df.iterrows():
        tuple_list.append((row['movieId'], row['title'], row['genres']))
    return tuple_list

recom=recommendationss

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

# Trier les films par ordre décroissant de score de diversité et afficher les 10 premiers
print(diversity_scores)
sorted_diversity_scores = sorted(diversity_scores, key=lambda x: x[3], reverse=True)
for i in range(10):
    print(sorted_diversity_scores[i][1], sorted_diversity_scores[i][3])

print(tuple_list_to_dataframe(sorted_diversity_scores))


def get_movie_title(movie_id):
    with open('movies.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['movieId']) == movie_id:
                return row['title']
    return None  # Movie id not found in the CSV file
import requests
import json
import webbrowser

def get_movie_trailer(api_key, movie_title):
    # Construct the API request URL
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"

    # Send the API request and parse the response
    response = requests.get(search_url)
    data = json.loads(response.text)
    
    if not data["results"] or data["results"][0]["title"].lower() != movie_title.lower():
        return None

    # Extract the movie ID of the first search result
    movie_id = data["results"][0]["id"]

    # Construct the API request URL to get the movie details
    movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=videos"

    # Send the API request and parse the response
    response = requests.get(movie_url)
    data = json.loads(response.text)

    # Extract the trailer key for the first video result
    trailer_key = data["videos"]["results"][0]["key"]

    # Construct the trailer URL on YouTube
    trailer_url = f"https://www.youtube.com/embed/{trailer_key}"
   
    # Return the movie title and trailer URL as a tuple
    return (trailer_url)