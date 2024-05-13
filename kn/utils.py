import math   
import csv
import random
import random
import re
import numpy as np
import pandas as pd


CSV_FILE = '/home/adel/Desktop/kn/movies.csv'
def get_next_id(CSV_FILE):
    # Lecture du fichier CSV et récupération de la valeur maximale de movieId
    with open(CSV_FILE, 'r') as file:
        reader = csv.DictReader(file)
        movie_ids = [int(row['movieId']) for row in reader]
    next_id = max(movie_ids) + 1 if movie_ids else 1
    return str(next_id)

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
    print(sorted_user_predictions.head(10))
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

def get_movie_title(movie_id):
    with open('tables/movies.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['movieId']) == movie_id:
                return row['title']
    return None  # Movie id not found in the CSV file
