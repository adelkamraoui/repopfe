import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import csr_matrix

# Read the rating and movie data
rating = pd.read_csv('tables/ratings.csv')
movie = pd.read_csv('tables/movies.csv')
df = pd.merge(rating, movie, on='movieId')

# Create the pivot table
mtrx_df = rating.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Get the indices of non-zero ratings
non_zero_indices = np.nonzero(mtrx_df.values)

# Check if there are at least 20 non-zero ratings
if len(non_zero_indices[0]) >= 20:
    # Randomly select 20 indices
    random_indices = random.sample(range(len(non_zero_indices[0])), 20)

    # Get the row and column indices separately
    rows = non_zero_indices[0][random_indices]
    cols = non_zero_indices[1][random_indices]

    # Store the original values for later comparison
    original_ratings = mtrx_df.iloc[rows, cols].copy()

    # Replace the selected ratings with zeros
    mtrx_df.iloc[rows, cols] = 0

    # Convert the NumPy array to a sparse CSR matrix
    mtrx = csr_matrix(mtrx_df.values)

    # Perform matrix factorization using ALS
    model = implicit.als.AlternatingLeastSquares(factors=50)
    model.fit(mtrx.T)

    # Adjust the size of item factors matrix
    max_col_index = max(cols)
    item_factors = np.zeros((mtrx_df.shape[1], model.factors))
    item_factors[cols, :] = model.item_factors[:max_col_index+1, :]

    # Calculate predicted ratings
    user_factors = model.user_factors[rows, :]
    predicted_ratings = (user_factors @ item_factors.T).flatten()

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(original_ratings, predicted_ratings))

    # Print the modified DataFrame after replacing 20 ratings with zeros
    print('Modified DataFrame:')
    print(mtrx_df.head(30))

    # Print the predicted ratings
    print('Predicted Ratings:')
    print(predicted_ratings)

    # Print RMSE
    print('RMSE:', rmse)
else:
    print('Error: There are fewer than 20 non-zero ratings in the DataFrame.')



