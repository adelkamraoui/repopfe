import requests
import json
import csv
from utils import*

api_key = 'ccfc2af2a0cd4597bf0472fab1af2f02'  # Replace with your actual TMDb API key


def get_posters(titles):
    poster_urls = []

    for title in titles:
        pre=remove_year(title)
        url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={pre}'
        response = requests.get(url)
        data = response.json()
        if data['total_results'] > 0:
            poster_path = data['results'][0]['poster_path']
            poster_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'
            print("+1")
            poster_urls.append(poster_url)
        else:
            poster_url = 'https://i.quotev.com/b2gtjqawaaaa.jpg'  # Replace with your actual error image path
            poster_urls.append(poster_url)    
            print("+1 taa errror")

    return poster_urls    
    
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
    print(data)

    # Extract the trailer key for the first video result
    trailer_key = data["videos"]["results"][0]["key"]

    # Construct the trailer URL on YouTube
    trailer_url = f"https://www.youtube.com/embed/{trailer_key}"
   
    # Return the movie title and trailer URL as a tuple
    return (trailer_url)

def get_movie_description(movie_name, api_key):
    # Construct the URL for the TMDB API search endpoint
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}'

    # Make a GET request to the API
    response = requests.get(url)

    # Get the JSON data from the response
    data = response.json()
    print(data)
    # Check if any results were found
    if 'total_results' not in data or data['total_results'] == 0:
        return 'No results found for the given movie name.'
    
    # Get the ID of the first movie in the search results
    movie_id = data['results'][0]['id']
    
    # Construct the URL for the TMDB API movie detail endpoint
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'

    # Make another GET request to the API
    response = requests.get(url)

    # Get the JSON data from the response
    data = response.json()

    # Get the movie description from the data
    description = data['overview']

    return description

def get_movie_url_from_csv(csv_file, movie_title):
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header row
        for row in reader:
            if row[1] == movie_title:
                return row[3]
    return None  # return None if movie not found
