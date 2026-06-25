import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.extract_dir = os.path.join(self.base_dir, "ml-latest-small")
        
        # Ensure the Bollywood dataset exists and is loaded
        self._ensure_dataset_initialized()
        self._load_and_preprocess_data()
        self._init_models()

    def _ensure_dataset_initialized(self):
        """Generates the Bollywood movie and ratings files in the workspace."""
        # We always overwrite or generate the Bollywood dataset to satisfy Bollywood requirements
        self._create_bollywood_dataset()

    def _create_bollywood_dataset(self):
        """Creates the Bollywood dataset files (movies.csv and ratings.csv) with 500 movies."""
        print("Generating 500 Bollywood movie dataset...")
        os.makedirs(self.extract_dir, exist_ok=True)
        
        # Real Bollywood movies (around 120)
        real_movies = [
            (1, "Sholay (1975)", "Action|Adventure|Drama|Musical"),
            (2, "Dilwale Dulhania Le Jayenge (1995)", "Drama|Musical|Romance"),
            (3, "3 Idiots (2009)", "Comedy|Drama"),
            (4, "Dangal (2016)", "Action|Drama|Sport"),
            (5, "Lagaan: Once Upon a Time in India (2001)", "Drama|Musical|Romance|Sport"),
            (6, "Dil Chahta Hai (2001)", "Comedy|Drama|Romance"),
            (7, "Zindagi Na Milegi Dobara (2011)", "Comedy|Drama|Romance"),
            (8, "Shershaah (2021)", "Action|Drama|War"),
            (9, "Bajrangi Bhaijaan (2015)", "Adventure|Drama"),
            (10, "Queen (2013)", "Comedy|Drama"),
            (11, "My Name Is Khan (2010)", "Drama|Romance"),
            (12, "PK (2014)", "Comedy|Drama|Sci-Fi"),
            (13, "Barfi! (2012)", "Comedy|Drama|Romance"),
            (14, "Yeh Jawaani Hai Deewani (2013)", "Comedy|Drama|Romance"),
            (15, "Hera Pheri (2000)", "Comedy"),
            (16, "Drishyam (2015)", "Crime|Drama|Mystery|Thriller"),
            (17, "Andhadhun (2018)", "Comedy|Crime|Mystery|Thriller"),
            (18, "Kahaani (2012)", "Drama|Mystery|Thriller"),
            (19, "Swades (2004)", "Drama"),
            (20, "Chak De! India (2007)", "Drama|Sport"),
            (21, "Gangs of Wasseypur (2012)", "Action|Comedy|Crime|Drama|Thriller"),
            (22, "Devdas (2002)", "Drama|Musical|Romance"),
            (23, "Kal Ho Naa Ho (2003)", "Comedy|Drama|Romance"),
            (24, "Kuch Kuch Hota Hai (1998)", "Comedy|Drama|Musical|Romance"),
            (25, "Kabhi Khushi Kabhie Gham... (2001)", "Drama|Musical|Romance"),
            (26, "Om Shanti Om (2007)", "Action|Comedy|Drama|Musical|Romance"),
            (27, "Bajirao Mastani (2015)", "Drama|Romance"),
            (28, "Padmaavat (2018)", "Drama|Romance"),
            (29, "Chennai Express (2013)", "Action|Comedy|Romance"),
            (30, "Jab We Met (2007)", "Comedy|Romance"),
            (31, "Golmaal: Fun Unlimited (2006)", "Comedy"),
            (32, "Uri: The Surgical Strike (2019)", "Action|Drama|War"),
            (33, "Tanhaji: The Unsung Warrior (2020)", "Action|Drama|War"),
            (34, "Kabir Singh (2019)", "Drama|Romance"),
            (35, "Rockstar (2011)", "Drama|Musical|Romance"),
            (36, "Dil Se.. (1998)", "Drama|Romance|Thriller"),
            (37, "Rang De Basanti (2006)", "Comedy|Drama"),
            (38, "Ghajini (2008)", "Action|Drama|Mystery|Romance|Thriller"),
            (39, "Bahubali: The Beginning (2015)", "Action|Adventure|Drama|Fantasy"),
            (40, "Bahubali 2: The Conclusion (2017)", "Action|Adventure|Drama|Fantasy"),
            (41, "Krrish (2006)", "Action|Adventure|Fantasy|Sci-Fi"),
            (42, "Ra.One (2011)", "Action|Sci-Fi"),
            (43, "Dhoom 3 (2013)", "Action|Thriller"),
            (44, "Munna Bhai M.B.B.S. (2003)", "Comedy|Drama"),
            (45, "Lage Raho Munna Bhai (2006)", "Comedy|Drama"),
            (46, "Garam Masala (2005)", "Comedy|Romance"),
            (47, "Bhool Bhulaiyaa (2007)", "Comedy|Horror|Mystery|Thriller"),
            (48, "Stree (2018)", "Comedy|Horror"),
            (49, "Special 26 (2013)", "Comedy|Crime|Drama|Thriller"),
            (50, "Super 30 (2019)", "Drama"),
            (51, "Dhoom 2 (2006)", "Action|Thriller"),
            (52, "Dhoom (2004)", "Action|Thriller"),
            (53, "Koi... Mil Gaya (2003)", "Drama|Sci-Fi|Fantasy"),
            (54, "Krrish 3 (2013)", "Action|Sci-Fi|Fantasy"),
            (55, "Sanju (2018)", "Biography|Drama"),
            (56, "Border (1997)", "Action|Drama|War"),
            (57, "Gadar: Ek Prem Katha (2001)", "Action|Drama|Romance"),
            (58, "Gadar 2 (2023)", "Action|Drama"),
            (59, "Pathaan (2023)", "Action|Thriller"),
            (60, "Jawan (2023)", "Action|Thriller"),
            (61, "Animal (2023)", "Action|Drama|Thriller"),
            (62, "Tiger Zinda Hai (2017)", "Action|Thriller"),
            (63, "Ek Tha Tiger (2012)", "Action|Thriller"),
            (64, "War (2019)", "Action|Thriller"),
            (65, "Don (2006)", "Action|Thriller"),
            (66, "Don 2 (2011)", "Action|Thriller"),
            (67, "Dil To Pagal Hai (1997)", "Drama|Musical|Romance"),
            (68, "Mohabbatein (2000)", "Drama|Musical|Romance"),
            (69, "Veer-Zaara (2004)", "Drama|Musical|Romance"),
            (70, "Rab Ne Bana Di Jodi (2008)", "Comedy|Drama|Romance"),
            (71, "Raees (2017)", "Action|Crime|Drama"),
            (72, "Dilwale (2015) ", "Action|Comedy|Romance"),
            (73, "Taare Zameen Par (2007)", "Drama"),
            (74, "Talaash (2012)", "Drama|Mystery|Thriller"),
            (75, "Sultan (2016)", "Action|Drama|Sport"),
            (76, "Bharat (2019)", "Drama|Action"),
            (77, "Kick (2014)", "Action|Comedy|Thriller"),
            (78, "Wanted (2009)", "Action|Crime"),
            (79, "Ready (2011)", "Comedy|Romance"),
            (80, "Bodyguard (2011)", "Action|Comedy|Romance"),
            (81, "Agneepath (2012)", "Action|Drama"),
            (82, "Bang Bang! (2014)", "Action|Thriller"),
            (83, "Guzaarish (2010)", "Drama|Romance"),
            (84, "Jodhaa Akbar (2008)", "Drama|History|Romance"),
            (85, "Vikram Vedha (2022)", "Action|Crime|Thriller"),
            (86, "Wake Up Sid (2009)", "Comedy|Drama|Romance"),
            (87, "Ae Dil Hai Mushkil (2016)", "Drama|Romance"),
            (88, "Brahmastra: Part One - Shiva (2022)", "Action|Adventure|Fantasy|Sci-Fi"),
            (89, "Tamasha (2015)", "Drama|Romance"),
            (90, "Band Baaja Baaraat (2010)", "Comedy|Drama|Romance"),
            (91, "Simmba (2018)", "Action|Comedy|Crime"),
            (92, "Gully Boy (2019)", "Drama|Musical"),
            (93, "Sooryavanshi (2021)", "Action|Thriller"),
            (94, "Rowdy Rathore (2012)", "Action|Comedy"),
            (95, "Rustom (2016)", "Crime|Drama|Mystery"),
            (96, "Mission Mangal (2019)", "Drama|Sci-Fi"),
            (97, "Kesari (2019)", "Action|Drama|History"),
            (98, "Housefull (2010)", "Comedy"),
            (99, "Housefull 2 (2012)", "Comedy"),
            (100, "Airlift (2016)", "Drama|History|Thriller"),
            (101, "Baby (2015)", "Action|Thriller"),
            (102, "Jolly LLB (2013)", "Comedy|Drama"),
            (103, "Jolly LLB 2 (2017)", "Comedy|Drama"),
            (104, "Singham (2011)", "Action|Crime"),
            (105, "Singham Returns (2014)", "Action|Crime"),
            (106, "Golmaal Returns (2008)", "Comedy"),
            (107, "Golmaal 3 (2010)", "Comedy"),
            (108, "Golmaal Again (2017)", "Comedy|Horror"),
            (109, "Drishyam 2 (2022)", "Crime|Drama|Mystery"),
            (110, "Raid (2018)", "Action|Crime|Drama"),
            (111, "De De Pyaar De (2019)", "Comedy|Romance"),
            (112, "Article 15 (2019)", "Crime|Drama|Mystery"),
            (113, "Vicky Donor (2012)", "Comedy|Romance"),
            (114, "Badhaai Ho (2018)", "Comedy|Drama"),
            (115, "Bala (2019)", "Comedy|Drama"),
            (116, "Dream Girl (2019)", "Comedy|Romance"),
            (117, "Bareilly Ki Barfi (2017)", "Comedy|Romance"),
            (118, "Newton (2017)", "Comedy|Drama"),
            (119, "Gangs of Wasseypur - Part 2 (2012)", "Action|Comedy|Crime|Drama|Thriller"),
            (120, "Hera Pheri 2: Phir Hera Pheri (2006)", "Comedy")
        ]
        
        # Use set to ensure unique titles
        movie_titles_set = {re.sub(r'\s*\(\d{4}\)\s*$', '', m[1]).strip().lower() for m in real_movies}
        
        # Lists to generate programmatic Bollywood movies
        first_names = [
            "Dil", "Prem", "Kabhi", "Kuch", "Pyaar", "Mohabbat", "Raja", "Rani", "Khushi", "Gham", 
            "Yaar", "Dost", "Dushman", "Aashiq", "Babu", "Hero", "Villain", "Sarkar", "Jung", 
            "Karan", "Arjun", "Ram", "Lakhan", "Om", "Dev", "Jai", "Veer", "Raj", "Simran", "Rahul", 
            "Anjali", "Sanam", "Jaan", "Zindagi", "Saathiya", "Partner", "Aashiqui", "Baazigar", 
            "Soldier", "Badshah", "Koyla", "Khiladi", "Dhadkan", "Fanaa", "Kranti", "Beta", "Hum", 
            "Tum", "Hamara", "Apna", "Pardes", "Swarg", "Naseeb", "Kismat", "Pukar", "Josh", "Mela", 
            "Jodhaa", "Akbar", "Laila", "Majnu", "Heer", "Ranjha", "Bobby", "Gupt", "Soldier", 
            "Race", "Baaghi", "Dangal", "Gadar", "Jawan", "Tiger", "Pathaan", "Animal", "War"
        ]
        
        suffixes = [
            "Hota Hai", "Ke Liye", "Se", "Ke Sath", "Ki Kahani", "Ka Sikandar", "Ki Khoj", "Express", 
            "Dhamaka", "Khiladi", "No. 1", "Ki Dulhania", "Ki Prem Kahani", "Zindabad", "Mubarak", 
            "Ki Jung", "Ka Khiladi", "Aur Pyar", "Ki Kasam", "Ka Jadoo", "Ho Gaya", "Re", 
            "Na Milega Dobara", "Ke Deewane", "Dilwale", "Sajna", "Ki Barat", "Unlimited", "Returns", 
            "The Legend", "Revolutions", "Reloaded", "Ki Awaaz", "Ka Rishta", "Ki Dulhan", "Ki Aag",
            "Ka Faisla", "Deewane", "Mastana", "Chala Amerika", "Ki Duniya", "Ki Dhadkan", "Zindagi"
        ]
        
        genres_pool = [
            "Comedy|Drama",
            "Comedy|Romance",
            "Drama|Romance",
            "Action|Thriller",
            "Action|Comedy|Thriller",
            "Crime|Drama|Thriller",
            "Drama|Musical|Romance",
            "Action|Adventure|Drama",
            "Action|Drama|War",
            "Comedy|Horror",
            "Action|Sci-Fi",
            "Action|Adventure|Sci-Fi|Fantasy",
            "Drama|Mystery|Thriller"
        ]
        
        bollywood_movies = list(real_movies)
        next_movieId = 121
        import random
        random.seed(42)
        
        while len(bollywood_movies) < 500:
            name = random.choice(first_names)
            suff = random.choice(suffixes)
            year = random.randint(1975, 2025)
            clean_title = f"{name} {suff}"
            
            # Ensure uniqueness
            if clean_title.lower() not in movie_titles_set:
                movie_titles_set.add(clean_title.lower())
                title_with_year = f"{clean_title} ({year})"
                genre = random.choice(genres_pool)
                # 5% chance of adding Musical to genre if not already there
                if "Musical" not in genre and random.random() < 0.15:
                    genre += "|Musical"
                
                bollywood_movies.append((next_movieId, title_with_year, genre))
                next_movieId += 1
                
        movies_df = pd.DataFrame(bollywood_movies, columns=['movieId', 'title', 'genres'])
        movies_df.to_csv(os.path.join(self.extract_dir, 'movies.csv'), index=False)

        # Generate simulated user rating records
        # 60 simulated users rating a wider range of movies
        np.random.seed(42)
        ratings_data = []
        for user_id in range(1, 61):
            # Assign user profiles: 0: Action/War lover, 1: Romance/Musical, 2: Comedy/Drama, 3: Mixed
            profile = user_id % 4
            
            # Each user rates between 60 and 120 movies to keep matrix dense enough
            num_to_rate = np.random.randint(60, 120)
            movies_to_rate_indices = np.random.choice(len(bollywood_movies), num_to_rate, replace=False)
            
            for idx in movies_to_rate_indices:
                movie_id, title, genres = bollywood_movies[idx]
                genres_list = genres.split('|')
                
                # Base rating and rating probabilities
                base_rating = 3.0
                
                if profile == 0:  # Action/War/Sci-Fi lover
                    if any(g in genres_list for g in ['Action', 'War', 'Adventure', 'Sci-Fi']):
                        base_rating = 4.0
                elif profile == 1:  # Romance/Musical lover
                    if any(g in genres_list for g in ['Romance', 'Musical']):
                        base_rating = 4.0
                elif profile == 2:  # Comedy/Drama/Crime lover
                    if any(g in genres_list for g in ['Comedy', 'Drama', 'Crime']):
                        base_rating = 4.0
                else:  # Mixed profile
                    base_rating = 3.5

                # Add noise
                rating = base_rating + np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                rating = max(1.0, min(5.0, rating))
                
                ratings_data.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': float(rating),
                    'timestamp': 1234567890
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        ratings_df.to_csv(os.path.join(self.extract_dir, 'ratings.csv'), index=False)
        print(f"Bollywood Dataset generated. {len(bollywood_movies)} movies and {len(ratings_df)} ratings saved.")

    def _load_and_preprocess_data(self):
        """Loads dataset files and performs necessary pre-processing and data cleaning."""
        movies_file = os.path.join(self.extract_dir, "movies.csv")
        ratings_file = os.path.join(self.extract_dir, "ratings.csv")
        
        self.movies_df = pd.read_csv(movies_file)
        self.ratings_df = pd.read_csv(ratings_file)
        
        # Clean title and extract release year
        def parse_title_and_year(title):
            title = str(title).strip()
            match = re.search(r'\s*\((\d{4})\)\s*$', title)
            if match:
                year = match.group(1)
                clean_title = title[:match.start()].strip()
                return pd.Series([clean_title, year])
            return pd.Series([title, "Unknown"])

        self.movies_df[['clean_title', 'year']] = self.movies_df.apply(lambda r: parse_title_and_year(r['title']), axis=1)
        
        # Format genres for TF-IDF (space-separated) and save original for display
        self.movies_df['genres_display'] = self.movies_df['genres']
        self.movies_df['genres'] = self.movies_df['genres'].fillna('').str.replace('|', ' ', regex=False)
        
        # Calculate average ratings and ratings count for fallback/popularity scoring
        rating_stats = self.ratings_df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()
        
        # Merge stats back into movies dataframe
        self.movies_df = self.movies_df.merge(rating_stats, on='movieId', how='left')
        self.movies_df['avg_rating'] = self.movies_df['avg_rating'].fillna(3.0)
        self.movies_df['num_ratings'] = self.movies_df['num_ratings'].fillna(0).astype(int)

    def _init_models(self):
        """Initializes features and models for Content-Based and Collaborative Filtering."""
        # 1. Content Filtering Initialization (TF-IDF on Genres)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['genres'])
        
        # Map movieId to its index in tfidf_matrix
        self.movieId_to_idx = {movieId: idx for idx, movieId in enumerate(self.movies_df['movieId'])}
        
        # 2. Collaborative Filtering Initialization (Item-Item Ratings Cosine Similarity)
        # Create ratings pivot table
        self.ratings_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values='rating')
        # Reindex column headers to align with the complete movie list row order
        self.ratings_matrix = self.ratings_matrix.reindex(columns=self.movies_df['movieId'])
        
        # Fill NaN values with 0 representing unrated to compute similarity
        ratings_filled = self.ratings_matrix.fillna(0)
        
        # Compute Cosine Similarity between movies (columns of ratings_filled)
        print("Computing item-item similarity matrix for Collaborative Filtering...")
        self.item_similarity = cosine_similarity(ratings_filled.T)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.movies_df['movieId'],
            columns=self.movies_df['movieId']
        )
        print("Similarity calculations completed.")

    def search_movies(self, query, limit=10):
        """Searches movies by matching the title substring (case-insensitive)."""
        if not query or not query.strip():
            # Return top popular movies if query is empty
            return self.movies_df.sort_values(by='num_ratings', ascending=False).head(limit)[
                ['movieId', 'clean_title', 'year', 'genres_display', 'avg_rating']
            ].to_dict(orient='records')
            
        query = query.lower().strip()
        matches = self.movies_df[self.movies_df['clean_title'].str.lower().str.contains(query, regex=False)]
        
        # Sort matches: movies starting with the query first, then by popularity
        matches = matches.copy()
        matches['starts_with'] = matches['clean_title'].str.lower().str.startswith(query)
        sorted_matches = matches.sort_values(by=['starts_with', 'num_ratings'], ascending=[False, False])
        
        return sorted_matches.head(limit)[
            ['movieId', 'clean_title', 'year', 'genres_display', 'avg_rating']
        ].to_dict(orient='records')

    def get_movie_details(self, movieId):
        """Retrieves details of a movie by ID."""
        movie_row = self.movies_df[self.movies_df['movieId'] == movieId]
        if movie_row.empty:
            return None
        return movie_row.iloc[0].to_dict()

    def get_popular_recommendations(self, limit=5):
        """Gets top movies ranked by rating count and average score for users without ratings."""
        # Select movies with at least 5 ratings in our smaller Bollywood dataset
        candidates = self.movies_df[self.movies_df['num_ratings'] >= 5]
        if candidates.empty:
            candidates = self.movies_df
            
        top_popular = candidates.sort_values(by=['avg_rating', 'num_ratings'], ascending=[False, False]).head(limit)
        
        results = []
        for _, row in top_popular.iterrows():
            results.append({
                'movieId': int(row['movieId']),
                'title': row['clean_title'],
                'year': row['year'],
                'genres': row['genres_display'],
                'content_score': 0.0,
                'collaborative_score': float(row['avg_rating']) / 5.0, # scale to [0,1]
                'hybrid_score': float(row['avg_rating']) / 5.0
            })
        return results

    def get_hybrid_recommendations(self, user_ratings, limit=5):
        """
        Generates hybrid movie recommendations for a user given their rated movies.
        user_ratings: dict of {movieId: rating} (rating scale 1.0 - 5.0)
        """
        if not user_ratings:
            return self.get_popular_recommendations(limit)
            
        # Clean user ratings: convert keys to ints
        user_ratings = {int(k): float(v) for k, v in user_ratings.items()}
        rated_movie_ids = list(user_ratings.keys())
        
        # 1. CONTENT-BASED FILTERING
        # Create user profile vector from TF-IDF vectors of rated movies, weighted by rating centering
        user_profile = np.zeros(self.tfidf_matrix.shape[1])
        total_weight = 0
        
        for movieId, rating in user_ratings.items():
            if movieId in self.movieId_to_idx:
                idx = self.movieId_to_idx[movieId]
                # Center rating around 2.5 (positive weight for liking, negative for disliking)
                weight = rating - 2.5
                user_profile += weight * self.tfidf_matrix[idx].toarray()[0]
                total_weight += abs(weight)
                
        # Fallback if weights cancel out or sum to 0
        if total_weight == 0:
            for movieId, rating in user_ratings.items():
                if movieId in self.movieId_to_idx:
                    idx = self.movieId_to_idx[movieId]
                    user_profile += rating * self.tfidf_matrix[idx].toarray()[0]
                    
        # Compute Cosine Similarity between user profile and all movies
        user_profile_2d = user_profile.reshape(1, -1)
        content_similarities = cosine_similarity(user_profile_2d, self.tfidf_matrix)[0]
        # Normalize to [0, 1]
        content_scores = np.clip(content_similarities, 0, 1)
        
        # 2. COLLABORATIVE FILTERING (Item-Based Recommendation)
        ratings_vector = np.array([user_ratings[m] for m in rated_movie_ids])
        
        # Get the similarities of all movies to the rated movies
        # S_sub is shape (num_movies, num_rated)
        available_rated = [m for m in rated_movie_ids if m in self.item_similarity_df.columns]
        
        if len(available_rated) > 0:
            S_sub = self.item_similarity_df[available_rated].values
            # Filter ratings_vector to align with available items
            ratings_vector = np.array([user_ratings[m] for m in available_rated])
            
            # Clip similarities to positive only to prevent negative rating propagation
            S_sub_pos = np.clip(S_sub, 0, None)
            
            # Compute predicted collaborative rating: sum(sim * rating) / sum(sim)
            weighted_sum = S_sub_pos @ ratings_vector
            sum_sim = S_sub_pos @ np.ones_like(ratings_vector)
            
            collaborative_ratings = np.zeros(len(self.movies_df))
            mask = sum_sim > 0
            collaborative_ratings[mask] = weighted_sum[mask] / sum_sim[mask]
            
            # Fallback for items with no similarity to rated movies: use movie's global average rating
            avg_ratings = self.movies_df['avg_rating'].values
            collaborative_ratings[~mask] = avg_ratings[~mask]
        else:
            # Fallback if none of the rated movies are in the similarity matrix
            collaborative_ratings = self.movies_df['avg_rating'].values
            
        # Scale predicted collaborative ratings from [1, 5] to [0, 1]
        collaborative_scores = (collaborative_ratings - 1.0) / 4.0
        collaborative_scores = np.clip(collaborative_scores, 0, 1)
        
        # 3. HYBRID SCORE CALCULATION
        # Combine using specified weights: 0.6 Content + 0.4 Collaborative
        hybrid_scores = 0.6 * content_scores + 0.4 * collaborative_scores
        
        # Build results dataframe
        scores_df = pd.DataFrame({
            'movieId': self.movies_df['movieId'],
            'title': self.movies_df['clean_title'],
            'year': self.movies_df['year'],
            'genres': self.movies_df['genres_display'],
            'content_score': content_scores,
            'collaborative_score': collaborative_scores,
            'hybrid_score': hybrid_scores
        })
        
        # Filter out movies that the user has already rated
        scores_df = scores_df[~scores_df['movieId'].isin(rated_movie_ids)]
        
        # Get top 5 recommendations sorted by hybrid score descending
        top_recommendations = scores_df.sort_values(by='hybrid_score', ascending=False).head(limit)
        
        results = []
        for _, row in top_recommendations.iterrows():
            results.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'year': row['year'],
                'genres': row['genres'],
                'content_score': round(float(row['content_score']), 4),
                'collaborative_score': round(float(row['collaborative_score']), 4),
                'hybrid_score': round(float(row['hybrid_score']), 4)
            })
            
        return results

    def recommend(self, movie_name, limit=5):
        """
        Generates hybrid movie recommendations similar to a specific movie title.
        movie_name: str, name of the movie to find recommendations for.
        """
        # Find movie matching the query name (case insensitive)
        query = str(movie_name).lower().strip()
        
        # Try exact clean title match first
        matches = self.movies_df[self.movies_df['clean_title'].str.lower() == query]
        
        # Try full title with year exact match if exact clean title fails
        if matches.empty:
            matches = self.movies_df[self.movies_df['title'].str.lower() == query]
            
        # Try substring match if exact matches fail
        if matches.empty:
            matches = self.movies_df[self.movies_df['clean_title'].str.lower().str.contains(query, regex=False)]
            
        if matches.empty:
            raise ValueError(f"Movie '{movie_name}' not found in the dataset.")
            
        # If multiple matches, select the one with the highest number of ratings (most popular)
        selected_movie = matches.sort_values(by='num_ratings', ascending=False).iloc[0]
        selected_movieId = selected_movie['movieId']
        
        # 1. Content-based Score: Cosine similarity of the target movie's TF-IDF vector with all other movies
        target_idx = self.movieId_to_idx[selected_movieId]
        target_vector = self.tfidf_matrix[target_idx]
        content_similarities = cosine_similarity(target_vector, self.tfidf_matrix)[0]
        content_scores = np.clip(content_similarities, 0, 1)
        
        # 2. Collaborative Score: Retrieve cosine similarities from the ratings item-item similarity matrix
        if selected_movieId in self.item_similarity_df.columns:
            collaborative_scores = self.item_similarity_df[selected_movieId].values
        else:
            collaborative_scores = np.zeros(len(self.movies_df))
            
        collaborative_scores = np.clip(collaborative_scores, 0, 1)
        
        # 3. Hybrid Score (0.6 * content_score + 0.4 * collaborative_score)
        hybrid_scores = 0.6 * content_scores + 0.4 * collaborative_scores
        
        # Build results DataFrame
        scores_df = pd.DataFrame({
            'movieId': self.movies_df['movieId'],
            'title': self.movies_df['clean_title'],
            'year': self.movies_df['year'],
            'genres': self.movies_df['genres_display'],
            'content_score': content_scores,
            'collaborative_score': collaborative_scores,
            'hybrid_score': hybrid_scores
        })
        
        # Exclude the target movie itself
        scores_df = scores_df[scores_df['movieId'] != selected_movieId]
        
        # Get top 5 recommendations sorted by hybrid score descending
        top_recs = scores_df.sort_values(by='hybrid_score', ascending=False).head(limit)
        
        results = []
        for _, row in top_recs.iterrows():
            results.append({
                'movieId': int(row['movieId']),
                'title': row['title'],
                'year': row['year'],
                'genres': row['genres'],
                'content_score': round(float(row['content_score']), 4),
                'collaborative_score': round(float(row['collaborative_score']), 4),
                'hybrid_score': round(float(row['hybrid_score']), 4)
            })
            
        return results

# Module level API instantiation and wrapper
_recommender_instance = None

def _get_recommender():
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = MovieRecommender()
    return _recommender_instance

def recommend(movie_name):
    """
    Module level function to recommend movies similar to the given movie name.
    Usage:
        from recommendation import recommend
        results = recommend("Sholay")
    """
    return _get_recommender().recommend(movie_name)

if __name__ == "__main__":
    # Quick sanity test
    print("Dataset processed successfully!")
    print("Testing recommend() function for 'Sholay'...")
    try:
        recs = recommend("Sholay")
        for idx, rec in enumerate(recs):
            print(f"{idx+1}. {rec['title']} ({rec['year']}) - Hybrid: {rec['hybrid_score']:.4f} [Content: {rec['content_score']:.4f}, Collab: {rec['collaborative_score']:.4f}]")
    except Exception as e:
        print(f"Error: {e}")
