import os
from flask import Flask, render_template, request, jsonify, session
from recommendation import MovieRecommender

app = Flask(__name__)
app.secret_key = "hybrid_rec_sys_secret_key_12345"

# Initialize the recommender engine
# Note: This will download the MovieLens small dataset if it's not present.
print("Initializing Movie Recommender Engine...")
recommender = MovieRecommender()
print("Movie Recommender Engine initialized successfully.")

@app.before_request
def ensure_session_ratings():
    """Ensure ratings dictionary exists in user session."""
    if 'ratings' not in session:
        session['ratings'] = {}

@app.route("/")
def index():
    """Serves the main application landing page."""
    return render_template("index.html")

@app.route("/api/search", methods=["GET"])
def search_movies():
    """
    Search endpoint that accepts a query string and returns matching movies.
    Query parameter: ?q=movie_title
    """
    query = request.args.get("q", "")
    results = recommender.search_movies(query)
    return jsonify(results)

@app.route("/api/rate", methods=["POST"])
def rate_movie():
    """
    Rates a movie and updates the active user session.
    Payload: { "movieId": 123, "rating": 4.5 }
    """
    data = request.get_json()
    if not data or "movieId" not in data or "rating" not in data:
        return jsonify({"success": False, "error": "Invalid payload parameters"}), 400
    
    try:
        movie_id = str(data["movieId"]) # Store as string in session (JSON serialization limitation for dict keys)
        rating = float(data["rating"])
        
        if not (0.5 <= rating <= 5.0):
            return jsonify({"success": False, "error": "Rating must be between 0.5 and 5.0"}), 400
            
        # Get details to ensure the movie exists
        movie_details = recommender.get_movie_details(int(movie_id))
        if not movie_details:
            return jsonify({"success": False, "error": "Movie not found"}), 404
            
        # Update ratings in session
        ratings = session.get("ratings", {})
        ratings[movie_id] = rating
        session["ratings"] = ratings
        session.modified = True
        
        return jsonify({
            "success": True, 
            "message": f"Rated '{movie_details['clean_title']}' as {rating}",
            "ratings_count": len(ratings)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ratings", methods=["GET"])
def get_ratings():
    """Retrieves all movies rated in the active session with details."""
    ratings = session.get("ratings", {})
    results = []
    
    for movie_id_str, rating in ratings.items():
        movie_id = int(movie_id_str)
        details = recommender.get_movie_details(movie_id)
        if details:
            results.append({
                "movieId": movie_id,
                "title": details["clean_title"],
                "year": details["year"],
                "genres": details["genres_display"],
                "rating": rating
            })
            
    return jsonify(results)

@app.route("/api/recommend", methods=["GET"])
def recommend_movies():
    """Generates the hybrid recommendations based on current user ratings."""
    ratings = session.get("ratings", {})
    try:
        recommendations = recommender.get_hybrid_recommendations(ratings)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/clear_ratings", methods=["POST"])
def clear_ratings():
    """Resets the active user session ratings."""
    session["ratings"] = {}
    session.modified = True
    return jsonify({"success": True, "message": "All ratings cleared successfully."})

if __name__ == "__main__":
    # Start Flask server using environment PORT for deployment compatibility
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
