import time
from recommendation import MovieRecommender

def main():
    print("==========================================")
    print("Testing Bollywood Movie Recommender...")
    print("==========================================")
    
    start_time = time.time()
    recommender = MovieRecommender()
    print(f"Initialization took {time.time() - start_time:.2f} seconds.")
    
    print("\n------------------------------------------")
    print("Testing Movie Search for 'Sholay'")
    print("------------------------------------------")
    search_results = recommender.search_movies("Sholay")
    for r in search_results[:3]:
        print(f"- ID: {r['movieId']}, Title: {r['clean_title']} ({r['year']}), Genres: {r['genres_display']}, Avg Rating: {r['avg_rating']:.2f}")

    print("\n------------------------------------------")
    print("Testing Movie Search for '3 Idiots'")
    print("------------------------------------------")
    search_results = recommender.search_movies("3 Idiots")
    for r in search_results[:3]:
        print(f"- ID: {r['movieId']}, Title: {r['clean_title']} ({r['year']}), Genres: {r['genres_display']}, Avg Rating: {r['avg_rating']:.2f}")

    print("\n------------------------------------------")
    print("Testing Popular Recommendations (Empty profile)")
    print("------------------------------------------")
    recs = recommender.get_hybrid_recommendations({})
    for idx, rec in enumerate(recs):
        print(f"{idx+1}. {rec['title']} ({rec['year']}) - Score: {rec['hybrid_score']:.4f}")

    print("\n------------------------------------------")
    print("Testing Hybrid Recommendations (With profile)")
    print("------------------------------------------")
    # 1: Sholay - rated 5.0 (Likes action/adventure/drama/musical)
    # 3: 3 Idiots - rated 5.0 (Likes comedy/drama)
    # 15: Hera Pheri - rated 2.0 (Low comedy rating)
    test_profile = {
        1: 5.0,
        3: 5.0,
        15: 2.0
    }
    
    recs = recommender.get_hybrid_recommendations(test_profile)
    for idx, rec in enumerate(recs):
        print(f"{idx+1}. {rec['title']} ({rec['year']})")
        print(f"   - Genres: {rec['genres']}")
        print(f"   - Hybrid Score:       {rec['hybrid_score']:.4f}")
        print(f"   - Content Score:      {rec['content_score']:.4f} (60% weight)")
        print(f"   - Collaborative Score: {rec['collaborative_score']:.4f} (40% weight)")

if __name__ == "__main__":
    main()
