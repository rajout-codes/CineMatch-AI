import requests

def test_api_flow():
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    print("==========================================")
    print("Testing CineMatch AI Bollywood Flask APIs...")
    print("==========================================")
    
    # 1. Test Search API
    print("\n1. Testing Movie Search API for 'Sholay'...")
    res = session.get(f"{base_url}/api/search?q=Sholay")
    assert res.status_code == 200, f"Failed search: {res.text}"
    search_results = res.json()
    assert len(search_results) > 0, "No search results returned"
    sholay = search_results[0]
    print(f"   Success! Found movie: '{sholay['clean_title']}' with ID {sholay['movieId']}")
    
    # 2. Test Rate Movie API
    print(f"\n2. Rating '{sholay['clean_title']}' as 5.0 stars...")
    rate_payload = {"movieId": sholay["movieId"], "rating": 5.0}
    res = session.post(f"{base_url}/api/rate", json=rate_payload)
    assert res.status_code == 200, f"Failed rating: {res.text}"
    rate_response = res.json()
    assert rate_response["success"] is True, "Rate response returned failure status"
    print(f"   Success! Response message: {rate_response['message']}")
    
    # Rate another movie (3 Idiots - ID: 3)
    print("\n3. Rating '3 Idiots' as 5.0 stars...")
    rate_payload_2 = {"movieId": 3, "rating": 5.0}
    res = session.post(f"{base_url}/api/rate", json=rate_payload_2)
    assert res.status_code == 200, f"Failed rating 3 Idiots: {res.text}"
    print(f"   Success! Response message: {res.json()['message']}")

    # 3. Test Active Ratings API
    print("\n4. Retrieving current active ratings list...")
    res = session.get(f"{base_url}/api/ratings")
    assert res.status_code == 200, f"Failed getting ratings: {res.text}"
    active_ratings = res.json()
    assert len(active_ratings) == 2, f"Expected 2 rated movies, got {len(active_ratings)}"
    print("   Active profile contains:")
    for r in active_ratings:
        print(f"   - {r['title']} ({r['year']}): {r['rating']} stars")
        
    # 4. Test Recommendations API
    print("\n5. Generating hybrid recommendations...")
    res = session.get(f"{base_url}/api/recommend")
    assert res.status_code == 200, f"Failed recommendations: {res.text}"
    recs = res.json()
    assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"
    print("   Success! Top 5 recommendations generated:")
    for idx, rec in enumerate(recs):
        print(f"   {idx+1}. {rec['title']} ({rec['year']})")
        print(f"      - Genres: {rec['genres']}")
        print(f"      - Hybrid score: {rec['hybrid_score']:.4f} [Content: {rec['content_score']:.4f}, Collab: {rec['collaborative_score']:.4f}]")
        
    # 5. Clear Ratings API
    print("\n6. Cleaning up and clearing ratings...")
    res = session.post(f"{base_url}/api/clear_ratings")
    assert res.status_code == 200, f"Failed clearing ratings: {res.text}"
    assert res.json()["success"] is True
    
    # Confirm they are cleared
    res = session.get(f"{base_url}/api/ratings")
    assert len(res.json()) == 0, "Ratings list should be empty after clear"
    print("   Success! Active ratings cleared.")
    print("\n==========================================")
    print("All API tests passed successfully!")
    print("==========================================")

if __name__ == "__main__":
    test_api_flow()
