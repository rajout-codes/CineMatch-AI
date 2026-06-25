document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // Elements
    const searchInput = document.getElementById("movie-search-input");
    const clearSearchBtn = document.getElementById("clear-search-btn");
    const searchResultsContainer = document.getElementById("search-results-container");
    const ratingWidget = document.getElementById("rating-widget");
    const selectedMovieTitle = document.getElementById("selected-movie-title");
    const selectedMovieGenres = document.getElementById("selected-movie-genres");
    const ratingDropdown = document.getElementById("rating-dropdown");
    const cancelRatingBtn = document.getElementById("cancel-rating-btn");
    const submitRatingBtn = document.getElementById("submit-rating-btn");
    const ratingsCountBadge = document.getElementById("ratings-count-badge");
    const ratingsListContainer = document.getElementById("ratings-list-container");
    const ratingsEmptyState = document.getElementById("ratings-empty-state");
    const ratingsActionsContainer = document.getElementById("ratings-actions-container");
    const clearAllRatingsBtn = document.getElementById("clear-all-ratings-btn");
    const getRecommendationsBtn = document.getElementById("get-recommendations-btn");
    const recommendationsContainer = document.getElementById("recommendations-container");

    // State Variables
    let selectedMovieId = null;
    let currentRatingValue = 0.0;
    let searchDebounceTimeout = null;

    // Load active ratings on startup
    loadActiveRatings();

    // ==========================================================================
    // Search & Autocomplete
    // ==========================================================================
    
    searchInput.addEventListener("input", () => {
        const query = searchInput.value.trim();
        
        if (query.length > 0) {
            clearSearchBtn.classList.remove("hidden");
        } else {
            clearSearchBtn.classList.add("hidden");
            searchResultsContainer.classList.add("hidden");
            return;
        }

        // Debounce API requests to reduce server load
        clearTimeout(searchDebounceTimeout);
        searchDebounceTimeout = setTimeout(() => {
            fetch(`/api/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(data => {
                    displaySearchResults(data);
                })
                .catch(err => console.error("Error searching movies:", err));
        }, 250);
    });

    clearSearchBtn.addEventListener("click", () => {
        searchInput.value = "";
        clearSearchBtn.classList.add("hidden");
        searchResultsContainer.classList.add("hidden");
        searchInput.focus();
    });

    // Close autocomplete on click outside
    document.addEventListener("click", (e) => {
        if (!searchInput.contains(e.target) && !searchResultsContainer.contains(e.target)) {
            searchResultsContainer.classList.add("hidden");
        }
    });

    function displaySearchResults(movies) {
        searchResultsContainer.innerHTML = "";
        
        if (movies.length === 0) {
            const noResults = document.createElement("div");
            noResults.className = "search-item";
            noResults.style.color = "var(--text-muted)";
            noResults.style.justifyContent = "center";
            noResults.textContent = "No movies found matching search criteria";
            searchResultsContainer.appendChild(noResults);
            searchResultsContainer.classList.remove("hidden");
            return;
        }

        movies.forEach(movie => {
            const item = document.createElement("div");
            item.className = "search-item";
            
            // Format rating for display
            const ratingText = movie.avg_rating ? parseFloat(movie.avg_rating).toFixed(1) : "3.0";
            
            item.innerHTML = `
                <div class="search-item-info">
                    <span class="search-item-title">${movie.clean_title} <span class="search-item-year">(${movie.year})</span></span>
                    <span class="search-item-genres">${movie.genres_display}</span>
                </div>
                <div class="search-item-avg">
                    <i data-lucide="star"></i>
                    <span>${ratingText}</span>
                </div>
            `;
            
            item.addEventListener("click", () => {
                selectMovieForRating(movie);
            });
            
            searchResultsContainer.appendChild(item);
        });

        lucide.createIcons({
            attrs: {
                class: 'star-icon-small'
            },
            nameAttr: 'data-lucide'
        });
        
        // Custom styling for tiny icons in autocomplete
        document.querySelectorAll('.star-icon-small').forEach(icon => {
            icon.style.width = '0.85rem';
            icon.style.height = '0.85rem';
            icon.style.fill = 'var(--warning)';
        });
        
        searchResultsContainer.classList.remove("hidden");
    }

    function selectMovieForRating(movie) {
        selectedMovieId = movie.movieId;
        selectedMovieTitle.textContent = `${movie.clean_title} (${movie.year})`;
        selectedMovieGenres.textContent = movie.genres_display;
        
        // Reset dropdown
        resetRatingDropdown();
        
        // Hide search list and clear search input
        searchResultsContainer.classList.add("hidden");
        searchInput.value = "";
        clearSearchBtn.classList.add("hidden");
        
        // Slide rating widget in
        ratingWidget.classList.remove("hidden");
        ratingWidget.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // ==========================================================================
    // Interactive Rating Dropdown Selection
    // ==========================================================================
    
    ratingDropdown.addEventListener("change", () => {
        currentRatingValue = parseFloat(ratingDropdown.value);
    });

    function resetRatingDropdown() {
        currentRatingValue = 0.0;
        ratingDropdown.value = "";
    }

    cancelRatingBtn.addEventListener("click", () => {
        ratingWidget.classList.add("hidden");
        selectedMovieId = null;
        resetRatingDropdown();
    });

    submitRatingBtn.addEventListener("click", () => {
        if (!selectedMovieId) return;
        
        if (currentRatingValue === 0.0 || isNaN(currentRatingValue)) {
            alert("Please select a rating before submitting!");
            return;
        }

        submitRatingBtn.disabled = true;
        submitRatingBtn.textContent = "Submitting...";

        fetch("/api/rate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                movieId: selectedMovieId,
                rating: currentRatingValue
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                ratingWidget.classList.add("hidden");
                selectedMovieId = null;
                resetRatingDropdown();
                loadActiveRatings();
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(err => {
            console.error("Error submitting rating:", err);
            alert("An error occurred while submitting the rating.");
        })
        .finally(() => {
            submitRatingBtn.disabled = false;
            submitRatingBtn.textContent = "Submit Rating";
        });
    });

    // ==========================================================================
    // Profile Management (Active Ratings)
    // ==========================================================================
    
    function loadActiveRatings() {
        fetch("/api/ratings")
            .then(res => res.json())
            .then(ratings => {
                renderRatings(ratings);
            })
            .catch(err => console.error("Error loading ratings:", err));
    }

    function renderRatings(ratings) {
        // Update badge count
        ratingsCountBadge.textContent = `${ratings.length} Rated`;
        
        // Clear ratings container except empty state
        const items = ratingsListContainer.querySelectorAll(".rated-movie-item");
        items.forEach(item => item.remove());

        if (ratings.length === 0) {
            ratingsEmptyState.classList.remove("hidden");
            ratingsActionsContainer.classList.add("hidden");
            return;
        }

        ratingsEmptyState.classList.add("hidden");
        ratingsActionsContainer.classList.remove("hidden");

        // Loop in reverse order to show newest ratings at the top
        ratings.slice().reverse().forEach(item => {
            const row = document.createElement("div");
            row.className = "rated-movie-item";
            row.innerHTML = `
                <div class="rated-movie-meta">
                    <div class="rated-movie-title" title="${item.title}">${item.title} <span style="font-size:0.8rem; color:var(--text-muted);">(${item.year})</span></div>
                    <div class="rated-movie-genres">${item.genres}</div>
                </div>
                <div class="rated-movie-score">
                    <span class="rated-score-badge">
                        <i data-lucide="star"></i>
                        <span>${parseFloat(item.rating).toFixed(1)}</span>
                    </span>
                </div>
            `;
            ratingsListContainer.appendChild(row);
        });

        lucide.createIcons();
    }

    clearAllRatingsBtn.addEventListener("click", () => {
        if (!confirm("Are you sure you want to clear all your ratings? This will reset your recommendations.")) {
            return;
        }

        fetch("/api/clear_ratings", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    loadActiveRatings();
                    // Clear recommendations
                    showEmptyRecommendations();
                }
            })
            .catch(err => console.error("Error clearing ratings:", err));
    });

    function showEmptyRecommendations() {
        recommendationsContainer.innerHTML = `
            <div class="empty-state recommendations-empty">
                <i data-lucide="clapperboard" class="empty-icon highlight"></i>
                <h3>Awaiting Taste Profile</h3>
                <p>We combine your genres preferences (60% weight) with similar users' reviews (40% weight) to recommend perfect movies. Start rating or click above to see popular choices!</p>
            </div>
        `;
        lucide.createIcons();
    }

    // ==========================================================================
    // Hybrid Recommendation Engine
    // ==========================================================================
    
    getRecommendationsBtn.addEventListener("click", () => {
        getRecommendationsBtn.disabled = true;
        const originalText = getRecommendationsBtn.innerHTML;
        getRecommendationsBtn.innerHTML = `<i data-lucide="loader" class="btn-icon animate-spin"></i> Generating...`;
        lucide.createIcons();

        // Custom style spinner
        const spinner = getRecommendationsBtn.querySelector('.animate-spin');
        if (spinner) {
            spinner.style.animation = 'spin 1s linear infinite';
            if (!document.getElementById('spin-style')) {
                const style = document.createElement('style');
                style.id = 'spin-style';
                style.innerHTML = '@keyframes spin { 100% { transform: rotate(360deg); } }';
                document.head.appendChild(style);
            }
        }

        // Fetch top recommendations
        fetch("/api/recommend")
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert(`Error generating recommendations: ${data.error}`);
                    showEmptyRecommendations();
                } else {
                    renderRecommendations(data);
                }
            })
            .catch(err => {
                console.error("Error loading recommendations:", err);
                alert("Failed to load recommendations. Please verify Flask console.");
            })
            .finally(() => {
                getRecommendationsBtn.disabled = false;
                getRecommendationsBtn.innerHTML = originalText;
                lucide.createIcons();
            });
    });

    function renderRecommendations(movies) {
        recommendationsContainer.innerHTML = "";
        
        if (movies.length === 0) {
            recommendationsContainer.innerHTML = `
                <div class="empty-state recommendations-empty">
                    <i data-lucide="alert-circle" class="empty-icon highlight"></i>
                    <h3>No Recommendations Found</h3>
                    <p>We couldn't generate recommendations. Try rating some other popular movies!</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        movies.forEach((movie, index) => {
            const card = document.createElement("div");
            card.className = `recommendation-card delay-${index}`;
            
            // Format scores as percentages
            const hybridPercent = Math.round(movie.hybrid_score * 100);
            const contentPercent = Math.round(movie.content_score * 100);
            const collabPercent = Math.round(movie.collaborative_score * 100);
            
            // Calculate SVG dashoffset (perimeter is 220, offset goes from 220 down to 0)
            const strokeOffset = Math.max(0, Math.min(220, 220 - (220 * movie.hybrid_score)));
            
            // Format genres as pills
            const genrePills = movie.genres.split("|").map(genre => 
                `<span class="genre-pill">${genre.trim()}</span>`
            ).join("");

            card.innerHTML = `
                <div class="rec-rank-badge">#${index + 1}</div>
                
                <!-- Circular Radial Gauge -->
                <div class="rec-gauge-container">
                    <div class="circle-gauge">
                        <svg>
                            <defs>
                                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="var(--accent)" />
                                    <stop offset="100%" stop-color="var(--primary)" />
                                </linearGradient>
                            </defs>
                            <circle class="circle-bg" cx="36" cy="36" r="30"></circle>
                            <circle class="circle-progress" cx="36" cy="36" r="30" style="stroke-dashoffset: 220;"></circle>
                        </svg>
                        <div class="gauge-percentage">
                            ${hybridPercent}%
                            <span>Match</span>
                        </div>
                    </div>
                </div>
                
                <!-- Details & Score Breakdown -->
                <div class="rec-details">
                    <div class="rec-title-wrapper">
                        <h3>${movie.title}<span>(${movie.year})</span></h3>
                    </div>
                    <div class="rec-genres-pills">
                        ${genrePills}
                    </div>
                    
                    <!-- Progress bar breakdown -->
                    <div class="rec-breakdown">
                        <!-- Content score (60% weight) -->
                        <div class="breakdown-row">
                            <span class="breakdown-label content-label">
                                <i data-lucide="tag"></i> Genre Match (Content-based)
                            </span>
                            <div class="breakdown-bar-wrapper">
                                <div class="breakdown-track">
                                    <div class="breakdown-bar content-bar" style="width: 0%;"></div>
                                </div>
                                <span class="breakdown-val">${contentPercent}%</span>
                            </div>
                        </div>
                        <!-- Collaborative score (40% weight) -->
                        <div class="breakdown-row">
                            <span class="breakdown-label collab-label">
                                <i data-lucide="users"></i> User Taste (Collaborative)
                            </span>
                            <div class="breakdown-bar-wrapper">
                                <div class="breakdown-track">
                                    <div class="breakdown-bar collab-bar" style="width: 0%;"></div>
                                </div>
                                <span class="breakdown-val">${collabPercent}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            recommendationsContainer.appendChild(card);
            
            // Trigger animation of gauges/progress bars after a tiny delay
            setTimeout(() => {
                const progressCircle = card.querySelector(".circle-progress");
                const contentBar = card.querySelector(".content-bar");
                const collabBar = card.querySelector(".collab-bar");
                
                if (progressCircle) progressCircle.style.strokeDashoffset = strokeOffset;
                if (contentBar) contentBar.style.width = `${contentPercent}%`;
                if (collabBar) collabBar.style.width = `${collabPercent}%`;
            }, 100);
        });

        lucide.createIcons();
    }
});
