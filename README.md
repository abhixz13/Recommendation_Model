PM_Lens-Recommendation System
Overview
This Jupyter notebook builds two recommendation systems using the MovieLens dataset and analyzes them from a Product Manager's perspective. The goal is to understand the gap between algorithmic performance and product value.

Dataset
Source: MovieLens dataset

Files Used:

ratings.csv - User ratings (userId, movieId, rating, timestamp)
movies.csv - Movie metadata (movieId, title, genres)
tags.csv - User-generated tags (userId, movieId, tag, timestamp)
Scale:

671 users
7,062 movies
97,243 ratings
Sparsity: 97.95% (only 2.05% of possible user-movie combinations have ratings)
What We Built
1. Cluster-Based Collaborative Filtering
Approach:

Applied K-Means clustering to group users with similar rating patterns
Used elbow analysis (SSE) to select optimal k=7 clusters
Generated recommendations by averaging ratings from users in the same cluster
Key Steps:

Created user-item interaction matrix (671 users × 1,000 most-rated movies)
Filled missing values with 0, stored as sparse matrix (CSR format)
Calculated SSE for k ranging from 2 to 47
Selected k=7 based on elbow plot
Assigned each user to a cluster
For a given user (e.g., User 619), recommended movies with highest average ratings from their cluster
Results:

Cluster 6: 350 users (52.2%)
Cluster 0: 118 users (17.6%)
Cluster 4: 86 users (12.8%)
Clusters 1,2,3,5: 117 users (17.4%)
Example Output (User 619 recommendations):

Training Day (2001) - 5.0 avg rating
Ran (1985) - 5.0
Knight's Tale, A (2001) - 5.0
Lock, Stock & Two Smoking Barrels (1998) - 4.83
Old Boy (2003) - 4.80
2. Content-Based Filtering
Approach:

Combined movie title, genres, and user tags into text features
Applied TF-IDF vectorization to extract 2,642 features
Calculated cosine similarity between all movies (1,459 × 1,459 matrix)
Recommended movies with highest similarity scores
Key Steps:

Merged ratings, movies, and tags datasets
Cleaned text: lowercase, removed non-alphabetic characters
Tokenized and lemmatized using NLTK
Applied TF-IDF vectorization
Computed cosine similarity matrix
For a given movie, sorted by similarity and returned top 10
Example Output ("The Usual Suspects" recommendations):

Game, The (1997)
Andalusian Dog, An (1929)
Town, The (2010)
Now You See Me (2013)
Charade (1963)
Negotiator, The (1998)
Following (1998)
21 Grams (2003)
Inception (2010)
Insomnia (2002)
Genre alignment: All recommendations were Crime/Mystery/Thriller movies, matching the input movie's genres.

Technical Implementation
Libraries:

pandas - Data manipulation
numpy - Numerical operations
matplotlib, seaborn - Visualizations
scipy.sparse - Sparse matrix operations (CSR format)
sklearn.cluster.KMeans - Clustering algorithm
sklearn.metrics.pairwise.cosine_similarity - Similarity calculation
sklearn.feature_extraction.text.TfidfVectorizer - Text feature extraction
nltk - Text preprocessing (tokenization, lemmatization, stopwords)
Key Techniques:

Sparse matrix storage: Used CSR (Compressed Sparse Row) format to handle 97.95% sparsity efficiently
Elbow method: Plotted SSE vs. k to identify optimal cluster count
TF-IDF: Term Frequency-Inverse Document Frequency for text feature extraction
Cosine similarity: Measured angle between feature vectors (range: 0-1, where 1 = identical)
Lemmatization: Reduced words to base form (e.g., "running" → "run")
Model Performance
What We Measured:

✅ SSE (Sum of Squared Errors) for cluster optimization
✅ Cosine similarity scores (0.0 - 1.0 range)
✅ Cluster distribution balance
✅ Genre overlap in content-based recommendations
What We Did NOT Measure:

❌ RMSE (Root Mean Squared Error) - would require train/test split and rating predictions
❌ Precision@K - would require labeled "relevant" items
❌ Coverage - % of catalog that can be recommended
❌ Click-through rate or conversion metrics
Why: This is a baseline implementation focused on understanding the models, not optimizing for production metrics.

Key Observations
From Clustering Approach:
Severe cluster imbalance: 52% of users in one cluster means most users get similar recommendations
No personalization within cluster: User 619 gets the same recommendations as the other 349 users in cluster 6
Cold start handled: New users can be assigned to nearest cluster centroid
From Content-Based Approach:
Perfect genre matching: All "Usual Suspects" recommendations were Crime/Mystery/Thriller
Filter bubble problem: Zero cross-genre discovery for users with diverse tastes
No collaborative signal: Recommendations based solely on item features, ignoring user behavior patterns
Explainable: Easy to justify why a movie was recommended (genre/tag similarity)
Sparsity Challenge:
Original matrix: 671 × 7,062 = 4,738,602 possible ratings
Actual ratings: 97,243 (2.05% density)
Even after filtering to top 1,000 movies: still very sparse
Heatmap visualization shows most cells are empty (NaN values)
Notebook Structure
Data Loading & Exploration

Load ratings, movies, tags
Merge datasets
Create interaction matrix
Calculate sparsity
Clustering-Based System

Select top 1,000 most-rated movies
Create sparse matrix (CSR format)
Run K-Means for k=2 to k=47
Plot elbow curve (SSE vs. k)
Select k=7 and assign clusters
Visualize cluster distribution
Generate recommendations for User 619
Content-Based System

Combine title + genres + tags into text field
Preprocess text (tokenize, lemmatize, remove stopwords)
Apply TF-IDF vectorization (2,642 features)
Calculate cosine similarity matrix
Generate recommendations for "Usual Suspects"
Product Commentary

Why dataset sparsity matters
Cluster imbalance implications
Discovery vs. trust tradeoff
Limitations of both approaches
Reproducing the Analysis
Requirements:

pip install pandas numpy matplotlib seaborn scikit-learn scipy nltk
Download NLTK data:

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
Run the notebook:

Ensure ratings.csv, movies.csv, and tags.csv are in the same directory
Execute cells sequentially
Charts will be generated inline
Files Generated
chart1_sparsity_heatmap.png - Visualization of sparse interaction matrix (18 users × 30 movies)
chart2_elbow_plot.png - SSE vs. k for cluster selection
chart3_cluster_distribution.png - Bar chart showing user distribution across 7 clusters
Limitations
No train/test split: All analysis done on full dataset, no held-out validation
No hyperparameter tuning: Used default parameters for TF-IDF, K-Means
Limited evaluation metrics: No RMSE, precision, recall, or business metrics
Static recommendations: No consideration of temporal patterns, user context, or real-time signals
No hybrid approach: Didn't combine collaborative and content-based methods
Cluster assignment is hard: Users can't belong to multiple clusters
Next Steps for Production
To make this production-ready, you would need:

Evaluation framework: Train/test split, calculate RMSE, Precision@K, Coverage
A/B testing: Measure CTR, conversion, revenue impact
Hybrid model: Combine collaborative + content-based + business rules
Real-time updates: Refresh recommendations based on recent user behavior
Explainability layer: Generate human-readable reasons for each recommendation
Constraint handling: Check inventory, pricing, compatibility, lead times
Diversity optimization: Balance relevance with discovery
Cold-start strategy: Handle new users and new items systematically
This notebook is a learning exercise to understand recommendation algorithms and their product implications, not a production system.
