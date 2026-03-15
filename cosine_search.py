import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Model initialization
model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

# Dataset
data = [
    # Food & Recipes
    "How to cook quick chicken dinner with vegetables?",
    "Traditional borscht recipe with beets.",
    "Where to buy fresh fish near me?",
    "How to bake chocolate cake without flour?",
    "Best Italian restaurants in downtown.",
    "How to freeze soup for the week?",
    "Quick breakfast ideas in 5 minutes.",
    "What can replace eggs in baking?",

    # Sports & Fitness
    "How to start morning jogging for beginners?",
    "Back exercises for sedentary work.",
    "What vitamins to take in winter for immunity?",
    "Gym workout program for weight loss.",
    "How to choose good running shoes?",
    "Yoga for beginners at home video tutorials.",
    "How much water should I drink per day?",
    "How to get abs in one month?",

    # Travel
    "Where to go for summer vacation on a budget?",
    "How to pack a suitcase for beach vacation?",
    "Best all-inclusive hotels in Turkey.",
    "Do I need a visa for Georgia?",
    "What to see in Paris in 3 days?",
    "How to find cheap flights online?",
    "Travel insurance for abroad.",
    "Europe road trip itinerary.",

    # Entertainment
    "Top best movies of 2024.",
    "Where to watch TV series for free?",
    "Science fiction book recommendations.",
    "Movie theater showings this weekend.",
    "How to learn oil painting for beginners?",
    "Best board games for friends party.",
    "Rock concerts this month.",
    "Cartoon recommendations for 5 year olds."
]

df = pd.DataFrame(data, columns=["text"])
print(f"Dataset created: {len(df)} records.")

# Vectorization
embeddings = embedding_model.encode(df["text"].tolist(), convert_to_numpy=True)
print(f"Vector shape: {embeddings.shape}")

# Semantic search
def find_similar_texts(query, embeddings, original_texts, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": original_texts[idx],
            "score": float(similarities[idx])
        })
    return results


# Test queries
test_queries = [
    "I want to cook something with meat",
    "How to exercise at home?",
    "Planning a trip abroad",
    "What to watch with family tonight?"
]

print("\nSemantic search test")

for query in test_queries:
    print(f"\nQuery: '{query}'")
    recommendations = find_similar_texts(query, embeddings, df["text"].tolist())
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. [Score: {rec['score']:.4f}] {rec['text']}")

# Clustering
def cluster_texts(embeddings, texts, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(embeddings)

    df_with_clusters = pd.DataFrame({"text": texts, "cluster_id": clusters})

    print(f"Clustering results (K={n_clusters})")

    for cluster_id in range(n_clusters):
        cluster_docs = df_with_clusters[df_with_clusters["cluster_id"] == cluster_id]["text"].tolist()
        print(f"\nCluster {cluster_id}")
        for doc in cluster_docs:
            print(f"    {doc}")


cluster_texts(embeddings, df["text"].tolist(), n_clusters=4)
