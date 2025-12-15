import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity[ 1 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEeGfxMi18yOht6ARqgc5sUIhVZlOukZrh9LmKVfa1THuMYhtG9_hCw5fBh0k2poSql_1pwz4HIewXTCmXuR4RYVVJ19hc0oUNrE7BdzPj1ygco8c-uLesGk9HyS7eEFADmsr5DIRBNow2TqTqdpWuMJphGptFmVAFxyLllDB5buDr_EeV75JwyFLUDmtSqaFTQnSLTBQ==)][ 4 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFZ4pcuGJj6C5ow_Yt-KG83JC97ZWmQOGzroAbqgKTWwS7OdHhhv9NYJ7VOL3EtEjgtfk8ERJwjt-Z0HK0wNSn2wF9_KvtrgZltRLnZ4ceUoRWiFma8gFRMu_Ow7A6bWXQ5PhDxRUmhOk0ikKXfEaf4uXglf8ZxNotLHGRcAT1ltLzve3SiwxzsUuCE8mqQBDgDEBMK9Rmj3vM=)][ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFefYiSWsuYXxdk16KtUwG8cg1-0mIWGOso9hlFlpoGnJmjhhP-qIOI9ye4iJ681yNk07vL1Kmw-mr1xDQi7PONkjvPJYPmBOoBCp9LfWgvPaEvlkN94YQbXlfEpcI1EsgAujct4LPvMjVTX0SYRQentyrDQHQgOmRxhGCQuz8E0NYF0Zk7JU2CFf34YNY=)][ 6 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFw0cZclR36_wz4VUJc_h7BOfeqxvDp8G5Zar-OKbFNSEjBgORarTe57QFYzHXpQjoXobif69B3XCmXmApzOWpN7v_qEMIVL4udTC8YbaFaJT_-08j3Mur_IZgGGjpMgT02rnlCKaAPmeLao6gL23kDj2ipMVPOgp7WIQcYCEQMzBOlt6qYrYc0cn-N2bI=)][ 7 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGQ5KRqxThmmnF7GeqVLEDgHVHiaKDtQ5e6mi96ZpkMXcGpGkMkGrc_Vcc8JaFIkilXUxU41-JAM-YutrVQf9LAmHljzeQz1CVkeu2AgTp6L2ADwSzSpCWqRTahNPgCdlBwCtseZf8=)][ 8 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH3OEgXXI4JPm0OsoMtxujzIyhFzzFSm3gpoMnsgOZ-CGw1rSmj4yzpMsGor0eE5qA5JgapuuQA3tZs4xq99yBSaVCMMz_QoBiYEQTYsMSVsrAJGd9KJhiZb53j5_CGVZv8oHn0vG6mUlwDhZYmrThUwKS5N-0gvw==)]

# 1. THE DATASET (A dummy e-commerce catalog)
data = [
    {"id": 1, "name": "Wireless Headphones", "category": "Electronics", "tags": "audio music wireless bluetooth bass"},
    {"id": 2, "name": "Running Shoes", "category": "Footwear", "tags": "sports running sneakers gym comfort"},
    {"id": 3, "name": "Gaming Laptop", "category": "Electronics", "tags": "computer gaming graphics fast performance"},
    {"id": 4, "name": "Coffee Maker", "category": "Home", "tags": "kitchen coffee brew morning appliance"},
    {"id": 5, "name": "Yoga Mat", "category": "Sports", "tags": "gym yoga exercise floor comfort"},
    {"id": 6, "name": "Bluetooth Speaker", "category": "Electronics", "tags": "audio music wireless party portable"},
    {"id": 7, "name": "Trail Sneakers", "category": "Footwear", "tags": "sports hiking outdoor running durable"},
    {"id": 8, "name": "Smart Watch", "category": "Electronics", "tags": "tech fitness time bluetooth tracking"},
    {"id": 9, "name": "Blender", "category": "Home", "tags": "kitchen smoothy cooking appliance food"},
    {"id": 10, "name": "Dumbbells Pair", "category": "Sports", "tags": "gym weights exercise muscle fitness"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# 2. THE RECOMMENDATION ENGINE
# We create a "soup" of metadata (category + tags) for the algorithm to read
df['soup'] = df['category'] + " " + df['tags']

# Initialize CountVectorizer to convert text into numbers (vectors)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

# Compute Cosine Similarity (how similar is every product to every other product?)
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Map product names to their index in the dataframe
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    """
    Takes a product title, finds its index, looks up similarity scores,
    and returns the top 3 most similar items.
    """
    # Get the index of the product that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x 1 , reverse=True)

    # Get the scores of the 3 most similar products (ignoring itself at index 0)
    sim_scores = sim_scores[1:4]

    # Get the product indices
    movie_indices = [i 0  for i in sim_scores]

    # Return the top 3 most similar products
    return df['name'].iloc[movie_indices]

# 3. THE USER INTERFACE (Streamlit)
st.title("🛍️ AI Shopping Assistant")
st.write("Select a product you like, and I'll recommend similar items!")

# Dropdown menu
selected_product = st.selectbox("I liked this product:", df['name'].values)

# Button to generate recommendations
if st.button('Recommend'):
    recommendations = get_recommendations(selected_product)
    
    st.subheader("You might also like:")
    for i, product in enumerate(recommendations):
        st.write(f"{i+1}. **{product}**")
