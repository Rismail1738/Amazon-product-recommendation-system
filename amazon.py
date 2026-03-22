import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Amazon Product Recommender", layout="wide")

st.title("Amazon Product Recommendation System")
st.write("Enter a product name to get similar product recommendations.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Ruweida Ali\PYTHON\Amazon Sales Dataset\amazon.csv")

    # Clean numeric columns
    df["discounted_price"] = pd.to_numeric(
        df["discounted_price"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )

    df["actual_price"] = pd.to_numeric(
        df["actual_price"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )

    df["discount_percentage"] = pd.to_numeric(
        df["discount_percentage"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    )

    df["rating_count"] = pd.to_numeric(
        df["rating_count"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Fill missing values
    df["rating_count"] = df["rating_count"].fillna(df["rating_count"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Remove duplicates
    df = df.drop_duplicates()

    # Combined text for recommendations
    df["combined_text"] = (
        df["product_name"].astype(str) + " " +
        df["about_product"].astype(str)
    )

    return df


@st.cache_resource
def build_recommender(text_data):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(text_data)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def recommend_products(product_name, df, cosine_sim, top_n=5):
    product_name = product_name.lower()

    matches = df[df["product_name"].str.lower().str.contains(product_name, na=False)]

    if matches.empty:
        return None

    idx = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in similarity_scores[1:top_n + 10]]

    recommendations = df.iloc[similar_indices][
        ["product_name", "rating", "discounted_price", "about_product"]
    ].drop_duplicates(subset=["product_name"]).head(top_n)

    return recommendations


df = load_data()
cosine_sim = build_recommender(df["combined_text"])

product_input = st.text_input("Enter product name", placeholder="e.g. boAt, Samsung, Ambrane")

if st.button("Get Recommendations"):
    if not product_input.strip():
        st.warning("Please enter a product name.")
    else:
        results = recommend_products(product_input, df, cosine_sim)

        if results is None:
            st.error("Product not found. Try another keyword.")
        else:
            st.success(f"Showing recommendations for: {product_input}")
            st.dataframe(results, use_container_width=True)