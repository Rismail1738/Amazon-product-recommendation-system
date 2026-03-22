#  Amazon Product Recommendation System

This project is an end-to-end Machine Learning application that analyzes Amazon product data and provides product recommendations using a content-based filtering approach.

The system includes data preprocessing, exploratory data analysis (EDA), feature engineering, sentiment analysis, a machine learning model, and a Streamlit web application for interactive recommendations.

##  Features

*  Exploratory Data Analysis (EDA)
*  Data Cleaning and Preprocessing
*  Feature Engineering (discount, engagement, etc.)
*  Sentiment Analysis using TextBlob
*  Machine Learning Model (Random Forest Regressor)
*  Content-Based Recommendation System (TF-IDF + Cosine Similarity)
*  Streamlit Web Application

##  How It Works

The recommendation system is based on **content similarity**:

1. Product names and descriptions are combined into a single text feature
2. TF-IDF is used to convert text into numerical vectors
3. Cosine similarity measures how similar products are
4. The system recommends products with the highest similarity scores

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TextBlob (NLP)
* Streamlit

## 📁 Project Structure

amazon-product-recommendation-system/

├── app.py
├── requirements.txt
├── README.md
└── data/
  └── amazon.csv

## ⚠️ Dataset Path (IMPORTANT)

The dataset was originally stored locally at:

C:\Users\Ruweida Ali\PYTHON\Amazon Sales Dataset\amazon.csv

To run this project:

 Either:

* Move `amazon.csv` into the `data/` folder

 OR:

* Update the file path inside `app.py`

Recommended approach:

```python
df = pd.read_csv("data/amazon.csv")
```

## ▶ How to Run the App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the browser link provided (usually http://localhost:8503/)

##  Key Insights

* Most product ratings fall between 3.5 and 4.5
* Discount percentage has little impact on product ratings
* Lower-priced products tend to have higher engagement
* Positive sentiment is strongly associated with higher engagement

