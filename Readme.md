# Amazon Product Recommendation System

This is a Streamlit app that recommends similar Amazon products based on product name and description.

## Features
- Product recommendation using TF-IDF and cosine similarity
- Streamlit web interface
- Content-based recommendation approach

## How it works
The app combines product names and descriptions, converts them into numerical vectors using TF-IDF, and computes cosine similarity to recommend similar products.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

## 3. Make sure your code reads the file correctly

Best if `amazon.csv` is in the same folder as `amazon.py`:

```python
df = pd.read_csv("amazon.csv")
## Dataset

The dataset is located in the `data/` folder.

Make sure the folder structure is maintained when running the project.