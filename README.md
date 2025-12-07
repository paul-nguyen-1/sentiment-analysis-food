# Recipe Retrieval and Sentiment Analysis
## Overview
This project combines information retrieval and sentiment analysis to build an interactive recipe search system. Within this project you are able to search 10,000+ recipes using natural language queries and explore sentiment patterns across different cuisines and dietary types.

## Features
- **Recipe Search:** BM25-based retrieval using Pyserini (Lucene)
- **Sentiment Analysis:** TextBlob analysis on recipe descriptions
- **Cuisine Classification:** Automatic categorization for Italian, Mexican, Asian, Indian, Mediterranean, French, American cuisines
- **Dietary Classification:** Identifies dietary types such as Vegan, Vegetarian, Keto, Gluten-Free, Paleo, Seafood, etc
- **Interactive Dashboard:** Streamlit web interface with filtering and visualization
- **Retrieval Evaluation:** Compared TF-IDF, BM25, BM25+RM3, and Query Likelihood

## Results
### Retrieval Performance
- **Best Model:** BM25 (k1=0.9, b=0.4)
- **Precision@10:** 0.56
- **Recall@10:** 0.86
- **MAP:** 0.81
- **Dataset:** 10,000 recipes indexed with Lucene

### Sentiment Findings
- American and French cuisines show most positive descriptions
- Vegan recipes have the highest sentiment scores (0.123)
- Gluten-Free and Keto recipes also demonstrate positive language patterns
- Indian cuisine shows slightly negative descriptions (-0.008)

## Dataset
- [Kaggle Recipe Dataset (2M+ Recipes)](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)

### Setup
1. Download `recipes_data.csv` from Kaggle
2. Place it in the `data/` folder

## Installation
### Prerequisites
- Python 3.8+
- Java 11+ 

### Setup

1. Clone the repository
```bash
git clone https://github.com/paul-nguyen-1/sentiment-analysis-food.git
cd sentiment-analysis-food
```

2. Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

3. Verify Java installation
```bash
java -version
```

## Usage
### Run Complete Pipeline
```bash
python main.py --full
```

### Pipeline Steps
``` bash
python main.py --step 1 # Preprocess data
python main.py --step 2 # Explore data
python main.py --step 3 # Prepare for Pyserini
python main.py --step 4 # Build index and search
python main.py --step 5 # Generate relevance judgments
python main.py --step 6 # Final evaluation
python sentiment.py # Sentiment analysis
```

### Launch Dashboard
```
 streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Streamlit Dashboard Features
### Search Tab
- Enter natural language queries
- Get BM25-ranked results with scores
- View sentiment scores for each recipe
- See cuisine and dietary classifications

### Sentiment Tab
- Overall sentiment statistics
- Filter by cuisine type, dietary category, sentiment
- Interactive bar charts showing patterns
- Top positive recipes list

### Retrieval Models
- **TF-IDF:** Term frequency-inverse document frequency baseline
- **BM25:** Best performing model (k1=0.9, b=0.4)
- **BM25+RM3:** BM25 with relevance feedback
- **Query Likelihood:** Language model approach

### Sentiment Analysis
- **Library:** TextBlob
- **Features:** Polarity (-1 to 1), Subjectivity (0 to 1)
- **Classification:** Positive (>0.1), Negative (<0.1), Neutral

### Cuisine Classification
Keyword-based matching for: Italian, Mexican, Asian, Indian, Mediterranean, French, American

### Dietary Classification
Identifies:
- Dietary restrictions (Vegan, Vegetarian, Gluten-Free, Keto, Paleo)
- Protein types (Poultry, Beef, Pork, Seafood)

## Evaluation Metrics
- **Precision@k:** Fraction of relevant recipes in top-k results
- **Recall@k:** Fraction of all relevant recipes found in top-k
- **MAP:** Average precision across all queries
- **nDCG@10:** Normalized discounted cumulative gain at rank 10

## Data Sources
- **Recipe Dataset:** Kaggle Recipe Dataset (10,000+ recipes)
- **Fields Used:** Title, ingredients, directions, NER tags, source

## Demo
Example searches to use:
- "chocolate chip cookies"
- "healthy chicken dinner"
- "vegan pasta"
- "quick breakfast ideas"
- "keto dessert"
- "gluten free pizza"
