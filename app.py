import streamlit as st
import pandas as pd
import os
from pyserini.search.lucene import LuceneSearcher
import json

DATA_DIR = "data"
INDEX_DIR = "indices/recipes_lucene"

st.set_page_config(page_title="Recipe Search", layout="wide")

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "recipes_sentiment.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_searcher():
    if os.path.exists(INDEX_DIR):
        return LuceneSearcher(INDEX_DIR)
    return None

df = load_data()
searcher = load_searcher()

st.title("Recipe Search & Sentiment Analysis")
st.markdown("---")

search_tab, sentiment_tab, about_tab = st.tabs(["Search", "Sentiment"])
with search_tab:
    st.header("Search Recipes")
    
    if not searcher:
        st.error("Index not found. run the pipeline first")
    else:
        query = st.text_input("Enter query:", placeholder="chocolate cookies, vegan pasta, etc.")
        num_results = st.slider("Number of results", 5, 50, 10)
        
        if query:
            searcher.set_bm25(k1=0.9, b=0.4)
            results = searcher.search(query, k=num_results)
            
            st.write(f"Found {len(results)} recipes")
            
            for rank, result in enumerate(results, 1):
                doc = json.loads(searcher.doc(result.docid).raw())
                title = doc.get('title', 'No title')
                ingredients = doc.get('ingredients', '')
                
                with st.expander(f"{rank}. {title} - Score: {result.score:.3f}"):
                    st.write(f"Ingredients: {ingredients[:200]}...")
                    
                    if df is not None:
                        match = df[df['title'] == title]
                        if not match.empty:
                            recipe_data = match.iloc[0]
                            
                            col_sentiment = st.columns(4)
                            col_sentiment[0].metric("Sentiment", recipe_data['sentiment_label'])
                            col_sentiment[1].metric("Score", f"{recipe_data['sentiment_polarity']:.2f}")
                            col_sentiment[2].metric("Cuisine", recipe_data['cuisine_type'])
                            col_sentiment[3].metric("Diet", recipe_data['primary_dietary'])

with sentiment_tab:
    st.header("Sentiment Analysis")
    
    if df is None:
        st.error("No sentiment data. run sentiment.py first")
    else:
        col_total, col_avg, col_pct = st.columns(3)
        
        total_recipes = len(df)
        avg_polarity = df['sentiment_polarity'].mean()
        positive_pct = (df['sentiment_label'] == 'positive').sum() / len(df) * 100
        
        col_total.metric("Total", f"{total_recipes:,}")
        col_avg.metric("Avg Polarity", f"{avg_polarity:.3f}")
        col_pct.metric("Positive %", f"{positive_pct:.1f}%")
        
        st.markdown("---")
        
        st.subheader("Filter")
        col_cuisine, col_diet, col_sentiment = st.columns(3)
        
        cuisines = ['All'] + sorted(df['cuisine_type'].unique().tolist())
        cuisine_filter = col_cuisine.selectbox("Cuisine", cuisines)
        
        diets = ['All'] + sorted(df['primary_dietary'].unique().tolist())
        diet_filter = col_diet.selectbox("Diet", diets)
        
        sent_filter = col_sentiment.selectbox("Sentiment", ['All', 'Positive', 'Neutral', 'Negative'])
        
        filtered = df.copy()
        if cuisine_filter != 'All':
            filtered = filtered[filtered['cuisine_type'] == cuisine_filter]
        if diet_filter != 'All':
            filtered = filtered[filtered['primary_dietary'] == diet_filter]
        if sent_filter != 'All':
            filtered = filtered[filtered['sentiment_label'] == sent_filter.lower()]
        
        st.info(f"Showing {len(filtered):,} recipes")
        
        st.markdown("---")
        
        col_cuisine_chart, col_diet_chart = st.columns(2)
        
        with col_cuisine_chart:
            st.subheader("Sentiment by Cuisine")
            by_cuisine = filtered.groupby('cuisine_type')['sentiment_polarity'].mean()
            by_cuisine = by_cuisine.sort_values(ascending=False)
            st.bar_chart(by_cuisine)
        
        with col_diet_chart:
            st.subheader("Sentiment by Diet")
            by_diet = filtered.groupby('primary_dietary')['sentiment_polarity'].mean()
            by_diet = by_diet.sort_values(ascending=False)
            st.bar_chart(by_diet)
        
        st.markdown("---")
        
        col_counts, col_distribution = st.columns(2)
        
        with col_counts:
            st.subheader("Recipe Counts")
            counts = filtered['cuisine_type'].value_counts()
            st.bar_chart(counts)
        
        with col_distribution:
            st.subheader("Sentiment Split")
            sent_counts = filtered['sentiment_label'].value_counts()
            st.bar_chart(sent_counts)
        
        st.markdown("---")
        st.subheader("Most Positive Recipes")
        
        top = filtered.nlargest(10, 'sentiment_polarity')
        for idx, row in top.iterrows():
            with st.expander(f"{row['title']} ({row['sentiment_polarity']:.3f})"):
                st.write(f"Cuisine: {row['cuisine_type']}")
                st.write(f"Diet: {row['primary_dietary']}")
                st.write(f"Sentiment: {row['sentiment_label']}")
