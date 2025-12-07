import pandas as pd
import json
import os
from tqdm import tqdm


def create_corpus(input_file, output_dir):
    print("Loading recipes")
    df = pd.read_csv(input_file)
    
    os.makedirs(output_dir, exist_ok=True)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating corpus"):
        doc = {
            "id": str(i),
            "contents": row['document'],
            "title": row['title'],
            "ingredients": row.get('ner_text', ''),
        }
        
        with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as f:
            json.dump(doc, f)
    
    print(f"Corpus created: {len(df)} documents in {output_dir}")


def create_sample_queries(output_file):
    sample_queries = [
        "chocolate chip cookies",
        "chicken alfredo pasta",
        "beef tacos",
        "apple pie",
        "caesar salad",
        "garlic butter shrimp",
        "avocado toast",
        "basil tomato mozzarella",
        "salmon lemon",
        "chicken broccoli",
        "vegan breakfast",
        "gluten free dessert",
        "keto dinner",
        "low carb lunch",
        "vegetarian pasta",
        "quick easy dinner",
        "slow cooker chicken",
        "no bake dessert",
        "one pot meal",
        "grilled vegetables",
        "italian pasta",
        "mexican chicken",
        "asian stir fry",
        "french dessert",
        "mediterranean salad",
        "healthy breakfast",
        "comfort food dinner",
        "party appetizer",
        "weekend brunch",
        "simple lunch",
    ]
    
    with open(output_file, 'w') as f:
        for query in sample_queries:
            f.write(query + '\n')    
    return sample_queries


def create_placeholder_qrels(queries_file, output_file, num_docs):
    with open(queries_file, 'r') as f:
        queries = [line.strip() for line in f]
    
    with open(output_file, 'w') as f:
        for qid in range(len(queries)):
            f.write(f"{qid + 1} {qid} 1\n")

def main():
    RECIPES_CSV = "/Users/swaggy/Desktop/sentiment-analysis-food/data/recipes_processed.csv"
    OUTPUT_DIR = "/Users/swaggy/Desktop/sentiment-analysis-food/data/recipes"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    corpus_dir = os.path.join(OUTPUT_DIR, "corpus")
    queries_file = os.path.join(OUTPUT_DIR, "recipes-queries.txt")
    qrels_file = os.path.join(OUTPUT_DIR, "recipes-qrels.txt")
    
    if not os.path.exists(RECIPES_CSV):
        print(f" Error: {RECIPES_CSV} not found")
        print("Please run recipe_preprocessing.py first")
        return
    
    create_corpus(RECIPES_CSV, corpus_dir)
    create_sample_queries(queries_file)
    df = pd.read_csv(RECIPES_CSV)
    create_placeholder_qrels(queries_file, qrels_file, len(df))

if __name__ == "__main__":
    main()
