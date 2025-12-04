import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
import ast

class SentimentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        print(f"Loading {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Got {len(self.df):,} recipes")
        
        if 'ner_list' in self.df.columns:
            try:
                self.df['ner_list'] = self.df['ner_list'].apply(
                    lambda x: ast.literal_eval(str(x)) if pd.notna(x) else []
                )
            except:
                pass
        
        return self
    
    def get_sentiment(self, text):
        if pd.isna(text) or not text:
            return 0.0, 0.0, 'neutral'
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            return polarity, subjectivity, 'positive'
        elif polarity < -0.1:
            return polarity, subjectivity, 'negative'
        else:
            return polarity, subjectivity, 'neutral'
    
    def categorize_cuisine(self, title, ingredients):
        text = f"{title} {ingredients}".lower()
        
        if any(word in text for word in ['pasta', 'italian', 'parmesan', 'marinara', 'pesto']):
            return 'Italian'
        if any(word in text for word in ['taco', 'burrito', 'salsa', 'enchilada', 'tortilla']):
            return 'Mexican'
        if any(word in text for word in ['stir fry', 'soy sauce', 'ginger', 'sesame', 'thai']):
            return 'Asian'
        if any(word in text for word in ['curry', 'masala', 'tikka', 'naan']):
            return 'Indian'
        if any(word in text for word in ['hummus', 'falafel', 'greek', 'feta', 'olive']):
            return 'Mediterranean'
        if any(word in text for word in ['french', 'croissant', 'quiche']):
            return 'French'
        if any(word in text for word in ['bbq', 'burger', 'southern', 'cajun']):
            return 'American'
        
        return 'Other'
    
    def categorize_diet(self, title, ingredients):
        text = f"{title} {ingredients}".lower()
        tags = []
        
        if 'vegan' in text:
            tags.append('Vegan')
        elif 'vegetarian' in text or 'veggie' in text:
            tags.append('Vegetarian')
        
        if 'gluten free' in text:
            tags.append('Gluten-Free')
        if 'keto' in text or 'low carb' in text:
            tags.append('Keto')
        if 'paleo' in text:
            tags.append('Paleo')
        
        if 'chicken' in text or 'turkey' in text:
            tags.append('Poultry')
        elif 'beef' in text or 'steak' in text:
            tags.append('Beef')
        elif 'pork' in text or 'bacon' in text:
            tags.append('Pork')
        elif any(word in text for word in ['fish', 'salmon', 'shrimp', 'seafood']):
            tags.append('Seafood')
        
        return tags if tags else ['Standard']
    
    def analyze(self):
        print("\nAnalyzing recipes")
        
        polarities = []
        subjectivities = []
        labels = []
        cuisines = []
        dietaries = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            polarity, subjectivity, label = self.get_sentiment(row.get('directions_clean', ''))
            polarities.append(polarity)
            subjectivities.append(subjectivity)
            labels.append(label)
            
            cuisine = self.categorize_cuisine(row.get('title', ''), row.get('ner_text', ''))
            cuisines.append(cuisine)
            
            dietary = self.categorize_diet(row.get('title', ''), row.get('ner_text', ''))
            dietaries.append(dietary)
        
        self.df['sentiment_polarity'] = polarities
        self.df['sentiment_subjectivity'] = subjectivities
        self.df['sentiment_label'] = labels
        self.df['cuisine_type'] = cuisines
        self.df['dietary_categories'] = dietaries
        self.df['primary_dietary'] = [diet[0] for diet in dietaries]
        return self
    
    def show_stats(self):
        print("\nStats:\n")
        
        avg_polarity = self.df['sentiment_polarity'].mean()
        avg_subjectivity = self.df['sentiment_subjectivity'].mean()
        print(f"\nAvg polarity: {avg_polarity:.3f}")
        print(f"Avg subjectivity: {avg_subjectivity:.3f}")
        
        print("\nSentiment breakdown:")
        for label in ['positive', 'neutral', 'negative']:
            count = (self.df['sentiment_label'] == label).sum()
            pct = count / len(self.df) * 100
            print(f"{label}: {count:,} ({pct:.1f}%)")
        
        print("\nTop 5 cuisines:")
        for cuisine, count in self.df['cuisine_type'].value_counts().head(5).items():
            print(f"{cuisine}: {count:,}")
        
        print("\nTop 5 dietary types:")
        for diet, count in self.df['primary_dietary'].value_counts().head(5).items():
            print(f"{diet}: {count:,}")
    
    def show_by_cuisine(self):
        print("\nCuisine Analysis\n")
        
        stats = self.df.groupby('cuisine_type')['sentiment_polarity'].agg(['mean', 'count'])
        stats = stats.sort_values('mean', ascending=False)
        
        print(f"\n{'Cuisine':<20} {'Avg':<10} {'Count':<10}")
        print("-" * 40)
        for cuisine, row in stats.iterrows():
            avg_score = row['mean']
            recipe_count = int(row['count'])
            print(f"{cuisine:<20} {avg_score:<10.3f} {recipe_count:<10}")
    
    def show_by_dietary(self):
        print("\nDietary Type:\n")
        
        stats = self.df.groupby('primary_dietary')['sentiment_polarity'].agg(['mean', 'count'])
        stats = stats.sort_values('mean', ascending=False)
        
        print(f"\n{'Type':<20} {'Avg':<10} {'Count':<10}\n")
        for diet, row in stats.iterrows():
            avg_score = row['mean']
            recipe_count = int(row['count'])
            print(f"{diet:<20} {avg_score:<10.3f} {recipe_count:<10}")
    
    def show_top(self, n=3):
        print(f"\nTop {n} Recipes\n")
        
        top = self.df.nlargest(n, 'sentiment_polarity')
        for idx, row in top.iterrows():
            print(f"\n{row['title']}")
            print(f"Score: {row['sentiment_polarity']:.3f}")
            print(f"{row['cuisine_type']} / {row['primary_dietary']}")
    
    def save(self, output_path):
        keep_cols = [
            'title', 'sentiment_polarity', 'sentiment_subjectivity', 
            'sentiment_label', 'cuisine_type', 'primary_dietary', 
            'dietary_categories'
        ]
        for col in ['ner_text', 'ner_list', 'directions_clean', 'link', 'source']:
            if col in self.df.columns:
                keep_cols.append(col)
        
        self.df[keep_cols].to_csv(output_path, index=False)

def main():
    INPUT = "/Users/swaggy/Desktop/sentiment-analysis-food/data/recipes_processed.csv"
    OUTPUT = "/Users/swaggy/Desktop/sentiment-analysis-food/data/recipes_sentiment.csv"
    
    analyzer = SentimentAnalyzer(INPUT)
    analyzer.load_data()
    analyzer.analyze()
    analyzer.show_stats()
    analyzer.show_by_cuisine()
    analyzer.show_by_dietary()
    analyzer.show_top(n=3)
    analyzer.save(OUTPUT)

if __name__ == "__main__":
    main()
