import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast
import os

class RecipeExplorer:
    def __init__(self, data_path: str, sample_size: int = 50000):
        print(f"Loading {sample_size:,} recipes from {data_path}")
        self.df = pd.read_csv(data_path, nrows=sample_size)
        print(f"Loaded {len(self.df):,} recipes")
    
    def stats(self):
        print("Stats")
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing'] > 0])
        
        print("\nDuplicates:")
        print(f"Duplicate titles: {self.df['title'].duplicated().sum()}")
        print(f"Duplicate links: {self.df['link'].duplicated().sum()}")
    
    def source_distribution(self):
        print("\nSource distribution:")
        if 'source' in self.df.columns:
            print("\nRecipes by source:")
            print(self.df['source'].value_counts())
        
        if 'site' in self.df.columns:
            print("\nTop 10 sites:")
            print(self.df['site'].value_counts().head(10))
    
    def text_length_analysis(self):
        
        self.df['title_len'] = self.df['title'].str.len()
        print("\nTitle length statistics:")
        print(self.df['title_len'].describe())
        
        self.df['directions_len'] = self.df['directions'].str.len()
        print("\nDirections length statistics:")
        print(self.df['directions_len'].describe())
        
        def count_ingredients(ner_field):
            try:
                parsed = ast.literal_eval(str(ner_field))
                return len(parsed) if isinstance(parsed, list) else 0
            except:
                return 0
        
        self.df['ingredient_count'] = self.df['NER'].apply(count_ingredients)
        print("Ingredient count statistics:")
        print(self.df['ingredient_count'].describe())
    
    def sample_recipes(self, n: int = 3):
        for idx in range(min(n, len(self.df))):
            recipe = self.df.iloc[idx]
            print(f"\nRecipe {idx + 1}:")
            print(f"Title: {recipe['title']}")
            
            ingredients = ast.literal_eval(str(recipe['NER']))
            print(f"Ingredients ({len(ingredients)}): {', '.join(ingredients[:5])}")
            
            print(f"Directions: {str(recipe['directions'])[:200]}")
            print(f"Source: {recipe.get('source', 'N/A')} | Site: {recipe.get('site', 'N/A')}")
    
    def ingredient_analysis(self, top_n: int = 20):
        print("Top ingredients analysis")
        all_ingredients = []
        for ner_field in self.df['NER']:
            try:
                parsed = ast.literal_eval(str(ner_field))
                if isinstance(parsed, list):
                    all_ingredients.extend([ing.lower().strip() for ing in parsed])
            except:
                continue
        
        ingredient_counts = Counter(all_ingredients)
        print("\nMost common ingredients:")
        for ingredient, count in ingredient_counts.most_common(top_n):
            pct = (count / len(self.df)) * 100
            print(f"{ingredient:.<30} {count:>6,} recipes ({pct:.1f}%)")
    
    def create_visualizations(self, output_dir: str = "."):
        print("Creating visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        has_multiple_sources = False
        if 'source' in self.df.columns:
            source_counts = self.df['source'].value_counts()
            has_multiple_sources = len(source_counts) > 1
        
        if has_multiple_sources:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes_flat = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes_flat = axes
        
        # Title length distribution
        axes_flat[0].hist(self.df['title_len'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes_flat[0].set_xlabel('Title Length (characters)')
        axes_flat[0].set_ylabel('Frequency')
        axes_flat[0].set_title('Distribution of Recipe Title Lengths')
        axes_flat[0].axvline(self.df['title_len'].median(), color='red', linestyle='--', label=f'Median: {self.df["title_len"].median():.0f}')
        axes_flat[0].legend()
        axes_flat[0].grid(alpha=0.3)
        
        # Ingredient count distribution
        ingredient_95th = self.df['ingredient_count'].quantile(0.95)
        axes_flat[1].hist(self.df['ingredient_count'], bins=30, edgecolor='black', alpha=0.7, color='seagreen')
        axes_flat[1].set_xlabel('Number of Ingredients')
        axes_flat[1].set_ylabel('Frequency')
        axes_flat[1].set_title('Distribution of Ingredient Counts')
        axes_flat[1].axvline(self.df['ingredient_count'].median(), color='red', linestyle='--', label=f'Median: {self.df["ingredient_count"].median():.0f}')
        axes_flat[1].set_xlim(0, min(25, ingredient_95th + 2))
        axes_flat[1].legend()
        axes_flat[1].grid(alpha=0.3)
        
        axes_flat[2].hist(self.df['directions_len'], bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes_flat[2].set_xlabel('Directions Length (characters)')
        axes_flat[2].set_ylabel('Frequency')
        axes_flat[2].set_title('Distribution of Recipe Directions Lengths')
        axes_flat[2].axvline(self.df['directions_len'].median(), color='red', linestyle='--', label=f'Median: {self.df["directions_len"].median():.0f}')
        axes_flat[2].legend()
        axes_flat[2].grid(alpha=0.3)
        
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "recipe_exploration.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
    
    def run_full_exploration(self):
        self.stats()
        self.source_distribution()
        self.text_length_analysis()
        self.ingredient_analysis(top_n=20)
        self.sample_recipes(n=3)
        self.create_visualizations()
        
        print("Completed")

def main():
    DATA_PATH = "data/recipes_data.csv"
    
    if not os.path.exists(DATA_PATH):
        print("Data file not found")
        print(f"Please place recipes_data.csv at: {DATA_PATH}")
        return
    
    explorer = RecipeExplorer(DATA_PATH, sample_size=50000)
    explorer.run_full_exploration()


if __name__ == "__main__":
    main()
