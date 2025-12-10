import pandas as pd
import numpy as np
import re
import ast

class RecipePreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self, sample_size=None, random_state=42):
        print(f"Loading data from {self.data_path}")
        
        if sample_size:
            total = sum(1 for _ in open(self.data_path)) - 1
            skip = np.random.RandomState(random_state).choice(
                range(1, total), 
                total - sample_size, 
                replace=False
            )
            self.df = pd.read_csv(self.data_path, skiprows=skip)
            print(f"Loaded sample of {len(self.df)} recipes")
        else:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} recipes")
            
        return self
    
    def explore_data(self):
        print("Dataset Overview:")
        print(f"Total recipes: {len(self.df)}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nSample recipe:")
        print(self.df.iloc[0])
            
        return self
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = text.strip()
        
        return text.lower()
    
    def parse_list_field(self, field):
        if pd.isna(field):
            return []
        try:
            parsed = ast.literal_eval(str(field))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except:
            pass
        return [item.strip() for item in str(field).split(',')]
    
    def process_recipes(self):
        print("Processing recipes")
        
        print("Cleaning titles")
        self.df['title_clean'] = self.df['title'].apply(self.clean_text)
        
        self.df['ingredients_list'] = self.df['ingredients'].apply(self.parse_list_field)
        self.df['ingredients_text'] = self.df['ingredients_list'].apply(
            lambda x: ' '.join(x) if x else ""
        )
        
        self.df['ner_list'] = self.df['NER'].apply(self.parse_list_field)
        self.df['ner_text'] = self.df['ner_list'].apply(
            lambda x: ' '.join(x) if x else ""
        )
        
        self.df['directions_clean'] = self.df['directions'].apply(self.clean_text)
        
        print("Creating combined document field")
        self.df['document'] = (
            self.df['title_clean'] + ' ' + 
            self.df['ner_text'] + ' ' +
            self.df['directions_clean']
        ).str.strip()
        
        before = len(self.df)
        self.df = self.df[self.df['document'].str.len() > 0]
        dropped = before - len(self.df)
        if dropped > 0:
            print(f"Removed {dropped} recipes with empty content")
        
        print(f"Processing complete {len(self.df)} recipes ready for indexing")
        
        return self
    
    def get_statistics(self):
        print("Processing Statistics")
        print(f"Average title length: {self.df['title_clean'].str.len().mean():.1f} chars")
        print(f"Average ingredients count: {self.df['ner_list'].apply(len).mean():.1f}")
        print(f"Average document length: {self.df['document'].str.len().mean():.1f} chars")
        print(f"Average word count per document: {self.df['document'].str.split().apply(len).mean():.1f}")
        
        return self
    
    def save_processed_data(self, output_path):
        cols = [
            'title',
            'title_clean', 
            'ner_list',
            'ner_text',
            'ingredients_text',
            'directions_clean',
            'document',
            'link',
            'source',
            'site'
        ]
        
        output_cols = [c for c in cols if c in self.df.columns]
        output_df = self.df[output_cols]
        output_df.to_csv(output_path, index=False)
        
        print(f"\nSaved processed data to {output_path}")
        return output_path


def main():
    INPUT_FILE = "data/recipes_data.csv"
    OUTPUT_FILE = "data/recipes_processed.csv"
    SAMPLE_SIZE = 10000
    
    print("Data Preprocessing:")
    preprocessor = RecipePreprocessor(INPUT_FILE)
    
    preprocessor.load_data(sample_size=SAMPLE_SIZE)
    preprocessor.explore_data()
    preprocessor.process_recipes()
    preprocessor.get_statistics()
    preprocessor.save_processed_data(OUTPUT_FILE)
    
    print("Preprocessing complete")

if __name__ == "__main__":
    main()
