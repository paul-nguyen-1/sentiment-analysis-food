import os
import argparse
import subprocess
import json

class RecipeRetrievalPipeline:
    def __init__(self, config):
        self.config = config
        self.base_dir = config.get('base_dir', os.getcwd())
        
    def print_header(self, title):
        print(f"{title}")
    
    def run_command(self, description, command):
        print(f"\n{description}")
        print(f"Running: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True, capture_output=False)
            print(f"{description} complete\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error in {description}")
            print(f"{e}")
            return False
        except FileNotFoundError:
            print(f"Script not found: {command[1]}")
            return False
    
    def step1_preprocess(self):
        self.print_header("STEP 1: Data Preprocessing")
        return self.run_command(
            "Preprocessing raw recipe data",
            ['python', 'recipe_preprocessing.py']
        )
    
    def step2_explore(self):
        self.print_header("STEP 2: Data Exploration")
        return self.run_command(
            "Exploring dataset",
            ['python', 'explore_data.py']
        )
    
    def step3_prepare_pyserini(self):
        self.print_header("STEP 3: Prepare Data for Pyserini")
        return self.run_command(
            "Converting recipes to Pyserini format",
            ['python', 'prepare_recipe.py']
        )
    
    def step4_search(self):
        self.print_header("STEP 4: Build Index & Run Search")
        return self.run_command(
            "Running search and evaluation",
            ['python', 'search_recipe.py']
        )
    
    def step5_label_qrels(self):
        self.print_header("STEP 5: Generate Relevance Judgments")
        
        qrels_file = self.config['qrels_file']
        index_dir = self.config['index_dir']
        queries_file = self.config['queries_file']
        
        print("Generating qrels based on query-title matching")
        
        if not os.path.exists(index_dir):
            print("Index not found. Run Step 4 first")
            return False
        
        try:
            from pyserini.search.lucene import LuceneSearcher
            
            searcher = LuceneSearcher(index_dir)
            searcher.set_bm25(k1=0.9, b=0.4)
            
            queries = open(queries_file).read().strip().split('\n')
            
            with open(qrels_file, 'w') as f:
                for i, query in enumerate(queries, 1):
                    hits = searcher.search(query, k=10)
                    for hit in hits[:10]:
                        doc = json.loads(searcher.doc(hit.docid).raw())
                        title = doc.get('title', '').lower()
                        query_lower = query.lower()
                        
                        query_words = set(query_lower.split())
                        title_words = set(title.split())
                        overlap = len(query_words & title_words)
                        
                        rel = 2 if overlap >= 2 else (1 if overlap >= 1 else 0)
                        f.write(f"{i} {hit.docid} {rel}\n")
            
            print(f"Auto-generated qrels for {len(queries)} queries")
            print(f"Saved to: {qrels_file}")
            return True
            
        except Exception as e:
            print(f"Error generating qrels: {e}")
            return False
    
    def step6_final_eval(self):
        self.print_header("STEP 6: Final Evaluation")
        
        qrels_file = self.config['qrels_file']
        
        if not os.path.exists(qrels_file):
            print("No qrels file found")
            return False
        
        with open(qrels_file, 'r') as f:
            num_labels = len([l for l in f if l.strip()])
        
        print(f"Found {num_labels} relevance judgments\n")
        
        return self.run_command(
            "Final evaluation with labeled qrels",
            ['python', 'search_recipe.py']
        )
        
    def step7_sentiment_analysis(self):
        self.print_header("STEP 7: Sentiment Analysis")
        return self.run_command(
            "Running sentiment analysis",
            ['python', 'sentiment.py']
        )
    
    def run_full_pipeline(self):
        steps = [
            self.step1_preprocess,
            self.step2_explore,
            self.step3_prepare_pyserini,
            self.step4_search,
            self.step5_label_qrels,
            self.step6_final_eval,
            self.step7_sentiment_analysis,
        ]
        
        for i, step in enumerate(steps, 1):
            if not step():
                print(f"\nPipeline stopped at step {i}")
                return
        
        self.print_header("PIPELINE COMPLETE")
        print("Output files:")
        print(f"bm25_b_tuning.png, bm25_k1_tuning.png")
        print(f"recipe_search_results.json")
        print(f"{self.config['index_dir']}")
    
    def run_step(self, step_num):
        steps = {
            1: self.step1_preprocess,
            2: self.step2_explore,
            3: self.step3_prepare_pyserini,
            4: self.step4_search,
            5: self.step5_label_qrels,
            6: self.step6_final_eval,
        }
        
        if step_num not in steps:
            print(f"Invalid step: {step_num}")
            return
        
        steps[step_num]()


def main():
    parser = argparse.ArgumentParser(description='Recipe Retrieval System Pipeline')
    
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5, 6, 7], help='Run specific step')
    
    args = parser.parse_args()
    
    config = {
        'raw_data': 'data/recipes_data.csv',
        'processed_data': 'data/recipes_processed.csv',
        'corpus_dir': 'data/recipes/corpus',
        'queries_file': 'data/recipes/recipes-queries.txt',
        'qrels_file': 'data/recipes/recipes-qrels.txt',
        'index_dir': 'indices/recipes_lucene',
    }
    
    pipeline = RecipeRetrievalPipeline(config)
    
    if args.full:
        pipeline.run_full_pipeline()
    elif args.step:
        pipeline.run_step(args.step)
    else:
        print("Usage: python main.py --full  OR  python main.py --step N")


if __name__ == "__main__":
    main()
