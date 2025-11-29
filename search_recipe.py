import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
import numpy as np
import subprocess

def build_index(input_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}")
        return
    
    print(f"Building index from {input_dir}")
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(cmd, check=True)
    print("Index built successfully")

def load_queries(query_file):
    with open(query_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(queries)} queries")
    return queries

def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            qid, docid, rel = parts
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels

def search_tfidf(searcher, queries, top_k=10, query_id_start=1):
    searcher.set_bm25(k1=0, b=0)
    results = {}
    for i, query in enumerate(tqdm(queries, desc="Searching with TF-IDF")):
        qid = str(i + query_id_start)
        hits = searcher.search(query, k=top_k)
        results[qid] = [(hit.docid, hit.score) for hit in hits]
    return results

def search_bm25(searcher, queries, top_k=10, query_id_start=1, k1=1.2, b=0.75):
    searcher.set_bm25(k1=k1, b=b)
    results = {}
    for i, query in enumerate(tqdm(queries, desc="Searching with BM25")):
        qid = str(i + query_id_start)
        hits = searcher.search(query, k=top_k)
        results[qid] = [(hit.docid, hit.score) for hit in hits]
    return results

def compute_precision_at_k(results, qrels, k=10, rel_threshold=1):
    precision_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        topk = query_results[:k]
        relevances = [
            1 if qrels[qid].get(docid, 0) >= rel_threshold else 0
            for docid, _ in topk
        ]
        if relevances:
            precision_scores.append(sum(relevances) / k)
    return np.mean(precision_scores) if precision_scores else 0.0

def compute_recall_at_k(results, qrels, k=10, rel_threshold=1):
    recall_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        
        total_relevant = sum(1 for rel in qrels[qid].values() if rel >= rel_threshold)
        if total_relevant == 0:
            continue
            
        topk = query_results[:k]
        retrieved_relevant = sum(
            1 for docid, _ in topk if qrels[qid].get(docid, 0) >= rel_threshold
        )
        
        recall_scores.append(retrieved_relevant / total_relevant)
    
    return np.mean(recall_scores) if recall_scores else 0.0

def compute_map(results, qrels, k=10):
    ap_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        
        relevances = [qrels[qid].get(docid, 0) for docid, _ in query_results[:k]]
        num_relevant = 0
        sum_precisions = 0.0
        
        for i, rel in enumerate(relevances, 1):
            if rel > 0:
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precisions += precision_at_i
        
        if num_relevant > 0:
            ap_scores.append(sum_precisions / num_relevant)
    
    return np.mean(ap_scores) if ap_scores else 0.0

def print_sample_results(searcher, results, queries, n_queries=3, n_results=5):
    print("SAMPLE SEARCH RESULTS")
    
    for i in range(min(n_queries, len(queries))):
        qid = str(i + 1)
        query = queries[i]
        print(f"\nQuery {qid}: \"{query}\"")
        print("-" * 80)
        
        if qid in results:
            for rank, (docid, score) in enumerate(results[qid][:n_results], 1):
                doc = json.loads(searcher.doc(docid).raw())
                title = doc.get('title', 'No title')
                print(f"{rank}. {title}")
                print(f"   Score: {score:.4f}")
        print()

def compare_retrieval_algorithms(index_dir, queries, qrels, top_k=10, query_id_start=1):
    print("COMPARING RETRIEVAL ALGORITHMS: TF-IDF vs BM25")
    searcher = LuceneSearcher(index_dir)
    
    print("\n[1/2] Running TF-IDF retrieval")
    tfidf_results = search_tfidf(searcher, queries, top_k=top_k, query_id_start=query_id_start)
    
    tfidf_metrics = {
        'name': 'TF-IDF',
        'precision@k': compute_precision_at_k(tfidf_results, qrels, k=top_k),
        'recall@k': compute_recall_at_k(tfidf_results, qrels, k=top_k),
        'MAP': compute_map(tfidf_results, qrels, k=top_k)
    }
    
    print("\n[2/2] Running BM25 retrieval")
    bm25_results = search_bm25(searcher, queries, top_k=top_k, query_id_start=query_id_start)
    
    bm25_metrics = {
        'name': 'BM25',
        'precision@k': compute_precision_at_k(bm25_results, qrels, k=top_k),
        'recall@k': compute_recall_at_k(bm25_results, qrels, k=top_k),
        'MAP': compute_map(bm25_results, qrels, k=top_k)
    }
    
    print("RESULTS SUMMARY")
    print(f"\n{'Algorithm':<15} {'Precision@'+str(top_k):<20} {'Recall@'+str(top_k):<20} {'MAP':<10}")
    print(f"{tfidf_metrics['name']:<15} {tfidf_metrics['precision@k']:<20.4f} {tfidf_metrics['recall@k']:<20.4f} {tfidf_metrics['MAP']:<10.4f}")
    print(f"{bm25_metrics['name']:<15} {bm25_metrics['precision@k']:<20.4f} {bm25_metrics['recall@k']:<20.4f} {bm25_metrics['MAP']:<10.4f}")
    
    print_sample_results(searcher, tfidf_results, queries, n_queries=3)
    
    return {
        'tfidf': tfidf_metrics,
        'bm25': bm25_metrics,
        'tfidf_results': tfidf_results,
        'bm25_results': bm25_results
    }

def main():
    BASE_DIR = "/Users/swaggy/Desktop/sentiment-analysis-food/data/recipes"
    CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
    INDEX_DIR = "/Users/swaggy/Desktop/sentiment-analysis-food/indices/recipes_lucene"
    QUERIES_FILE = os.path.join(BASE_DIR, "recipes-queries.txt")
    QRELS_FILE = os.path.join(BASE_DIR, "recipes-qrels.txt")
    
    build_index(CORPUS_DIR, INDEX_DIR)
    
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)
    
    comparison = compare_retrieval_algorithms(
        INDEX_DIR, queries, qrels, top_k=10, query_id_start=1
    )
    
    output = {
        "algorithm_comparison": {
            "tfidf": comparison['tfidf'],
            "bm25": comparison['bm25']
        },
        "num_queries": len(queries),
        "num_qrels": len(qrels),
        "evaluation_metric": "k=10"
    }
    
    with open("retrieval_comparison_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to: retrieval_comparison_results.json")

if __name__ == "__main__":
    main()
