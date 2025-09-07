# hybrid_search.py - Hybrid Search Engine with BM25 and Vector Search

import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path
import json
import math

class BM25:
    """BM25 implementation for keyword-based search"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

    def fit(self, corpus: List[str]):
        """Fit BM25 parameters to corpus"""
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate document frequencies
        df = defaultdict(int)
        for doc in corpus:
            words = set(self._tokenize(doc))
            for word in words:
                df[word] += 1
        
        # Calculate IDF values
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((len(corpus) - freq + 0.5) / (freq + 0.5))
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return re.findall(r'\w+', text.lower())
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        query_words = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.corpus):
            doc_words = self._tokenize(doc)
            doc_word_counts = Counter(doc_words)
            score = 0
            
            for word in query_words:
                if word in self.idf:
                    tf = doc_word_counts.get(word, 0)
                    idf = self.idf[word]
                    score += idf * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    )
            
            scores.append(score)
        
        return scores

class HybridSearchEngine:
    """Hybrid search engine combining BM25 and vector search"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid search engine
        
        Args:
            alpha: Weight for vector search (1-alpha for BM25)
        """
        self.alpha = alpha
        self.bm25 = BM25()
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.is_fitted = False

    def add_documents(self, doc_data: List[Tuple[str, str, List[Dict], List[List[float]]]]):
        """
        Add documents to the search engine
        
        Args:
            doc_data: List of tuples (doc_id, full_text, chunks, embeddings)
        """
        corpus_texts = []
        
        for doc_id, full_text, chunks, embeddings_list in doc_data:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
                if embedding:  # Only add chunks with valid embeddings
                    self.documents.append({
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {})
                    })
                    self.embeddings.append(embedding)
                    corpus_texts.append(chunk["content"])
        
        # Fit BM25 on the corpus
        if corpus_texts:
            self.bm25.fit(corpus_texts)
            self.is_fitted = True
            print(f"✅ Hybrid search fitted with {len(corpus_texts)} documents")

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a = np.array(v1, dtype=float)
            b = np.array(v2, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except Exception:
            return 0.0

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def search(self, query_embedding: List[float], query_text: str, 
               max_results: int = 5) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25 search
        
        Args:
            query_embedding: Vector representation of the query
            query_text: Text query for BM25 search
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with relevance scores
        """
        if not self.is_fitted or not self.documents:
            return []
        
        try:
            # Vector search scores
            vector_scores = []
            for embedding in self.embeddings:
                similarity = self._cosine_similarity(query_embedding, embedding)
                vector_scores.append(similarity)
            
            # BM25 scores
            bm25_scores = self.bm25.get_scores(query_text)
            
            # Normalize both score sets
            normalized_vector_scores = self._normalize_scores(vector_scores)
            normalized_bm25_scores = self._normalize_scores(bm25_scores)
            
            # Combine scores using RRF (Reciprocal Rank Fusion) approach
            combined_results = []
            
            for i, (doc, vector_score, bm25_score) in enumerate(
                zip(self.documents, normalized_vector_scores, normalized_bm25_scores)
            ):
                # Hybrid score with configurable weighting
                hybrid_score = (
                    self.alpha * vector_score + 
                    (1 - self.alpha) * bm25_score
                )
                
                combined_results.append({
                    "source": doc["doc_id"],
                    "chunk_index": doc["chunk_index"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": hybrid_score,
                    "vector_score": vector_score,
                    "bm25_score": bm25_score
                })
            
            # Sort by hybrid score
            combined_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Apply reranking if we have more sophisticated scoring
            reranked_results = self._rerank_results(combined_results[:max_results*2], 
                                                  query_text)
            
            return reranked_results[:max_results]
            
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            return []

    def _rerank_results(self, results: List[Dict], query_text: str) -> List[Dict]:
        """
        Simple reranking based on additional criteria
        """
        try:
            query_words = set(re.findall(r'\w+', query_text.lower()))
            
            for result in results:
                content_words = set(re.findall(r'\w+', result["content"].lower()))
                
                # Boost score based on exact word matches
                exact_matches = len(query_words.intersection(content_words))
                match_boost = exact_matches / len(query_words) if query_words else 0
                
                # Boost score based on content length (prefer comprehensive answers)
                length_score = min(len(result["content"]) / 500, 1.0)  # Normalize to 500 chars
                
                # Boost score based on metadata (if document type is relevant)
                metadata_boost = 0
                metadata = result.get("metadata", {})
                doc_type = metadata.get("document_type", "")
                
                if any(word in doc_type.lower() for word in ["technical", "research", "analysis"]):
                    metadata_boost = 0.1
                
                # Apply boosts
                boosted_score = (
                    result["similarity"] * 0.7 +  # Original hybrid score
                    match_boost * 0.2 +           # Exact word matches
                    length_score * 0.05 +         # Content comprehensiveness  
                    metadata_boost                # Document type relevance
                )
                
                result["similarity"] = boosted_score
            
            # Re-sort after reranking
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results
            
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "bm25_vocabulary_size": len(self.bm25.idf),
            "is_fitted": self.is_fitted,
            "alpha_weight": self.alpha
        }

# Cohere Reranking Integration (if available)
class CohereReranker:
    """Cohere reranking integration for improved results"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                import cohere
                self.client = cohere.Client(self.api_key)
                print("✅ Cohere reranker initialized")
            except ImportError:
                print("⚠️ Cohere not available. Install: pip install cohere")
            except Exception as e:
                print(f"❌ Cohere initialization failed: {e}")

    def rerank(self, query: str, documents: List[Dict], top_n: int = 5) -> List[Dict]:
        """Rerank documents using Cohere's rerank API"""
        if not self.client or not documents:
            return documents
        
        try:
            # Extract document texts
            doc_texts = [doc["content"] for doc in documents]
            
            # Call Cohere rerank API
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_n=min(top_n, len(documents))
            )
            
            # Reorder results based on Cohere scores
            reranked_results = []
            for result in response.results:
                original_doc = documents[result.index]
                original_doc["similarity"] = result.relevance_score
                original_doc["cohere_score"] = result.relevance_score
                reranked_results.append(original_doc)
            
            return reranked_results
            
        except Exception as e:
            print(f"❌ Cohere reranking failed: {e}")
            return documents

# Enhanced Hybrid Search with Reranking
class EnhancedHybridSearch(HybridSearchEngine):
    """Enhanced hybrid search with optional Cohere reranking"""
    
    def __init__(self, alpha: float = 0.5, use_cohere: bool = True):
        super().__init__(alpha)
        self.reranker = CohereReranker() if use_cohere else None

    def search(self, query_embedding: List[float], query_text: str, 
               max_results: int = 5, use_reranking: bool = True) -> List[Dict]:
        """
        Enhanced search with optional Cohere reranking
        """
        # Get initial results from hybrid search
        results = super().search(query_embedding, query_text, max_results * 2)
        
        # Apply Cohere reranking if available and requested
        if use_reranking and self.reranker and self.reranker.client:
            try:
                reranked_results = self.reranker.rerank(query_text, results, max_results)
                return reranked_results
            except Exception as e:
                print(f"⚠️ Reranking failed, using original results: {e}")
        
        return results[:max_results]

# Example usage
if __name__ == "__main__":
    # Example of how to use the hybrid search engine
    
    # Initialize search engine
    search_engine = EnhancedHybridSearch(alpha=0.6, use_cohere=True)
    
    # Add some example documents (you would get this from your document processor)
    example_docs = [
        ("doc1", "This is a technical document about machine learning", 
         [{"content": "Machine learning algorithms are powerful"}], 
         [[0.1, 0.2, 0.3, 0.4]]),  # Example embedding
        ("doc2", "Business strategy and planning document",
         [{"content": "Strategic planning is essential for business success"}],
         [[0.2, 0.3, 0.1, 0.5]])
    ]
    
    # Add documents to search engine
    search_engine.add_documents(example_docs)
    
    # Perform search
    query_text = "machine learning strategies"
    query_embedding = [0.15, 0.25, 0.2, 0.45]  # Example query embedding
    
    results = search_engine.search(query_embedding, query_text, max_results=5)
    
    # Print results
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Content: {result['content']}")
        print(f"  Score: {result['similarity']:.4f}")
        print(f"  Source: {result['source']}")
        print()
    
    # Print statistics
    stats = search_engine.get_stats()
    print("Search Engine Stats:")
    print(json.dumps(stats, indent=2))