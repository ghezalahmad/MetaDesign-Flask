
import pandas as pd
import numpy as np
import logging
import os
import requests

class SemanticMatcher:
    """
    Matches text descriptions (from LLM) to concrete rows in the candidate dataset.
    Prioritizes Cohere Rerank if API key matches, otherwise falls back to basic TF-IDF/Cosine similarity.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.use_cohere = bool(api_key)

    def find_best_match(self, text_proposal, candidates_df, input_columns):
        """
        Finds the row in candidates_df that best matches the text_proposal.
        """
        if candidates_df.empty:
            return None, 0.0

        # Create textual representation of candidates for matching
        # "Feature: Value, Feature: Value..."
        candidate_texts = []
        for idx, row in candidates_df.iterrows():
            desc = ", ".join([f"{col}: {row[col]}" for col in input_columns])
            candidate_texts.append(desc)

        if self.use_cohere:
            try:
                return self._match_cohere(text_proposal, candidate_texts, candidates_df)
            except Exception as e:
                logging.error(f"Cohere Rerank failed: {e}. Falling back to local matching.")
                # Fallback to local
        
        return self._match_local(text_proposal, candidate_texts, candidates_df)

    def _match_cohere(self, query, documents, df):
        url = "https://api.cohere.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Client-Name": "MetaDesign-Flask"
        }
        data = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": documents,
            "top_n": 1
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if result['results']:
            top_match = result['results'][0]
            index = top_match['index']
            score = top_match['relevance_score']
            return df.iloc[index], score
            
        return None, 0.0

    def _match_local(self, query, documents, df):
        """
        Simple TF-IDF Cosine Similarity for fallback.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer(stop_words='english')
        # Combine query and docs to ensure vocabulary overlap
        all_text = [query] + documents
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        # Query is at index 0, docs start at 1
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        return df.iloc[best_idx], float(best_score)
