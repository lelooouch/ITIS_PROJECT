"""
Similarity calculation algorithms.
"""
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance

from .main import SimilarityResult


class SimilarityCalculator:
    """Implements various similarity algorithms."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            min_df=1,
            max_df=0.8
        )
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except:
            return 0.0
    
    def lcs_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Longest Common Subsequence similarity.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            LCS similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Split into words
        words1 = text1.split()
        words2 = text2.split()
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate LCS length
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Normalize by shorter text length
        min_length = min(m, n)
        return lcs_length / min_length if min_length > 0 else 0.0
    
    def ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Calculate n-gram similarity.
        
        Args:
            text1: First text
            text2: Second text
            n: Size of n-grams
        
        Returns:
            N-gram similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = text1.split()
        words2 = text2.split()
        
        if len(words1) < n or len(words2) < n:
            # Fall back to Jaccard similarity for short texts
            set1 = set(words1)
            set2 = set(words2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        
        # Generate n-grams
        ngrams1 = [' '.join(words1[i:i+n]) for i in range(len(words1)-n+1)]
        ngrams2 = [' '.join(words2[i:i+n]) for i in range(len(words2)-n+1)]
        
        # Jaccard similarity of n-gram sets
        set1 = set(ngrams1)
        set2 = set(ngrams2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str,
        algorithm: str = 'ensemble'
    ) -> float:
        """
        Calculate similarity using specified algorithm.
        
        Args:
            text1: First text
            text2: Second text
            algorithm: One of 'cosine', 'lcs', 'ngram', 'ensemble'
        
        Returns:
            Similarity score (0-1)
        """
        if algorithm == 'cosine':
            return self.cosine_similarity(text1, text2)
        elif algorithm == 'lcs':
            return self.lcs_similarity(text1, text2)
        elif algorithm == 'ngram':
            return self.ngram_similarity(text1, text2)
        elif algorithm == 'ensemble':
            cosine = self.cosine_similarity(text1, text2)
            lcs = self.lcs_similarity(text1, text2)
            ngram = self.ngram_similarity(text1, text2)
            # Weighted average
            return 0.4 * cosine + 0.3 * lcs + 0.3 * ngram
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def calculate_all_similarities
