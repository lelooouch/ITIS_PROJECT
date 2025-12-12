"""
Main module for the Plagiarism Detection System.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .document_processor import DocumentProcessor
from .similarity_calculator import SimilarityCalculator
from .visualizer import ResultVisualizer


@dataclass
class SimilarityResult:
    """Represents a similarity comparison between two documents."""
    document_a: str
    document_b: str
    similarity: float
    cosine_similarity: float
    lcs_similarity: float
    ngram_similarity: float
    algorithm: str


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    timestamp: str
    documents_analyzed: int
    suspicious_pairs: List[SimilarityResult]
    average_similarity: float
    high_risk_pairs: int
    medium_risk_pairs: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'timestamp': self.timestamp,
            'documents_analyzed': self.documents_analyzed,
            'suspicious_pairs': [
                asdict(pair) for pair in self.suspicious_pairs
            ],
            'average_similarity': self.average_similarity,
            'high_risk_pairs': self.high_risk_pairs,
            'medium_risk_pairs': self.medium_risk_pairs
        }
    
    def save_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class PlagiarismDetector:
    """Main class for plagiarism detection."""
    
    def __init__(self, language: str = 'english'):
        self.document_processor = DocumentProcessor(language=language)
        self.similarity_calculator = SimilarityCalculator()
        self.visualizer = ResultVisualizer()
    
    def analyze_directory(
        self,
        directory_path: str,
        threshold: float = 0.7,
        algorithm: str = 'ensemble'
    ) -> AnalysisResults:
        """
        Analyze all documents in a directory for plagiarism.
        
        Args:
            directory_path: Path to directory with documents
            threshold: Similarity threshold (0-1)
            algorithm: One of 'cosine', 'lcs', 'ngram', 'ensemble'
        
        Returns:
            AnalysisResults object
        """
        print(f"Analyzing directory: {directory_path}")
        
        # Load and preprocess documents
        documents = self.document_processor.load_directory(directory_path)
        
        if len(documents) < 2:
            raise ValueError("Need at least 2 documents for analysis")
        
        print(f"Loaded {len(documents)} documents")
        
        # Calculate similarities
        doc_names = list(documents.keys())
        doc_texts = list(documents.values())
        
        similarities = self.similarity_calculator.calculate_all_similarities(
            doc_texts, doc_names, algorithm
        )
        
        # Filter suspicious pairs
        suspicious_pairs = [
            pair for pair in similarities
            if pair.similarity >= threshold
        ]
        
        # Calculate statistics
        all_similarities = [pair.similarity for pair in similarities]
        avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0
        
        high_risk = len([p for p in suspicious_pairs if p.similarity > 0.8])
        medium_risk = len([p for p in suspicious_pairs if 0.5 <= p.similarity <= 0.8])
        
        return AnalysisResults(
            timestamp=datetime.now().isoformat(),
            documents_analyzed=len(documents),
            suspicious_pairs=suspicious_pairs,
            average_similarity=avg_similarity,
            high_risk_pairs=high_risk,
            medium_risk_pairs=medium_risk
        )
    
    def compare_files(
        self,
        file1: str,
        file2: str,
        algorithm: str = 'ensemble'
    ) -> SimilarityResult:
        """
        Compare two specific files.
        
        Args:
            file1: Path to first file
            file2: Path to second file
            algorithm: Similarity algorithm to use
        
        Returns:
            SimilarityResult object
        """
        text1 = self.document_processor.load_single_file(file1)
        text2 = self.document_processor.load_single_file(file2)
        
        if algorithm == 'ensemble':
            cosine = self.similarity_calculator.cosine_similarity(text1, text2)
            lcs = self.similarity_calculator.lcs_similarity(text1, text2)
            ngram = self.similarity_calculator.ngram_similarity(text1, text2)
            similarity = (0.4 * cosine + 0.3 * lcs + 0.3 * ngram)
        else:
            similarity = self.similarity_calculator.calculate_similarity(
                text1, text2, algorithm
            )
        
        return SimilarityResult(
            document_a=Path(file1).name,
            document_b=Path(file2).name,
            similarity=similarity,
            cosine_similarity=self.similarity_calculator.cosine_similarity(text1, text2),
            lcs_similarity=self.similarity_calculator.lcs_similarity(text1, text2),
            ngram_similarity=self.similarity_calculator.ngram_similarity(text1, text2),
            algorithm=algorithm
        )
    
    def generate_visualization(
        self,
        results: AnalysisResults,
        output_path: str
    ) -> None:
        """
        Generate visualization of results.
        
        Args:
            results: AnalysisResults object
            output_path: Path to save visualization
        """
        self.visualizer.create_similarity_matrix(
            results, output_path
        )
