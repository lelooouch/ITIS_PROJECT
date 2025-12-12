"""
Unit тесты для основного модуля обнаружения плагиата.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from plagiarism_detector import PlagiarismDetector, AnalysisResults, SimilarityResult


class TestPlagiarismDetector:
    """Тестирование основного класса PlagiarismDetector."""
    
    @pytest.fixture
    def detector(self):
        """Создать экземпляр детектора для тестирования."""
        return PlagiarismDetector()
    
    @pytest.fixture
    def temp_directory(self):
        """Создать временную директорию с тестовыми файлами."""
        temp_dir = tempfile.mkdtemp()
        
        # Создаем тестовые файлы
        files = {
            "essay1.txt": "Машинное обучение это интересная тема для изучения.",
            "essay2.txt": "Изучение машинного обучения может быть интересным.",
            "essay3.txt": "Совершенно другой текст на другую тему.",
        }
        
        for filename, content in files.items():
            filepath = Path(temp_dir) / filename
            filepath.write_text(content, encoding='utf-8')
        
        yield temp_dir
        
        # Очистка
        shutil.rmtree(temp_dir)
    
    def test_detector_initialization(self, detector):
        """Тест инициализации детектора."""
        assert detector is not None
        assert hasattr(detector, 'language')
        assert detector.language == 'english'
    
    def test_analyze_directory_basic(self, detector, temp_directory):
        """Базовый тест анализа директории."""
        results = detector.analyze_directory(
            temp_directory,
            threshold=0.3
        )
        
        assert isinstance(results, AnalysisResults)
        assert results.documents_analyzed == 3
        assert results.timestamp is not None
        assert 0 <= results.average_similarity <= 1
    
    def test_analyze_directory_threshold(self, detector, temp_directory):
        """Тест анализа с разными порогами."""
        # Низкий порог - должно найти больше пар
        results_low = detector.analyze_directory(
            temp_directory,
            threshold=0.1
        )
        
        # Высокий порог - должно найти меньше пар
        results_high = detector.analyze_directory(
            temp_directory,
            threshold=0.9
        )
        
        assert len(results_low.suspicious_pairs) >= len(results_high.suspicious_pairs)
    
    def test_compare_files(self, detector, temp_directory):
        """Тест сравнения двух файлов."""
        file1 = Path(temp_directory) / "essay1.txt"
        file2 = Path(temp_directory) / "essay2.txt"
        
        result = detector.compare_files(str(file1), str(file2))
        
        assert isinstance(result, SimilarityResult)
        assert result.document_a == "essay1"
        assert result.document_b == "essay2"
        assert 0 <= result.similarity <= 1
        assert result.algorithm == 'ensemble'
    
    def test_empty_directory(self, detector, tmp_path):
        """Тест анализа пустой директории."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="Нужно как минимум 2 документа"):
            detector.analyze_directory(str(empty_dir))
    
    def test_single_file_directory(self, detector, tmp_path):
        """Тест анализа директории с одним файлом."""
        single_dir = tmp_path / "single"
        single_dir.mkdir()
        
        file_path = single_dir / "single.txt"
        file_path.write_text("Один файл", encoding='utf-8')
        
        with pytest.raises(ValueError, match="Нужно как минимум 2 документа"):
            detector.analyze_directory(str(single_dir))
    
    def test_save_results(self, detector, temp_directory, tmp_path):
        """Тест сохранения результатов в JSON."""
        results = detector.analyze_directory(temp_directory)
        
        # Сохраняем результаты
        output_file = tmp_path / "test_results.json"
        results.save_json(str(output_file))
        
        # Загружаем и проверяем
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['documents_analyzed'] == 3
        assert 'suspicious_pairs' in loaded_data
        assert 'average_similarity' in loaded_data
        assert 'timestamp' in loaded_data


class TestAnalysisResults:
    """Тестирование класса AnalysisResults."""
    
    def test_analysis_results_creation(self):
        """Тест создания экземпляра AnalysisResults."""
        from datetime import datetime
        
        results = AnalysisResults(
            timestamp=datetime.now().isoformat(),
            documents_analyzed=10,
            suspicious_pairs=[],
            average_similarity=0.5,
            high_risk_pairs=2,
            medium_risk_pairs=3
        )
        
        assert results.documents_analyzed == 10
        assert results.average_similarity == 0.5
        assert results.high_risk_pairs == 2
        assert results.medium_risk_pairs == 3
    
    def test_to_dict_method(self):
        """Тест метода to_dict()."""
        results = AnalysisResults(
            timestamp="2023-12-15T10:00:00",
            documents_analyzed=5,
            suspicious_pairs=[],
            average_similarity=0.3,
            high_risk_pairs=1,
            medium_risk_pairs=2
        )
        
        data = results.to_dict()
        
        assert isinstance(data, dict)
        assert data['documents_analyzed'] == 5
        assert data['average_similarity'] == 0.3
        assert data['high_risk_pairs'] == 1
        assert data['medium_risk_pairs'] == 2
        assert 'timestamp' in data
        assert 'suspicious_pairs' in data
    
    def test_with_similarity_results(self):
        """Тест с объектами SimilarityResult."""
        from datetime import datetime
        
        pairs = [
            SimilarityResult(
                document_a="doc1",
                document_b="doc2",
                similarity=0.8,
                cosine_similarity=0.75,
                lcs_similarity=0.85,
                ngram_similarity=0.78,
                algorithm='ensemble'
            )
        ]
        
        results = AnalysisResults(
            timestamp=datetime.now().isoformat(),
            documents_analyzed=2,
            suspicious_pairs=pairs,
            average_similarity=0.8,
            high_risk_pairs=1,
            medium_risk_pairs=0
        )
        
        data = results.to_dict()
        assert len(data['suspicious_pairs']) == 1
        assert data['suspicious_pairs'][0]['similarity'] == 0.8
        assert data['suspicious_pairs'][0]['document_a'] == 'doc1'


class TestSimilarityResult:
    """Тестирование класса SimilarityResult."""
    
    def test_similarity_result_creation(self):
        """Тест создания экземпляра SimilarityResult."""
        result = SimilarityResult(
            document_a="essay1.txt",
            document_b="essay2.txt",
            similarity=0.75,
            cosine_similarity=0.70,
            lcs_similarity=0.80,
            ngram_similarity=0.72,
            algorithm='ensemble'
        )
        
        assert result.document_a == "essay1.txt"
        assert result.document_b == "essay2.txt"
        assert result.similarity == 0.75
        assert result.cosine_similarity == 0.70
        assert result.lcs_similarity == 0.80
        assert result.ngram_similarity == 0.72
        assert result.algorithm == 'ensemble'
    
    def test_similarity_range(self):
        """Тест что схожесть в диапазоне 0-1."""
        result = SimilarityResult(
            document_a="a",
            document_b="b",
            similarity=0.
