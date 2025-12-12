"""
Document loading and preprocessing module.
"""
import re
import string
from pathlib import Path
from typing import Dict, Optional
import PyPDF2
from docx import Document
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class DocumentProcessor:
    """Handles document loading and preprocessing."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.doc'}
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self._download_nltk_resources()
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Regex patterns for cleaning
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\b\d+\b')
    
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def load_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
        
        Returns:
            Dictionary mapping filename to processed text
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        documents = {}
        
        for file_path in directory.glob('*'):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    content = self.load_single_file(str(file_path))
                    if content:
                        processed = self.preprocess_text(content)
                        documents[file_path.stem] = processed
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return documents
    
    def load_single_file(self, filepath: str) -> str:
        """
        Load a single file based on its extension.
        
        Args:
            filepath: Path to file
        
        Returns:
            Extracted text content
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        extension = path.suffix.lower()
        
        if extension == '.txt':
            return self._load_txt(path)
        elif extension == '.pdf':
            return self._load_pdf(path)
        elif extension in ['.docx', '.doc']:
            return self._load_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _load_txt(self, path: Path) -> str:
        """Load text from .txt file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_pdf(self, path: Path) -> str:
        """Extract text from .pdf file."""
        text = ""
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _load_docx(self, path: Path) -> str:
        """Extract text from .docx/.doc file."""
        doc = Document(path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for similarity comparison.
        
        Args:
            text: Raw text string
        
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.number_pattern.sub(' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
