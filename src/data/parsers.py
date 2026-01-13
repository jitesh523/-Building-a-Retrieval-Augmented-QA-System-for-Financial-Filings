"""Document parsers for PDF and HTML financial filings."""
import re
from pathlib import Path
from typing import Dict, Optional
import PyPDF2
from bs4 import BeautifulSoup


class PDFParser:
    """Parser for PDF financial filings."""
    
    def __init__(self):
        self.boilerplate_patterns = [
            r'^Table of Contents',
            r'^Page \d+',
            r'^\d+$',  # Page numbers alone
        ]
    
    def parse(self, file_path: str) -> Dict[str, any]:
        """
        Parse PDF document and extract text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Extract metadata
            metadata = {
                'num_pages': len(reader.pages),
                'file_name': path.name,
                'file_type': 'PDF'
            }
            
            # Extract text from all pages
            full_text = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    full_text.append({
                        'page_number': page_num,
                        'text': cleaned_text
                    })
            
            return {
                'text': full_text,
                'metadata': metadata
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove boilerplate
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip boilerplate patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in self.boilerplate_patterns):
                continue
            
            cleaned_lines.append(line)
        
        # Join lines and normalize whitespace
        cleaned = ' '.join(cleaned_lines)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()


class HTMLParser:
    """Parser for HTML financial filings."""
    
    def __init__(self):
        # Tags to remove completely
        self.skip_tags = ['script', 'style', 'head', 'meta', 'link']
    
    def parse(self, file_path: str) -> Dict[str, any]:
        """
        Parse HTML document and extract text.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove unwanted tags
        for tag in soup.find_all(self.skip_tags):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
        cleaned_text = self._clean_text(text)
        
        metadata = {
            'file_name': path.name,
            'file_type': 'HTML'
        }
        
        return {
            'text': [{'text': cleaned_text}],
            'metadata': metadata
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()


class DocumentParser:
    """Factory for document parsers."""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.html_parser = HTMLParser()
    
    def parse(self, file_path: str) -> Dict[str, any]:
        """
        Parse document based on file extension.
        
        Args:
            file_path: Path to document
            
        Returns:
            Parsed document with text and metadata
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.pdf':
            return self.pdf_parser.parse(file_path)
        elif ext in ['.html', '.htm']:
            return self.html_parser.parse(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
