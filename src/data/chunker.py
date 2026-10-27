"""Text chunking strategies for document processing."""
import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    index: int
    page_number: int = None
    section: str = None
    token_count: int = 0


class FixedSizeChunker:
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Raises:
            ValueError: If chunk_size is not positive or overlap is outside
                the range 0 <= overlap < chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, page_number: int = None) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to chunk
            page_number: Optional page number for metadata
            
        Returns:
            List of Chunk objects
        """
        # Simple tokenization (split on whitespace)
        tokens = text.split()
        
        if not tokens:
            return []
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append(Chunk(
                text=chunk_text,
                index=chunk_index,
                page_number=page_number,
                token_count=len(chunk_tokens)
            ))
            
            chunk_index += 1
            
            # Move start index considering overlap
            start_idx += self.chunk_size - self.overlap
            
            # Stop if we've covered all tokens
            if end_idx >= len(tokens):
                break
        
        return chunks


class SemanticChunker:
    """Semantic chunking based on paragraphs and sections."""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Section headers pattern (simplified)
        self.section_pattern = re.compile(
            r'^(ITEM \d+[A-Z]?\.|'
            r'Part [IVX]+\.|'
            r'Risk Factors|'
            r'Business|'
            r'Financial Statements|'
            r'Management.*Discussion)',
            re.IGNORECASE | re.MULTILINE
        )
    
    def chunk(self, text: str, page_number: int = None) -> List[Chunk]:
        """
        Split text into semantic chunks based on structure.
        
        Args:
            text: Text to chunk
            page_number: Optional page number
            
        Returns:
            List of Chunk objects
        """
        # Split by potential section headers
        sections = self._split_by_sections(text)
        
        chunks = []
        chunk_index = 0
        
        for section_text, section_name in sections:
            # Further split large sections
            section_chunks = self._split_section(section_text)
            
            for chunk_text in section_chunks:
                token_count = len(chunk_text.split())
                
                # Skip very small chunks
                if token_count < 20:
                    continue
                
                chunks.append(Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    page_number=page_number,
                    section=section_name,
                    token_count=token_count
                ))
                
                chunk_index += 1
        
        return chunks
    
    def _split_by_sections(self, text:str) -> List[tuple]:
        """Split text by section headers."""
        sections = []
        current_section = ""
        current_name = None
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            match = self.section_pattern.search(line)
            
            if match:
                # Save previous section
                if current_section.strip():
                    sections.append((current_section.strip(), current_name))
                
                # Start new section
                current_name = line.strip()
                current_section = ""
            else:
                current_section += line + " "
        
        # Add final section
        if current_section.strip():
            sections.append((current_section.strip(), current_name))
        
        # If no sections found, treat whole text as one section
        if not sections:
            sections = [(text, None)]
        
        return sections
    
    def _split_section(self, text: str) -> List[str]:
        """Split large sections into smaller chunks."""
        tokens = text.split()
        
        if len(tokens) <= self.max_chunk_size:
            return [text]
        
        # Split by sentences/paragraphs
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_size + sentence_tokens > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class DocumentChunker:
    """Main document chunker with multiple strategies."""
    
    def __init__(self, strategy: str = 'fixed', **kwargs):
        """
        Initialize document chunker.
        
        Args:
            strategy: 'fixed' or 'semantic'
            **kwargs: Arguments for the chosen strategy
        """
        if strategy == 'fixed':
            self.chunker = FixedSizeChunker(**kwargs)
        elif strategy == 'semantic':
            self.chunker = SemanticChunker(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def chunk_document(self, parsed_doc: Dict) -> List[Chunk]:
        """
        Chunk a parsed document.
        
        Args:
            parsed_doc: Document from DocumentParser
            
        Returns:
            list of Chunk objects
        """
        all_chunks = []
        
        # Handle both PDF (list of pages) and HTML (single text)
        text_parts = parsed_doc.get('text', [])
        
        for part in text_parts:
            text = part.get('text', '')
            page_num = part.get('page_number')
            
            chunks = self.chunker.chunk(text, page_number=page_num)
            all_chunks.extend(chunks)
        
        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.index = i
        
        return all_chunks
