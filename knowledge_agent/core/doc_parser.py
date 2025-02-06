"""Module for parsing various documentation formats."""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import PyPDF2
import toml
import yaml
import latex2mathml.converter
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

@dataclass
class DocSection:
    """A section of a document with metadata."""
    content: str
    title: str = 'Untitled Section'
    source_file: Optional[Path] = None
    line_number: int = 0
    content_type: str = 'text'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure metadata is a dictionary."""
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            self.metadata = {'value': str(self.metadata)}
        if self.source_file is not None and not isinstance(self.source_file, Path):
            self.source_file = Path(str(self.source_file))

class DocParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a document file and return a list of sections."""
        pass
    
    @abstractmethod
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from document content."""
        pass

class MarkdownParser(DocParser):
    """Parser for Markdown documents."""
    
    def __init__(self):
        """Initialize the Markdown parser."""
        self.md = MarkdownIt()
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a Markdown file into sections."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the markdown content
            tokens = self.md.parse(content)
            current_section = []
            current_title = None
            current_line = 0
            
            for token in tokens:
                if token.type == 'heading_open':
                    # If we have a previous section, add it
                    if current_section:
                        section_content = '\n'.join(current_section)
                        if section_content.strip():
                            sections.append(DocSection(
                                content=section_content,
                                title=current_title or 'Untitled Section',
                                source_file=file_path,
                                line_number=current_line,
                                content_type='markdown',
                                metadata={'type': 'heading', 'level': token.tag[1]}
                            ))
                        current_section = []
                    current_line = token.map[0] + 1 if token.map else 0
                elif token.type == 'heading_close':
                    current_title = '\n'.join(current_section)
                    current_section = []
                elif token.type == 'inline':
                    current_section.append(token.content)
                elif token.type == 'paragraph_open':
                    if current_section:
                        section_content = '\n'.join(current_section)
                        if section_content.strip():
                            sections.append(DocSection(
                                content=section_content,
                                title=current_title or 'Untitled Section',
                                source_file=file_path,
                                line_number=current_line,
                                content_type='markdown',
                                metadata={'type': 'paragraph'}
                            ))
                        current_section = []
                    current_line = token.map[0] + 1 if token.map else 0
            
            # Add the last section
            if current_section:
                section_content = '\n'.join(current_section)
                if section_content.strip():
                    sections.append(DocSection(
                        content=section_content,
                        title=current_title or 'Untitled Section',
                        source_file=file_path,
                        line_number=current_line,
                        content_type='markdown',
                        metadata={'type': 'paragraph'}
                    ))
            
            # If no sections were created, create one for the entire content
            if not sections and content.strip():
                sections.append(DocSection(
                    content=content,
                    title='Document',
                    source_file=file_path,
                    line_number=1,
                    content_type='markdown',
                    metadata={'type': 'document'}
                ))
                
        except Exception as e:
            logger.error(f"Failed to parse markdown file {file_path}: {str(e)}")
            # Return the entire content as a single section if it's not empty
            if content.strip():
                sections = [DocSection(
                    content=content,
                    title='Document',
                    source_file=file_path,
                    line_number=1,
                    content_type='markdown',
                    metadata={'type': 'document'}
                )]
        
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from Markdown content."""
        metadata = {}
        try:
            # Look for YAML frontmatter
            if content.startswith('---'):
                end_marker = content.find('---', 3)
                if end_marker != -1:
                    frontmatter = content[3:end_marker]
                    try:
                        metadata = yaml.safe_load(frontmatter)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed to extract Markdown metadata: {str(e)}")
        return metadata or {}

class TextParser(DocParser):
    """Parser for plain text documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a text file into sections."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split content into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                sections.append(DocSection(
                    content=paragraph,
                    source_file=file_path,
                    line_number=sum(len(p.split('\n')) for p in paragraphs[:i]) + i + 1,
                    content_type='text',
                    metadata={'type': 'paragraph', 'index': i}
                ))
        except Exception as e:
            logger.error(f"Failed to parse text file {file_path}: {str(e)}")
            # Return the entire content as a single section
            sections = [DocSection(
                content=content,
                source_file=file_path,
                content_type='text',
                metadata={'type': 'document'}
            )]
        
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from text content."""
        metadata = {}
        try:
            lines = content.split('\n')
            if lines:
                # Look for "Key: Value" patterns
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key and value:
                            metadata[key] = value
        except Exception as e:
            logger.error(f"Failed to extract text metadata: {str(e)}")
        return metadata

class ReStructuredTextParser(DocParser):
    """Parser for reStructuredText documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a reStructuredText file."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract metadata first
                metadata = self.extract_metadata(content)
                
                # Split into sections based on headers
                current_section = ""
                current_title = "Main"
                line_number = 1
                
                lines = content.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i]
                    next_line = lines[i + 1] if i + 1 < len(lines) else ''
                    
                    # Check for section headers (underlined with =, -, ~, etc.)
                    if next_line and set(next_line) <= set('=-~^'):
                        # Save previous section if it exists
                        if current_section:
                            sections.append(DocSection(
                                content=current_section.strip(),
                                title=current_title,
                                source_file=file_path,
                                line_number=line_number,
                                content_type='rst',
                                metadata=metadata
                            ))
                        
                        # Start new section
                        current_title = line.strip()
                        current_section = ""
                        i += 2  # Skip the underline
                    else:
                        current_section += line + "\n"
                    line_number += 1
                
                # Add the last section
                if current_section:
                    sections.append(DocSection(
                        content=current_section.strip(),
                        title=current_title,
                        source_file=file_path,
                        line_number=line_number,
                        content_type='rst',
                        metadata=metadata
                    ))
        except Exception as e:
            logger.error(f"Failed to parse RST {file_path}: {str(e)}")
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from reStructuredText content."""
        metadata = {}
        try:
            # Look for field lists at the start of the document
            lines = content.split('\n')
            for line in lines:
                if not line.strip():
                    break
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lstrip(':').lower()  # Remove leading : and convert to lowercase
                    value = value.strip()
                    if key and value:
                        metadata[key] = value
        except Exception as e:
            logger.error(f"Failed to extract RST metadata: {str(e)}")
        return metadata

class PdfParser(DocParser):
    """Parser for PDF documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a PDF file."""
        sections = []
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                # Extract text from each page
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        sections.append(DocSection(
                            content=text,
                            title=f"Page {i+1}",
                            source_file=file_path,
                            line_number=None,  # PDFs don't have line numbers
                            content_type='pdf',
                            metadata=self.extract_metadata(text)
                        ))
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {str(e)}")
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from PDF content."""
        # TODO: Implement PDF metadata extraction
        return {}

class LatexParser(DocParser):
    """Parser for LaTeX documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a LaTeX file."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract metadata first
                metadata = self.extract_metadata(content)
                
                # Split into sections based on \section commands
                current_section = ""
                current_title = "Main"
                line_number = 1
                
                # Handle preamble (everything before \begin{document})
                doc_start = content.find('\\begin{document}')
                if doc_start != -1:
                    preamble = content[:doc_start].strip()
                    if preamble:
                        sections.append(DocSection(
                            content=self._convert_latex(preamble),
                            title="Preamble",
                            source_file=file_path,
                            line_number=1,
                            content_type='tex',
                            metadata=metadata
                        ))
                    content = content[doc_start:]
                
                for line in content.split('\n'):
                    # Check for section commands
                    section_match = re.match(r'\\(section|chapter|subsection)\{(.*?)\}', line)
                    if section_match:
                        # Save previous section if it exists
                        if current_section:
                            sections.append(DocSection(
                                content=self._convert_latex(current_section.strip()),
                                title=current_title,
                                source_file=file_path,
                                line_number=line_number,
                                content_type='tex',
                                metadata=metadata
                            ))
                        
                        # Start new section
                        current_title = section_match.group(2)
                        current_section = ""
                    else:
                        current_section += line + "\n"
                    line_number += 1
                
                # Add the last section
                if current_section:
                    sections.append(DocSection(
                        content=self._convert_latex(current_section.strip()),
                        title=current_title,
                        source_file=file_path,
                        line_number=line_number,
                        content_type='tex',
                        metadata=metadata
                    ))
        except Exception as e:
            logger.error(f"Failed to parse LaTeX {file_path}: {str(e)}")
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from LaTeX content."""
        metadata = {}
        try:
            # Extract common LaTeX metadata commands
            patterns = {
                'title': r'\\title\{([^}]*)\}',
                'author': r'\\author\{([^}]*)\}',
                'date': r'\\date\{([^}]*)\}',
                'documentclass': r'\\documentclass(?:\[.*?\])?\{([^}]*)\}'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Remove any remaining LaTeX commands
                    value = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})?', '', value)
                    metadata[key] = value.strip()
        except Exception as e:
            logger.error(f"Failed to extract LaTeX metadata: {str(e)}")
        return metadata
    
    def _convert_latex(self, content: str) -> str:
        """Convert LaTeX content to plain text with MathML for equations."""
        try:
            # Replace math environments with MathML
            def replace_math(match):
                try:
                    math = match.group(1)
                    return latex2mathml.converter.convert(math)
                except:
                    return match.group(0)
            
            # Handle both inline and display math
            content = re.sub(r'\$\$(.*?)\$\$', replace_math, content)
            content = re.sub(r'\$(.*?)\$', replace_math, content)
            
            # Remove common LaTeX commands
            content = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', content)
            content = re.sub(r'\\[a-zA-Z]+', '', content)
            
            return content.strip()
        except Exception as e:
            logger.error(f"Failed to convert LaTeX content: {str(e)}")
            return content

class YamlParser(DocParser):
    """Parser for YAML documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a YAML file into sections."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = yaml.safe_load(content)
                
                # Convert YAML data to sections
                if isinstance(data, dict):
                    for key, value in data.items():
                        sections.append(DocSection(
                            content=str(value),
                            title=str(key),
                            source_file=file_path,
                            content_type='yaml',
                            metadata={'type': 'key_value', 'key': key}
                        ))
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        sections.append(DocSection(
                            content=str(item),
                            title=f"Item {i+1}",
                            source_file=file_path,
                            content_type='yaml',
                            metadata={'type': 'list_item', 'index': i}
                        ))
                else:
                    sections.append(DocSection(
                        content=str(data),
                        source_file=file_path,
                        content_type='yaml',
                        metadata={'type': 'document'}
                    ))
        except Exception as e:
            logger.error(f"Failed to parse YAML file {file_path}: {str(e)}")
            # Return the entire content as a single section
            sections = [DocSection(
                content=content,
                source_file=file_path,
                content_type='yaml',
                metadata={'type': 'document'}
            )]
        
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from YAML content."""
        # YAML content is already structured, so no additional metadata extraction needed
        return {}

class JsonParser(DocParser):
    """Parser for JSON documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a JSON file."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.loads(content)
                
                # Convert JSON data to sections
                sections.extend(self._process_json_data(data, file_path))
        except Exception as e:
            logger.error(f"Failed to parse JSON {file_path}: {str(e)}")
        return sections
    
    def _process_json_data(self, data: Union[Dict, List], file_path: Path, parent_key: str = "") -> List[DocSection]:
        """Process JSON data recursively."""
        sections = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                section_title = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, (dict, list)):
                    sections.extend(self._process_json_data(value, file_path, section_title))
                else:
                    sections.append(DocSection(
                        content=str(value),
                        title=section_title,
                        source_file=file_path,
                        content_type='json',
                        metadata=self.extract_metadata(str(value))
                    ))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                section_title = f"{parent_key}[{i}]" if parent_key else f"item_{i}"
                if isinstance(item, (dict, list)):
                    sections.extend(self._process_json_data(item, file_path, section_title))
                else:
                    sections.append(DocSection(
                        content=str(item),
                        title=section_title,
                        source_file=file_path,
                        content_type='json',
                        metadata=self.extract_metadata(str(item))
                    ))
        
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from JSON content."""
        # JSON content is already structured, so no additional metadata extraction needed
        return {}

class TomlParser(DocParser):
    """Parser for TOML documents."""
    
    def parse_file(self, file_path: Path) -> List[DocSection]:
        """Parse a TOML file."""
        sections = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = toml.loads(content)
                
                # Convert TOML data to sections
                sections.extend(self._process_toml_data(data, file_path))
        except Exception as e:
            logger.error(f"Failed to parse TOML {file_path}: {str(e)}")
        return sections
    
    def _process_toml_data(self, data: Dict, file_path: Path, parent_key: str = "") -> List[DocSection]:
        """Process TOML data recursively."""
        sections = []
        
        for key, value in data.items():
            section_title = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                sections.extend(self._process_toml_data(value, file_path, section_title))
            else:
                sections.append(DocSection(
                    content=str(value),
                    title=section_title,
                    source_file=file_path,
                    content_type='toml',
                    metadata=self.extract_metadata(str(value))
                ))
        
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from TOML content."""
        # TOML content is already structured, so no additional metadata extraction needed
        return {}

class ParserFactory:
    """Factory for creating document format-specific parsers."""
    
    _parsers = {
        '.md': MarkdownParser,
        '.markdown': MarkdownParser,
        '.txt': TextParser,
        '.rst': ReStructuredTextParser,
        '.pdf': PdfParser,
        '.tex': LatexParser,
        '.yaml': YamlParser,
        '.yml': YamlParser,
        '.json': JsonParser,
        '.toml': TomlParser,
    }
    
    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[DocParser]:
        """Get the appropriate parser for a file based on its extension."""
        parser_class = cls._parsers.get(file_path.suffix.lower())
        if parser_class:
            return parser_class()
        return None 