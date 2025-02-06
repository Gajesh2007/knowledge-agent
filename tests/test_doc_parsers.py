"""Tests for document format parsers."""

import pytest
from pathlib import Path
from textwrap import dedent

from knowledge_agent.core.doc_parser import (
    DocSection,
    MarkdownParser,
    TextParser,
    ReStructuredTextParser,
    LatexParser,
    ParserFactory
)

@pytest.fixture
def sample_markdown(tmp_path):
    """Create a sample Markdown file."""
    content = dedent("""
        ---
        title: Test Document
        author: Test Author
        ---
        
        # Section 1
        
        This is the first section.
        
        ## Subsection
        
        This is a subsection.
        
        # Section 2
        
        This is the second section.
    """).strip()
    
    file_path = tmp_path / "test.md"
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

@pytest.fixture
def sample_rst(tmp_path):
    """Create a sample reStructuredText file."""
    content = dedent("""
        :title: Test Document
        :author: Test Author
        
        Section 1
        =========
        
        This is the first section.
        
        Subsection
        ----------
        
        This is a subsection.
        
        Section 2
        =========
        
        This is the second section.
    """).strip()
    
    file_path = tmp_path / "test.rst"
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

@pytest.fixture
def sample_text(tmp_path):
    """Create a sample text file."""
    content = dedent("""
        Title: Important Document
        
        This is the first section.
        It has multiple lines.
        
        This is the second section.
        Another multi-line section.
        
        This is the third section.
    """).strip()
    
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

@pytest.fixture
def sample_latex(tmp_path):
    """Create a sample LaTeX file."""
    content = dedent(r"""
        \documentclass{article}
        \title{Test Document}
        \author{Test Author}
        \date{\today}
        
        \begin{document}
        
        \section{Introduction}
        This is the introduction with an equation: $E = mc^2$
        
        \section{Methods}
        Here's a display equation:
        $$
        F = ma
        $$
        
        \subsection{Details}
        More details here.
        
        \end{document}
    """).strip()
    
    file_path = tmp_path / "test.tex"
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

def test_markdown_parser(sample_markdown):
    """Test Markdown parser functionality."""
    parser = MarkdownParser()
    sections = parser.parse_file(sample_markdown)
    
    # Should find: Main (with frontmatter), Section 1, Subsection, Section 2
    assert len(sections) == 4
    assert sections[0].title == "Main"  # Contains frontmatter
    assert sections[1].title == "Section 1"
    assert sections[2].title == "Subsection"
    assert sections[3].title == "Section 2"
    
    # Test metadata extraction
    metadata = parser.extract_metadata(sections[0].content)
    assert metadata.get("title") == "Test Document"
    assert metadata.get("author") == "Test Author"

def test_rst_parser(sample_rst):
    """Test reStructuredText parser functionality."""
    parser = ReStructuredTextParser()
    sections = parser.parse_file(sample_rst)
    
    # Should find: Main (with metadata), Section 1, Subsection, Section 2
    assert len(sections) == 4
    assert sections[0].title == "Main"  # Contains metadata
    assert sections[1].title == "Section 1"
    assert sections[2].title == "Subsection"
    assert sections[3].title == "Section 2"
    
    # Test metadata extraction
    metadata = parser.extract_metadata(sections[0].content)
    assert metadata.get("title") == "Test Document"
    assert metadata.get("author") == "Test Author"

def test_text_parser(sample_text):
    """Test plain text parser functionality."""
    parser = TextParser()
    sections = parser.parse_file(sample_text)
    
    # Should find sections separated by blank lines
    assert len(sections) == 4
    assert "Title: Important Document" in sections[0].content
    assert "first section" in sections[1].content.lower()
    assert "second section" in sections[2].content.lower()
    assert "third section" in sections[3].content.lower()
    
    # Test basic metadata extraction
    metadata = parser.extract_metadata(sections[0].content)
    assert metadata.get("title") == "Important Document"

def test_latex_parser(sample_latex):
    """Test LaTeX parser functionality."""
    parser = LatexParser()
    sections = parser.parse_file(sample_latex)
    
    # Should find: Main (with preamble), Introduction, Methods, Details
    assert len(sections) >= 3
    assert any(s.title == "Introduction" for s in sections)
    assert any(s.title == "Methods" for s in sections)
    assert any(s.title == "Details" for s in sections)
    
    # Test metadata extraction
    metadata = parser.extract_metadata(sections[0].content)
    assert metadata.get("title") == "Test Document"
    assert metadata.get("author") == "Test Author"
    assert metadata.get("documentclass") == "article"
    
    # Test math conversion
    intro_section = next(s for s in sections if s.title == "Introduction")
    assert "E = mc^2" in intro_section.content
    assert "<math" in intro_section.content  # Should contain MathML

def test_parser_factory():
    """Test parser factory functionality."""
    # Test supported extensions
    assert isinstance(ParserFactory.get_parser(Path("test.md")), MarkdownParser)
    assert isinstance(ParserFactory.get_parser(Path("test.rst")), ReStructuredTextParser)
    assert isinstance(ParserFactory.get_parser(Path("test.txt")), TextParser)
    assert isinstance(ParserFactory.get_parser(Path("test.tex")), LatexParser)
    
    # Test unsupported extension
    assert ParserFactory.get_parser(Path("test.unknown")) is None

def test_doc_section_creation():
    """Test DocSection creation and attributes."""
    section = DocSection(
        title="Test Section",
        content="Test content",
        source_file=Path("test.md"),
        line_number=1,
        metadata={"key": "value"}
    )
    
    assert section.title == "Test Section"
    assert section.content == "Test content"
    assert section.line_number == 1
    assert section.metadata["key"] == "value" 