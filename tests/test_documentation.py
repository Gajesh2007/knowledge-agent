"""Tests for documentation generation functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from knowledge_agent.core.documentation import (
    DocumentationGenerator,
    DocSection,
    DocTemplate
)
from knowledge_agent.core.llm import LLMHandler

@pytest.fixture
def mock_llm_handler():
    """Create a mock LLM handler."""
    handler = Mock(spec=LLMHandler)
    handler.generate_response.return_value = "Generated content"
    return handler

@pytest.fixture
def doc_generator(mock_llm_handler, tmp_path):
    """Create a documentation generator with mocked dependencies."""
    return DocumentationGenerator(
        llm_handler=mock_llm_handler,
        output_dir=str(tmp_path)
    )

def test_generate_documentation(doc_generator, tmp_path):
    """Test basic documentation generation."""
    # Setup
    root_path = tmp_path / "src"
    root_path.mkdir()
    
    # Execute
    sections = doc_generator.generate_documentation(root_path)
    
    # Assert
    assert len(sections) > 0
    assert all(isinstance(s, DocSection) for s in sections)
    assert all(s.content == "Generated content" for s in sections)

def test_generate_architecture_diagram(doc_generator, tmp_path):
    """Test architecture diagram generation."""
    # Setup
    root_path = tmp_path / "src"
    root_path.mkdir()
    output_path = tmp_path / "diagram.png"
    
    # Execute
    result_path = doc_generator.generate_architecture_diagram(
        root_path,
        output_path=output_path
    )
    
    # Assert
    assert result_path == output_path
    assert output_path.exists()

def test_export_mkdocs(doc_generator, tmp_path):
    """Test MkDocs export functionality."""
    # Setup
    sections = [
        DocSection(
            title="Overview",
            content="# Overview\n\nTest content",
            order=0,
            metadata={"type": "overview"}
        ),
        DocSection(
            title="API",
            content="# API\n\nAPI documentation",
            order=1,
            metadata={"type": "api"}
        )
    ]
    
    # Execute
    doc_generator.export_mkdocs(sections)
    
    # Assert
    assert (Path(doc_generator.output_dir) / "mkdocs.yml").exists()
    assert (Path(doc_generator.output_dir) / "overview.md").exists()
    assert (Path(doc_generator.output_dir) / "api.md").exists()

def test_custom_template(doc_generator):
    """Test using a custom documentation template."""
    # Setup
    custom_template = DocTemplate(
        name="custom",
        sections=["summary", "details"],
        prompts={
            "summary": "Generate a summary",
            "details": "Generate details"
        },
        metadata={"type": "custom"}
    )
    doc_generator.templates["custom"] = custom_template
    
    # Execute
    sections = doc_generator.generate_documentation(
        Path("."),
        template_name="custom"
    )
    
    # Assert
    assert len(sections) == 2
    assert {s.title for s in sections} == {"Summary", "Details"}

@patch('knowledge_agent.core.documentation.Diagram')
def test_architecture_diagram_with_dependencies(
    mock_diagram,
    doc_generator,
    tmp_path
):
    """Test architecture diagram generation with dependencies."""
    # Setup
    root_path = tmp_path / "src"
    root_path.mkdir()
    
    # Mock analyzer's dependency graph
    doc_generator.analyzer.dependency_graph.nodes = {
        "module1": Mock(
            name="module1",
            type="module",
            outgoing={"module2"}
        ),
        "module2": Mock(
            name="module2",
            type="module",
            outgoing=set()
        )
    }
    
    # Execute
    doc_generator.generate_architecture_diagram(root_path)
    
    # Assert
    mock_diagram.assert_called_once()
    assert mock_diagram.call_args[1]["show"] is False 