"""Module for automated documentation generation and maintenance."""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.generic.storage import Storage
from diagrams.generic.database import SQL
from mkdocs.config import load_config
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import get_files
from mkdocs.structure.pages import Page
from mkdocs.utils import write_file

from knowledge_agent.core.code_analysis import CodeAnalyzer
from knowledge_agent.core.code_parser import CodeEntity, parse_codebase
from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.logging import logger
from knowledge_agent.core.version_control import VersionManager

@dataclass
class DocSection:
    """Represents a section of documentation."""
    title: str
    content: str
    order: int
    metadata: Dict[str, str]

@dataclass
class DocTemplate:
    """Template for generating documentation."""
    name: str
    sections: List[str]
    prompts: Dict[str, str]
    metadata: Dict[str, str]

class DocumentationGenerator:
    """Generates and maintains documentation for a codebase."""
    
    DEFAULT_TEMPLATES = {
        "module": DocTemplate(
            name="module",
            sections=[
                "overview",
                "architecture",
                "installation",
                "usage",
                "api",
                "examples"
            ],
            prompts={
                "overview": (
                    "Generate a clear overview of this module. Include:\n"
                    "1. Main purpose and functionality\n"
                    "2. Key features and capabilities\n"
                    "3. High-level architecture\n"
                    "4. Target users/use cases"
                ),
                "architecture": (
                    "Describe the module's architecture. Include:\n"
                    "1. Component diagram\n"
                    "2. Data flow\n"
                    "3. Key design decisions\n"
                    "4. Integration points"
                ),
                "installation": (
                    "Provide installation instructions. Include:\n"
                    "1. Prerequisites\n"
                    "2. Step-by-step installation\n"
                    "3. Configuration options\n"
                    "4. Troubleshooting"
                ),
                "usage": (
                    "Explain how to use the module. Include:\n"
                    "1. Basic usage examples\n"
                    "2. Common patterns\n"
                    "3. Configuration options\n"
                    "4. Best practices"
                ),
                "api": (
                    "Document the module's API. Include:\n"
                    "1. Public classes and methods\n"
                    "2. Parameters and return types\n"
                    "3. Examples for each method\n"
                    "4. Error handling"
                ),
                "examples": (
                    "Provide detailed examples. Include:\n"
                    "1. Common use cases\n"
                    "2. Code snippets\n"
                    "3. Expected output\n"
                    "4. Edge cases"
                )
            },
            metadata={"type": "module"}
        ),
        "api": DocTemplate(
            name="api",
            sections=[
                "endpoints",
                "models",
                "authentication",
                "errors"
            ],
            prompts={
                "endpoints": (
                    "Document API endpoints. Include:\n"
                    "1. URL and method\n"
                    "2. Request/response format\n"
                    "3. Authentication requirements\n"
                    "4. Example requests"
                ),
                "models": (
                    "Document data models. Include:\n"
                    "1. Schema definitions\n"
                    "2. Field descriptions\n"
                    "3. Validation rules\n"
                    "4. Example objects"
                ),
                "authentication": (
                    "Explain authentication. Include:\n"
                    "1. Authentication methods\n"
                    "2. Token formats\n"
                    "3. Security considerations\n"
                    "4. Example flows"
                ),
                "errors": (
                    "Document error handling. Include:\n"
                    "1. Error codes\n"
                    "2. Error messages\n"
                    "3. Recovery steps\n"
                    "4. Example scenarios"
                )
            },
            metadata={"type": "api"}
        )
    }
    
    def __init__(
        self,
        llm_handler: LLMHandler,
        output_dir: str = "./docs",
        template_dir: Optional[str] = None
    ):
        """Initialize the documentation generator.
        
        Args:
            llm_handler: LLM handler for generating content
            output_dir: Directory for generated documentation
            template_dir: Optional directory for custom templates
        """
        self.llm_handler = llm_handler
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir) if template_dir else None
        self.analyzer = CodeAnalyzer(llm_handler)
        
        # Load custom templates if available
        self.templates = self.DEFAULT_TEMPLATES.copy()
        if self.template_dir and self.template_dir.exists():
            self._load_custom_templates()
    
    def generate_documentation(
        self,
        root_path: Path,
        template_name: str = "module",
        metadata: Optional[Dict[str, str]] = None
    ) -> List[DocSection]:
        """Generate documentation for a codebase.
        
        Args:
            root_path: Root directory of the codebase
            template_name: Name of the template to use
            metadata: Optional additional metadata
            
        Returns:
            List of documentation sections
        """
        # Get template
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Analyze codebase
        self.analyzer.analyze_codebase(root_path)
        
        # Generate sections
        sections = []
        for section_name in template.sections:
            prompt = template.prompts[section_name]
            content = self._generate_section(
                section_name,
                prompt,
                root_path,
                metadata or {}
            )
            
            sections.append(DocSection(
                title=section_name.title(),
                content=content,
                order=template.sections.index(section_name),
                metadata={"type": section_name, **(metadata or {})}
            ))
        
        return sections
    
    def generate_architecture_diagram(
        self,
        root_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """Generate an architecture diagram for the codebase.
        
        Args:
            root_path: Root directory of the codebase
            output_path: Optional custom output path
            
        Returns:
            Path to the generated diagram
        """
        if not output_path:
            output_path = self.output_dir / "architecture.png"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Analyze dependencies
        self.analyzer.analyze_codebase(root_path)
        
        # Create diagram
        with Diagram(
            "Architecture Overview",
            filename=str(output_path),
            show=False
        ):
            with Cluster("Core"):
                modules = {
                    node.name: Python(node.name)
                    for node in self.analyzer.dependency_graph.nodes.values()
                    if node.type == "module"
                }
            
            # Add connections
            for name, node in modules.items():
                for dep in self.analyzer.dependency_graph.get_dependencies(name):
                    if dep in modules:
                        node >> modules[dep]
        
        return output_path
    
    def export_mkdocs(
        self,
        sections: List[DocSection],
        config: Optional[Dict] = None
    ):
        """Export documentation to MkDocs format.
        
        Args:
            sections: List of documentation sections
            config: Optional MkDocs configuration
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write sections
        nav = []
        for section in sorted(sections, key=lambda s: s.order):
            file_name = f"{section.title.lower()}.md"
            file_path = self.output_dir / file_name
            
            # Write content
            with open(file_path, "w") as f:
                f.write(f"# {section.title}\n\n")
                f.write(section.content)
            
            nav.append({section.title: file_name})
        
        # Create mkdocs.yml
        mkdocs_config = {
            "site_name": "Documentation",
            "theme": {
                "name": "material",
                "features": [
                    "navigation.tabs",
                    "navigation.sections",
                    "toc.integrate",
                    "search.suggest",
                    "search.highlight"
                ]
            },
            "plugins": [
                "search",
                {
                    "mkdocstrings": {
                        "handlers": {
                            "python": {
                                "setup_commands": [
                                    "import os",
                                    "import sys",
                                    "sys.path.append(os.getcwd())"
                                ]
                            }
                        }
                    }
                }
            ],
            "nav": nav,
            **(config or {})
        }
        
        with open(self.output_dir / "mkdocs.yml", "w") as f:
            import yaml
            yaml.dump(mkdocs_config, f, default_flow_style=False)
    
    def _generate_section(
        self,
        section_name: str,
        prompt: str,
        root_path: Path,
        metadata: Dict[str, str]
    ) -> str:
        """Generate content for a documentation section.
        
        Args:
            section_name: Name of the section
            prompt: Prompt for the LLM
            root_path: Root directory of the codebase
            metadata: Additional metadata
            
        Returns:
            Generated content
        """
        # Prepare context
        context = []
        
        # Add codebase structure
        context.append("Codebase structure:")
        for node in self.analyzer.dependency_graph.nodes.values():
            context.append(
                f"- {node.name} ({node.type})\n"
                f"  File: {node.file_path}\n"
                f"  Dependencies: {', '.join(node.outgoing)}"
            )
        
        # Add section-specific context
        if section_name == "api":
            # Add API-specific information
            context.append("\nAPI Information:")
            for node in self.analyzer.dependency_graph.nodes.values():
                if "api" in node.name.lower() or "endpoint" in node.name.lower():
                    context.append(f"- {node.name}")
        elif section_name == "architecture":
            # Add architectural information
            context.append("\nArchitectural Components:")
            for node in self.analyzer.dependency_graph.nodes.values():
                if node.type in ["module", "class"]:
                    deps = self.analyzer.find_dependencies(node.name)
                    context.append(
                        f"- {node.name}:\n"
                        f"  Depends on: {', '.join(deps['dependencies'])}\n"
                        f"  Used by: {', '.join(deps['dependents'])}"
                    )
        
        # Generate content
        try:
            content = self.llm_handler.generate_response(
                prompt,
                ["\n".join(context)],
                role_name="engineer"
            )
            return content
        except Exception as e:
            logger.error(f"Failed to generate {section_name} section: {str(e)}")
            return f"Error generating {section_name} section"
    
    def _load_custom_templates(self):
        """Load custom documentation templates."""
        try:
            for file in self.template_dir.glob("*.json"):
                with open(file) as f:
                    import json
                    data = json.load(f)
                    template = DocTemplate(**data)
                    self.templates[template.name] = template
        except Exception as e:
            logger.error(f"Failed to load custom templates: {str(e)}") 