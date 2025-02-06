"""Module for AST-based code parsing and analysis."""

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

@dataclass
class CodeEntity:
    """Represents a parsed code entity (function, class, etc.)."""
    name: str
    type: str  # 'function', 'class', 'method', etc.
    docstring: Optional[str]
    code: str
    start_line: int
    end_line: int
    parent: Optional[str]  # Parent class/module name
    dependencies: Set[str]  # Import dependencies
    metadata: Dict[str, str]  # Additional metadata (language, file path, etc.)

class LanguageParser(ABC):
    """Abstract base class for language-specific parsers."""
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a file and return a list of code entities."""
        pass
    
    @abstractmethod
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract import dependencies from code content."""
        pass

class PythonParser(LanguageParser):
    """Parser for Python code using the ast module."""
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a Python file using AST."""
        entities = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            # Track the current class context
            current_class = None
            
            # Extract file-level docstring
            if (isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Str)):
                module_doc = ast.get_docstring(tree)
                if module_doc:
                    entities.append(CodeEntity(
                        name=file_path.stem,
                        type='module',
                        docstring=module_doc,
                        code='',  # No code for module-level docstring
                        start_line=1,
                        end_line=tree.body[0].end_lineno,
                        parent=None,
                        dependencies=self.extract_dependencies(content),
                        metadata={'language': 'python', 'path': str(file_path)}
                    ))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    current_class = node.name
                    entities.append(self._parse_class(node, file_path))
                    
                elif isinstance(node, ast.FunctionDef):
                    entities.append(self._parse_function(node, current_class, file_path))
                    
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {str(e)}")
            
        return entities
    
    def _parse_class(self, node: ast.ClassDef, file_path: Path) -> CodeEntity:
        """Parse a class definition node."""
        return CodeEntity(
            name=node.name,
            type='class',
            docstring=ast.get_docstring(node),
            code=self._get_node_source(node),
            start_line=node.lineno,
            end_line=node.end_lineno,
            parent=None,
            dependencies=set(),  # Class-level dependencies handled at module level
            metadata={'language': 'python', 'path': str(file_path)}
        )
    
    def _parse_function(self, node: ast.FunctionDef, class_name: Optional[str], file_path: Path) -> CodeEntity:
        """Parse a function definition node."""
        return CodeEntity(
            name=node.name,
            type='method' if class_name else 'function',
            docstring=ast.get_docstring(node),
            code=self._get_node_source(node),
            start_line=node.lineno,
            end_line=node.end_lineno,
            parent=class_name,
            dependencies=set(),  # Function-level dependencies handled at module level
            metadata={'language': 'python', 'path': str(file_path)}
        )
    
    def _get_node_source(self, node: Union[ast.ClassDef, ast.FunctionDef]) -> str:
        """Get the source code for a node."""
        return ast.unparse(node)
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract import dependencies from Python code."""
        dependencies = set()
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        dependencies.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.add(node.module)
        except Exception as e:
            logger.error(f"Failed to extract dependencies: {str(e)}")
        return dependencies

class ParserFactory:
    """Factory for creating language-specific parsers."""
    
    _parsers = {
        '.py': PythonParser,
        # Add more parsers as they're implemented:
        # '.go': GoParser,
        # '.rs': RustParser,
        # '.sol': SolidityParser,
    }
    
    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[LanguageParser]:
        """Get the appropriate parser for a file based on its extension."""
        parser_class = cls._parsers.get(file_path.suffix.lower())
        if parser_class:
            return parser_class()
        return None

def parse_codebase(root_path: Path) -> List[CodeEntity]:
    """Parse an entire codebase and return all code entities."""
    entities = []
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            parser = ParserFactory.get_parser(file_path)
            if parser:
                try:
                    file_entities = parser.parse_file(file_path)
                    entities.extend(file_entities)
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {str(e)}")
                    continue
    
    return entities 