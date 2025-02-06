"""Module for AST-based code parsing and analysis."""

import ast
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import tree_sitter
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

# Initialize tree-sitter languages
LANGUAGE_DIR = Path(__file__).parent / "build" / "languages"
LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)

# Clone language repositories if they don't exist
LANGUAGE_REPOS = {
    'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript',
    'typescript': 'https://github.com/tree-sitter/tree-sitter-typescript',
    'cpp': 'https://github.com/tree-sitter/tree-sitter-cpp',
    'html': 'https://github.com/tree-sitter/tree-sitter-html',
    'css': 'https://github.com/tree-sitter/tree-sitter-css',
}

# Initialize languages
LANGUAGES = {}

def init_languages():
    """Initialize tree-sitter languages."""
    global LANGUAGES
    
    # Clone repositories if needed
    for lang_name, repo_url in LANGUAGE_REPOS.items():
        lang_dir = LANGUAGE_DIR / f"tree-sitter-{lang_name}"
        if not lang_dir.exists():
            try:
                subprocess.run(['git', 'clone', repo_url, str(lang_dir)], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone {lang_name} grammar: {str(e)}")
    
    # Build language library
    try:
        Language.build_library(
            str(LANGUAGE_DIR / "languages.so"),
            [str(LANGUAGE_DIR / d) for d in LANGUAGE_DIR.iterdir() if d.is_dir()]
        )
        
        # Load all languages
        for lang_name in LANGUAGE_REPOS:
            try:
                LANGUAGES[lang_name] = Language(str(LANGUAGE_DIR / "languages.so"), lang_name)
            except Exception as e:
                logger.error(f"Failed to load {lang_name} language: {str(e)}")
                LANGUAGES[lang_name] = None
    except Exception as e:
        logger.error(f"Failed to build tree-sitter languages: {str(e)}")

# Initialize languages on module load
init_languages()

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

class TreeSitterParser(LanguageParser):
    """Base class for tree-sitter based parsers."""
    
    def __init__(self, language_name: str):
        """Initialize the parser with a specific language."""
        self.parser = Parser()
        self.language_name = language_name
        language = LANGUAGES.get(language_name)
        if language is None:
            raise ValueError(f"Language {language_name} not available")
        self.parser.set_language(language)
    
    def _get_node_text(self, node, content: str) -> str:
        """Get the text content of a node."""
        return content[node.start_byte:node.end_byte].decode('utf-8')
    
    def _get_docstring(self, node, content: str) -> Optional[str]:
        """Extract docstring/comments for a node."""
        comments = []
        for child in node.children:
            if child.type == 'comment':
                comments.append(self._get_node_text(child, content))
        return '\n'.join(comments) if comments else None

class JavaScriptParser(TreeSitterParser):
    """Parser for JavaScript and JSX code."""
    
    def __init__(self):
        """Initialize the JavaScript parser."""
        super().__init__("javascript")
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a JavaScript file."""
        entities = []
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                tree = self.parser.parse(content)
                
                # Process functions and classes
                for node in self._traverse(tree.root_node):
                    if node.type in ('function_declaration', 'method_definition', 'class_declaration'):
                        name = None
                        for child in node.children:
                            if child.type == 'identifier':
                                name = self._get_node_text(child, content)
                                break
                        
                        if name:
                            entities.append(CodeEntity(
                                name=name,
                                type=node.type.replace('_declaration', '').replace('_definition', ''),
                                docstring=self._get_docstring(node, content),
                                code=self._get_node_text(node, content),
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                parent=None,  # TODO: Handle nested classes/functions
                                dependencies=self.extract_dependencies(content),
                                metadata={'language': self.language_name, 'path': str(file_path)}
                            ))
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {str(e)}")
        return entities
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract import dependencies from JavaScript code."""
        dependencies = set()
        try:
            tree = self.parser.parse(content)
            for node in self._traverse(tree.root_node):
                if node.type == 'import_statement':
                    for child in node.children:
                        if child.type == 'string':
                            dependencies.add(self._get_node_text(child, content).strip('"\''))
        except Exception as e:
            logger.error(f"Failed to extract dependencies: {str(e)}")
        return dependencies
    
    def _traverse(self, node):
        """Traverse the AST."""
        yield node
        for child in node.children:
            yield from self._traverse(child)

class TypeScriptParser(TreeSitterParser):
    """Parser for TypeScript and TSX code."""
    
    def __init__(self):
        """Initialize the TypeScript parser."""
        super().__init__("typescript")
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a TypeScript file."""
        # Similar to JavaScriptParser but with TypeScript-specific handling
        return []  # TODO: Implement TypeScript parsing
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract import dependencies from TypeScript code."""
        # Similar to JavaScriptParser
        return set()  # TODO: Implement TypeScript dependency extraction

class CppParser(TreeSitterParser):
    """Parser for C++ code."""
    
    def __init__(self):
        """Initialize the C++ parser."""
        super().__init__("cpp")
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a C++ file."""
        # TODO: Implement C++ parsing
        return []
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract include dependencies from C++ code."""
        # TODO: Implement C++ dependency extraction
        return set()

class HtmlParser(TreeSitterParser):
    """Parser for HTML code."""
    
    def __init__(self):
        """Initialize the HTML parser."""
        super().__init__("html")
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse an HTML file."""
        # TODO: Implement HTML parsing
        return []
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract dependencies (scripts, stylesheets) from HTML code."""
        # TODO: Implement HTML dependency extraction
        return set()

class CssParser(TreeSitterParser):
    """Parser for CSS code."""
    
    def __init__(self):
        """Initialize the CSS parser."""
        super().__init__("css")
    
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        """Parse a CSS file."""
        # TODO: Implement CSS parsing
        return []
    
    def extract_dependencies(self, content: str) -> Set[str]:
        """Extract dependencies (@import) from CSS code."""
        # TODO: Implement CSS dependency extraction
        return set()

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
        '.js': JavaScriptParser,
        '.jsx': JavaScriptParser,  # JSX uses the JavaScript parser
        '.ts': TypeScriptParser,
        '.tsx': TypeScriptParser,  # TSX uses the TypeScript parser
        '.cpp': CppParser,
        '.c': CppParser,  # C files can use the C++ parser
        '.html': HtmlParser,
        '.css': CssParser,
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