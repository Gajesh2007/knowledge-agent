"""
Module for AST-based code parsing and analysis.
Languages:
- JavaScript, TypeScript(+TSX), C++, HTML, CSS, Solidity via Tree-Sitter
- Python via built-in AST
- Go via Tree-Sitter
"""

import ast
import json
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import tree_sitter
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser
import ctypes

logger = logging.getLogger(__name__)

#######################################################################
# 1) Build/Load Tree-Sitter Grammars
#######################################################################

# Only need custom directory for Solidity
LANGUAGE_DIR = Path.home() / ".knowledge-agent" / "tree-sitter-languages"
LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)

# Only need to define Solidity repo as others come from tree-sitter-languages
SOLIDITY_REPO = "https://github.com/JoranHonig/tree-sitter-solidity"

LANGUAGES: Dict[str, Optional[Language]] = {}

def _build_solidity(sol_dir: Path, so_path: Path):
    """Build the Solidity parser with the correct settings."""
    try:
        # Clone if needed
        if not (sol_dir / "src").exists():
            logger.info(f"Cloning Solidity from {SOLIDITY_REPO}")
            subprocess.run(["git", "clone", SOLIDITY_REPO, str(sol_dir)], check=True)
            subprocess.run(["git", "checkout", "v1.2.11"], cwd=str(sol_dir), check=True)
        
        # First generate and build the parser
        subprocess.run(["tree-sitter", "generate"], cwd=str(sol_dir), check=True)
        subprocess.run(["tree-sitter", "build"], cwd=str(sol_dir), check=True)
        
        # Copy the built library - handle different OS file extensions
        if sys.platform == "darwin":
            lib_name = "solidity.dylib"
        elif sys.platform == "linux":
            lib_name = "solidity.so"
        elif sys.platform == "win32":
            lib_name = "solidity.dll"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
            
        lib_path = sol_dir / lib_name
        if not lib_path.exists():
            raise FileNotFoundError(f"Parser library not found: {lib_path}")
        
        shutil.copy2(lib_path, so_path)
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to build Solidity parser: {e}")
        return False

def init_languages():
    """Initialize all language parsers, using tree-sitter-languages where possible."""
    global LANGUAGES
    
    # If we've already loaded languages, skip
    if LANGUAGES:
        logger.info("Tree-sitter languages already initialized.")
        return

    # Load standard languages from tree-sitter-languages first
    standard_langs = ['javascript', 'typescript', 'tsx', 'cpp', 'c', 'html', 'css', 'go', 'rust']
    for lang in standard_langs:
        try:
            LANGUAGES[lang] = get_language(lang)
            logger.info(f"Loaded '{lang}' from tree-sitter-languages")
        except Exception as e:
            logger.error(f"Failed to load {lang} from tree-sitter-languages: {e}")
            LANGUAGES[lang] = None

    # Build Solidity since it's not in tree-sitter-languages
    try:
        # Build Solidity from source
        logger.info("Building Solidity from source...")
        sol_dir = LANGUAGE_DIR / "tree-sitter-solidity"
        so_path = LANGUAGE_DIR / "languages_solidity.so"
        
        if not _build_solidity(sol_dir, so_path):
            logger.error("Failed to build Solidity parser")
            LANGUAGES["solidity"] = None
            return
            
        # Load the built library
        LANGUAGES["solidity"] = Language(str(so_path), "solidity")
        logger.info("Loaded 'solidity' language from custom build")
    except Exception as e:
        logger.error(f"Failed to load Solidity: {e}")
        LANGUAGES["solidity"] = None

def _npm_install_if_exists(path_dir: Path):
    """If package.json is present, do 'npm install' if node_modules doesn't exist"""
    pkg = path_dir / "package.json"
    if pkg.exists():
        nm = path_dir / "node_modules"
        if not nm.exists():
            logger.info(f"npm install in {path_dir} ...")
            subprocess.run(["npm", "install"], check=True)

#######################################################################
# 2) CodeEntity model
#######################################################################

@dataclass
class CodeEntity:
    name: str
    type: str
    docstring: Optional[str]
    code: str
    start_line: int
    end_line: int
    parent: Optional[str]
    dependencies: Set[str]
    metadata: Dict[str, str]
    # Enhanced context fields
    surrounding_context: Optional[str] = None  # Code before and after this entity
    related_entities: Set[str] = field(default_factory=set)  # Names of related code entities
    semantic_type: Optional[str] = None  # More specific semantic type info
    full_qualified_name: Optional[str] = None  # Package/module qualified name
    symbols: Set[str] = field(default_factory=set)  # Symbols used within this entity
    # New fields for better search relevance
    file_context: Optional[str] = None  # Important surrounding file context (imports, configs)
    module_doc: Optional[str] = None  # Module/package level documentation
    related_files: Set[str] = field(default_factory=set)  # Related files (tests, configs)
    semantic_summary: Optional[str] = None  # AI-generated summary of what this entity does
    usage_examples: List[str] = field(default_factory=list)  # Code examples showing usage
    api_endpoints: Set[str] = field(default_factory=set)  # For web services/APIs
    config_settings: Dict[str, str] = field(default_factory=dict)  # Related configuration
    relationships: Dict[str, Set[str]] = field(default_factory=lambda: {
        'calls': set(),  # Functions/methods this entity calls
        'called_by': set(),  # Functions/methods that call this entity
        'implements': set(),  # Interfaces/traits implemented
        'extended_by': set(),  # Classes that extend this one
        'extends': set(),  # Classes this one extends
        'uses': set(),  # Types/modules used by this entity
        'used_by': set(),  # Types/modules that use this entity
        'tested_by': set(),  # Test files that test this entity
        'configures': set(),  # Config files that configure this entity
    })

    def to_search_text(self) -> str:
        """Convert the entity to a rich text representation for semantic search."""
        sections = []
        
        # Basic information
        sections.append(f"Name: {self.name}")
        sections.append(f"Type: {self.type}")
        if self.semantic_type:
            sections.append(f"Semantic Type: {self.semantic_type}")
        if self.full_qualified_name:
            sections.append(f"Full Name: {self.full_qualified_name}")
            
        # Documentation
        if self.docstring:
            sections.append(f"Documentation: {self.docstring}")
        if self.module_doc:
            sections.append(f"Module Documentation: {self.module_doc}")
        if self.semantic_summary:
            sections.append(f"Summary: {self.semantic_summary}")
            
        # Code and context
        sections.append(f"Code: {self.code}")
        if self.surrounding_context:
            sections.append(f"Context: {self.surrounding_context}")
        if self.file_context:
            sections.append(f"File Context: {self.file_context}")
            
        # Relationships
        if self.parent:
            sections.append(f"Parent: {self.parent}")
        if self.related_entities:
            sections.append(f"Related Entities: {', '.join(self.related_entities)}")
        if self.symbols:
            sections.append(f"Uses Symbols: {', '.join(self.symbols)}")
        if self.dependencies:
            sections.append(f"Dependencies: {', '.join(self.dependencies)}")
            
        # Examples and API info
        if self.usage_examples:
            sections.append("Usage Examples:\n" + "\n".join(self.usage_examples))
        if self.api_endpoints:
            sections.append(f"API Endpoints: {', '.join(self.api_endpoints)}")
            
        # Configuration
        if self.config_settings:
            sections.append("Configuration Settings:\n" + 
                          "\n".join(f"{k}: {v}" for k, v in self.config_settings.items()))
            
        # Detailed relationships
        for rel_type, entities in self.relationships.items():
            if entities:
                sections.append(f"{rel_type.replace('_', ' ').title()}: {', '.join(entities)}")
        
        # Metadata
        sections.append("Metadata:\n" + 
                      "\n".join(f"{k}: {v}" for k, v in self.metadata.items()))
        
        return "\n\n".join(sections)

    def get_related_context(self) -> List[str]:
        """Get a list of related context strings that should be searched together."""
        contexts = []
        
        # Add direct code and documentation
        if self.docstring:
            contexts.append(self.docstring)
        if self.code:
            contexts.append(self.code)
            
        # Add important contextual information
        if self.module_doc:
            contexts.append(self.module_doc)
        if self.semantic_summary:
            contexts.append(self.semantic_summary)
        if self.surrounding_context:
            contexts.append(self.surrounding_context)
        if self.file_context:
            contexts.append(self.file_context)
            
        # Add usage examples and configuration
        contexts.extend(self.usage_examples)
        if self.config_settings:
            contexts.append("\n".join(f"{k}: {v}" for k, v in self.config_settings.items()))
            
        return contexts

#######################################################################
# 3) Abstract base classes for code parsers
#######################################################################

class LanguageParser(ABC):
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        pass

    @abstractmethod
    def extract_dependencies(self, content: str) -> Set[str]:
        pass

class TreeSitterParser(LanguageParser):
    def __init__(self, language_name: str):
        if LANGUAGES.get(language_name) is None:
            raise ValueError(f"Grammar '{language_name}' not available.")
        self.parser = Parser()
        self.parser.set_language(LANGUAGES[language_name])
        self.language_name = language_name

    def _get_node_text(self, node, content: bytes) -> str:
        return content[node.start_byte : node.end_byte].decode("utf-8")

    def _get_docstring(self, node, content: bytes) -> Optional[str]:
        comments = []
        # Look for comments before the node
        prev = node.prev_sibling
        while prev and prev.type == "comment":
            comments.insert(0, self._get_node_text(prev, content))
            prev = prev.prev_sibling
        
        # Look for comments inside the node
        for child in node.children:
            if child.type == "comment":
                comments.append(self._get_node_text(child, content))
                
        return "\n".join(comments) if comments else None

    def _get_surrounding_context(self, node, content: bytes, context_lines: int = 5) -> str:
        """Get code context before and after the node."""
        file_lines = content.decode("utf-8").splitlines()
        start = max(0, node.start_point[0] - context_lines)
        end = min(len(file_lines), node.end_point[0] + context_lines + 1)
        return "\n".join(file_lines[start:end])

    def _get_symbols(self, node, content: bytes) -> Set[str]:
        """Extract symbols (variables, types, etc.) used in the code."""
        symbols = set()
        for child in self._traverse(node):
            if child.type in ("identifier", "type_identifier", "field_identifier"):
                symbols.add(self._get_node_text(child, content))
        return symbols

#######################################################################
# 4) Language-specific TreeSitter Parsers
#######################################################################

class JavaScriptParser(TreeSitterParser):
    def __init__(self):
        super().__init__("javascript")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        out = []
        try:
            data = file_path.read_bytes()
            tree = self.parser.parse(data)
            for node in self._traverse(tree.root_node):
                if node.type in ("function_declaration","method_definition","class_declaration"):
                    name = None
                    for c in node.children:
                        if c.type == "identifier":
                            name = self._get_node_text(c, data)
                            break
                    if name:
                        out.append(CodeEntity(
                            name=name,
                            type=node.type.replace("_declaration","").replace("_definition",""),
                            docstring=self._get_docstring(node, data),
                            code=self._get_node_text(node, data),
                            start_line=node.start_point[0]+1,
                            end_line=node.end_point[0]+1,
                            parent=None,
                            dependencies=self.extract_dependencies(data.decode("utf-8")),
                            metadata={"language":"javascript","path":str(file_path)}
                        ))
        except Exception as e:
            logger.error(f"Failed to parse JS {file_path}: {e}", exc_info=True)
        return out

    def _traverse(self, node):
        yield node
        for c in node.children:
            yield from self._traverse(c)

    def extract_dependencies(self, content: str) -> Set[str]:
        s = set()
        try:
            t = self.parser.parse(content.encode("utf-8"))
            for n in self._traverse(t.root_node):
                if n.type == "import_statement":
                    for c in n.children:
                        if c.type == "string":
                            s.add(c.text.decode("utf-8").strip("'\""))
        except Exception as e:
            logger.error(f"JS deps extraction error: {e}", exc_info=True)
        return s

class TypeScriptParser(TreeSitterParser):
    def __init__(self):
        super().__init__("typescript")
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        return []
    def extract_dependencies(self, content:str)->Set[str]:
        return set()

class CppParser(TreeSitterParser):
    def __init__(self):
        super().__init__("cpp")
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        return []
    def extract_dependencies(self, content:str)->Set[str]:
        return set()

class HtmlParser(TreeSitterParser):
    def __init__(self):
        super().__init__("html")
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        return []
    def extract_dependencies(self, content:str)->Set[str]:
        return set()

class CssParser(TreeSitterParser):
    def __init__(self):
        super().__init__("css")
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        return []
    def extract_dependencies(self, content:str)->Set[str]:
        return set()

class GoParser(TreeSitterParser):
    def __init__(self):
        super().__init__("go")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        out = []
        try:
            data = file_path.read_bytes()
            tree = self.parser.parse(data)
            
            # First pass: collect all declarations and their relationships
            type_info = {}
            func_info = {}
            var_info = {}
            const_info = {}
            interface_info = {}
            
            # Track package info
            package_name = None
            imports = set()
            module_doc = None
            
            # First collect package and imports
            for node in self._traverse(tree.root_node):
                if node.type == "package_clause":
                    for c in node.children:
                        if c.type == "package_identifier":
                            package_name = self._get_node_text(c, data)
                            break
                    # Get package documentation
                    module_doc = self._get_package_doc(tree.root_node, data)
                elif node.type == "import_declaration":
                    imports.update(self.extract_dependencies(self._get_node_text(node, data)))
            
            # Collect all declarations first
            for node in self._traverse(tree.root_node):
                if node.type == "type_declaration":
                    for c in node.children:
                        if c.type == "type_spec":
                            name = None
                            for sc in c.children:
                                if sc.type == "type_identifier":
                                    name = self._get_node_text(sc, data)
                                    break
                            if name:
                                type_info[name] = {
                                    'node': node,
                                    'methods': set(),
                                    'implements': set(),
                                    'implemented_by': set(),
                                    'used_by': set(),
                                    'fields': set(),
                                    'doc': self._get_docstring(node, data)
                                }
                                
                elif node.type == "function_declaration":
                    name = None
                    for c in node.children:
                        if c.type == "identifier":
                            name = self._get_node_text(c, data)
                            break
                    if name:
                        func_info[name] = {
                            'node': node,
                            'calls': set(),
                            'called_by': set(),
                            'doc': self._get_docstring(node, data)
                        }
                        
                elif node.type == "method_declaration":
                    name = None
                    receiver = None
                    for c in node.children:
                        if c.type == "identifier":
                            name = self._get_node_text(c, data)
                        elif c.type == "parameter_list":
                            for param in c.children:
                                if param.type == "type_identifier":
                                    receiver = self._get_node_text(param, data)
                    if name and receiver:
                        if receiver in type_info:
                            type_info[receiver]['methods'].add(name)
                            
            # Second pass: collect relationships and usage
            for node in self._traverse(tree.root_node):
                # Track function calls
                if node.type == "call_expression":
                    caller = None
                    callee = None
                    
                    # Get the caller context
                    parent = node
                    while parent and parent.type not in ("function_declaration", "method_declaration"):
                        parent = parent.parent
                    if parent:
                        for c in parent.children:
                            if c.type == "identifier":
                                caller = self._get_node_text(c, data)
                                
                    # Get the called function
                    for c in node.children:
                        if c.type == "identifier":
                            callee = self._get_node_text(c, data)
                            
                    if caller and callee:
                        if caller in func_info:
                            func_info[caller]['calls'].add(callee)
                        if callee in func_info:
                            func_info[callee]['called_by'].add(caller)
                            
                # Track interface implementations
                elif node.type == "type_spec":
                    type_name = None
                    for c in node.children:
                        if c.type == "type_identifier":
                            type_name = self._get_node_text(c, data)
                        elif c.type == "interface_type":
                            if type_name:
                                # Look for method specifications
                                for method in c.children:
                                    if method.type == "method_spec":
                                        method_text = self._get_node_text(method, data)
                                        for impl in type_info.values():
                                            if method_text in impl['methods']:
                                                impl['implements'].add(type_name)
                                                if type_name in type_info:
                                                    type_info[type_name]['implemented_by'].add(
                                                        self._get_node_text(impl['node'], data)
                                                    )
            
            # Find related test files
            test_files = set()
            base_name = file_path.stem
            test_file = file_path.parent / f"{base_name}_test.go"
            if test_file.exists():
                test_files.add(str(test_file))
            
            # Find related config files
            config_files = set()
            for config in file_path.parent.glob("*.yaml"):
                config_files.add(str(config))
            for config in file_path.parent.glob("*.json"):
                config_files.add(str(config))
            
            # Process all declarations with enhanced context
            for node in self._traverse(tree.root_node):
                if node.type in (
                    "function_declaration",
                    "method_declaration", 
                    "type_declaration",
                    "const_declaration",
                    "var_declaration",
                    "interface_type",
                    "struct_type"
                ):
                    name = None
                    parent = package_name
                    semantic_type = node.type.replace("_declaration", "").replace("_type", "")
                    
                    # Get name from different node types
                    for c in node.children:
                        if c.type in ("identifier", "field_identifier", "type_identifier"):
                            name = self._get_node_text(c, data)
                            break
                            
                    if name:
                        # Build fully qualified name
                        fqn = f"{package_name}.{name}" if package_name else name
                        if parent and parent != package_name:
                            fqn = f"{package_name}.{parent}.{name}"
                        
                        # Get documentation with context
                        docstring = self._get_docstring(node, data)
                        
                        # Get file context (imports, package info)
                        file_context = self._get_file_context(tree.root_node, data)
                        
                        # Build relationships dictionary
                        relationships = {
                            'calls': set(),
                            'called_by': set(),
                            'implements': set(),
                            'extended_by': set(),
                            'extends': set(),
                            'uses': set(),
                            'used_by': set(),
                            'tested_by': test_files,
                            'configures': config_files,
                        }
                        
                        # Add relationship information
                        if name in func_info:
                            relationships['calls'] = func_info[name]['calls']
                            relationships['called_by'] = func_info[name]['called_by']
                        elif name in type_info:
                            relationships['implements'] = type_info[name]['implements']
                            relationships['extended_by'] = type_info[name]['implemented_by']
                        
                        # Extract API endpoints if this is an HTTP handler
                        api_endpoints = set()
                        if semantic_type == "function" and self._is_http_handler(node, data):
                            api_endpoints.add(self._extract_route(node, data))
                        
                        # Try to generate a semantic summary
                        semantic_summary = f"This {semantic_type} {name} in package {package_name}"
                        if docstring:
                            semantic_summary += f" {docstring.split('.')[0]}."
                        
                        out.append(CodeEntity(
                            name=name,
                            type=semantic_type,
                            docstring=docstring,
                            code=self._get_node_text(node, data),
                            start_line=node.start_point[0]+1,
                            end_line=node.end_point[0]+1,
                            parent=parent,
                            dependencies=imports,
                            metadata={
                                "language": "go",
                                "path": str(file_path),
                                "package": package_name
                            },
                            surrounding_context=self._get_surrounding_context(node, data),
                            related_entities=set(type_info.keys()) | set(func_info.keys()),
                            semantic_type=semantic_type,
                            full_qualified_name=fqn,
                            symbols=self._get_symbols(node, data),
                            file_context=file_context,
                            module_doc=module_doc,
                            related_files=test_files | config_files,
                            semantic_summary=semantic_summary,
                            api_endpoints=api_endpoints,
                            relationships=relationships
                        ))
                        
        except Exception as e:
            logger.error(f"Failed to parse Go {file_path}: {e}", exc_info=True)
        return out

    def _get_file_context(self, root_node, content: bytes) -> str:
        """Extract important file-level context like imports and package info."""
        context_parts = []
        for node in root_node.children:
            if node.type in ("package_clause", "import_declaration"):
                context_parts.append(self._get_node_text(node, content))
        return "\n".join(context_parts)

    def _is_http_handler(self, node, content: bytes) -> bool:
        """Check if a function is an HTTP handler."""
        if node.type != "function_declaration":
            return False
        # Look for http.Handler interface implementation
        for child in node.children:
            if child.type == "parameter_list":
                for param in child.children:
                    param_text = self._get_node_text(param, content)
                    if "http.Request" in param_text:
                        return True
        return False

    def _extract_route(self, node, content: bytes) -> str:
        """Extract the route path from an HTTP handler function."""
        # Look for string literals that look like routes
        for child in self._traverse(node):
            if child.type == "interpreted_string_literal":
                route = self._get_node_text(child, content).strip('"\'')
                if route.startswith("/"):
                    return route
        return ""

    def _get_package_doc(self, root_node, content: bytes) -> Optional[str]:
        """Extract package-level documentation."""
        comments = []
        for node in root_node.children:
            if node.type == "comment" and node.start_point[0] < root_node.start_point[0]:
                comments.append(self._get_node_text(node, content))
            else:
                break
        return "\n".join(comments) if comments else None

    def _traverse(self, node):
        yield node
        for c in node.children:
            yield from self._traverse(c)

    def extract_dependencies(self, content: str) -> Set[str]:
        s = set()
        try:
            t = self.parser.parse(content.encode("utf-8"))
            for n in self._traverse(t.root_node):
                if n.type == "import_declaration":
                    for c in n.children:
                        if c.type == "import_spec_list":
                            for spec in c.children:
                                if spec.type == "import_spec":
                                    for sc in spec.children:
                                        if sc.type == "interpreted_string_literal":
                                            s.add(self._get_node_text(sc, content.encode("utf-8")).strip('"'))
                        elif c.type == "import_spec":
                            for sc in c.children:
                                if sc.type == "interpreted_string_literal":
                                    s.add(self._get_node_text(sc, content.encode("utf-8")).strip('"'))
        except Exception as e:
            logger.error(f"Go deps extraction error: {e}", exc_info=True)
        return s

class RustParser(TreeSitterParser):
    def __init__(self):
        super().__init__("rust")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        out = []
        try:
            data = file_path.read_bytes()
            tree = self.parser.parse(data)
            
            # Track module name for parent relationship
            module_name = None
            for node in self._traverse(tree.root_node):
                if node.type == "mod_item":
                    for c in node.children:
                        if c.type == "identifier":
                            module_name = self._get_node_text(c, data)
                            break
            
            for node in self._traverse(tree.root_node):
                # Parse more Rust items
                if node.type in (
                    "function_item",
                    "struct_item",
                    "trait_item",
                    "impl_item",
                    "enum_item",
                    "const_item",
                    "static_item",
                    "type_item",
                    "union_item",
                    "macro_definition"
                ):
                    name = None
                    parent = module_name
                    
                    # Handle impl blocks
                    if node.type == "impl_item":
                        for c in node.children:
                            if c.type == "type_identifier":
                                parent = self._get_node_text(c, data)
                                break
                    
                    # Get name from different node types
                    for c in node.children:
                        if c.type in ("identifier", "type_identifier"):
                            name = self._get_node_text(c, data)
                            break
                            
                    if name:
                        # Get full docstring including any attributes
                        docstring = self._get_docstring(node, data)
                        if not docstring:
                            # Look for doc attributes
                            for c in node.children:
                                if c.type == "attribute_item":
                                    attr = self._get_node_text(c, data)
                                    if "#[doc =" in attr:
                                        docstring = (docstring or "") + "\n" + attr.strip()
                        
                        out.append(CodeEntity(
                            name=name,
                            type=node.type.replace("_item", ""),
                            docstring=docstring,
                            code=self._get_node_text(node, data),
                            start_line=node.start_point[0]+1,
                            end_line=node.end_point[0]+1,
                            parent=parent,
                            dependencies=self.extract_dependencies(data.decode("utf-8")),
                            metadata={
                                "language": "rust",
                                "path": str(file_path),
                                "module": module_name
                            }
                        ))
        except Exception as e:
            logger.error(f"Failed to parse Rust {file_path}: {e}", exc_info=True)
        return out

    def _traverse(self, node):
        yield node
        for c in node.children:
            yield from self._traverse(c)

    def extract_dependencies(self, content: str) -> Set[str]:
        s = set()
        try:
            t = self.parser.parse(content.encode("utf-8"))
            for n in self._traverse(t.root_node):
                # Handle use declarations (imports)
                if n.type == "use_declaration":
                    path = []
                    for c in n.children:
                        if c.type == "scoped_identifier":
                            s.add(self._get_node_text(c, content.encode("utf-8")))
                        elif c.type == "identifier":
                            path.append(self._get_node_text(c, content.encode("utf-8")))
                    if path:
                        s.add("::".join(path))
                # Handle extern crate declarations
                elif n.type == "extern_crate_declaration":
                    for c in n.children:
                        if c.type == "identifier":
                            s.add(self._get_node_text(c, content.encode("utf-8")))
                # Handle macro imports
                elif n.type == "macro_use_declaration":
                    for c in n.children:
                        if c.type == "identifier":
                            s.add(self._get_node_text(c, content.encode("utf-8")))
        except Exception as e:
            logger.error(f"Rust deps extraction error: {e}", exc_info=True)
        return s

#######################################################################
# 5) Python doesn't need Tree-Sitter
#######################################################################

class PythonParser(LanguageParser):
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        out = []
        try:
            code_str = file_path.read_text(encoding="utf-8")
            mod = ast.parse(code_str)
            
            # First pass: collect class and function info
            class_info = {}
            func_info = {}
            module_doc = ast.get_docstring(mod)
            
            for node in ast.walk(mod):
                if isinstance(node, ast.ClassDef):
                    class_info[node.name] = {
                        'node': node,
                        'methods': set(),
                        'bases': {b.id for b in node.bases if isinstance(b, ast.Name)},
                        'attributes': set(),
                    }
                elif isinstance(node, ast.FunctionDef):
                    func_info[node.name] = {
                        'node': node,
                        'calls': set(),
                        'decorators': {d.id for d in node.decorator_list if isinstance(d, ast.Name)},
                    }
            
            # Second pass: collect relationships
            for node in ast.walk(mod):
                if isinstance(node, ast.ClassDef):
                    # Track method-class relationships
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info[node.name]['methods'].add(item.name)
                            if item.name in func_info:
                                func_info[item.name]['parent'] = node.name
                                
                    # Track attribute access
                    for item in ast.walk(node):
                        if isinstance(item, ast.Attribute):
                            class_info[node.name]['attributes'].add(item.attr)
                            
                elif isinstance(node, ast.FunctionDef):
                    # Track function calls
                    for item in ast.walk(node):
                        if isinstance(item, ast.Call):
                            if isinstance(item.func, ast.Name):
                                func_info[node.name]['calls'].add(item.func.id)
            
            # Process classes
            for node in ast.walk(mod):
                if isinstance(node, ast.ClassDef):
                    # Get surrounding lines for context
                    file_lines = code_str.splitlines()
                    start = max(0, node.lineno - 5)
                    end = min(len(file_lines), node.end_lineno + 5)
                    surrounding_context = "\n".join(file_lines[start:end])
                    
                    # Collect related entities
                    related = set()
                    related.update(class_info[node.name]['methods'])
                    related.update(class_info[node.name]['bases'])
                    
                    # Extract symbols used
                    symbols = set()
                    for item in ast.walk(node):
                        if isinstance(item, (ast.Name, ast.Attribute)):
                            if isinstance(item, ast.Name):
                                symbols.add(item.id)
                            else:
                                symbols.add(item.attr)
                    
                    # Build fully qualified name
                    module_name = file_path.stem
                    fqn = f"{module_name}.{node.name}"
                    
                    out.append(CodeEntity(
                        name=node.name,
                        type="class",
                        docstring=ast.get_docstring(node),
                        code=ast.unparse(node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        parent=None,
                        dependencies=self.extract_dependencies(code_str),
                        metadata={
                            "language": "python",
                            "path": str(file_path),
                            "module": module_name,
                            "bases": list(class_info[node.name]['bases']),
                            "methods": list(class_info[node.name]['methods']),
                            "attributes": list(class_info[node.name]['attributes'])
                        },
                        surrounding_context=surrounding_context,
                        related_entities=related,
                        semantic_type="class",
                        full_qualified_name=fqn,
                        symbols=symbols
                    ))
                    
                elif isinstance(node, ast.FunctionDef):
                    # Similar context gathering for functions
                    file_lines = code_str.splitlines()
                    start = max(0, node.lineno - 5)
                    end = min(len(file_lines), node.end_lineno + 5)
                    surrounding_context = "\n".join(file_lines[start:end])
                    
                    # Determine semantic type (method vs function)
                    semantic_type = "method" if node.name in func_info and 'parent' in func_info[node.name] else "function"
                    parent = func_info[node.name].get('parent')
                    
                    # Collect related entities and symbols
                    related = set()
                    related.update(func_info[node.name]['calls'])
                    related.update(func_info[node.name]['decorators'])
                    
                    symbols = set()
                    for item in ast.walk(node):
                        if isinstance(item, (ast.Name, ast.Attribute)):
                            if isinstance(item, ast.Name):
                                symbols.add(item.id)
                            else:
                                symbols.add(item.attr)
                    
                    # Build fully qualified name
                    module_name = file_path.stem
                    fqn = f"{module_name}.{node.name}"
                    if parent:
                        fqn = f"{module_name}.{parent}.{node.name}"
                    
                    out.append(CodeEntity(
                        name=node.name,
                        type=semantic_type,
                        docstring=ast.get_docstring(node),
                        code=ast.unparse(node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        parent=parent,
                        dependencies=self.extract_dependencies(code_str),
                        metadata={
                            "language": "python",
                            "path": str(file_path),
                            "module": module_name,
                            "decorators": list(func_info[node.name]['decorators']),
                            "calls": list(func_info[node.name]['calls'])
                        },
                        surrounding_context=surrounding_context,
                        related_entities=related,
                        semantic_type=semantic_type,
                        full_qualified_name=fqn,
                        symbols=symbols
                    ))
                    
        except Exception as e:
            logger.error(f"Python parse error {file_path}: {e}", exc_info=True)
        return out

    def extract_dependencies(self, content: str) -> Set[str]:
        s = set()
        try:
            mod = ast.parse(content)
            for n in ast.walk(mod):
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        s.add(alias.name)
                elif isinstance(n, ast.ImportFrom):
                    if n.module:
                        s.add(n.module)
                # Also track runtime imports
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'import_module':
                    for arg in n.args:
                        if isinstance(arg, ast.Str):
                            s.add(arg.s)
        except Exception as e:
            logger.error(f"extract deps python: {e}", exc_info=True)
        return s

#######################################################################
# 6) ParserFactory for each file extension
#######################################################################

class ParserFactory:
    _parsers={
        ".py": PythonParser,
        ".js": JavaScriptParser,
        ".jsx": JavaScriptParser,
        ".ts": TypeScriptParser,
        ".tsx": TypeScriptParser,
        ".cpp": CppParser,
        ".c": CppParser,
        ".html": HtmlParser,
        ".css": CssParser,
        ".go": GoParser,
        ".rs": RustParser,
    }
    
    @classmethod
    def get_parser(cls, file_path: Path):
        ext = file_path.suffix.lower()
        parser_cls = cls._parsers.get(ext)
        if not parser_cls:
            return None
            
        try:
            return parser_cls()
        except ValueError as e:
            logger.warning(f"Parser not available for {ext}: {e}")
            return None

def parse_codebase(root_path: Path)->List["CodeEntity"]:
    """
    Recursively parse recognized file types in `root_path`.
    """
    results=[]
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            p=ParserFactory.get_parser(file_path)
            if p:
                results.extend(p.parse_file(file_path))
    return results

# Initialize at import
init_languages()

def get_languages_dir() -> Path:
    """Get the directory where language libraries are stored."""
    return LANGUAGE_DIR

LANGUAGE_BUILDERS = {
    "solidity": _build_solidity,
}

def get_language(language: str) -> Language:
    """Get a tree-sitter language by name."""
    # For standard languages, use tree-sitter-languages
    if language not in ["solidity"]:
        return tree_sitter_languages.get_language(language)
        
    # For Solidity, we need to build it
    languages_dir = get_languages_dir()
    so_path = languages_dir / f"languages_{language}.so"
    
    # Build if needed
    if not so_path.exists():
        sol_dir = languages_dir / "tree-sitter-solidity"
        if not _build_solidity(sol_dir, so_path):
            raise RuntimeError(f"Failed to build {language} parser")
    
    # Load the library
    return Language(str(so_path), language)
