"""
Module for AST-based code parsing and analysis.
Languages:
- JavaScript, TypeScript(+TSX), C++, HTML, CSS, Solidity via Tree-Sitter
- Python via built-in AST
"""

import ast
import json
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    standard_langs = ['javascript', 'typescript', 'tsx', 'cpp', 'c', 'html', 'css']
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
        for child in node.children:
            if child.type == "comment":
                comments.append(self._get_node_text(child, content))
        return "\n".join(comments) if comments else None

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
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        return []
    def extract_dependencies(self, content:str)->Set[str]:
        return set()
    
class RustParser(TreeSitterParser):
    def __init__(self):
        super().__init__("rust")
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        return []
    def extract_dependencies(self, content:str)->Set[str]:
        return set()

#######################################################################
# 5) Python doesn't need Tree-Sitter
#######################################################################

class PythonParser(LanguageParser):
    def parse_file(self, file_path: Path)->List[CodeEntity]:
        out=[]
        try:
            code_str = file_path.read_text(encoding="utf-8")
            mod = ast.parse(code_str)
            for node in ast.walk(mod):
                if isinstance(node, ast.ClassDef):
                    out.append(CodeEntity(
                        name=node.name,
                        type="class",
                        docstring=ast.get_docstring(node),
                        code=ast.unparse(node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        parent=None,
                        dependencies=set(),
                        metadata={"language":"python","path":str(file_path)}
                    ))
                elif isinstance(node, ast.FunctionDef):
                    out.append(CodeEntity(
                        name=node.name,
                        type="function",
                        docstring=ast.get_docstring(node),
                        code=ast.unparse(node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        parent=None,
                        dependencies=set(),
                        metadata={"language":"python","path":str(file_path)}
                    ))
        except Exception as e:
            logger.error(f"Python parse error {file_path}: {e}", exc_info=True)
        return out
    def extract_dependencies(self, content:str)->Set[str]:
        s=set()
        try:
            mod=ast.parse(content)
            for n in ast.walk(mod):
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        s.add(alias.name)
                elif isinstance(n, ast.ImportFrom):
                    if n.module:
                        s.add(n.module)
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
