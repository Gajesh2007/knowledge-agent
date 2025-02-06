"""
Module for AST-based code parsing and analysis, supporting:
- JavaScript (tree-sitter-javascript)
- TypeScript + TSX (tree-sitter-typescript)
- C++ (tree-sitter-cpp)
- HTML (tree-sitter-html)
- CSS (tree-sitter-css)
- Solidity (tree-sitter-solidity)
- Python (using built-in `ast`)
"""

import ast
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import tree_sitter
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) Build/Load Tree-Sitter Grammars
# ---------------------------------------------------------------------------

# Change from package directory to user's home directory for builds
LANGUAGE_DIR = Path.home() / ".knowledge-agent" / "tree-sitter-languages"
LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)

# Git repos for each grammar we want
LANGUAGE_REPOS = {
    # JS is sometimes needed for TS to build fully
    "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
    "typescript": "https://github.com/tree-sitter/tree-sitter-typescript",
    "cpp":        "https://github.com/tree-sitter/tree-sitter-cpp",
    "html":       "https://github.com/tree-sitter/tree-sitter-html",
    "css":        "https://github.com/tree-sitter/tree-sitter-css",
    "solidity":   "https://github.com/JoranHonig/tree-sitter-solidity",
}

# We'll store each loaded language object in LANGUAGES
LANGUAGES: Dict[str, Optional[Language]] = {}

def init_languages() -> None:
    """
    Main entry point: clone all grammar repos, build them into languages.so,
    then load each grammar. If something fails, we skip that grammar.
    """
    global LANGUAGES
    lib_path = LANGUAGE_DIR / "languages.so"

    # If we've already built "languages.so" and populated LANGUAGES, skip:
    if lib_path.exists() and LANGUAGES:
        logger.info("Tree-sitter languages are already initialized, skipping rebuild.")
        return

    # If Rust is missing, some grammars might fail:
    try:
        subprocess.run(["rustc", "--version"], check=True, capture_output=True)
        logger.debug("Rust compiler found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("Rust compiler NOT found; building TS or other grammars may fail.")

    # 1) Clone or update all grammar repos
    _clone_all_grammars()

    # 2) Build them each in a workable order
    _setup_javascript_grammar()
    _setup_typescript_grammar()
    _setup_generic_grammar("cpp")       # calls tree-sitter generate for c++ if needed
    _setup_generic_grammar("html")
    _setup_generic_grammar("css")
    _setup_solidity_grammar()

    # 3) Build combined library languages.so
    if not lib_path.exists():
        try:
            logger.info("Building unified languages.so library ...")
            # Get all grammar source directories that have src/parser.c
            grammar_dirs = []
            for d in LANGUAGE_DIR.iterdir():
                if d.is_dir():
                    if d.name == "tree-sitter-typescript":
                        # For typescript, we need both typescript and tsx
                        ts_dir = d / "typescript" / "src"
                        tsx_dir = d / "tsx" / "src"
                        if ts_dir.exists() and (ts_dir / "parser.c").exists():
                            grammar_dirs.append(str(ts_dir.parent))
                        if tsx_dir.exists() and (tsx_dir / "parser.c").exists():
                            grammar_dirs.append(str(tsx_dir.parent))
                    else:
                        src_dir = d / "src"
                        if src_dir.exists() and (src_dir / "parser.c").exists():
                            grammar_dirs.append(str(d))
            
            Language.build_library(
                str(lib_path),
                grammar_dirs
            )
            logger.info("Successfully built languages.so")
        except Exception as e:
            logger.error(f"Failed to build languages.so: {e}")

    # 4) Finally, load each grammar into LANGUAGES (if languages.so is present)
    for lang_name in LANGUAGE_REPOS:
        if not lib_path.exists():
            logger.warning(f"languages.so not found, skipping {lang_name}")
            LANGUAGES[lang_name] = None
            continue
        try:
            if lang_name == "typescript":
                # TS also includes TSX
                LANGUAGES["typescript"] = Language(str(lib_path), "typescript")
                LANGUAGES["tsx"] = Language(str(lib_path), "tsx")
                logger.info("Loaded 'typescript' and 'tsx' languages.")
            elif lang_name == "solidity":
                LANGUAGES["solidity"] = Language(str(lib_path), "solidity")
                logger.info("Loaded 'solidity' language.")
            else:
                LANGUAGES[lang_name] = Language(str(lib_path), lang_name)
                logger.info(f"Loaded '{lang_name}' language.")
        except Exception as e:
            logger.error(f"Failed to load {lang_name}: {e}")
            LANGUAGES[lang_name] = None

def _clone_all_grammars() -> None:
    """
    Clones each grammar repository if not present already.
    """
    for lang_name, repo_url in LANGUAGE_REPOS.items():
        lang_dir = LANGUAGE_DIR / f"tree-sitter-{lang_name}"
        if not lang_dir.exists():
            try:
                logger.info(f"Cloning {lang_name} from {repo_url} => {lang_dir}")
                subprocess.run(["git", "clone", repo_url, str(lang_dir)], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone {lang_name}: {e}")
        else:
            logger.debug(f"{lang_name} grammar already present at {lang_dir}")

def _setup_javascript_grammar() -> None:
    """
    For JS, typically: cd tree-sitter-javascript; npm install; tree-sitter generate
    """
    js_dir = LANGUAGE_DIR / "tree-sitter-javascript"
    if not js_dir.exists():
        return
    logger.info("Setting up JavaScript grammar ...")
    os.chdir(str(js_dir))
    try:
        _npm_install_if_needed(js_dir)
        subprocess.run(["tree-sitter", "generate"], check=True)
        logger.info("JavaScript grammar build completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build JavaScript grammar: {e}")
    finally:
        os.chdir(str(LANGUAGE_DIR))

def _setup_typescript_grammar() -> None:
    """
    For TS, we need to set up both typescript and tsx parsers.
    Each needs its own npm install and tree-sitter generate.
    First ensure JavaScript grammar is built as TypeScript depends on it.
    """
    ts_dir = LANGUAGE_DIR / "tree-sitter-typescript"
    if not ts_dir.exists():
        return
    logger.info("Setting up TypeScript grammar ...")
    
    # First ensure JavaScript grammar is properly set up
    js_dir = LANGUAGE_DIR / "tree-sitter-javascript"
    if not (js_dir / "src" / "grammar.json").exists():
        _setup_javascript_grammar()
    
    # Create a node_modules symlink to access JS grammar
    ts_modules = ts_dir / "node_modules"
    if not ts_modules.exists():
        ts_modules.mkdir(parents=True)
    js_module_link = ts_modules / "tree-sitter-javascript"
    if not js_module_link.exists():
        try:
            os.symlink(str(js_dir), str(js_module_link))
        except Exception as e:
            logger.error(f"Failed to create symlink to JS grammar: {e}")
            raise
    
    # Handle each subfolder (typescript and tsx) separately
    for sub in ["typescript", "tsx"]:
        sub_dir = ts_dir / sub
        if not sub_dir.exists():
            logger.warning(f"TypeScript subfolder not found: {sub_dir}")
            continue
            
        try:
            logger.info(f"Setting up {sub} parser...")
            os.chdir(str(sub_dir))
            
            # Clean previous builds
            if (sub_dir / "build").exists():
                shutil.rmtree(sub_dir / "build")
            if (sub_dir / "node_modules").exists() and not (sub_dir / "node_modules").is_symlink():
                shutil.rmtree(sub_dir / "node_modules")
            
            # Create symlink to node_modules in subfolder
            sub_modules = sub_dir / "node_modules"
            if not sub_modules.exists():
                os.symlink(str(ts_modules), str(sub_modules))
            
            # Install dependencies
            _npm_install_if_needed(sub_dir)
            
            # Generate the parser
            subprocess.run(["tree-sitter", "generate"], check=True)
            
            # Copy necessary files to src directory
            src_dir = sub_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy parser files if they exist
            for file_name in ["parser.c", "parser.h", "tree_sitter/parser.h"]:
                src_file = sub_dir / file_name
                if src_file.exists():
                    dst_file = src_dir / os.path.basename(file_name)
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_file), str(dst_file))
                    logger.info(f"Copied {file_name} to {dst_file}")
            
            logger.info(f"Successfully set up {sub} parser")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {sub} parser: {e}")
            raise
        except Exception as e:
            logger.error(f"Error setting up {sub} parser: {e}")
            raise
        finally:
            os.chdir(str(LANGUAGE_DIR))

def _setup_generic_grammar(lang_name: str) -> None:
    """
    For grammars like c++ / html / css that typically only need 'tree-sitter generate'.
    We'll also do npm install if there's a package.json
    """
    gram_dir = LANGUAGE_DIR / f"tree-sitter-{lang_name}"
    if not gram_dir.exists():
        return
    logger.info(f"Setting up {lang_name} grammar ...")
    os.chdir(str(gram_dir))
    try:
        _npm_install_if_needed(gram_dir)
        # If there's a 'grammar.js' or 'src/grammar.json', we do 'tree-sitter generate'
        # Some grammars might just store `src/parser.c`, but let's attempt:
        subprocess.run(["tree-sitter", "generate"], check=True)
        logger.info(f"{lang_name} grammar build completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed building {lang_name} grammar: {e}")
    finally:
        os.chdir(str(LANGUAGE_DIR))

def _setup_solidity_grammar() -> None:
    """
    For solidity, typically: npm install; tree-sitter generate
    """
    sol_dir = LANGUAGE_DIR / "tree-sitter-solidity"
    if not sol_dir.exists():
        return
    logger.info("Setting up Solidity grammar ...")
    os.chdir(str(sol_dir))
    try:
        _npm_install_if_needed(sol_dir)
        subprocess.run(["tree-sitter", "generate"], check=True)
        logger.info("Solidity grammar build completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed building Solidity grammar: {e}")
    finally:
        os.chdir(str(LANGUAGE_DIR))

def _npm_install_if_needed(path_dir: Path) -> None:
    """
    If there's a package.json, do 'npm install'
    """
    pkg_json = path_dir / "package.json"
    if pkg_json.exists():
        if (path_dir / "node_modules").exists():
            logger.debug(f"node_modules already present in {path_dir}, skipping install.")
        else:
            logger.info(f"npm install in {path_dir} ...")
            subprocess.run(["npm", "install"], check=True)

# Immediately initialize on module load
init_languages()

# ---------------------------------------------------------------------------
# 2) Data Model: CodeEntity
# ---------------------------------------------------------------------------

@dataclass
class CodeEntity:
    """Represents a parsed code entity (function, class, etc.)."""
    name: str
    type: str            # e.g. 'function', 'class', 'method', ...
    docstring: Optional[str]
    code: str
    start_line: int
    end_line: int
    parent: Optional[str]
    dependencies: Set[str]
    metadata: Dict[str, str]

# ---------------------------------------------------------------------------
# 3) Abstract base for each parser
# ---------------------------------------------------------------------------

class LanguageParser(ABC):
    """Abstract base class for code parsers (tree-sitter or otherwise)."""

    @abstractmethod
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        pass

    @abstractmethod
    def extract_dependencies(self, content: str) -> Set[str]:
        pass

# ---------------------------------------------------------------------------
# 4) TreeSitterParser base class
# ---------------------------------------------------------------------------

class TreeSitterParser(LanguageParser):
    """
    Base class for grammars we've loaded with tree-sitter.
    Subclasses override parse_file with the actual logic
    for collecting CodeEntity objects.
    """

    def __init__(self, language_name: str):
        lang_obj = LANGUAGES.get(language_name)
        if lang_obj is None:
            raise ValueError(f"Grammar for '{language_name}' not available.")
        self.parser = Parser()
        self.parser.set_language(lang_obj)
        self.language_name = language_name

    def _get_node_text(self, node, content: bytes) -> str:
        return content[node.start_byte: node.end_byte].decode("utf-8")

    def _get_docstring(self, node, content: bytes) -> Optional[str]:
        # Example: gather child comments
        comments = []
        for child in node.children:
            if child.type == "comment":
                comments.append(self._get_node_text(child, content))
        return "\n".join(comments) if comments else None

# ---------------------------------------------------------------------------
# 5) Language-specific Parsers
# ---------------------------------------------------------------------------

class JavaScriptParser(TreeSitterParser):
    def __init__(self):
        super().__init__("javascript")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        entities: List[CodeEntity] = []
        try:
            content = file_path.read_bytes()
            tree = self.parser.parse(content)
            for node in self._traverse(tree.root_node):
                if node.type in ("function_declaration", "method_definition", "class_declaration"):
                    name = None
                    for child in node.children:
                        if child.type == "identifier":
                            name = self._get_node_text(child, content)
                            break
                    if name:
                        entities.append(CodeEntity(
                            name=name,
                            type=node.type.replace("_declaration", "").replace("_definition", ""),
                            docstring=self._get_docstring(node, content),
                            code=self._get_node_text(node, content),
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent=None,
                            dependencies=self.extract_dependencies(content.decode("utf-8")),
                            metadata={"language": "javascript", "path": str(file_path)},
                        ))
        except Exception as e:
            logger.error(f"Failed to parse JS file {file_path}: {e}", exc_info=True)
        return entities

    def _traverse(self, node):
        yield node
        for child in node.children:
            yield from self._traverse(child)

    def extract_dependencies(self, content: str) -> Set[str]:
        deps = set()
        try:
            tree = self.parser.parse(content.encode("utf-8"))
            for node in self._traverse(tree.root_node):
                if node.type == "import_statement":
                    for child in node.children:
                        if child.type == "string":
                            deps.add(child.text.decode("utf-8").strip("'\""))
        except Exception as e:
            logger.error(f"Failed extracting JS dependencies: {e}", exc_info=True)
        return deps

class TypeScriptParser(TreeSitterParser):
    def __init__(self):
        super().__init__("typescript")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        # Basic stub
        return []

    def extract_dependencies(self, content: str) -> Set[str]:
        return set()

# For C++:
class CppParser(TreeSitterParser):
    def __init__(self):
        super().__init__("cpp")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        # Minimal stub
        return []

    def extract_dependencies(self, content: str) -> Set[str]:
        return set()

# For HTML:
class HtmlParser(TreeSitterParser):
    def __init__(self):
        super().__init__("html")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        # Minimal stub
        return []

    def extract_dependencies(self, content: str) -> Set[str]:
        return set()

# For CSS:
class CssParser(TreeSitterParser):
    def __init__(self):
        super().__init__("css")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        return []

    def extract_dependencies(self, content: str) -> Set[str]:
        return set()

# For Solidity:
class SolidityParser(TreeSitterParser):
    def __init__(self):
        super().__init__("solidity")

    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        entities = []
        try:
            content = file_path.read_bytes()
            tree = self.parser.parse(content)
            for node in self._traverse(tree.root_node):
                # Example: look for 'contract_declaration', 'function_definition'
                if node.type in ("contract_declaration", "function_definition", "modifier_definition"):
                    name = None
                    for child in node.children:
                        if child.type == "identifier":
                            name = self._get_node_text(child, content)
                            break
                    if name:
                        entities.append(CodeEntity(
                            name=name,
                            type=node.type.replace("_declaration", "").replace("_definition", ""),
                            docstring=self._get_docstring(node, content),
                            code=self._get_node_text(node, content),
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            parent=None,
                            dependencies=self.extract_dependencies(content.decode("utf-8")),
                            metadata={"language": "solidity", "path": str(file_path)},
                        ))
        except Exception as e:
            logger.error(f"Failed to parse Solidity file {file_path}: {e}", exc_info=True)
        return entities

    def _traverse(self, node):
        yield node
        for child in node.children:
            yield from self._traverse(child)

    def extract_dependencies(self, content: str) -> Set[str]:
        return set()

# Python doesn't need TreeSitter:
class PythonParser(LanguageParser):
    def parse_file(self, file_path: Path) -> List[CodeEntity]:
        results: List[CodeEntity] = []
        try:
            import ast
            code_str = file_path.read_text(encoding="utf-8")
            mod = ast.parse(code_str)
            for node in ast.walk(mod):
                if isinstance(node, ast.ClassDef):
                    results.append(CodeEntity(
                        name=node.name,
                        type="class",
                        docstring=ast.get_docstring(node),
                        code=ast.unparse(node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        parent=None,
                        dependencies=set(),
                        metadata={"language": "python", "path": str(file_path)},
                    ))
                elif isinstance(node, ast.FunctionDef):
                    results.append(CodeEntity(
                        name=node.name,
                        type="function",
                        docstring=ast.get_docstring(node),
                        code=ast.unparse(node),
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        parent=None,
                        dependencies=set(),
                        metadata={"language": "python", "path": str(file_path)},
                    ))
        except Exception as e:
            logger.error(f"Failed to parse Python file {file_path}: {e}", exc_info=True)
        return results

    def extract_dependencies(self, content: str) -> Set[str]:
        # Example: parse import statements from Python
        deps = set()
        try:
            import ast
            mod = ast.parse(content)
            for node in ast.walk(mod):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        deps.add(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        deps.add(node.module)
        except Exception as e:
            logger.error(f"Failed to extract python deps: {e}", exc_info=True)
        return deps

# ---------------------------------------------------------------------------
# 6) ParserFactory: returns the correct parser for each extension
# ---------------------------------------------------------------------------

class ParserFactory:
    _parsers = {
        ".py":  PythonParser,
        ".js":  JavaScriptParser,
        ".jsx": JavaScriptParser,
        ".ts":  TypeScriptParser,
        ".tsx": TypeScriptParser,
        ".cpp": CppParser,
        ".c":   CppParser,   # handle c/cpp similarly
        ".html":HtmlParser,
        ".css": CssParser,
        ".sol": SolidityParser,
    }

    @classmethod
    def get_parser(cls, file_path: Path):
        ext = file_path.suffix.lower()
        parser_cls = cls._parsers.get(ext)
        if parser_cls is None:
            return None
        try:
            return parser_cls()
        except ValueError as e:
            # Means we tried to use a grammar that wasn't built
            logger.warning(f"Parser not available for {ext}: {e}")
            return None

def parse_codebase(root_path: Path) -> List[CodeEntity]:
    """
    Recursively parse recognized file types in `root_path`.
    Returns a list of CodeEntity objects for all parseable code.
    """
    all_entities: List[CodeEntity] = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            parser = ParserFactory.get_parser(file_path)
            if parser:
                all_entities.extend(parser.parse_file(file_path))
    return all_entities
