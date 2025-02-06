"""Module for ingesting code and documentation files into the vector store."""

import logging
from pathlib import Path
from typing import List, Optional, Set, Union

from knowledge_agent.core.code_parser import CodeEntity, ParserFactory as CodeParserFactory
from knowledge_agent.core.doc_parser import (
    DocSection as ParserDocSection,
    ParserFactory as DocParserFactory,
)
from knowledge_agent.core.documentation import DocSection as DocumentationDocSection
from knowledge_agent.core.vector_store import VectorStore
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

logger = logging.getLogger(__name__)

SUPPORTED_CODE_EXTENSIONS = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript (React)',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript (React)',
    '.cpp': 'C++',
    '.c': 'C',
    '.html': 'HTML',
    '.css': 'CSS',
    '.go': 'Go',
    '.rs': 'Rust',
}

SUPPORTED_DOC_EXTENSIONS = {
    '.md': 'Markdown',  # ensure .md is recognized as doc
    '.txt': 'Text',
    '.rst': 'reStructuredText',
    '.pdf': 'PDF',
    '.tex': 'LaTeX',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.toml': 'TOML',
    '.sol': 'Solidity',  # Treat Solidity files as documentation
}

EXCLUDED_DIRECTORIES = {
    'node_modules',
    '.git',
    '__pycache__',
    'build',
    'dist',
    'venv',
    '.env',
    '.venv',
    '.idea',
    '.vscode',
    'coverage',
    '.github',  # GitHub specific files
    # 'certora',  # Allow Certora verification files for Solidity
    # 'test',     # Allow test files for Solidity
    # 'tests',    # Allow test files for Solidity
    # 'script',   # Allow build/deployment scripts for Solidity
}

EXCLUDED_FILES = {
    'package.json',
    'package-lock.json',
    'yarn.lock',
    '.gitignore',
    '.npmrc',
    '.env',
    '.DS_Store',
    'Thumbs.db',
    '.eslintrc',
    '.prettierrc',
    'tsconfig.json',
    'jest.config.js',
    'README.md',   # You can remove this line if you WANT to ingest README
    'LICENSE',
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    'docker-compose.yml',
    'Dockerfile',
}

def is_supported_file(file_path: Path) -> bool:
    """Check if a file is supported for ingestion."""
    # Skip excluded directories (but allow test directories for Solidity)
    if file_path.suffix.lower() != '.sol':
        for parent in file_path.parents:
            if parent.name.lower() in EXCLUDED_DIRECTORIES:
                return False

    # Skip excluded files (but allow test files for Solidity)
    if file_path.suffix.lower() != '.sol' and file_path.name.lower() in EXCLUDED_FILES:
        return False

    # Skip typical test file naming patterns (but allow for Solidity)
    if file_path.suffix.lower() != '.sol':
        fname = file_path.name.lower()
        if fname.startswith('test_') or fname.endswith('.test.js'):
            return False

    # Check if recognized code or doc extension
    suffix = file_path.suffix.lower()
    return (
        suffix in SUPPORTED_CODE_EXTENSIONS
        or suffix in SUPPORTED_DOC_EXTENSIONS
    )

def ingest_file(file_path: Path, vector_store: VectorStore) -> int:
    """Ingest a single file into the vector store."""
    if not is_supported_file(file_path):
        logger.warning(f"Unsupported or excluded file type: {file_path}")
        return 0

    try:
        # 1) Try code parser
        code_parser = CodeParserFactory.get_parser(file_path)
        if code_parser:
            return _ingest_code_file(code_parser, file_path, vector_store)

        # 2) No code parser -> try doc parser
        doc_parser = DocParserFactory.get_parser(file_path)
        if doc_parser:
            return _ingest_doc_file(doc_parser, file_path, vector_store)

        # 3) If no parser found
        logger.warning(f"No parser found for file: {file_path}")
        return 0

    except Exception as e:
        logger.error(f"Failed to ingest file {file_path}: {e}", exc_info=True)
        return 0

def _ingest_code_file(parser, file_path: Path, vector_store: VectorStore) -> int:
    """Use a code parser to ingest a file."""
    try:
        entities = parser.parse_file(file_path)
        # Filter out anything that's not CodeEntity
        valid_entities = [e for e in entities if isinstance(e, CodeEntity)]
        if len(valid_entities) < len(entities):
            logger.warning(
                f"Parser returned some non-CodeEntity items for {file_path}, ignoring them."
            )
        return _ingest_code_entities(valid_entities, vector_store)
    except Exception as e:
        logger.error(f"Failed parsing code file {file_path}: {str(e)}", exc_info=True)
        return 0

def _ingest_doc_file(parser, file_path: Path, vector_store: VectorStore) -> int:
    """Use a doc parser to ingest a file."""
    try:
        sections = parser.parse_file(file_path)
        return _ingest_doc_sections(sections, vector_store, debug_filepath=file_path)
    except Exception as e:
        logger.error(f"Failed parsing doc file {file_path}: {str(e)}", exc_info=True)
        return 0

def _ingest_code_entities(entities: List[CodeEntity], vector_store: VectorStore) -> int:
    """Ingest code entities into the vector store as separate chunks."""
    chunks_created = 0
    for entity in entities:
        try:
            # Force metadata to be dict
            if not isinstance(entity.metadata, dict):
                logger.warning(f"Entity {entity.name} has non‐dict metadata, forcing to dict.")
                entity.metadata = {'_forced': str(entity.metadata)}

            # Build and filter metadata
            code_meta = _filter_metadata({
                'type': 'code',
                'entity_type': entity.type,
                'name': entity.name,
                'language': entity.metadata.get('language', 'unknown'),
                'file_path': entity.metadata.get('path', ''),
                'start_line': entity.start_line,
                'end_line': entity.end_line,
                'parent': entity.parent or '',
                'dependencies': ','.join(sorted(entity.dependencies)) if entity.dependencies else '',
            })
            # Make a Document for the code
            doc_code = Document(page_content=entity.code, metadata=code_meta)
            vector_store.add_chunk(doc_code)
            chunks_created += 1

            # If docstring exists, store it separately
            if entity.docstring:
                docstring_meta = _filter_metadata({
                    'type': 'docstring',
                    'entity_name': entity.name,
                    'entity_type': entity.type,
                    'language': entity.metadata.get('language', 'unknown'),
                    'file_path': entity.metadata.get('path', ''),
                    'start_line': entity.start_line,
                })
                doc_docstring = Document(page_content=entity.docstring, metadata=docstring_meta)
                vector_store.add_chunk(doc_docstring)
                chunks_created += 1

        except Exception as e:
            logger.error(f"Failed to ingest code entity {entity.name}: {str(e)}", exc_info=True)
    return chunks_created

def _filter_metadata(metadata: dict) -> dict:
    """Filter out complex types from metadata dictionary."""
    filtered = {}
    allowed_types = (str, int, float, bool)
    for key, value in metadata.items():
        if isinstance(value, allowed_types):
            filtered[key] = value
        elif value is not None:
            filtered[key] = str(value)
    return filtered

def _ingest_doc_sections(
    sections: List[Union[str, ParserDocSection, DocumentationDocSection]],
    vector_store: VectorStore,
    debug_filepath: Optional[Path] = None
) -> int:
    """Ingest documentation sections into the vector store."""
    chunks_created = 0
    for i, section in enumerate(sections):
        try:
            # Debug info
            logger.debug(f"[DOC] In file {debug_filepath} at index {i}, got section type={type(section)}: {repr(section)[:200]}")

            # 1. Convert to ParserDocSection if needed
            if isinstance(section, str):
                if not section.strip():
                    continue
                section = ParserDocSection(
                    content=section,
                    title="Text Section",
                    content_type="text",
                    metadata={"type": "text_section"},
                )
            elif isinstance(section, DocumentationDocSection):
                section = ParserDocSection(
                    content=section.content,
                    title=section.title,
                    source_file=section.source_file,
                    line_number=section.line_number,
                    content_type=section.content_type,
                    metadata=section.metadata or {},
                )
            elif not isinstance(section, ParserDocSection):
                section = ParserDocSection(
                    content=str(section),
                    title="Object Section",
                    content_type="text",
                    metadata={"type": "object_section", "object_type": type(section).__name__},
                )

            # 2. Validate the section
            if not isinstance(section, ParserDocSection):
                logger.error(f"Post-conversion is not ParserDocSection. Skipping: {section}")
                continue

            # 3. Process content
            content = section.content.strip()
            if not content:
                continue

            # 4. Ensure metadata is a dict
            if not isinstance(section.metadata, dict):
                section.metadata = {"_forced": str(section.metadata)}

            # 5. Build metadata
            metadata = {
                "type": "documentation",
                "title": section.title or "Untitled Section",
                "file_path": str(section.source_file or ""),
                "line_number": getattr(section, "line_number", 0) or 0,
                "content_type": section.content_type or "text",
            }

            # 6. Merge with section metadata
            for k, v in section.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dict one level
                    for subk, subv in v.items():
                        if isinstance(subv, (str, int, float, bool)):
                            metadata[f"{k}.{subk}"] = subv
                elif v is not None:
                    metadata[k] = str(v)

            # 7. Create and store the chunk
            doc_chunk = Document(
                page_content=content,
                metadata=_filter_metadata(metadata)
            )
            vector_store.add_chunk(doc_chunk)
            chunks_created += 1

        except Exception as e:
            logger.error(
                f"Failed to ingest doc section: {e}, section type={type(section)} index={i}", 
                exc_info=True
            )
            continue

    return chunks_created

def ingest_directory(
    directory: Path,
    vector_store: VectorStore,
    exclude_patterns: Optional[Set[str]] = None
) -> int:
    """Recursively ingest all supported files in a directory."""
    if exclude_patterns is None:
        exclude_patterns = set()

    total_chunks = 0
    try:
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            # Skip user-specified exclude patterns
            if any(pat in str(file_path) for pat in exclude_patterns):
                continue

            # Skip if in an excluded directory
            if any(parent.name.lower() in EXCLUDED_DIRECTORIES for parent in file_path.parents):
                logger.debug(f"Skipping file in excluded directory: {file_path}")
                continue

            # Skip excluded filenames
            if file_path.name.lower() in EXCLUDED_FILES:
                logger.debug(f"Skipping excluded file: {file_path}")
                continue

            # If recognized as code/doc, ingest
            if is_supported_file(file_path):
                try:
                    chunks = ingest_file(file_path, vector_store)
                    total_chunks += chunks
                    logger.info(f"Ingested {file_path}: {chunks} chunks created")
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Failed to ingest directory {directory}: {str(e)}", exc_info=True)

    return total_chunks
