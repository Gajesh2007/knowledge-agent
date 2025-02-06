"""
CLI Knowledge Base Agent - Main CLI interface
"""
import os
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from knowledge_agent import __version__
from knowledge_agent.core.ingestion import ingest_directory
from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.vector_store import VectorStore
from knowledge_agent.core.logging import logger

# Load environment variables
load_dotenv()

# Set HuggingFace tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

console = Console()

# Initialize components
try:
    logger.debug("Initializing vector store")
    vector_store = VectorStore(
        persist_directory=os.getenv("VECTOR_STORE_PATH", "./.vectorstore")
    )
except Exception as e:
    logger.error("Failed to initialize vector store", exc_info=e)
    vector_store = None

try:
    logger.debug("Initializing LLM handler")
    llm_handler = LLMHandler()
except ValueError as e:
    logger.error(str(e))
    llm_handler = None
except Exception as e:
    logger.error("Failed to initialize LLM handler", exc_info=e)
    llm_handler = None

@click.group()
@click.version_option(version=__version__)
def cli():
    """CLI Knowledge Base Agent - Index and query your codebase with natural language."""
    pass

@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    help="Path to the code directory to ingest",
)
@click.option(
    "--docs",
    "-d",
    type=click.Path(exists=True),
    help="Path to the documentation directory to ingest",
)
def ingest(path: Optional[str], docs: Optional[str]):
    """Ingest code and documentation into the knowledge base."""
    if not path and not docs:
        logger.error("No input paths provided")
        return
    
    if not vector_store:
        logger.error("Vector store not initialized")
        return

    with logger.section("Ingestion", "Starting ingestion process..."):
        # Process code files
        if path:
            logger.info("Processing code files")
            code_chunks = ingest_directory(Path(path), is_code=True)
            vector_store.add_documents(code_chunks)

        # Process documentation files
        if docs:
            logger.info("Processing documentation files")
            doc_chunks = ingest_directory(Path(docs), is_code=False)
            vector_store.add_documents(doc_chunks)

        # Show stats
        stats = vector_store.get_collection_stats()
        logger.success(
            f"Ingestion complete\n"
            f"Total chunks in vector store: {stats['total_chunks']}\n"
            f"Vector store location: {stats['persist_directory']}"
        )

@cli.command()
@click.argument("query")
def search(query: str):
    """Search the knowledge base with a natural language query."""
    if not vector_store:
        logger.error("Vector store not initialized")
        return
    
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return

    with logger.section("Search", "Processing query..."):
        # Search for relevant chunks
        results = vector_store.similarity_search(query)
        if not results:
            logger.warning("No relevant documents found in the knowledge base")
            return

        # Generate response
        logger.info("Generating response")
        response = llm_handler.generate_response(query, results)
        
        # Display response in a nicely formatted panel
        logger.section("Answer", "")
        # Try to parse the response as markdown for better formatting
        try:
            markdown = Markdown(response)
            logger.console.print(Panel(
                markdown,
                title="Claude's Response",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True
            ))
        except Exception:
            # Fallback to plain text if markdown parsing fails
            logger.console.print(Panel(
                response,
                title="Claude's Response",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True
            ))

if __name__ == "__main__":
    cli() 