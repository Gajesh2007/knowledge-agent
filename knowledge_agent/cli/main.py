"""
CLI Knowledge Base Agent - Main CLI interface
"""
import os
from pathlib import Path
from typing import Optional
import cmd

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.theme import Theme
from rich.syntax import Syntax
from rich.prompt import Prompt
from knowledge_agent import __version__
from knowledge_agent.core.ingestion import ingest_directory
from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.vector_store import VectorStore
from knowledge_agent.core.logging import logger
from knowledge_agent.core.role_prompts import get_role_prompt, DEFAULT_ROLE

# Load environment variables
load_dotenv()

# Set HuggingFace tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

# Configure Rich theming
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "code": "green",
    "heading": "blue bold",
    "bullet": "yellow",
    "role": "magenta",
    "prompt": "cyan bold",
})

console = Console(theme=custom_theme)

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

def format_response(response: str, role: str) -> str:
    """Format the response with proper styling based on role.
    
    Args:
        response: The raw response text
        role: The current role (beginner, engineer, or bd)
    
    Returns:
        Formatted response text with proper styling
    """
    # Split the response into sections
    sections = response.split("\n\n")
    formatted_sections = []
    
    for section in sections:
        # Format code blocks with syntax highlighting
        if section.startswith("```"):
            lang = section.split("\n")[0][3:]
            code = "\n".join(section.split("\n")[1:-1])
            syntax = Syntax(code, lang or "python", theme="monokai", line_numbers=True)
            formatted_sections.append(str(syntax))
        
        # Format summaries with bullets
        elif any(marker in section.lower() for marker in ["quick summary", "technical summary", "business summary"]):
            title = section.split("\n")[0]
            bullets = section.split("\n")[1:]
            formatted_sections.append(f"[heading]{title}[/heading]")
            for bullet in bullets:
                if bullet.strip():
                    formatted_sections.append(f"[bullet]{bullet}[/bullet]")
        
        # Format step-by-step sections for beginners
        elif role == "beginner" and "Step " in section:
            steps = section.split("\n")
            for step in steps:
                if step.strip():
                    if step.startswith("Step"):
                        formatted_sections.append(f"[heading]{step}[/heading]")
                    else:
                        formatted_sections.append(step)
        
        # Default formatting
        else:
            formatted_sections.append(section)
    
    return "\n\n".join(formatted_sections)

class InteractiveSession(cmd.Cmd):
    """Interactive CLI session for the knowledge base agent."""
    
    intro = """Welcome to the Knowledge Base Agent Interactive Session!
Type 'help' for a list of commands, or start asking questions directly.
Use Ctrl+D or 'exit' to end the session."""
    
    def __init__(self, role: str = DEFAULT_ROLE):
        """Initialize the interactive session.
        
        Args:
            role: The role to use for responses
        """
        super().__init__()
        self.role = role
        self.prompt = f"[prompt]{role}>[/prompt] "
        self.console = console
        
        # Print role-specific welcome message
        role_msg = {
            "beginner": "I'll explain things in simple terms with step-by-step instructions.",
            "engineer": "I'll provide technical details and implementation insights.",
            "bd": "I'll focus on business impact and high-level functionality."
        }
        self.console.print(f"\n[info]Using {role} mode: {role_msg[role]}[/info]\n")
    
    def default(self, query: str) -> None:
        """Handle queries that don't match any command."""
        if not query.strip():
            return
        
        if not vector_store or not llm_handler:
            self.console.print("[error]System not properly initialized. Please check your configuration.[/error]")
            return
        
        with logger.section("Search", "Processing query..."):
            # Search for relevant chunks
            results = vector_store.similarity_search(query)
            if not results:
                self.console.print("[warning]No relevant documents found in the knowledge base[/warning]")
                return
            
            # Get role-specific prompt
            role_prompt = get_role_prompt(self.role)
            
            # Generate and display response
            response = llm_handler.generate_response(query, results, role_prompt)
            formatted_response = format_response(response, self.role)
            
            try:
                markdown = Markdown(formatted_response)
                self.console.print(Panel(
                    markdown,
                    title=f"[role]Response ({self.role.title()})[/role]",
                    title_align="left",
                    border_style="cyan",
                    padding=(1, 2),
                    expand=True
                ))
            except Exception:
                self.console.print(Panel(
                    formatted_response,
                    title=f"[role]Response ({self.role.title()})[/role]",
                    title_align="left",
                    border_style="cyan",
                    padding=(1, 2),
                    expand=True
                ))
    
    def do_role(self, arg: str) -> None:
        """Change the current role (beginner, engineer, bd)."""
        if not arg:
            self.console.print(f"[info]Current role: {self.role}[/info]")
            return
        
        try:
            role = arg.lower()
            get_role_prompt(role)  # Validate role
            self.role = role
            self.prompt = f"[prompt]{role}>[/prompt] "
            self.console.print(f"[success]Switched to {role} mode[/success]")
        except ValueError as e:
            self.console.print(f"[error]{str(e)}[/error]")
    
    def do_clear(self, arg: str) -> None:
        """Clear the conversation history."""
        llm_handler.clear_memory()
        self.console.print("[info]Conversation history cleared[/info]")
    
    def do_exit(self, arg: str) -> bool:
        """Exit the interactive session."""
        return True
    
    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D to exit."""
        print()  # Add newline
        return True

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
@click.option(
    "--role",
    "-r",
    type=click.Choice(["beginner", "engineer", "bd"], case_sensitive=False),
    default=lambda: os.getenv("DEFAULT_ROLE", DEFAULT_ROLE),
    help="Role-based expertise mode for tailored responses",
)
def search(query: str, role: str):
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

        # Get role-specific prompt
        role_prompt = get_role_prompt(role)
        logger.debug(f"Using role: {role}")

        # Generate response
        logger.info("Generating response")
        response = llm_handler.generate_response(query, results, role_prompt)
        
        # Format the response with proper styling
        formatted_response = format_response(response, role)
        
        # Display response in a nicely formatted panel
        logger.section("Answer", "")
        try:
            # Try to parse as markdown first
            markdown = Markdown(formatted_response)
            console.print(Panel(
                markdown,
                title=f"[role]Response ({role.title()})[/role]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True
            ))
        except Exception:
            # Fallback to plain text with our custom formatting
            console.print(Panel(
                formatted_response,
                title=f"[role]Response ({role.title()})[/role]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True
            ))

@cli.command()
@click.option(
    "--role",
    "-r",
    type=click.Choice(["beginner", "engineer", "bd"], case_sensitive=False),
    default=lambda: os.getenv("DEFAULT_ROLE", DEFAULT_ROLE),
    help="Role-based expertise mode for tailored responses",
)
def session(role: str):
    """Start an interactive session with the knowledge base agent."""
    if not vector_store:
        logger.error("Vector store not initialized")
        return
    
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    InteractiveSession(role=role).cmdloop()

if __name__ == "__main__":
    cli() 