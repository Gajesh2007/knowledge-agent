"""
CLI Knowledge Base Agent - Main CLI interface
"""
import os
from pathlib import Path
from typing import Optional, List
import cmd
from datetime import datetime

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.theme import Theme
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.table import Table
from knowledge_agent import __version__
from knowledge_agent.core.ingestion import ingest_directory, SUPPORTED_CODE_EXTENSIONS, SUPPORTED_DOC_EXTENSIONS
from knowledge_agent.core.vector_store import VectorStore
from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.logging import logger
from knowledge_agent.core.role_prompts import get_role_prompt, DEFAULT_ROLE
from knowledge_agent.core.repo_manager import RepoManager
from knowledge_agent.core.code_analysis import CodeAnalyzer
from knowledge_agent.core.conversation import ConversationMemory
from knowledge_agent.core.version_control import VersionManager
from knowledge_agent.core.role_manager import RoleManager
from knowledge_agent.core.retrieval import SearchResult
import git

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
    
    def __init__(self, role: str = DEFAULT_ROLE, session_id: Optional[str] = None):
        """Initialize the interactive session.
        
        Args:
            role: The role to use for responses
            session_id: Optional session ID to resume
        """
        super().__init__()
        self.role = role
        self.prompt = f"[prompt]{role}>[/prompt] "
        self.console = console
        
        # Initialize conversation memory
        self.memory = ConversationMemory(
            llm_handler=llm_handler,
            session_id=session_id
        )
        
        # Print role-specific welcome message
        role_msg = {
            "beginner": "I'll explain things in simple terms with step-by-step instructions.",
            "engineer": "I'll provide technical details and implementation insights.",
            "bd": "I'll focus on business impact and high-level functionality."
        }
        self.console.print(f"\n[info]Using {role} mode: {role_msg[role]}[/info]\n")
        
        if session_id:
            self.console.print(f"[info]Resumed session: {session_id}[/info]\n")
    
    def default(self, query: str) -> None:
        """Handle queries that don't match any command."""
        if not query.strip():
            return
        
        if not vector_store or not llm_handler:
            self.console.print("[error]System not properly initialized. Please check your configuration.[/error]")
            return
        
        with logger.section("Search", "Processing query..."):
            # Add user's query to memory
            self.memory.add_message('user', query)
            
            # Search for relevant chunks
            results = vector_store.similarity_search(query)
            if not results:
                self.console.print("[warning]No relevant documents found in the knowledge base[/warning]")
                return
            
            # Get role-specific prompt and conversation context
            role_prompt = get_role_prompt(self.role)
            context = self.memory.get_context()
            
            # Generate and display response
            response = llm_handler.generate_response(
                query,
                results,
                role_prompt,
                conversation_context=context
            )
            
            # Add response to memory
            self.memory.add_message('assistant', response)
            
            # Display formatted response
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
            
            # Add role change to memory
            self.memory.add_message(
                'system',
                f"Role changed to {role}",
                metadata={'type': 'role_change', 'role': role}
            )
        except ValueError as e:
            self.console.print(f"[error]{str(e)}[/error]")
    
    def do_clear(self, arg: str) -> None:
        """Clear the conversation history."""
        self.memory = ConversationMemory(llm_handler=llm_handler)
        self.console.print("[info]Conversation history cleared[/info]")
    
    def do_sessions(self, arg: str) -> None:
        """List available conversation sessions."""
        sessions = ConversationMemory.list_sessions()
        
        if not sessions:
            self.console.print("[info]No saved sessions found[/info]")
            return
        
        table = Table(title="Saved Sessions")
        table.add_column("Session ID")
        table.add_column("Last Modified")
        
        for session_id, timestamp in sessions:
            table.add_row(
                session_id,
                datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            )
        
        self.console.print(table)
    
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
@click.option('--repo', multiple=True, help='GitHub repository URLs to fetch (can specify multiple)')
@click.option('--from-file', type=click.Path(exists=True, dir_okay=False), help='Text file containing "repository_url branch" pairs (one per line)')
@click.option('--branch', default='master', help='Default branch to fetch (used when not specified per repository)')
@click.option('--ingest', is_flag=True, help='Ingest the repositories after fetching')
@click.option('--exclude', multiple=True, help='Patterns to exclude from ingestion')
def fetch(repo: tuple, from_file: str, branch: str = 'master', ingest: bool = False, exclude: tuple = ()):
    """Fetch and optionally ingest GitHub repositories."""
    repositories = [(r, branch) for r in repo]  # Convert tuple to list with default branch
    
    # Read additional repos from file if provided
    if from_file:
        try:
            with open(from_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        repo_url, repo_branch = parts[0], parts[1]
                    else:
                        repo_url, repo_branch = parts[0], branch  # Use default branch if not specified
                    repositories.append((repo_url, repo_branch))
        except Exception as e:
            console.print(f"[error]Failed to read repositories from file: {str(e)}[/error]")
            return

    if not repositories:
        console.print("[error]Please specify at least one repository URL (via --repo or --from-file)[/error]")
        return

    repo_manager = RepoManager()
    
    # Show summary of what will be processed
    console.print(f"\n[info]Will process {len(repositories)} repositories:[/info]")
    for repo_url, repo_branch in repositories:
        console.print(f"  • {repo_url} (branch: {repo_branch})")
    console.print()
    
    for repository_url, repository_branch in repositories:
        try:
            # Extract repo name from URL
            repo_name = repository_url.split('/')[-1].replace('.git', '')
            logger.info(f"Updating repository {repo_name} (branch: {repository_branch})")
            
            # Fetch/clone the repository
            repo_path = repo_manager.fetch_repo(repository_url, repository_branch)
            
            if ingest:
                logger.info("Ingesting code files...")
                try:
                    count = ingest_directory(repo_path, vector_store, exclude_patterns=exclude)
                    console.print(f"[success]Successfully ingested {count} chunks from {repo_name}[/success]")
                except Exception as e:
                    console.print(f"[error]Failed to ingest repository {repo_name}: {str(e)}[/error]")
            
            console.print(f"[success]Successfully processed {repo_name}[/success]")
            
        except git.exc.GitCommandError as e:
            console.print(f"[error]Git error for {repository_url}: {str(e)}[/error]")
        except Exception as e:
            console.print(f"[error]Error processing {repository_url}: {str(e)}[/error]")
            continue

    if ingest:
        # Show final stats
        stats = vector_store.get_collection_stats()
        console.print(f"\n[info]Total chunks in vector store: {stats['total_chunks']}[/info]")

@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--exclude', multiple=True, help='Patterns to exclude from ingestion')
def ingest(path: str, exclude: tuple = ()):
    """Ingest files from a directory."""
    try:
        vector_store = VectorStore()
        directory = Path(path)
        
        # List supported file types
        logger.info("Supported code file types:")
        for ext, lang in SUPPORTED_CODE_EXTENSIONS.items():
            logger.info(f"  {ext}: {lang}")
        logger.info("\nSupported documentation file types:")
        for ext, doc_type in SUPPORTED_DOC_EXTENSIONS.items():
            logger.info(f"  {ext}: {doc_type}")
        
        # Ingest files
        chunks = ingest_directory(directory, vector_store, exclude_patterns=set(exclude))
        logger.info(f"Created {chunks} chunks")
    except Exception as e:
        logger.error(f"Failed to ingest directory: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('query')
@click.option('--limit', default=5, help='Maximum number of results to return')
def search(query: str, limit: int = 5):
    """Search the vector store."""
    try:
        vector_store = VectorStore()
        results = vector_store.search(query, limit=limit)
        
        if not results:
            logger.info("No results found")
            return
        
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Score: {result.score:.3f}")
            logger.info(f"File: {result.metadata.get('file_path', 'Unknown')}")
            if result.metadata.get('type') == 'code':
                logger.info(f"Language: {result.metadata.get('language', 'Unknown')}")
                logger.info(f"Entity: {result.metadata.get('name', 'Unknown')} ({result.metadata.get('entity_type', 'Unknown')})")
                logger.info(f"Lines: {result.metadata.get('start_line', '?')}-{result.metadata.get('end_line', '?')}")
            elif result.metadata.get('type') == 'documentation':
                logger.info(f"Title: {result.metadata.get('title', 'Unknown')}")
                if result.metadata.get('line_number'):
                    logger.info(f"Line: {result.metadata['line_number']}")
            logger.info("\nContent:")
            logger.info(result.content)
    except Exception as e:
        logger.error(f"Failed to search: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option(
    "--role",
    "-r",
    type=click.Choice(["beginner", "engineer", "bd"], case_sensitive=False),
    default=lambda: os.getenv("DEFAULT_ROLE", DEFAULT_ROLE),
    help="Role-based expertise mode for tailored responses",
)
@click.option(
    "--resume",
    help="Resume a previous session by ID",
)
def session(role: str, resume: Optional[str]):
    """Start an interactive session with the knowledge base agent."""
    if not vector_store:
        logger.error("Vector store not initialized")
        return
    
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    InteractiveSession(role=role, session_id=resume).cmdloop()

@cli.group()
def analyze():
    """Code analysis commands."""
    pass

@analyze.command()
@click.argument("path", type=click.Path(exists=True))
def dependencies(path: str):
    """Analyze code dependencies in a directory."""
    analyzer = CodeAnalyzer(llm_handler)
    
    with logger.section("Analysis", "Analyzing code dependencies..."):
        analyzer.analyze_codebase(Path(path))
        
        # Display dependency graph stats
        stats_table = Table(title="Dependency Analysis")
        stats_table.add_column("Entity Type")
        stats_table.add_column("Count")
        
        type_counts = {}
        for node in analyzer.dependency_graph.nodes.values():
            type_counts[node.type] = type_counts.get(node.type, 0) + 1
        
        for type_name, count in type_counts.items():
            stats_table.add_row(type_name, str(count))
        
        console.print(stats_table)

@analyze.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--name",
    "-n",
    help="Name of the entity to analyze",
    required=True
)
@click.option(
    "--transitive",
    "-t",
    is_flag=True,
    help="Include transitive dependencies",
)
def deps(path: str, name: str, transitive: bool):
    """Show dependencies for a specific entity."""
    analyzer = CodeAnalyzer(llm_handler)
    
    with logger.section("Analysis", f"Analyzing dependencies for {name}..."):
        analyzer.analyze_codebase(Path(path))
        deps = analyzer.find_dependencies(name, include_transitive=transitive)
        
        # Display results
        table = Table(title=f"Dependencies for {name}")
        table.add_column("Type")
        table.add_column("Entities")
        
        if deps['dependencies']:
            table.add_row(
                "Depends On",
                "\n".join(sorted(deps['dependencies']))
            )
        
        if deps['dependents']:
            table.add_row(
                "Used By",
                "\n".join(sorted(deps['dependents']))
            )
        
        console.print(table)

@analyze.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--name",
    "-n",
    help="Name of the entity to summarize",
    required=True
)
def summarize(path: str, name: str):
    """Generate a semantic summary for a code entity."""
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    analyzer = CodeAnalyzer(llm_handler)
    
    with logger.section("Analysis", f"Generating summary for {name}..."):
        analyzer.analyze_codebase(Path(path))
        
        # Find the entity
        entity = None
        for node in analyzer.dependency_graph.nodes.values():
            if node.name == name:
                entity = node
                break
        
        if not entity:
            logger.error(f"Entity '{name}' not found in codebase")
            return
        
        # Generate and display summary
        summary = analyzer.get_entity_summary(entity)
        
        try:
            markdown = Markdown(summary)
            console.print(Panel(
                markdown,
                title="[heading]Code Summary[/heading]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True
            ))
        except Exception:
            console.print(Panel(
                summary,
                title="[heading]Code Summary[/heading]",
                title_align="left",
                border_style="cyan",
                padding=(1, 2),
                expand=True
            ))

@analyze.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--focus",
    "-f",
    multiple=True,
    type=click.Choice(['style', 'performance', 'security', 'refactor']),
    help="Focus on specific types of suggestions",
)
def suggest(path: str, focus: List[str]):
    """Generate code improvement suggestions."""
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    from knowledge_agent.core.suggestions import SuggestionEngine
    suggestion_engine = SuggestionEngine(llm_handler)
    
    with logger.section("Analysis", "Generating code suggestions..."):
        # Convert focus areas to set
        focus_areas = set(focus) if focus else None
        
        # Generate suggestions
        suggestions_by_file = suggestion_engine.analyze_codebase(
            Path(path),
            focus_areas=focus_areas
        )
        
        if not suggestions_by_file:
            logger.info("No suggestions found")
            return
        
        # Display suggestions grouped by file
        for file_path, suggestions in suggestions_by_file.items():
            console.print(f"\n[heading]Suggestions for {file_path}[/heading]")
            
            # Group suggestions by priority
            by_priority = {
                'high': [],
                'medium': [],
                'low': []
            }
            
            for suggestion in suggestions:
                by_priority[suggestion.priority].append(suggestion)
            
            # Display suggestions by priority
            for priority in ['high', 'medium', 'low']:
                if by_priority[priority]:
                    priority_style = {
                        'high': 'red',
                        'medium': 'yellow',
                        'low': 'green'
                    }[priority]
                    
                    console.print(f"\n[{priority_style}]🔍 {priority.upper()} Priority Suggestions[/{priority_style}]")
                    
                    for suggestion in by_priority[priority]:
                        panel = Panel(
                            Markdown(
                                f"**Type:** {suggestion.type}\n"
                                f"**Description:** {suggestion.description}\n"
                                f"**Rationale:** {suggestion.rationale}\n"
                                f"**Impact:** {suggestion.impact}\n"
                                + (f"\n**Suggested Changes:**\n```python\n{suggestion.suggested_code}```"
                                   if suggestion.suggested_code else "")
                            ),
                            title=f"[code]{suggestion.entity_name}[/code]",
                            border_style=priority_style
                        )
                        console.print(panel)

@analyze.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--name",
    "-n",
    help="Name of the entity to refactor",
    required=True
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(['style', 'performance', 'security', 'refactor']),
    help="Type of refactoring to focus on",
)
def refactor(path: str, name: str, type: Optional[str]):
    """Generate refactoring suggestions for a specific entity."""
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    from knowledge_agent.core.suggestions import SuggestionEngine
    suggestion_engine = SuggestionEngine(llm_handler)
    
    with logger.section("Analysis", f"Analyzing {name} for refactoring..."):
        # First analyze the codebase to get the dependency graph
        suggestion_engine.analyzer.analyze_codebase(Path(path))
        
        # Find the target entity
        target_node = None
        for node in suggestion_engine.analyzer.dependency_graph.nodes.values():
            if node.name == name:
                target_node = node
                break
        
        if not target_node:
            logger.error(f"Entity '{name}' not found in codebase")
            return
        
        # Create a CodeEntity from the node
        entity = CodeEntity(
            name=target_node.name,
            type=target_node.type,
            docstring=None,
            code=suggestion_engine._get_entity_code(target_node),
            start_line=0,
            end_line=0,
            parent=None,
            dependencies=target_node.outgoing,
            metadata={'path': target_node.file_path}
        )
        
        # Generate suggestions
        suggestions = suggestion_engine.analyze_entity(entity)
        
        # Filter by type if specified
        if type:
            suggestions = [s for s in suggestions if s.type == type]
        
        if not suggestions:
            logger.info("No refactoring suggestions found")
            return
        
        # Display suggestions
        console.print(f"\n[heading]Refactoring Suggestions for {name}[/heading]")
        
        for suggestion in suggestions:
            priority_style = {
                'high': 'red',
                'medium': 'yellow',
                'low': 'green'
            }[suggestion.priority]
            
            panel = Panel(
                Markdown(
                    f"**Type:** {suggestion.type}\n"
                    f"**Priority:** {suggestion.priority.upper()}\n"
                    f"**Description:** {suggestion.description}\n"
                    f"**Rationale:** {suggestion.rationale}\n"
                    f"**Impact:** {suggestion.impact}\n"
                    + (f"\n**Suggested Changes:**\n```python\n{suggestion.suggested_code}```"
                       if suggestion.suggested_code else "")
                ),
                title=f"[code]{suggestion.entity_name}[/code]",
                border_style=priority_style
            )
            console.print(panel)
            
            # Ask if user wants to apply the suggestion
            if suggestion.suggested_code and Prompt.ask(
                "\nWould you like to apply this suggestion?",
                choices=["yes", "no"],
                default="no"
            ) == "yes":
                try:
                    # Create a backup first
                    import shutil
                    backup_path = Path(suggestion.file_path + ".bak")
                    shutil.copy2(suggestion.file_path, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                    
                    # Apply the change
                    with open(suggestion.file_path, 'r') as f:
                        content = f.read()
                    
                    # Replace the old code with the new code
                    # Note: This is a simple replacement. A more robust solution would
                    # use the AST to make precise changes.
                    new_content = content.replace(suggestion.current_code, suggestion.suggested_code)
                    
                    with open(suggestion.file_path, 'w') as f:
                        f.write(new_content)
                    
                    logger.success("Applied refactoring suggestion")
                    
                except Exception as e:
                    logger.error(f"Failed to apply suggestion: {str(e)}")
                    if backup_path.exists():
                        shutil.copy2(backup_path, suggestion.file_path)
                        logger.info("Restored from backup")
                finally:
                    if backup_path.exists():
                        backup_path.unlink()

@cli.group()
def version():
    """Version control commands."""
    pass

@version.command()
@click.argument("path", type=click.Path(exists=True))
def list(path: str):
    """List available versions (branches and tags)."""
    version_manager = VersionManager(Path(path))
    
    versions = version_manager.get_available_versions()
    if not versions:
        logger.warning("No versions found")
        return
    
    table = Table(title="Available Versions")
    table.add_column("Reference")
    table.add_column("Commit Hash")
    table.add_column("Author")
    table.add_column("Date")
    table.add_column("Message")
    
    for ref, meta in versions:
        table.add_row(
            ref,
            meta.commit_hash[:8],
            meta.author,
            meta.commit_date.strftime('%Y-%m-%d %H:%M:%S'),
            meta.message.split('\n')[0]  # First line of commit message
        )
    
    console.print(table)

@version.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--old",
    "-o",
    required=True,
    help="Old version reference (commit, branch, tag)",
)
@click.option(
    "--new",
    "-n",
    required=True,
    help="New version reference (commit, branch, tag)",
)
def diff(path: str, old: str, new: str):
    """Show changes between two versions."""
    version_manager = VersionManager(Path(path), llm_handler)
    
    with logger.section("Diff", f"Comparing {old} to {new}..."):
        # Get changed files
        changes = version_manager.get_changed_files(old, new)
        
        # Display changes
        table = Table(title=f"Changes from {old} to {new}")
        table.add_column("File")
        table.add_column("Type")
        table.add_column("Changes")
        
        for change in changes:
            table.add_row(
                change.file_path,
                change.change_type,
                f"+{change.additions} -{change.deletions}"
            )
        
        console.print(table)
        
        # Generate and display summary
        summary = version_manager.generate_diff_summary(old, new)
        console.print(Panel(
            Markdown(summary),
            title="[heading]Change Summary[/heading]",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
            expand=True
        ))

@version.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--ref",
    "-r",
    required=True,
    help="Version reference to ingest (commit, branch, tag)",
)
@click.option(
    "--docs/--no-docs",
    default=True,
    help="Whether to also ingest documentation",
)
def ingest(path: str, ref: str, docs: bool):
    """Ingest code at a specific version."""
    if not vector_store:
        logger.error("Vector store not initialized")
        return
    
    with logger.section("Ingestion", f"Ingesting version {ref}..."):
        # Ingest code
        logger.info("Processing code files")
        code_chunks = ingest_directory(Path(path), is_code=True, ref=ref)
        vector_store.add_documents(code_chunks)
        
        # Ingest docs if requested
        if docs:
            logger.info("Processing documentation files")
            docs_path = Path(path) / "docs"
            if docs_path.exists():
                doc_chunks = ingest_directory(docs_path, is_code=False, ref=ref)
                vector_store.add_documents(doc_chunks)
            else:
                logger.warning("No documentation directory found")
        
        # Show stats
        stats = vector_store.get_collection_stats()
        logger.success(
            f"Ingestion complete\n"
            f"Total chunks in vector store: {stats['total_chunks']}\n"
            f"Vector store location: {stats['persist_directory']}"
        )

@cli.group()
def role():
    """Role management commands."""
    pass

@role.command()
def list():
    """List available roles."""
    role_manager = RoleManager()
    
    table = Table(title="Available Roles")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Description")
    table.add_column("Style Guidelines")
    
    for template in role_manager.list_roles():
        table.add_row(
            template.name,
            "Default" if template.name in role_manager.DEFAULT_ROLES else "Custom",
            template.description,
            "\n".join(f"{k}: {v}" for k, v in template.response_style.items())
        )
    
    console.print(table)

@role.command()
@click.argument("name")
@click.option(
    "--description",
    "-d",
    required=True,
    help="Role description",
)
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="System prompt for the role",
)
@click.option(
    "--style",
    "-s",
    multiple=True,
    help="Style guideline in format 'type:description'",
)
@click.option(
    "--metadata",
    "-m",
    multiple=True,
    help="Additional metadata in format 'key:value'",
)
def create(name: str, description: str, prompt: str, style: List[str], metadata: List[str]):
    """Create a new custom role."""
    role_manager = RoleManager()
    
    # Parse style guidelines
    response_style = {}
    for s in style:
        try:
            type_name, desc = s.split(":", 1)
            response_style[type_name.strip()] = desc.strip()
        except ValueError:
            logger.error(f"Invalid style format: {s}")
            return
    
    # Parse metadata
    meta = {}
    for m in metadata:
        try:
            key, value = m.split(":", 1)
            meta[key.strip()] = value.strip()
        except ValueError:
            logger.error(f"Invalid metadata format: {m}")
            return
    
    try:
        template = role_manager.create_role(
            name=name,
            description=description,
            system_prompt=prompt,
            response_style=response_style,
            metadata=meta
        )
        logger.success(f"Created custom role: {template.name}")
    except ValueError as e:
        logger.error(str(e))

@role.command()
@click.argument("name")
@click.option(
    "--description",
    "-d",
    help="New role description",
)
@click.option(
    "--prompt",
    "-p",
    help="New system prompt",
)
@click.option(
    "--style",
    "-s",
    multiple=True,
    help="New style guideline in format 'type:description'",
)
@click.option(
    "--metadata",
    "-m",
    multiple=True,
    help="New metadata in format 'key:value'",
)
def update(name: str, description: str, prompt: str, style: List[str], metadata: List[str]):
    """Update an existing custom role."""
    role_manager = RoleManager()
    
    # Parse style guidelines
    response_style = {}
    for s in style:
        try:
            type_name, desc = s.split(":", 1)
            response_style[type_name.strip()] = desc.strip()
        except ValueError:
            logger.error(f"Invalid style format: {s}")
            return
    
    # Parse metadata
    meta = {}
    for m in metadata:
        try:
            key, value = m.split(":", 1)
            meta[key.strip()] = value.strip()
        except ValueError:
            logger.error(f"Invalid metadata format: {m}")
            return
    
    try:
        template = role_manager.update_role(
            name=name,
            description=description,
            system_prompt=prompt,
            response_style=response_style if style else None,
            metadata=meta if metadata else None
        )
        if template:
            logger.success(f"Updated custom role: {template.name}")
        else:
            logger.error(f"Role not found: {name}")
    except ValueError as e:
        logger.error(str(e))

@role.command()
@click.argument("name")
def delete(name: str):
    """Delete a custom role."""
    role_manager = RoleManager()
    
    try:
        if role_manager.delete_role(name):
            logger.success(f"Deleted custom role: {name}")
        else:
            logger.error(f"Role not found: {name}")
    except ValueError as e:
        logger.error(str(e))

@role.command()
@click.argument("name")
def show(name: str):
    """Show details of a role."""
    role_manager = RoleManager()
    
    template = role_manager.get_role(name)
    if not template:
        logger.error(f"Role not found: {name}")
        return
    
    # Display role details
    console.print(Panel(
        Markdown(f"""
# {template.name.title()} Role

**Type:** {"Default" if name in role_manager.DEFAULT_ROLES else "Custom"}
**Description:** {template.description}

## System Prompt
```
{template.system_prompt}
```

## Style Guidelines
{chr(10).join(f'- **{k}:** {v}' for k, v in template.response_style.items())}

## Metadata
{chr(10).join(f'- **{k}:** {v}' for k, v in template.metadata.items())}
"""),
        title="[heading]Role Details[/heading]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=True
    ))

@cli.group()
def docs():
    """Documentation generation commands."""
    pass

@docs.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--template",
    "-t",
    type=click.Choice(["module", "api"]),
    default="module",
    help="Documentation template to use",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./docs",
    help="Output directory for documentation",
)
@click.option(
    "--diagram/--no-diagram",
    default=True,
    help="Generate architecture diagram",
)
def generate(path: str, template: str, output: str, diagram: bool):
    """Generate documentation for a codebase."""
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    from knowledge_agent.core.documentation import DocumentationGenerator
    doc_generator = DocumentationGenerator(llm_handler, output_dir=output)
    
    with logger.section("Documentation", "Generating documentation..."):
        try:
            # Generate documentation sections
            sections = doc_generator.generate_documentation(
                Path(path),
                template_name=template
            )
            
            # Generate architecture diagram if requested
            if diagram:
                logger.info("Generating architecture diagram")
                diagram_path = doc_generator.generate_architecture_diagram(Path(path))
                logger.info(f"Generated diagram at {diagram_path}")
            
            # Export to MkDocs
            logger.info("Exporting to MkDocs format")
            doc_generator.export_mkdocs(sections)
            
            logger.success(
                f"Documentation generated successfully\n"
                f"Output directory: {output}\n"
                f"To view: Run 'mkdocs serve' in the output directory"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate documentation: {str(e)}")

@docs.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--name",
    "-n",
    help="Name of the entity to document",
    required=True
)
def entity(path: str, name: str):
    """Generate documentation for a specific code entity."""
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    analyzer = CodeAnalyzer(llm_handler)
    
    with logger.section("Documentation", f"Generating documentation for {name}..."):
        analyzer.analyze_codebase(Path(path))
        
        # Find the entity
        entity = None
        for node in analyzer.dependency_graph.nodes.values():
            if node.name == name:
                entity = node
                break
        
        if not entity:
            logger.error(f"Entity '{name}' not found in codebase")
            return
        
        # Generate and display documentation
        summary = analyzer.get_entity_summary(entity)
        
        # Get dependencies
        deps = analyzer.find_dependencies(name)
        
        # Display documentation
        console.print(Panel(
            Markdown(
                f"# {name} ({entity.type})\n\n"
                f"{summary}\n\n"
                f"## Dependencies\n\n"
                f"**Depends on:**\n"
                + "\n".join(f"- {dep}" for dep in sorted(deps['dependencies']))
                + "\n\n**Used by:**\n"
                + "\n".join(f"- {dep}" for dep in sorted(deps['dependents']))
            ),
            title="[heading]Documentation[/heading]",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
            expand=True
        ))

@docs.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for diagram",
)
def diagram(path: str, output: Optional[str]):
    """Generate an architecture diagram for the codebase."""
    if not llm_handler:
        logger.error("LLM handler not initialized. Please check your ANTHROPIC_API_KEY.")
        return
    
    from knowledge_agent.core.documentation import DocumentationGenerator
    doc_generator = DocumentationGenerator(llm_handler)
    
    with logger.section("Documentation", "Generating architecture diagram..."):
        try:
            output_path = doc_generator.generate_architecture_diagram(
                Path(path),
                output_path=Path(output) if output else None
            )
            logger.success(f"Generated diagram at {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate diagram: {str(e)}")

if __name__ == "__main__":
    cli() 