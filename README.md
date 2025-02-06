# CLI Knowledge Base Agent

A powerful command-line tool that helps you understand your codebase through natural language queries. Features role-based explanations tailored for different expertise levels (beginner, engineer, business development).

## Features

- 🎯 **Role-Based Responses**: Get explanations tailored to your expertise level
  - Beginner: Step-by-step explanations with simple language
  - Engineer: Technical details and implementation insights
  - Business Development: High-level functionality and business impact

- 💬 **Interactive Sessions**: Start a conversation with your codebase
  - Multi-turn queries with conversation memory
  - Switch roles on the fly
  - Color-coded, formatted output

- 🔍 **Smart Search**: Uses semantic search to find relevant code and documentation
  - Supports both code and documentation ingestion
  - Maintains context between queries
  - Provides relevant code snippets and explanations

### Advanced Retrieval & Context

The knowledge agent includes advanced retrieval capabilities that enhance search results through:

- **Semantic Clustering**: Results are automatically grouped into semantic clusters for better organization
- **Context-Aware Scoring**: Search results are scored based on conversation context and relevance
- **Metadata Filtering**: Filter results by specific metadata attributes
- **Minimum Relevance Thresholds**: Set minimum relevance scores to ensure quality results

#### Advanced Search Commands

```bash
# Basic search with default settings
knowledge-agent search "How does the vector store work?"

# Search with metadata filtering
knowledge-agent search "security features" -f type=code -f language=python

# Show semantic clusters in results
knowledge-agent search "database operations" --show-clusters

# Set minimum relevance threshold
knowledge-agent search "error handling" --min-relevance 0.7

# Specify number of results
knowledge-agent search "API endpoints" --k 10

# Use basic search without advanced features
knowledge-agent search "quick lookup" --basic

# Combine with role-based responses
knowledge-agent search "architecture overview" --role architect --show-clusters
```

Each search result includes:
- Source file and type
- Content snippet
- Relevance score
- Context score (when applicable)
- Semantic cluster information (when enabled)

### Documentation Generation

The knowledge agent includes powerful documentation generation capabilities:

- **Automated Documentation**: Generate comprehensive documentation from your codebase
  - Multiple templates (module, API)
  - Architecture diagrams
  - MkDocs integration with Material theme
  - Code entity documentation

- **Documentation Commands**:
```bash
# Generate full documentation
knowledge-agent docs generate ./src

# Use specific template
knowledge-agent docs generate ./src -t api

# Custom output directory
knowledge-agent docs generate ./src -o ./custom-docs

# Generate without architecture diagram
knowledge-agent docs generate ./src --no-diagram

# Document specific entity
knowledge-agent docs entity ./src -n MyClass

# Generate architecture diagram only
knowledge-agent docs diagram ./src -o architecture.png
```

- **Documentation Features**:
  - Semantic analysis of code structure
  - Dependency tracking and visualization
  - Integration with version control
  - Customizable templates
  - Beautiful, searchable documentation site

- **Generated Documentation Includes**:
  - Overview and architecture
  - Installation and usage guides
  - API documentation
  - Code examples
  - Architecture diagrams
  - Dependency information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gajesh2007/knowledge-agent.git
cd knowledge-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies using Poetry:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

4. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
# Edit .env with your settings:
# - ANTHROPIC_API_KEY: Your Claude API key
# - VECTOR_STORE_PATH: Path to store the vector database (default: ./.vectorstore)
# - DEFAULT_ROLE: Default role for responses (default: engineer)
# - LOG_LEVEL: Logging verbosity (default: INFO)
```

## Usage

1. **Ingest your codebase**:
```bash
# Ingest local code and documentation
knowledge-agent ingest --path ./your/code/path --docs ./your/docs/path

# Ingest from GitHub repositories
knowledge-agent fetch --repo https://github.com/user/repo1 --ingest
knowledge-agent fetch -r repo1 -r repo2 -b main --ingest  # Multiple repos with branch

# Fetch without ingesting
knowledge-agent fetch --repo https://github.com/user/repo1
knowledge-agent ingest --path ./.repos/repo1  # Ingest later

# Clean cached repositories
knowledge-agent fetch --clean  # Remove all cached repos
```

2. **Ask questions (one-off)**:
```bash
# Use default role (engineer)
knowledge-agent search "How does the vector store work?"

# Specify a role
knowledge-agent search --role beginner "How does the vector store work?"
knowledge-agent search -r bd "What's the business impact of the vector store?"
```

3. **Start an interactive session**:
```bash
# Start with default role
knowledge-agent session

# Start with specific role
knowledge-agent session --role beginner
```

### Interactive Session Commands

- Ask questions directly by typing them
- `role <role>`: Switch to a different role (beginner/engineer/bd)
- `clear`: Clear conversation history
- `exit` or Ctrl+D: End the session

### Working with GitHub Repositories

The knowledge agent can directly fetch and analyze code from GitHub repositories:

1. **Fetch and Ingest in One Step**:
```bash
# Single repository
knowledge-agent fetch --repo https://github.com/user/repo1 --ingest

# Multiple repositories
knowledge-agent fetch -r repo1 -r repo2 -r repo3 --ingest

# Specific branch
knowledge-agent fetch --repo https://github.com/user/repo1 -b develop --ingest
```

2. **Fetch First, Ingest Later**:
```bash
# Fetch repositories
knowledge-agent fetch --repo https://github.com/user/repo1
knowledge-agent fetch -r repo2 -b main

# Ingest when ready
knowledge-agent ingest --path ./.repos/repo1
knowledge-agent ingest --path ./.repos/repo2
```

3. **Repository Management**:
```bash
# Clean specific repository
knowledge-agent fetch --repo https://github.com/user/repo1 --clean

# Clean all cached repositories
knowledge-agent fetch --clean

# Update existing repositories
knowledge-agent fetch --repo https://github.com/user/repo1 --update
```

4. **Version Control Integration**:
```bash
# List available versions
knowledge-agent version list ./.repos/repo1

# Compare versions
knowledge-agent version diff ./.repos/repo1 -o v1.0.0 -n v2.0.0

# Ingest specific version
knowledge-agent version ingest ./.repos/repo1 -r v1.0.0
```

## Development

1. **Setup Development Environment**:
```bash
# Clone the repository
git clone https://github.com/Gajesh2007/knowledge-agent.git
cd knowledge-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
poetry install --with dev
```

2. **Run Tests**:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_retrieval.py

# Run tests by marker
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests

# Run tests with coverage
pytest --cov=knowledge_agent
```

3. **Code Quality**:
```bash
# Format code
black knowledge_agent tests

# Sort imports
isort knowledge_agent tests

# Lint code
flake8 knowledge_agent tests
```

4. **Documentation**:
```bash
# Generate documentation
knowledge-agent docs generate ./knowledge_agent -o ./docs

# Serve documentation locally
cd docs && mkdocs serve
```

## Project Structure

```
knowledge-agent/
├── knowledge_agent/          # Main package
│   ├── core/                # Core functionality
│   │   ├── retrieval.py     # Advanced retrieval
│   │   ├── documentation.py # Documentation generation
│   │   └── ...
│   └── cli/                 # CLI interface
├── tests/                   # Test suite
├── docs/                    # Documentation
└── pyproject.toml          # Project configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Create a Pull Request

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding or modifying tests
- `refactor:` Code changes that neither fix bugs nor add features
- `style:` Changes that don't affect code meaning (formatting, etc)
- `chore:` Changes to build process or auxiliary tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.