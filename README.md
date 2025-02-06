# Knowledge Agent

A powerful CLI tool for understanding codebases through natural language queries, featuring role-based explanations and advanced code analysis capabilities. Built with LangChain and Claude.

## Key Features

- 🤖 **Intelligent Code Understanding**
  - Natural language queries about your codebase
  - Semantic code search and analysis
  - Multi-language support with tree-sitter
  - Advanced code parsing and documentation generation

- 🎭 **Role-Based Interactions**
  - Customizable explanation styles for different roles
  - Dynamic role switching during sessions
  - Persistent conversation memory
  - Context-aware responses

- 🔍 **Advanced Search & Analysis**
  - Semantic code search with ChromaDB
  - Code structure analysis
  - Documentation parsing (Markdown, RST, PDF)
  - Version control integration

- 📚 **Documentation Features**
  - Automated documentation generation
  - Architecture diagram creation
  - MkDocs integration with Material theme
  - API documentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gajesh2007/knowledge-agent.git
cd knowledge-agent
```

2. Install using Poetry (Python 3.9+):
```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with required settings:
# - ANTHROPIC_API_KEY: Your Claude API key
# - Other optional configurations
```

## Quick Start

1. **Ingest your codebase**:
```bash
# Local codebase
knowledge-agent ingest --path ./your/code

# From GitHub
knowledge-agent fetch --repo https://github.com/user/repo --ingest
```

2. **Query your codebase**:
```bash
# One-off query
knowledge-agent search "How does the error handling work?"

# Interactive session
knowledge-agent session
```

3. **Generate documentation**:
```bash
knowledge-agent docs generate ./src
```

## Advanced Usage

### Role-Based Queries

```bash
# Specify role for targeted explanations
knowledge-agent search --role engineer "How is authentication implemented?"
knowledge-agent search --role beginner "Explain the basic architecture"
```

### Interactive Sessions

```bash
# Start session with specific role
knowledge-agent session --role architect

# Available commands in session:
# - /role <role>: Switch roles
# - /clear: Clear conversation history
# - /exit: End session
```

### Documentation Generation

```bash
# Generate with specific template
knowledge-agent docs generate ./src --template api

# Include architecture diagram
knowledge-agent docs generate ./src --with-diagram

# Custom output
knowledge-agent docs generate ./src --output ./custom-docs
```

### Version Control Integration

```bash
# Analyze specific version
knowledge-agent version analyze --repo ./repo --ref v1.0.0

# Compare versions
knowledge-agent version diff --repo ./repo --old v1.0.0 --new v2.0.0
```

## Project Structure

```
knowledge-agent/
├── knowledge_agent/
│   ├── core/                # Core functionality
│   │   ├── llm.py          # LLM integration
│   │   ├── retrieval.py    # Search functionality
│   │   ├── code_parser.py  # Code analysis
│   │   ├── doc_parser.py   # Documentation parsing
│   │   └── ...
│   └── cli/                # CLI interface
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest
pytest --cov=knowledge_agent

# Code quality
black knowledge_agent tests
isort knowledge_agent tests
flake8 knowledge_agent tests
mypy knowledge_agent
```

## Dependencies

- Python 3.9+
- Key packages:
  - LangChain 0.3.17
  - Anthropic Claude SDK 0.45.2
  - ChromaDB 0.6.3
  - Tree-sitter for code parsing
  - MkDocs with Material theme
  - Various documentation parsers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and run tests
4. Submit a pull request

Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## License

MIT License - See [LICENSE](LICENSE) file for details. 