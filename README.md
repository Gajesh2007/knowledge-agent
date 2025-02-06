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

The knowledge agent now includes advanced retrieval capabilities that enhance search results through:

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

The knowledge agent now includes powerful documentation generation capabilities:

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
git clone https://github.com/yourusername/knowledge-agent.git
cd knowledge-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

1. **Ingest your codebase**:
```bash
knowledge-agent ingest --path ./your/code/path --docs ./your/docs/path
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

## Configuration

The following environment variables can be configured in `.env`:

- `ANTHROPIC_API_KEY`: Your Claude API key
- `VECTOR_STORE_PATH`: Path to store the vector database
- `DEFAULT_ROLE`: Default role for responses (beginner/engineer/bd)
- `LOG_LEVEL`: Logging verbosity (INFO/DEBUG/WARNING)

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 