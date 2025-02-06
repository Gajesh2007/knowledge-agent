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