# CLI Knowledge Base Agent

A command-line tool that ingests code in multiple languages (Go, Python, Rust, Solidity) and related documentation, indexes them using LangChain's built-in tools for chunking and vector storage, and allows users to ask questions and retrieve context-based answers via a CLI interface.

## Features

- Code and documentation ingestion for multiple languages
- Vector-based semantic search using LangChain
- Natural language Q&A powered by Claude (Anthropic)
- Local vector storage using ChromaDB
- Command-line interface for easy interaction

## Installation

1. Make sure you have Python 3.9+ installed
2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/knowledge-agent.git
   cd knowledge-agent
   ```
4. Install dependencies:
   ```bash
   poetry install
   ```
5. Copy the example environment file and configure your settings:
   ```bash
   cp .env.example .env
   ```
6. Edit `.env` and add your Anthropic API key

## Usage

1. Ingest code and documentation:
   ```bash
   knowledge-agent ingest --path /path/to/code --docs /path/to/docs
   ```

2. Query the knowledge base:
   ```bash
   knowledge-agent search "How does function X work?"
   ```

## Development

- Run tests:
  ```bash
  poetry run pytest
  ```

- Format code:
  ```bash
  poetry run black .
  ```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 