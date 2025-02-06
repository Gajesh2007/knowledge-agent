"""LLM functionality for handling interactions with Claude."""
import os
from typing import List, Tuple

from anthropic import Anthropic

from knowledge_agent.core.logging import logger

DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that helps users understand code and documentation.
Your task is to provide accurate, helpful responses based on the context provided.
If you don't have enough information in the context to answer a question, say so.
Always cite the specific files and code snippets you reference in your answers."""

class LLMHandler:
    """Handles interactions with the Claude LLM."""

    def __init__(self, api_key: str = None):
        """Initialize the LLM handler.
        
        Args:
            api_key: Optional API key for Claude. If not provided, will look for ANTHROPIC_API_KEY in environment.
        """
        logger.debug("Initializing LLM handler")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment")
            raise ValueError("ANTHROPIC_API_KEY must be provided or set in environment")
        
        # Initialize the client
        try:
            logger.debug("Initializing Anthropic client")
            self.client = Anthropic(api_key=self.api_key)
            logger.success("LLM handler initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Anthropic client", exc_info=e)
            raise
    
    def format_context(self, documents: List[Tuple[dict, float]]) -> str:
        """Format document chunks and their relevance scores into a string.
        
        Args:
            documents: List of tuples containing (document, relevance_score)
        
        Returns:
            Formatted string containing the context
        """
        logger.debug(f"Formatting {len(documents)} documents for context")
        context_parts = []
        for doc, score in documents:
            context_parts.append(
                f"File: {doc.metadata['file_path']}\n"
                f"Language: {doc.metadata['language']}\n"
                f"Relevance Score: {score:.4f}\n"
                f"Content:\n{doc.page_content}\n"
            )
        return "\n---\n".join(context_parts)
    
    def generate_response(self, query: str, documents: List[Tuple[dict, float]]) -> str:
        """Generate a response using Claude.
        
        Args:
            query: The user's query
            documents: List of tuples containing (document, relevance_score)
        
        Returns:
            Claude's response
        """
        try:
            logger.info(f"Generating response for query: {query}")
            context = self.format_context(documents)
            
            logger.debug("Sending request to Claude")
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                system=DEFAULT_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"Based on the following code and documentation, please answer this question: {query}\n\nContext:\n{context}"
                    }
                ]
            )
            
            logger.debug("Response received from Claude")
            return message.content[0].text
        except Exception as e:
            logger.error("Error generating response", exc_info=e)
            return "Sorry, I encountered an error while generating a response." 