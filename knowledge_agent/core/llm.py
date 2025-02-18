"""LLM functionality for handling interactions with Claude."""
import os
from typing import List, Optional, Dict, Tuple, Union, TYPE_CHECKING
import time

from anthropic import Anthropic
from anthropic.types import RateLimitError
from langchain_core.documents import Document

from knowledge_agent.core.logging import logger
from knowledge_agent.core.role_manager import RoleManager

if TYPE_CHECKING:
    from knowledge_agent.core.retrieval import SearchResult

DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that helps users understand code and documentation.
Your task is to provide accurate, helpful responses based on the context provided.
If you don't have enough information in the context to answer a question, say so.
Always cite the specific files and code snippets you reference in your answers.
When answering follow-up questions, use the conversation history to maintain context."""

class LLMHandler:
    """Handles interactions with the Claude LLM."""

    def __init__(self):
        """Initialize the LLM handler."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = Anthropic(
            api_key=api_key,
            max_retries=3,  # Maximum number of retries
            timeout=30.0,   # Timeout in seconds
        )
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        self.role_manager = RoleManager()
        
        # Base system prompt
        self.base_prompt = (
            "You are a knowledgeable assistant helping users understand code and documentation. "
            "You have access to relevant code snippets and documentation that will be provided. "
            "Please provide clear, accurate, and helpful responses."
        )
    
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
    
    def generate_response(
        self,
        query: str,
        documents: List[Union[Document, 'SearchResult']],
        role_name: Optional[str] = None,
        context_type: Optional[str] = None,
        conversation_context: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """Generate a response using the LLM.
        
        Args:
            query: User's question
            documents: Relevant documents from the vector store
            role_name: Optional role name for response style
            context_type: Optional context type for style guidelines
            conversation_context: Optional conversation history
            max_tokens: Maximum tokens in the response
            
        Returns:
            Generated response text
        """
        from knowledge_agent.core.retrieval import SearchResult  # Import here to avoid circular import
        
        # Build the system prompt
        system_prompt = self.base_prompt
        
        # Add role-specific prompt if provided
        if role_name:
            role_prompt = self.role_manager.get_prompt(role_name, context_type)
            if role_prompt:
                system_prompt = f"{system_prompt}\n\n{role_prompt}"
        
        # Build the context
        context_parts = []
        
        # Add conversation context if available
        if conversation_context:
            context_parts.append("Previous conversation:")
            context_parts.append(conversation_context)
        
        # Add relevant documents
        context_parts.append("\nRelevant code and documentation:")
        for doc in documents:
            # Handle both Document and SearchResult objects
            if isinstance(doc, SearchResult):
                document = doc.document
                relevance = doc.relevance_score
            else:
                document = doc
                relevance = None
            
            # Add metadata context
            meta_context = []
            if 'language' in document.metadata:
                meta_context.append(f"Language: {document.metadata['language']}")
            if 'type' in document.metadata:
                meta_context.append(f"Type: {document.metadata['type']}")
            if 'name' in document.metadata:
                meta_context.append(f"File: {document.metadata['name']}")
            
            # Add relevance score if available
            if relevance is not None:
                meta_context.append(f"Relevance: {relevance:.2f}")
            
            meta_str = " | ".join(meta_context)
            if meta_str:
                context_parts.append(f"\n{meta_str}")
            
            # Add document content
            content = document.page_content.strip()
            if content:
                context_parts.append(f"```\n{content}\n```\n")
        
        # Combine everything
        try:
            logger.debug("Sending request to Claude")
            max_retries = 3
            retry_delay = 1.0  # Initial delay in seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        system=system_prompt,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Context:\n{chr(10).join(context_parts)}\n\n"
                                    f"Question: {query}"
                                )
                            }
                        ],
                        max_tokens=max_tokens
                    )
                    return response.content[0].text
                    
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            logger.error("Failed to generate response", exc_info=e)
            return f"Error generating response: {str(e)}"
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        logger.debug("Conversation memory cleared") 