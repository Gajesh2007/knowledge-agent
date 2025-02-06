"""LLM functionality for handling interactions with Claude."""
import os
from typing import List, Tuple, Optional

from anthropic import Anthropic

from knowledge_agent.core.logging import logger
from knowledge_agent.core.memory import ConversationMemory

DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that helps users understand code and documentation.
Your task is to provide accurate, helpful responses based on the context provided.
If you don't have enough information in the context to answer a question, say so.
Always cite the specific files and code snippets you reference in your answers.
When answering follow-up questions, use the conversation history to maintain context."""

class LLMHandler:
    """Handles interactions with the Claude LLM."""

    def __init__(self, api_key: str = None, max_conversation_history: int = 10):
        """Initialize the LLM handler.
        
        Args:
            api_key: Optional API key for Claude. If not provided, will look for ANTHROPIC_API_KEY in environment.
            max_conversation_history: Maximum number of messages to keep in conversation history
        """
        logger.debug("Initializing LLM handler")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment")
            raise ValueError("ANTHROPIC_API_KEY must be provided or set in environment")
        
        # Initialize the client and conversation memory
        try:
            logger.debug("Initializing Anthropic client")
            self.client = Anthropic(api_key=self.api_key)
            self.memory = ConversationMemory(max_messages=max_conversation_history)
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
    
    def generate_response(
        self, 
        query: str, 
        documents: List[Tuple[dict, float]], 
        role_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using Claude.
        
        Args:
            query: The user's query
            documents: List of tuples containing (document, relevance_score)
            role_prompt: Optional role-specific prompt instructions
        
        Returns:
            Claude's response
        """
        try:
            logger.info(f"Generating response for query: {query}")
            context = self.format_context(documents)
            
            # Add user message to memory
            self.memory.add_user_message(query)
            
            # Get conversation history
            conversation_context = self.memory.get_recent_context()
            
            # Combine prompts and context
            system_prompt = DEFAULT_SYSTEM_PROMPT
            if role_prompt:
                system_prompt = f"{system_prompt}\n\n{role_prompt}"
            
            # Build the user message with context
            user_message = f"Based on the following code and documentation, please answer this question: {query}\n\n"
            if conversation_context:
                user_message += f"Previous conversation:\n{conversation_context}\n\n"
            user_message += f"Context:\n{context}"
            
            logger.debug("Sending request to Claude")
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )
            
            response = message.content[0].text
            
            # Add assistant response to memory
            self.memory.add_assistant_message(response, context=[doc for doc, _ in documents])
            
            logger.debug("Response received from Claude")
            return response
        except Exception as e:
            logger.error("Error generating response", exc_info=e)
            return "Sorry, I encountered an error while generating a response."
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        logger.debug("Conversation memory cleared") 