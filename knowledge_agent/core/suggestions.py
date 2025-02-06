"""Module for automated code suggestions and refactoring."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from knowledge_agent.core.code_analysis import CodeAnalyzer
from knowledge_agent.core.code_parser import CodeEntity
from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.logging import logger

@dataclass
class CodeSuggestion:
    """Represents a code improvement suggestion."""
    type: str  # 'refactor', 'style', 'performance', 'security', etc.
    entity_name: str
    file_path: str
    description: str
    priority: str  # 'high', 'medium', 'low'
    current_code: str
    suggested_code: Optional[str] = None
    rationale: Optional[str] = None
    impact: Optional[str] = None

class SuggestionEngine:
    """Engine for generating automated code suggestions."""
    
    def __init__(self, llm_handler: LLMHandler):
        """Initialize the suggestion engine.
        
        Args:
            llm_handler: LLM handler for generating suggestions
        """
        self.llm_handler = llm_handler
        self.analyzer = CodeAnalyzer(llm_handler)
    
    def analyze_entity(self, entity: CodeEntity) -> List[CodeSuggestion]:
        """Analyze a code entity and generate improvement suggestions.
        
        Args:
            entity: The code entity to analyze
            
        Returns:
            List of code suggestions
        """
        suggestions = []
        
        # Prepare context for LLM
        context = (
            f"Code entity type: {entity.type}\n"
            f"Name: {entity.name}\n"
            f"File: {entity.metadata['path']}\n"
            f"Dependencies: {', '.join(entity.dependencies)}\n\n"
            f"Code:\n{entity.code}\n"
        )
        
        # Generate suggestions using LLM
        prompt = (
            "Please analyze this code and provide improvement suggestions. "
            "Consider the following aspects:\n"
            "1. Code style and readability\n"
            "2. Performance optimizations\n"
            "3. Security considerations\n"
            "4. Design patterns and best practices\n"
            "5. Error handling and edge cases\n\n"
            "For each suggestion, provide:\n"
            "- Type of improvement\n"
            "- Priority level (high/medium/low)\n"
            "- Detailed description\n"
            "- Suggested code changes (if applicable)\n"
            "- Rationale for the change\n"
            "- Expected impact\n"
        )
        
        try:
            response = self.llm_handler.generate_response(prompt, [context])
            suggestions.extend(self._parse_suggestions(response, entity))
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
        
        return suggestions
    
    def analyze_codebase(
        self,
        root_path: Path,
        focus_areas: Optional[Set[str]] = None
    ) -> Dict[str, List[CodeSuggestion]]:
        """Analyze an entire codebase and generate suggestions.
        
        Args:
            root_path: Root directory of the codebase
            focus_areas: Optional set of specific areas to focus on
                        (e.g., {'performance', 'security'})
            
        Returns:
            Dictionary mapping file paths to lists of suggestions
        """
        # First analyze the codebase structure
        self.analyzer.analyze_codebase(root_path)
        
        # Generate suggestions for each entity
        suggestions_by_file = {}
        
        for node in self.analyzer.dependency_graph.nodes.values():
            # Create a CodeEntity from the node
            entity = CodeEntity(
                name=node.name,
                type=node.type,
                docstring=None,  # We don't have this in the node
                code=self._get_entity_code(node),
                start_line=0,  # We don't have this in the node
                end_line=0,  # We don't have this in the node
                parent=None,
                dependencies=node.outgoing,
                metadata={'path': node.file_path}
            )
            
            # Generate suggestions for this entity
            suggestions = self.analyze_entity(entity)
            
            # Filter by focus areas if specified
            if focus_areas:
                suggestions = [
                    s for s in suggestions
                    if s.type in focus_areas
                ]
            
            # Group by file
            if suggestions:
                file_path = node.file_path
                if file_path not in suggestions_by_file:
                    suggestions_by_file[file_path] = []
                suggestions_by_file[file_path].extend(suggestions)
        
        return suggestions_by_file
    
    def _get_entity_code(self, node: 'DependencyNode') -> str:
        """Get the code for a dependency node.
        
        This is a placeholder - in a real implementation, we would need to:
        1. Parse the file
        2. Find the specific entity
        3. Extract its code
        
        For now, we return a placeholder message.
        """
        return f"# Code for {node.name} would be extracted here"
    
    def _parse_suggestions(
        self,
        llm_response: str,
        entity: CodeEntity
    ) -> List[CodeSuggestion]:
        """Parse LLM response into structured suggestions.
        
        This is a simple parser that expects suggestions to be separated by
        clear markers. A more robust parser would be needed for production.
        """
        suggestions = []
        current_suggestion = None
        
        for line in llm_response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Look for suggestion markers
            if line.startswith('Type:'):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                current_suggestion = CodeSuggestion(
                    type=line.split(':', 1)[1].strip().lower(),
                    entity_name=entity.name,
                    file_path=entity.metadata['path'],
                    description='',
                    priority='medium',  # Default priority
                    current_code=entity.code
                )
            elif current_suggestion:
                if line.startswith('Priority:'):
                    current_suggestion.priority = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Description:'):
                    current_suggestion.description = line.split(':', 1)[1].strip()
                elif line.startswith('Suggested Code:'):
                    current_suggestion.suggested_code = ''
                elif line.startswith('Rationale:'):
                    current_suggestion.rationale = line.split(':', 1)[1].strip()
                elif line.startswith('Impact:'):
                    current_suggestion.impact = line.split(':', 1)[1].strip()
                elif current_suggestion.suggested_code is not None:
                    # Accumulate suggested code
                    current_suggestion.suggested_code += line + '\n'
        
        # Add the last suggestion
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions 