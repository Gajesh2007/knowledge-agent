"""Module for advanced code analysis features like dependency mapping and semantic summaries."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from knowledge_agent.core.code_parser import CodeEntity, parse_codebase
from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.logging import logger

@dataclass
class DependencyNode:
    """Node in the dependency graph."""
    name: str
    type: str  # 'module', 'class', 'function', etc.
    file_path: str
    incoming: Set[str]  # Things that depend on this
    outgoing: Set[str]  # Things this depends on
    metadata: Dict[str, str]

class DependencyGraph:
    """Graph representation of code dependencies."""
    
    def __init__(self):
        """Initialize an empty dependency graph."""
        self.nodes: Dict[str, DependencyNode] = {}
        self._incoming_edges = defaultdict(set)  # name -> set of names that depend on it
        self._outgoing_edges = defaultdict(set)  # name -> set of names it depends on
    
    def add_entity(self, entity: CodeEntity):
        """Add a code entity to the graph."""
        # Create or update node
        if entity.name not in self.nodes:
            self.nodes[entity.name] = DependencyNode(
                name=entity.name,
                type=entity.type,
                file_path=entity.metadata['path'],
                incoming=set(),
                outgoing=set(),
                metadata=entity.metadata
            )
        
        # Add dependencies
        node = self.nodes[entity.name]
        for dep in entity.dependencies:
            node.outgoing.add(dep)
            self._outgoing_edges[entity.name].add(dep)
            self._incoming_edges[dep].add(entity.name)
    
    def get_dependents(self, name: str) -> Set[str]:
        """Get all entities that depend on the given name."""
        return self._incoming_edges[name]
    
    def get_dependencies(self, name: str) -> Set[str]:
        """Get all entities that the given name depends on."""
        return self._outgoing_edges[name]
    
    def get_transitive_dependents(self, name: str) -> Set[str]:
        """Get all entities that directly or indirectly depend on the given name."""
        visited = set()
        to_visit = {name}
        
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                to_visit.update(self.get_dependents(current) - visited)
        
        return visited - {name}
    
    def get_transitive_dependencies(self, name: str) -> Set[str]:
        """Get all entities that the given name directly or indirectly depends on."""
        visited = set()
        to_visit = {name}
        
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                to_visit.update(self.get_dependencies(current) - visited)
        
        return visited - {name}

class CodeAnalyzer:
    """High-level code analysis functionality."""
    
    def __init__(self, llm_handler: Optional[LLMHandler] = None):
        """Initialize the code analyzer.
        
        Args:
            llm_handler: Optional LLM handler for generating summaries
        """
        self.llm_handler = llm_handler
        self.dependency_graph = DependencyGraph()
    
    def analyze_codebase(self, root_path: Path):
        """Analyze an entire codebase.
        
        Args:
            root_path: Root directory of the codebase
        """
        logger.info(f"Analyzing codebase at {root_path}")
        
        # Parse the codebase
        entities = parse_codebase(root_path)
        
        # Build dependency graph
        for entity in entities:
            self.dependency_graph.add_entity(entity)
        
        logger.info(f"Built dependency graph with {len(self.dependency_graph.nodes)} nodes")
    
    def get_entity_summary(self, entity: CodeEntity) -> str:
        """Generate a semantic summary for a code entity.
        
        Args:
            entity: The code entity to summarize
            
        Returns:
            A natural language summary of the entity
        """
        if not self.llm_handler:
            return "LLM handler not available for generating summaries"
        
        # Prepare context about the entity
        context = (
            f"This is a {entity.type} named '{entity.name}'"
            f"{f' in class {entity.parent}' if entity.parent else ''}"
            f" from file {entity.metadata['path']}.\n\n"
        )
        
        # Add dependency information
        if entity.dependencies:
            context += f"It depends on: {', '.join(entity.dependencies)}\n\n"
        
        dependents = self.dependency_graph.get_dependents(entity.name)
        if dependents:
            context += f"It is used by: {', '.join(dependents)}\n\n"
        
        # Add the code and docstring
        if entity.docstring:
            context += f"Docstring:\n{entity.docstring}\n\n"
        context += f"Code:\n{entity.code}\n"
        
        # Generate summary using LLM
        prompt = (
            "Please provide a concise summary of this code entity. "
            "Focus on its purpose, key functionality, and relationships with other parts of the codebase. "
            "Keep the summary clear and technical, but avoid just repeating the docstring."
        )
        
        try:
            summary = self.llm_handler.generate_response(prompt, [context])
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return "Failed to generate summary"
    
    def find_dependencies(self, name: str, include_transitive: bool = False) -> Dict[str, Set[str]]:
        """Find dependencies for a given entity.
        
        Args:
            name: Name of the entity
            include_transitive: Whether to include indirect dependencies
            
        Returns:
            Dictionary with 'dependents' and 'dependencies' sets
        """
        if include_transitive:
            dependents = self.dependency_graph.get_transitive_dependents(name)
            dependencies = self.dependency_graph.get_transitive_dependencies(name)
        else:
            dependents = self.dependency_graph.get_dependents(name)
            dependencies = self.dependency_graph.get_dependencies(name)
        
        return {
            'dependents': dependents,
            'dependencies': dependencies
        } 