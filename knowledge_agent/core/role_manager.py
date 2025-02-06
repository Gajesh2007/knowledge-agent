"""Module for managing custom roles and role-based prompts."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from knowledge_agent.core.logging import logger

@dataclass
class RoleTemplate:
    """Template for a role's behavior and responses."""
    name: str
    description: str
    system_prompt: str
    response_style: Dict[str, str]  # Style guidelines for different response types
    metadata: Dict[str, str]  # Additional role configuration

class RoleManager:
    """Manages custom roles and role templates."""
    
    DEFAULT_ROLES = {
        "beginner": RoleTemplate(
            name="beginner",
            description="Explains concepts in simple terms with step-by-step instructions",
            system_prompt=(
                "You are a patient teacher helping beginners understand code. "
                "Break down complex concepts into simple terms and provide step-by-step explanations. "
                "Use analogies and examples when helpful. "
                "Avoid technical jargon unless necessary, and when used, explain it clearly."
            ),
            response_style={
                "code_explanation": "Step-by-step breakdown with comments",
                "error_handling": "Clear, actionable solutions with explanations",
                "concepts": "Simple analogies and real-world examples",
            },
            metadata={"expertise_level": "basic"}
        ),
        "engineer": RoleTemplate(
            name="engineer",
            description="Provides technical details and implementation insights",
            system_prompt=(
                "You are an experienced software engineer providing technical insights. "
                "Focus on implementation details, design patterns, and best practices. "
                "Be precise and thorough in technical explanations. "
                "Include relevant code examples and performance considerations."
            ),
            response_style={
                "code_explanation": "Technical analysis with design rationale",
                "error_handling": "Root cause analysis and solution approaches",
                "concepts": "Detailed technical explanations with examples",
            },
            metadata={"expertise_level": "advanced"}
        ),
        "bd": RoleTemplate(
            name="bd",
            description="Focuses on business impact and high-level functionality",
            system_prompt=(
                "You are a business-focused technical advisor. "
                "Emphasize business impact, use cases, and high-level functionality. "
                "Translate technical concepts into business value. "
                "Focus on ROI, efficiency gains, and strategic advantages."
            ),
            response_style={
                "code_explanation": "High-level functionality and business impact",
                "error_handling": "Impact assessment and business-focused solutions",
                "concepts": "Business context and value proposition",
            },
            metadata={"expertise_level": "business"}
        )
    }
    
    def __init__(self, config_dir: str = "./.roles"):
        """Initialize role manager.
        
        Args:
            config_dir: Directory for storing custom role configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load custom roles
        self.custom_roles: Dict[str, RoleTemplate] = {}
        self._load_custom_roles()
    
    def get_role(self, name: str) -> Optional[RoleTemplate]:
        """Get a role template by name.
        
        Args:
            name: Role name
            
        Returns:
            Role template or None if not found
        """
        return self.custom_roles.get(name) or self.DEFAULT_ROLES.get(name)
    
    def list_roles(self) -> List[RoleTemplate]:
        """Get all available roles.
        
        Returns:
            List of role templates
        """
        return list(self.DEFAULT_ROLES.values()) + list(self.custom_roles.values())
    
    def create_role(
        self,
        name: str,
        description: str,
        system_prompt: str,
        response_style: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> RoleTemplate:
        """Create a new custom role.
        
        Args:
            name: Role name
            description: Role description
            system_prompt: System prompt for the role
            response_style: Optional style guidelines
            metadata: Optional additional configuration
            
        Returns:
            Created role template
            
        Raises:
            ValueError: If role name already exists
        """
        if name in self.DEFAULT_ROLES:
            raise ValueError(f"Cannot override default role: {name}")
        
        template = RoleTemplate(
            name=name,
            description=description,
            system_prompt=system_prompt,
            response_style=response_style or {},
            metadata=metadata or {}
        )
        
        # Save to disk
        self._save_role(template)
        
        # Add to memory
        self.custom_roles[name] = template
        
        return template
    
    def update_role(
        self,
        name: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_style: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[RoleTemplate]:
        """Update an existing custom role.
        
        Args:
            name: Role name
            description: Optional new description
            system_prompt: Optional new system prompt
            response_style: Optional new style guidelines
            metadata: Optional new metadata
            
        Returns:
            Updated role template or None if not found
            
        Raises:
            ValueError: If trying to update a default role
        """
        if name in self.DEFAULT_ROLES:
            raise ValueError(f"Cannot modify default role: {name}")
        
        template = self.custom_roles.get(name)
        if not template:
            return None
        
        # Update fields
        if description:
            template.description = description
        if system_prompt:
            template.system_prompt = system_prompt
        if response_style:
            template.response_style.update(response_style)
        if metadata:
            template.metadata.update(metadata)
        
        # Save changes
        self._save_role(template)
        
        return template
    
    def delete_role(self, name: str) -> bool:
        """Delete a custom role.
        
        Args:
            name: Role name
            
        Returns:
            True if role was deleted, False if not found
            
        Raises:
            ValueError: If trying to delete a default role
        """
        if name in self.DEFAULT_ROLES:
            raise ValueError(f"Cannot delete default role: {name}")
        
        if name not in self.custom_roles:
            return False
        
        # Remove from disk
        role_file = self._get_role_file(name)
        if role_file.exists():
            role_file.unlink()
        
        # Remove from memory
        del self.custom_roles[name]
        
        return True
    
    def _get_role_file(self, name: str) -> Path:
        """Get the path to a role's configuration file."""
        return self.config_dir / f"{name}.json"
    
    def _save_role(self, template: RoleTemplate):
        """Save a role template to disk."""
        try:
            role_file = self._get_role_file(template.name)
            with open(role_file, 'w') as f:
                json.dump(asdict(template), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save role {template.name}: {str(e)}")
    
    def _load_custom_roles(self):
        """Load custom roles from disk."""
        for role_file in self.config_dir.glob("*.json"):
            try:
                with open(role_file) as f:
                    data = json.load(f)
                template = RoleTemplate(**data)
                self.custom_roles[template.name] = template
            except Exception as e:
                logger.error(f"Failed to load role from {role_file}: {str(e)}")
    
    def get_prompt(self, role_name: str, context_type: Optional[str] = None) -> str:
        """Get the complete prompt for a role.
        
        Args:
            role_name: Role name
            context_type: Optional context type for style guidelines
            
        Returns:
            Complete prompt including style guidelines
        """
        template = self.get_role(role_name)
        if not template:
            return ""
        
        prompt = template.system_prompt
        
        # Add style guidelines if context type is specified
        if context_type and context_type in template.response_style:
            prompt += f"\n\nFor this response, use this style: {template.response_style[context_type]}"
        
        return prompt 