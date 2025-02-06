"""Tests for role-based prompting functionality."""
import pytest
from knowledge_agent.core.role_prompts import get_role_prompt, ROLE_PROMPTS

def test_get_role_prompt_valid_roles():
    """Test that get_role_prompt returns correct prompts for valid roles."""
    for role in ["beginner", "engineer", "bd"]:
        prompt = get_role_prompt(role)
        assert prompt == ROLE_PROMPTS[role]
        assert "Focus on:" in prompt
        assert "Response Format:" in prompt
        assert "Example Summary Style:" in prompt

def test_get_role_prompt_case_insensitive():
    """Test that role names are case-insensitive."""
    variations = [
        ("BEGINNER", "beginner"),
        ("Engineer", "engineer"),
        ("BD", "bd"),
        ("BeGiNnEr", "beginner")
    ]
    
    for input_role, expected_role in variations:
        assert get_role_prompt(input_role) == ROLE_PROMPTS[expected_role]

def test_get_role_prompt_invalid_role():
    """Test that invalid roles raise ValueError with helpful message."""
    invalid_roles = ["developer", "user", "admin", ""]
    
    for role in invalid_roles:
        with pytest.raises(ValueError) as exc_info:
            get_role_prompt(role)
        
        error_msg = str(exc_info.value)
        assert "Invalid role" in error_msg
        assert "Must be one of:" in error_msg
        for valid_role in ROLE_PROMPTS.keys():
            assert valid_role in error_msg

def test_role_prompt_content():
    """Test that each role's prompt contains required sections and formatting."""
    required_sections = {
        "beginner": [
            "patient teacher",
            "simple, clear language",
            "concrete examples",
            "step-by-step explanation",
            "Quick Summary"
        ],
        "engineer": [
            "technical expert",
            "implementation details",
            "Design Pattern",
            "Performance",
            "Technical Summary"
        ],
        "bd": [
            "business-focused",
            "value proposition",
            "high-level functionality",
            "Business Summary",
            "ROI Factors"
        ]
    }
    
    for role, expected_content in required_sections.items():
        prompt = get_role_prompt(role)
        for content in expected_content:
            assert content in prompt, f"'{content}' not found in {role} prompt" 