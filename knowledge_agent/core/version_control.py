"""Module for Git version control integration and diff analysis."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from git import Repo, Commit, GitCommandError
from git.diff import Diff

from knowledge_agent.core.llm import LLMHandler
from knowledge_agent.core.logging import logger

@dataclass
class VersionMetadata:
    """Metadata about a specific version of code."""
    commit_hash: str
    branch: str
    commit_date: datetime
    author: str
    message: str

@dataclass
class DiffInfo:
    """Information about changes between versions."""
    file_path: str
    old_path: Optional[str]
    new_path: Optional[str]
    change_type: str  # 'added', 'modified', 'deleted', 'renamed'
    additions: int
    deletions: int
    diff_text: str

class VersionManager:
    """Manages Git version control integration."""
    
    def __init__(self, repo_path: Path, llm_handler: Optional[LLMHandler] = None):
        """Initialize version manager.
        
        Args:
            repo_path: Path to the Git repository
            llm_handler: Optional LLM handler for generating diff summaries
        """
        self.repo_path = Path(repo_path)
        self.llm_handler = llm_handler
        
        try:
            self.repo = Repo(repo_path)
            if self.repo.bare:
                raise ValueError(f"Repository at {repo_path} is bare")
        except Exception as e:
            logger.error(f"Failed to initialize repository: {str(e)}")
            raise
    
    def get_version_metadata(self, ref: str = "HEAD") -> VersionMetadata:
        """Get metadata about a specific version.
        
        Args:
            ref: Git reference (commit hash, branch name, tag)
            
        Returns:
            Version metadata
        """
        try:
            commit = self.repo.commit(ref)
            return VersionMetadata(
                commit_hash=commit.hexsha,
                branch=self.repo.active_branch.name,
                commit_date=datetime.fromtimestamp(commit.committed_date),
                author=commit.author.name,
                message=commit.message.strip()
            )
        except Exception as e:
            logger.error(f"Failed to get version metadata: {str(e)}")
            raise
    
    def get_file_at_version(self, file_path: str, ref: str = "HEAD") -> Optional[str]:
        """Get the contents of a file at a specific version.
        
        Args:
            file_path: Path to the file relative to repo root
            ref: Git reference
            
        Returns:
            File contents or None if file doesn't exist
        """
        try:
            commit = self.repo.commit(ref)
            try:
                blob = commit.tree / file_path
                return blob.data_stream.read().decode('utf-8')
            except KeyError:
                return None
        except Exception as e:
            logger.error(f"Failed to get file at version: {str(e)}")
            return None
    
    def get_changed_files(self, old_ref: str, new_ref: str) -> List[DiffInfo]:
        """Get information about files changed between versions.
        
        Args:
            old_ref: Old Git reference
            new_ref: New Git reference
            
        Returns:
            List of diff information objects
        """
        try:
            old_commit = self.repo.commit(old_ref)
            new_commit = self.repo.commit(new_ref)
            diffs = old_commit.diff(new_commit)
            
            diff_infos = []
            for diff in diffs:
                change_type = self._get_change_type(diff)
                diff_infos.append(DiffInfo(
                    file_path=diff.b_path or diff.a_path,
                    old_path=diff.a_path,
                    new_path=diff.b_path,
                    change_type=change_type,
                    additions=diff.stats.get('insertions', 0),
                    deletions=diff.stats.get('deletions', 0),
                    diff_text=diff.diff.decode('utf-8')
                ))
            
            return diff_infos
            
        except Exception as e:
            logger.error(f"Failed to get changed files: {str(e)}")
            return []
    
    def _get_change_type(self, diff: Diff) -> str:
        """Determine the type of change from a diff."""
        if diff.new_file:
            return 'added'
        elif diff.deleted_file:
            return 'deleted'
        elif diff.renamed_file:
            return 'renamed'
        else:
            return 'modified'
    
    def generate_diff_summary(self, old_ref: str, new_ref: str) -> str:
        """Generate a natural language summary of changes between versions.
        
        Args:
            old_ref: Old Git reference
            new_ref: New Git reference
            
        Returns:
            Summary of changes
        """
        if not self.llm_handler:
            return "LLM handler not available for generating diff summary"
        
        try:
            # Get version metadata
            old_meta = self.get_version_metadata(old_ref)
            new_meta = self.get_version_metadata(new_ref)
            
            # Get changed files
            diffs = self.get_changed_files(old_ref, new_ref)
            
            # Prepare context for the LLM
            context = [
                f"Changes from {old_meta.commit_hash[:8]} to {new_meta.commit_hash[:8]}",
                f"Old version: {old_meta.message}",
                f"New version: {new_meta.message}",
                "\nChanged files:"
            ]
            
            for diff in diffs:
                context.append(
                    f"\n{diff.file_path} ({diff.change_type}):\n"
                    f"  +{diff.additions} -{diff.deletions}\n"
                    f"{diff.diff_text}"
                )
            
            prompt = (
                "Please provide a clear, concise summary of these code changes. "
                "Focus on the key modifications, their purpose (based on commit messages), "
                "and any significant patterns in the changes. "
                "Include both technical details and high-level impact."
            )
            
            return self.llm_handler.generate_response(prompt, ["\n".join(context)])
            
        except Exception as e:
            logger.error(f"Failed to generate diff summary: {str(e)}")
            return f"Error generating diff summary: {str(e)}"
    
    def get_available_versions(self) -> List[Tuple[str, VersionMetadata]]:
        """Get a list of available versions (commits) in the repository.
        
        Returns:
            List of (ref, metadata) tuples
        """
        try:
            versions = []
            
            # Add branches
            for branch in self.repo.heads:
                versions.append((
                    branch.name,
                    self.get_version_metadata(branch.name)
                ))
            
            # Add tags
            for tag in self.repo.tags:
                versions.append((
                    tag.name,
                    self.get_version_metadata(tag.name)
                ))
            
            return versions
            
        except Exception as e:
            logger.error(f"Failed to get available versions: {str(e)}")
            return [] 