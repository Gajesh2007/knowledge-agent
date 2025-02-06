"""Module for managing GitHub repository fetching and updates."""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from git import Repo, GitCommandError
import logging

logger = logging.getLogger(__name__)

class RepoManager:
    """Manages GitHub repository cloning, updating, and tracking."""
    
    def __init__(self, cache_dir: str = "./.repos"):
        """Initialize the repo manager with a cache directory.
        
        Args:
            cache_dir: Directory to store cloned repositories
        """
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_repo_path(self, repo_url: str) -> Path:
        """Get the local path for a repository based on its URL.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Path object for the local repository directory
        """
        # Extract repo name from URL and create a safe directory name
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return self.cache_dir / repo_name
    
    def fetch_repo(self, repo_url: str, branch: Optional[str] = None) -> Path:
        """Clone or update a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            branch: Optional branch name to checkout
            
        Returns:
            Path to the local repository
            
        Raises:
            GitCommandError: If git operations fail
        """
        repo_path = self._get_repo_path(repo_url)
        
        try:
            if repo_path.exists():
                # Update existing repository
                logger.info(f"Updating existing repository at {repo_path}")
                repo = Repo(repo_path)
                origin = repo.remotes.origin
                origin.fetch()
                if branch:
                    repo.git.checkout(branch)
                origin.pull()
            else:
                # Clone new repository
                logger.info(f"Cloning new repository to {repo_path}")
                repo = Repo.clone_from(repo_url, repo_path)
                if branch:
                    repo.git.checkout(branch)
                    
            return repo_path
            
        except GitCommandError as e:
            logger.error(f"Git operation failed: {str(e)}")
            raise
            
    def fetch_multiple(self, repo_urls: List[str], branch: Optional[str] = None) -> List[Path]:
        """Fetch multiple repositories.
        
        Args:
            repo_urls: List of GitHub repository URLs
            branch: Optional branch name to checkout for all repos
            
        Returns:
            List of paths to local repositories
        """
        paths = []
        for url in repo_urls:
            try:
                path = self.fetch_repo(url, branch)
                paths.append(path)
            except GitCommandError as e:
                logger.error(f"Failed to fetch repo {url}: {str(e)}")
                continue
        return paths
    
    def cleanup(self, repo_url: Optional[str] = None):
        """Remove cached repositories.
        
        Args:
            repo_url: Optional specific repository URL to remove.
                     If None, removes all repositories.
        """
        if repo_url:
            repo_path = self._get_repo_path(repo_url)
            if repo_path.exists():
                shutil.rmtree(repo_path)
                logger.info(f"Removed repository at {repo_path}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                logger.info(f"Cleaned all repositories from {self.cache_dir}") 