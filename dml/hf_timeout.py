import timeout_decorator
from huggingface_hub import HfApi
from requests.exceptions import Timeout
import time

class TimeoutHfApi(HfApi):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @timeout_decorator.timeout(120, timeout_exception=Timeout)
    def list_repo_files_with_timeout(self, repo_id, **kwargs):
        """Wrapper for list_repo_files with timeout"""
        return super().list_repo_files(repo_id=repo_id, **kwargs)
    
    @timeout_decorator.timeout(30, timeout_exception=Timeout)
    def list_repo_tree_with_timeout(self, repo_id, **kwargs):
        """Wrapper for list_repo_files with timeout"""
        return super().list_repo_tree(repo_id=repo_id, **kwargs)

    @timeout_decorator.timeout(30, timeout_exception=Timeout)
    def hf_hub_download_with_timeout(self, repo_id, filename, **kwargs):
        """Wrapper for list_repo_files with timeout"""
        return super().hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        
    @timeout_decorator.timeout(30, timeout_exception=Timeout)
    def repo_info_with_timeout(self, repo_id, **kwargs):
        """Wrapper for list_repo_files with timeout"""
        return super().repo_info(repo_id=repo_id, **kwargs)
        
    