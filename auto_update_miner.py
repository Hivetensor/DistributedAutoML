import importlib
import logging
import os
import re
import subprocess
import sys
import time
from typing import Optional

# Define repository and script paths
REPO_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_PATH = "neurons/miner.py"
BRANCH = "main"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_git_command(command: list[str]) -> Optional[str]:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(
            ["git", "-C", REPO_PATH] + command,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        return None


def get_local_version() -> Optional[str]:
    """Read the local version from dml/chain/__init__.py."""
    try:
        sys.path.insert(0, REPO_PATH)
        import dml.chain as chain

        importlib.reload(chain)  # Ensure we're getting the latest version
        return getattr(chain, "__spec_version__", None)
    except ImportError:
        logging.error("Failed to import dml.chain module")
        return None
    finally:
        sys.path.pop(0)


def get_remote_version() -> Optional[str]:
    """More reliable remote version check."""
    if run_git_command(["fetch"]) is None:
        return None

    try:
        remote_content = run_git_command(
            ["show", f"origin/{BRANCH}:dml/chain/__init__.py"]
        )
        if not remote_content:
            return None

        # Use regex to reliably extract version number
        match = re.search(r'__spec_version__\s*=\s*["\']?(\d+)["\']?', remote_content)
        return match.group(1) if match else None

    except Exception as e:
        logging.error(f"Error fetching remote version: {e}")
        return None


def get_current_branch() -> Optional[str]:
    """Returns the current local git branch."""
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])


def stash_changes() -> bool:
    """Stash local changes to avoid conflicts."""
    return run_git_command(["stash"]) is not None


def apply_stash() -> bool:
    """Apply stashed changes after update."""
    return run_git_command(["stash", "pop"]) is not None


def switch_to_branch(branch_name: str) -> bool:
    """Switch to the specified branch."""
    return run_git_command(["checkout", branch_name]) is not None


def update_repo() -> bool:
    """Pull the latest changes from the repository."""
    return run_git_command(["pull"]) is not None


def run_main_script():
    """Run the main script."""
    try:
        subprocess.run([sys.executable, MAIN_SCRIPT_PATH], check=True)
        logging.info("Main script executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing main script: {e}")
        return False
    return True


def install_packages():
    """Runs pip install -e . to install the package in editable mode."""
    try:
        result = subprocess.run(
            ["pip", "install", "-e", "."], check=True, capture_output=True, text=True
        )
        logging.debug(result.stdout)
        logging.info("Package installed in editable mode successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred during installation: {e.stderr}")
        raise


def backup_current_state():
    """Create backup of current code state."""
    backup_dir = f"backup_{int(time.time())}"
    run_git_command(["stash", "push", "-m", f"Backup before update to {backup_dir}"])
    return backup_dir


def verify_update():
    """Verify the update succeeded."""
    try:
        # Try importing main modules
        import dml.chain

        # Try running basic initialization
        subprocess.run([sys.executable, "-c", "import dml.chain"], check=True)
        return True
    except Exception as e:
        logging.error(f"Update verification failed: {e}")
        return False


def rollback_update(backup_dir):
    """Rollback to previous state."""
    logging.info("Rolling back update...")
    run_git_command(["reset", "--hard", "HEAD@{1}"])
    run_git_command(["stash", "pop"])


def main():
    try:
        local_version = get_local_version()
        remote_version = get_remote_version()
        current_branch = get_current_branch()

        if not all([local_version, remote_version, current_branch]):
            logging.error("Failed to retrieve necessary information.")
            return 1

        logging.info(f"Local version: {local_version}")
        logging.info(f"Remote version: {remote_version}")

        if int(local_version) != int(remote_version):
            logging.info("New version detected, performing update...")

            # Create backup
            backup_dir = backup_current_state()

            try:
                if current_branch != BRANCH:
                    logging.info(f"Currently on branch {current_branch}.")

                    status = run_git_command(["status", "--porcelain"])

                    if status:
                        logging.info("Uncommitted changes found, stashing them.")
                        if not stash_changes():
                            raise Exception("Failed to stash changes.")

                    if not switch_to_branch(BRANCH):
                        raise Exception(f"Failed to switch to {BRANCH} branch.")

                    if not update_repo():
                        raise Exception("Failed to update repository.")

                    if not switch_to_branch(current_branch):
                        raise Exception(
                            f"Failed to switch back to {current_branch} branch."
                        )

                    if status and not apply_stash():
                        raise Exception("Failed to apply stashed changes.")
                else:
                    if not update_repo():
                        raise Exception("Failed to update repository.")

                install_packages()

                # Verify update
                if not verify_update():
                    raise Exception("Update verification failed")

                logging.info("Update completed successfully.")

                # Restart the script with the new version
                logging.info("Restarting script with updated version...")
                os.execv(sys.executable, ["python"] + sys.argv)

            except Exception as e:
                logging.error(f"Update failed: {e}")
                rollback_update(backup_dir)
                return 1

        else:
            logging.info("No update required.")
            return run_main_script()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
