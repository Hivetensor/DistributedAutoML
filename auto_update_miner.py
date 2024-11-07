import logging
import os
import re
import subprocess
import sys
from typing import Optional

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_PATH = "neurons/miner.py"
BRANCH = "main"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_git(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", REPO_PATH] + cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {e}")
        return None


def get_version(content):
    match = re.search(r"__spec_version__\s*=\s*['\"]?(\d+)", content)
    return int(match.group(1)) if match else None


def get_versions():
    try:
        with open(os.path.join(REPO_PATH, "dml/chain/__init__.py")) as f:
            local_ver = get_version(f.read())

        run_git(["fetch"])
        remote_content = run_git(["show", f"origin/{BRANCH}:dml/chain/__init__.py"])
        remote_ver = get_version(remote_content) if remote_content else None

        return local_ver, remote_ver
    except Exception as e:
        logging.error(f"Failed to get versions: {e}")
        return None, None


def main():
    try:
        old_commit = run_git(["rev-parse", "HEAD"])

        local_ver, remote_ver = get_versions()
        if not all([local_ver, remote_ver]):
            return 1

        logging.info(f"Local version: {local_ver}, Remote version: {remote_ver}")
        if local_ver == remote_ver:
            logging.info("No update needed")
            return 0

        result = subprocess.run(
            ["git", "-C", REPO_PATH, "pull"], capture_output=True, text=True
        )

        if result.returncode != 0 and "local changes" in result.stderr:
            #TODO: Potential issues regarding config files
            logging.info("Stashing local changes")
            run_git(["stash"])
            if not run_git(["pull"]):
                return 1
            run_git(["stash", "pop"])

        try:
            subprocess.run(["pip", "install", "-e", "."], check=True)
            subprocess.run([sys.executable, MAIN_SCRIPT_PATH], check=True)
        except subprocess.CalledProcessError:
            # Do git reset to last commit if update fails
            logging.error("Update failed, rolling back")
            run_git(["reset", "--hard", old_commit])
            return 1

        logging.info("Update successful")
        return 0

    except Exception as e:
        logging.error(f"Error during update: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
