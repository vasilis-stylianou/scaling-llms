from pathlib import Path
from dotenv import load_dotenv

def pytest_configure(config):
    """
    Load environment variables from the repository root .env file
    before any tests or other fixtures are initialized.
    """
    # Navigate from <repo_dir>/tests/conftest.py up to <repo_dir>/.env
    env_path = Path(__file__).parent.parent / ".env"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Optional: Warn if the .env is missing in a research environment
        print(f"\nWarning: .env file not found at {env_path}")