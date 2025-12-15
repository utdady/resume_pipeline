"""Setup script to install spaCy model for NLP-enhanced job description parsing."""
import subprocess
import sys

def install_spacy_model():
    """Install the spaCy English model."""
    print("Installing spaCy English model (en_core_web_sm)...")
    print("This may take a few minutes...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("\n✓ spaCy model installed successfully!")
        print("You can now use the NLP-enhanced job description parser.")
    except subprocess.CalledProcessError:
        print("\n✗ Error installing spaCy model.")
        print("Please try manually: python -m spacy download en_core_web_sm")
        sys.exit(1)
    except FileNotFoundError:
        print("\n✗ spaCy not found. Please install it first:")
        print("  pip install spacy")
        sys.exit(1)

if __name__ == "__main__":
    install_spacy_model()

