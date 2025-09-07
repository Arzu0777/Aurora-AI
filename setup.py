# setup.py - Installation script for MemoryPal Enhanced V2

"""
MemoryPal Enhanced V2 Setup Script
=================================

Automated setup script to install dependencies and configure the environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🧠 MemoryPal Enhanced V2 - Setup Script")
    print("=" * 50)
    print("This script will help you set up MemoryPal Enhanced V2")
    print("with all required dependencies and configuration.\n")

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {python_version.major}.{python_version.minor}")
        print("💡 Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} - Compatible")
    return True

def install_system_dependencies():
    """Install system dependencies"""
    print("\n📦 Installing system dependencies...")
    
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        print("💡 Installing ffmpeg...")
        try:
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
            print("✅ ffmpeg installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  Could not install ffmpeg automatically")
            print("💡 Please install manually: sudo apt install ffmpeg")
    
    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        print("💡 Installing ffmpeg with Homebrew...")
        try:
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
            print("✅ ffmpeg installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  Could not install ffmpeg automatically")
            print("💡 Please install Homebrew and run: brew install ffmpeg")
    
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("⚠️  Please install ffmpeg manually:")
        print("   1. Download from https://ffmpeg.org/download.html")
        print("   2. Extract to a folder (e.g., C:\\ffmpeg)")
        print("   3. Add C:\\ffmpeg\\bin to your PATH environment variable")
        print("💡 Alternatively, use conda: conda install -c conda-forge ffmpeg")
    
    else:
        print(f"⚠️  Unsupported system: {system}")
        print("💡 Please install ffmpeg manually for your system")

def install_python_dependencies():
    """Install Python dependencies"""
    print("\n🐍 Installing Python dependencies...")
    
    # Upgrade pip first
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ pip upgraded successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Could not upgrade pip")
    
    # Install core dependencies first
    core_packages = [
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "PyMuPDF>=1.23.0",
        "google-generativeai>=0.3.0"
    ]
    
    print("📦 Installing core packages...")
    for package in core_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"✅ {package.split('>=')[0]} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Install optional packages
    optional_packages = [
        "audio-recorder-streamlit>=0.0.8",
        "openai-whisper>=1.1.10",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "pyttsx3>=2.90"
    ]
    
    print("\n📦 Installing optional packages...")
    for package in optional_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package.split('>=')[0]} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Optional package {package.split('>=')[0]} failed - will continue")
    
    # Install from requirements.txt if available
    if Path("requirements.txt").exists():
        print("\n📋 Installing from requirements.txt...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("✅ All requirements installed successfully")
        except subprocess.CalledProcessError:
            print("⚠️  Some packages from requirements.txt failed to install")

def setup_environment():
    """Set up environment files and directories"""
    print("\n🔧 Setting up environment...")
    
    # Create .env file from template
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if not env_path.exists() and example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
        print("✅ Created .env file from template")
    elif not env_path.exists():
        # Create basic .env file
        basic_env = """# MemoryPal Enhanced V2 Environment Variables
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
COHERE_API_KEY=your_cohere_api_key_here
"""
        env_path.write_text(basic_env)
        print("✅ Created basic .env file")
    else:
        print("✅ .env file already exists")
    
    # Create necessary directories
    directories = ["temp", "outputs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_api_keys():
    """Help user set up API keys"""
    print("\n🔑 API Key Configuration")
    print("-" * 30)
    
    print("📝 You need to configure API keys in your .env file:")
    print()
    print("1. 🌟 GOOGLE_API_KEY (Required)")
    print("   - Get it from: https://makersuite.google.com/app/apikey")
    print("   - This is required for the main functionality")
    print()
    print("2. 🤗 HUGGINGFACE_TOKEN (Optional but recommended)")
    print("   - Get it from: https://huggingface.co/settings/tokens")
    print("   - Needed for speaker diarization")
    print()
    print("3. 🔄 COHERE_API_KEY (Optional)")
    print("   - Get it from: https://cohere.com/")
    print("   - Enables advanced search reranking")
    print()
    
    # Interactive key setup
    if input("Would you like to configure API keys now? (y/N): ").lower().strip() == 'y':
        setup_keys_interactively()
    else:
        print("💡 You can configure API keys later by editing the .env file")

def setup_keys_interactively():
    """Interactive API key setup"""
    env_path = Path(".env")
    env_content = env_path.read_text() if env_path.exists() else ""
    
    # Google API Key
    google_key = input("\n🌟 Enter your Google Gemini API key (or press Enter to skip): ").strip()
    if google_key:
        env_content = update_env_var(env_content, "GOOGLE_API_KEY", google_key)
        print("✅ Google API key configured")
    
    # Hugging Face Token
    hf_token = input("\n🤗 Enter your Hugging Face token (or press Enter to skip): ").strip()
    if hf_token:
        env_content = update_env_var(env_content, "HUGGINGFACE_TOKEN", hf_token)
        print("✅ Hugging Face token configured")
    
    # Cohere API Key
    cohere_key = input("\n🔄 Enter your Cohere API key (or press Enter to skip): ").strip()
    if cohere_key:
        env_content = update_env_var(env_content, "COHERE_API_KEY", cohere_key)
        print("✅ Cohere API key configured")
    
    # Save updated .env file
    env_path.write_text(env_content)
    print("\n💾 Configuration saved to .env file")

def update_env_var(env_content: str, var_name: str, var_value: str) -> str:
    """Update environment variable in .env content"""
    import re
    
    pattern = f"^{var_name}=.*$"
    replacement = f"{var_name}={var_value}"
    
    if re.search(pattern, env_content, re.MULTILINE):
        env_content = re.sub(pattern, replacement, env_content, flags=re.MULTILINE)
    else:
        env_content += f"\n{replacement}"
    
    return env_content

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Streamlit import failed")
        return False
    
    try:
        import google.generativeai
        print("✅ Google Generative AI imported successfully")
    except ImportError:
        print("❌ Google Generative AI import failed")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment loading works")
    except ImportError:
        print("❌ python-dotenv import failed")
        return False
    
    # Test optional imports
    optional_imports = [
        ("whisper", "OpenAI Whisper"),
        ("librosa", "Librosa audio processing"),
        ("transformers", "Hugging Face Transformers"),
        ("audio_recorder_streamlit", "Audio Recorder Streamlit")
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")
    
    return True

def print_final_instructions():
    """Print final setup instructions"""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("✅ MemoryPal Enhanced V2 is ready to use!")
    print()
    print("🚀 To start the application:")
    print("   python run.py")
    print("   # or")
    print("   streamlit run enhanced_rag_app_v2.py")
    print()
    print("💡 Next steps:")
    print("   1. Configure your API keys in the .env file")
    print("   2. Start the application")
    print("   3. Upload documents or record audio")
    print("   4. Ask questions and get insights!")
    print()
    print("📚 Need help? Check the README.md file")
    print("🐛 Issues? Report them on GitHub")
    print()
    print("Happy analyzing! 🧠✨")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Set up environment
    setup_environment()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed")
        print("💡 Please check error messages above and try manual installation")
        sys.exit(1)
    
    # API key setup
    setup_api_keys()
    
    # Final instructions
    print_final_instructions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("💡 Please check the error and try manual installation")
        sys.exit(1)