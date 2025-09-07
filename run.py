# run.py - Simple launcher script for MemoryPal Enhanced V2

"""
MemoryPal Enhanced V2 Launcher
==============================

Simple script to launch the Streamlit application with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if essential dependencies are installed"""
    required_packages = [
        'streamlit',
        'google.generativeai',
        'numpy',
        'pathlib',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('.', '/').replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("⚠️  .env file not found")
        print("💡 Copy .env.example to .env and add your API keys:")
        print("cp .env.example .env")
        
        # Try to create .env from .env.example
        example_path = Path('.env.example')
        if example_path.exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env from .env.example template")
            print("📝 Please edit .env file and add your API keys")
        
        return False
    
    # Check for essential environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    
    if not google_api_key:
        print("⚠️  GOOGLE_API_KEY not found in .env file")
        print("💡 Get your API key from: https://makersuite.google.com/app/apikey")
        print("📝 Add it to your .env file: GOOGLE_API_KEY=your_key_here")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['temp', 'outputs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def main():
    """Main launcher function"""
    print("🧠 MemoryPal Enhanced V2 Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    main_app = Path('enhanced_rag_app_v2.py')
    if not main_app.exists():
        print("❌ enhanced_rag_app_v2.py not found in current directory")
        print("💡 Make sure you're in the project root directory")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment configuration
    print("🔑 Checking environment configuration...")
    env_ok = check_env_file()
    
    # Create necessary directories
    print("📁 Setting up directories...")
    create_directories()
    
    if not env_ok:
        print("\n⚠️  Environment not fully configured")
        print("💡 The app will start but some features may not work")
        print("📝 Please configure your .env file for full functionality")
        
        response = input("\nContinue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("👋 Exiting. Please configure .env and try again.")
            sys.exit(0)
    
    # Launch Streamlit app
    print("\n🚀 Launching MemoryPal Enhanced V2...")
    print("🌐 The app will open in your default browser")
    print("📱 Access URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Upload documents for analysis")
    print("   - Record audio directly in browser")
    print("   - Ask questions about your content")
    print("   - Use quick tools for instant insights")
    print("\n" + "=" * 40)
    
    try:
        # Launch Streamlit with optimized configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "enhanced_rag_app_v2.py",
            "--server.maxUploadSize=200",  # 200MB max upload
            "--server.maxMessageSize=200",  # 200MB max message
            "--browser.gatherUsageStats=false",  # Disable usage stats
            "--theme.primaryColor=#ff6b35",  # Custom theme color
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 MemoryPal Enhanced V2 stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("💡 Try running directly: streamlit run enhanced_rag_app_v2.py")

if __name__ == "__main__":
    main()