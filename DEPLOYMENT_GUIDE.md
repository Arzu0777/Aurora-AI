# DEPLOYMENT_GUIDE.md - MemoryPal Enhanced V2 Deployment Guide

# 🚀 MemoryPal Enhanced V2 - Deployment Guide

This guide covers different deployment options for MemoryPal Enhanced V2, from local development to cloud deployment.

## 📋 Table of Contents

1. [Quick Start (Local)](#quick-start-local)
2. [Development Setup](#development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## 🏠 Quick Start (Local)

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd memorypal-enhanced-v2

# Run automated setup
python setup.py

# Start the application
python run.py
```

### Option 2: Manual Setup
```bash
# Clone and navigate
git clone <repository-url>
cd memorypal-enhanced-v2

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Create directories
mkdir temp outputs

# Start application
streamlit run enhanced_rag_app_v2.py
```

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- FFmpeg (for audio processing)
- 4GB+ RAM (8GB+ recommended)
- 2GB+ free disk space

### Detailed Installation Steps

1. **System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y ffmpeg python3-dev build-essential
   
   # macOS
   brew install ffmpeg
   
   # Windows (using chocolatey)
   choco install ffmpeg
   ```

2. **Python Environment**
   ```bash
   # Create virtual environment (recommended)
   python -m venv memorypal-env
   
   # Activate virtual environment
   # Linux/Mac:
   source memorypal-env/bin/activate
   # Windows:
   memorypal-env\Scripts\activate
   
   # Upgrade pip
   pip install --upgrade pip
   ```

3. **Install Dependencies**
   ```bash
   # Core dependencies
   pip install streamlit python-dotenv numpy PyMuPDF google-generativeai
   
   # Audio processing
   pip install audio-recorder-streamlit openai-whisper librosa soundfile
   
   # Advanced features (optional)
   pip install transformers torch pyannote.audio pyttsx3 cohere
   
   # Or install all at once
   pip install -r requirements.txt
   ```

4. **Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your API keys
   nano .env  # or use your preferred editor
   ```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Create docker-compose.yml**
   ```yaml
   version: '3.8'
   
   services:
     memorypal:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - ./temp:/app/temp
         - ./outputs:/app/outputs
         - ./.env:/app/.env
       environment:
         - STREAMLIT_SERVER_PORT=8501
         - STREAMLIT_SERVER_ADDRESS=0.0.0.0
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

2. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       ffmpeg \
       curl \
       build-essential \
       && rm -rf /var/lib/apt/lists/*
   
   # Set working directory
   WORKDIR /app
   
   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Create necessary directories
   RUN mkdir -p temp outputs
   
   # Expose port
   EXPOSE 8501
   
   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8501/_stcore/health || exit 1
   
   # Run application
   CMD ["streamlit", "run", "enhanced_rag_app_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

3. **Deploy with Docker Compose**
   ```bash
   # Build and start
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   
   # Stop
   docker-compose down
   ```

### Standalone Docker

```bash
# Build image
docker build -t memorypal-enhanced-v2 .

# Run container
docker run -d \
  --name memorypal \
  -p 8501:8501 \
  -v $(pwd)/temp:/app/temp \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/.env:/app/.env \
  memorypal-enhanced-v2
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `enhanced_rag_app_v2.py`
   - Add secrets in settings (API keys)

3. **Secrets Configuration**
   ```toml
   # In Streamlit Cloud secrets
   GOOGLE_API_KEY = "your_api_key_here"
   HUGGINGFACE_TOKEN = "your_token_here"
   COHERE_API_KEY = "your_cohere_key_here"
   ```

### Heroku

1. **Create Procfile**
   ```
   web: streamlit run enhanced_rag_app_v2.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create setup.sh**
   ```bash
   #!/bin/bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**
   ```bash
   # Install Heroku CLI and login
   heroku login
   
   # Create app
   heroku create memorypal-enhanced-v2
   
   # Set config vars (API keys)
   heroku config:set GOOGLE_API_KEY=your_key_here
   
   # Deploy
   git push heroku main
   ```

### Google Cloud Platform

1. **Create app.yaml**
   ```yaml
   runtime: python39
   
   env_variables:
     GOOGLE_API_KEY: "your_api_key_here"
     HUGGINGFACE_TOKEN: "your_token_here"
   
   automatic_scaling:
     min_instances: 0
     max_instances: 2
   
   resources:
     cpu: 1
     memory_gb: 2
   ```

2. **Deploy to App Engine**
   ```bash
   # Install Google Cloud SDK
   gcloud init
   
   # Deploy
   gcloud app deploy
   ```

### AWS EC2

1. **Launch EC2 Instance**
   - Use Ubuntu 20.04 LTS
   - t3.medium or larger recommended
   - Security group: Allow HTTP (80) and HTTPS (443)

2. **Set up on EC2**
   ```bash
   # Connect to instance
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install dependencies
   sudo apt install -y python3 python3-pip nginx ffmpeg
   
   # Clone repository
   git clone <your-repo-url>
   cd memorypal-enhanced-v2
   
   # Install Python dependencies
   pip3 install -r requirements.txt
   
   # Set up environment
   cp .env.example .env
   nano .env  # Add your API keys
   
   # Run with systemd (production)
   sudo nano /etc/systemd/system/memorypal.service
   ```

3. **Create systemd service**
   ```ini
   [Unit]
   Description=MemoryPal Enhanced V2
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/memorypal-enhanced-v2
   ExecStart=/usr/bin/python3 -m streamlit run enhanced_rag_app_v2.py --server.port=8501
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

4. **Configure Nginx (optional)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

## 🔧 Environment Configuration

### Required Environment Variables
```bash
# Essential
GOOGLE_API_KEY=your_google_gemini_api_key

# Optional but recommended
HUGGINGFACE_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_api_key

# Optional database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key

# Optional search
SERPAPI_KEY=your_serpapi_key
```

### Production Configuration
```bash
# Performance settings
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Security settings
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Audio processing
WHISPER_MODEL=base
MAX_AUDIO_DURATION=600  # 10 minutes
```

## ⚡ Performance Optimization

### Memory Management
```python
# In enhanced_rag_app_v2.py, add these optimizations:

# Limit conversation history
CONVERSATION_CONTEXT_WINDOW = 10

# Reduce chunk size for large documents
CHUNK_SIZE = 800
MAX_CHUNKS_PER_DOCUMENT = 30

# Use smaller Whisper model for faster processing
WHISPER_MODEL = "base"  # Instead of "large"
```

### Caching Strategies
```python
# Add caching decorators
@st.cache_data
def load_model():
    return whisper.load_model("base")

@st.cache_data
def process_document(file_path):
    # Document processing logic
    pass
```

### Database Optimization
```python
# Use connection pooling
import psycopg2.pool

# Create connection pool
pool = psycopg2.pool.SimpleConnectionPool(1, 20, **db_config)
```

## 🐛 Troubleshooting

### Common Issues

**1. Memory Errors**
```bash
# Solution: Increase system memory or reduce model size
export WHISPER_MODEL=tiny
export MAX_CHUNKS=20
```

**2. Audio Processing Fails**
```bash
# Check ffmpeg installation
ffmpeg -version

# Reinstall if needed
sudo apt install --reinstall ffmpeg
```

**3. Import Errors**
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version
```

**4. Streamlit Port Issues**
```bash
# Use different port
streamlit run enhanced_rag_app_v2.py --server.port=8502
```

**5. API Rate Limits**
```bash
# Add delays between requests
import time
time.sleep(1)  # Add to API calls
```

### Health Checks

Create a health check endpoint:
```python
# Add to enhanced_rag_app_v2.py
@st.cache_data
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }
```

### Monitoring

1. **Application Logs**
   ```bash
   # View Streamlit logs
   tail -f ~/.streamlit/logs/streamlit.log
   
   # Custom logging
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Performance Monitoring**
   ```python
   # Add performance tracking
   import time
   start_time = time.time()
   # ... process ...
   processing_time = time.time() - start_time
   st.sidebar.metric("Processing Time", f"{processing_time:.2f}s")
   ```

### Backup and Recovery

1. **Data Backup**
   ```bash
   # Backup outputs and config
   tar -czf backup-$(date +%Y%m%d).tar.gz outputs .env temp
   ```

2. **Database Backup** (if using Supabase)
   ```bash
   # Export data
   pg_dump --host=db.your-project.supabase.co --username=postgres --dbname=postgres > backup.sql
   ```

## 📊 Resource Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 5GB
- **Network**: 1 Mbps

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 20GB+ SSD
- **Network**: 10+ Mbps

### Scaling Considerations
- **Horizontal**: Use load balancer with multiple instances
- **Vertical**: Increase CPU/RAM for single instance
- **Database**: Separate database server for production
- **Storage**: Use cloud storage for large files

---

## 🎯 Next Steps

After successful deployment:

1. **Configure monitoring** and alerts
2. **Set up automated backups**
3. **Implement CI/CD pipeline**
4. **Add SSL certificates** for production
5. **Configure domain name** and DNS
6. **Set up user authentication** if needed
7. **Implement rate limiting**
8. **Add analytics tracking**

## 📞 Support

Need help with deployment?

- Check the main [README.md](README.md) for general information
- Review [troubleshooting](#troubleshooting) section above
- Open an issue on GitHub with deployment details
- Include system information and error logs

---

**Happy Deploying! 🚀✨**