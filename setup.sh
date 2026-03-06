#!/bin/bash
# ============================================
# SETUP SCRIPT — RAG Chatbot Project
# ============================================
# Run this script to set up everything:
#   chmod +x setup.sh && ./setup.sh
# ============================================

echo "🚀 Setting up RAG Chatbot Project..."
echo "======================================"

# Step 1: Create virtual environment
echo ""
echo "📦 Step 1: Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip
echo ""
echo "⬆️  Step 2: Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies
echo ""
echo "📥 Step 3: Installing dependencies..."
echo "This may take a few minutes (downloading ML models, etc.)"
pip install -r requirements.txt

# Step 4: Create .env file if it doesn't exist
echo ""
echo "🔑 Step 4: Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "   Created .env file from .env.example"
    echo "   ⚠️  IMPORTANT: Edit .env and add your Groq API key!"
    echo "   Get a FREE key at: https://console.groq.com/keys"
else
    echo "   .env file already exists"
fi

# Step 5: Create necessary directories
echo ""
echo "📁 Step 5: Creating directories..."
mkdir -p sample_docs
mkdir -p uploaded_docs
mkdir -p chroma_db

# Step 6: Verify installation
echo ""
echo "✅ Step 6: Verifying installation..."
python -c "
import sys
modules = {
    'PyPDF2': 'PDF loading',
    'chromadb': 'Vector database', 
    'sentence_transformers': 'Local embeddings',
    'fastapi': 'API framework',
    'streamlit': 'UI framework',
    'groq': 'LLM integration',
    'dotenv': 'Environment variables',
    'rich': 'Terminal output',
}
all_good = True
for module, purpose in modules.items():
    try:
        __import__(module)
        print(f'   ✅ {module} ({purpose})')
    except ImportError:
        print(f'   ❌ {module} ({purpose}) - MISSING!')
        all_good = False

if all_good:
    print('\n   🎉 All dependencies installed successfully!')
else:
    print('\n   ⚠️  Some dependencies are missing. Run: pip install -r requirements.txt')
"

echo ""
echo "======================================"
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit .env and add your GROQ_API_KEY"
echo "      (Get FREE key at: https://console.groq.com/keys)"
echo ""
echo "   2. Activate the virtual environment:"
echo "      source venv/bin/activate"
echo ""
echo "   3. Run the step-by-step walkthrough:"
echo "      python test_pipeline.py"
echo ""
echo "   4. Or run the Streamlit chat UI:"
echo "      streamlit run streamlit_app.py"
echo ""
echo "   5. Or run the FastAPI server:"
echo "      python app.py"
echo "      Then visit: http://localhost:8000/docs"
echo ""
echo "======================================"
