#!/bin/bash
# Setup Python 3.13 virtual environment using uv


set -e  # Exit on error

echo "🚀 Setting up Python 3.13 virtual environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed successfully"
fi

# Remove existing venv if it exists
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new virtual environment with Python 3.13
echo "📦 Creating virtual environment with Python 3.13..."
uv venv --python 3.13

# Activate the virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install the package with dev dependencies
echo "📚 Installing rf-detr-mask with development dependencies..."
sudo CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install --use-pep517 onnxsim
uv pip install -e ".[dev,metrics,onnxexport]"

echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Python version:"
python --version
