#!/bin/bash
# setup_macos.sh - Setup script for HiFiGAN training on macOS

echo "🍎 Setting up HiFiGAN for macOS training..."
echo "============================================"

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "❌ Error: train.py not found. Please run this script from the ParrotVocoder directory."
    exit 1
fi

echo "✅ Checking Python environment..."
python -c "import torch; print(f'PyTorch {torch.__version__} available')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

echo "📦 Checking required packages..."
python -c "
import sys
required_packages = ['torch', 'numpy', 'librosa', 'scipy', 'tensorboard', 'soundfile', 'matplotlib']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} - MISSING')
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    print('Please install them with: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('All required packages are available!')
"

echo "📁 Checking dataset..."
if [ ! -d "LJSpeech-1.1/wavs" ]; then
    echo "❌ LJSpeech dataset not found."
    echo "📥 Would you like to download it automatically? (This will take ~10-15 minutes)"
    read -p "Download LJSpeech dataset? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Starting dataset download..."
        python setup_dataset.py
    else
        echo "⚠️  Please download LJSpeech dataset manually:"
        echo "   1. Download from: https://keithito.com/LJ-Speech-Dataset/"
        echo "   2. Extract to this directory so you have: LJSpeech-1.1/wavs/"
        exit 1
    fi
else
    wav_count=$(find LJSpeech-1.1/wavs -name "*.wav" | wc -l)
    echo "✅ Found LJSpeech dataset with $wav_count audio files"
fi

echo "🔧 Creating checkpoint directory..."
mkdir -p cp_hifigan

echo "📋 Configuration summary:"
echo "  - Config file: config_v1.json"
echo "  - Training data: LJSpeech-1.1/training.txt"
echo "  - Validation data: LJSpeech-1.1/validation.txt"
echo "  - Audio files: LJSpeech-1.1/wavs/"
echo "  - Checkpoints: cp_hifigan/"
echo "  - Device: MPS (Apple Silicon GPU acceleration)"

echo ""
echo "🚀 Ready to start training!"
echo "Run the training with:"
echo "  python train.py --config config_v1.json"
echo ""
echo "💡 Training tips for macOS:"
echo "  - Monitor Activity Monitor for memory usage"
echo "  - Training will use MPS (Metal Performance Shaders) for acceleration"
echo "  - Reduce batch_size in config_v1.json if you run out of memory"
echo "  - Press Ctrl+C to stop training safely"
