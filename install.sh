#!/bin/bash
echo "Step 1: Installing main packages..."
pip install -r requirements.txt
echo "Step 2: Reinstalling pandas..."
pip install pandas==2.2.2 --force-reinstall
echo "Step 2: Installing mordred without dependencies..."
pip install mordred==1.2.0 --no-deps
echo "Step 3: Force-pinning numpy to 2.0.2..."
pip install "numpy==2.0.2" --force-reinstall
echo "Step 4: Installing torch..."
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
echo "Done!"