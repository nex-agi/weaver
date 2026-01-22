# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Weaver SDK Release Script
# This script builds and uploads the package to PyPI

set -e  # Exit on any error

echo "======================================"
echo "Weaver SDK Release Script"
echo "======================================"
echo ""

# Get the version from pyproject.toml
VERSION=$(grep "^version = " pyproject.toml | cut -d'"' -f2)
echo "📦 Package version: $VERSION"
echo ""

# Step 1: Clean up old builds
echo "🧹 Cleaning old builds..."
if [ -d "dist" ]; then
    rm -rf dist/*
    echo "   ✓ Removed dist/*"
else
    echo "   ℹ dist/ directory doesn't exist yet"
fi
echo ""

# Step 2: Build the package
echo "🔨 Building package..."
python3 -m build
if [ $? -eq 0 ]; then
    echo "   ✓ Build successful"
else
    echo "   ✗ Build failed"
    exit 1
fi
echo ""

# Step 3: List built files
echo "📋 Built files:"
ls -lh dist/
echo ""

# Step 4: Upload to PyPI
echo "🚀 Uploading to PyPI..."
read -p "Do you want to upload to PyPI? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    twine upload dist/*
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully released version $VERSION to PyPI!"
        echo ""
        echo "Install with: pip install --upgrade weaver==$VERSION"
    else
        echo ""
        echo "✗ Upload failed"
        exit 1
    fi
else
    echo ""
    echo "⏭️  Upload skipped. Built files are in dist/"
    echo ""
    echo "To upload manually, run:"
    echo "  twine upload dist/*"
fi

echo ""
echo "======================================"
echo "Done! 🎉"
echo "======================================"
