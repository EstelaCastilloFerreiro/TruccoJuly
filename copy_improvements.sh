#!/bin/bash

# Script to copy improved files to the forked repository
echo "🔄 Copying improved files to forked repository..."

# Define source and destination directories
SOURCE_DIR="/Users/carlotaperezprieto/Downloads/TRUCCONEW-main"
DEST_DIR="/Users/carlotaperezprieto/Downloads/TRUCCONEW"

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "❌ Destination directory $DEST_DIR does not exist!"
    echo "Please clone your forked repository first:"
    echo "git clone https://github.com/carlotapprieto/TRUCCONEW.git"
    exit 1
fi

# Copy all files and directories
echo "📁 Copying files..."
cp -r "$SOURCE_DIR"/* "$DEST_DIR/"

# Copy hidden files (like .gitignore)
echo "📁 Copying hidden files..."
cp -r "$SOURCE_DIR"/.* "$DEST_DIR/" 2>/dev/null || true

# Remove the .git directory if it was copied (to keep the fork's git history)
if [ -d "$DEST_DIR/.git" ]; then
    echo "🗑️ Removing copied .git directory to preserve fork history..."
    rm -rf "$DEST_DIR/.git"
fi

echo "✅ Files copied successfully!"
echo ""
echo "📋 Next steps:"
echo "1. cd $DEST_DIR"
echo "2. git add ."
echo "3. git commit -m \"Major improvements: Added advanced ML models, improved predictions, and enhanced dashboard features\""
echo "4. git push origin main"
echo ""
echo "🎉 Your improved version will be available on GitHub!" 