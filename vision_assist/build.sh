#!/usr/bin/env bash
set -e

echo "==> Installing system dependencies..."
apt-get update -qq
apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Running Django setup..."
python manage.py collectstatic --no-input
python manage.py migrate

echo "==> Build complete!"
