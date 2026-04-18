#!/usr/bin/env bash
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Running Django setup..."
python manage.py collectstatic --no-input
python manage.py migrate

echo "==> Build complete!"