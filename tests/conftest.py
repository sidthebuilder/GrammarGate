import sys

# Prevent outdated torchvision build from crashing transformers image_utils
sys.modules["torchvision"] = None
