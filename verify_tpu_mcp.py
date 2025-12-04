import sys
import os
import asyncio
import json
import traceback

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

def test_tpu_engine():
    print("--- Testing TPU Engine ---")
    try:
        from coral_tpu_mcp.tpu_engine import get_engine
        engine = get_engine()
        print(f"Engine initialized.")
        print(f"TPU Available: {engine.is_available}")
        
        health = engine.health_check()
        print(f"Health Check: {json.dumps(health, indent=2)}")
        
        if not engine.is_available:
            print("TPU is correctly reported as unavailable (expected without pycoral/hardware).")
        else:
            print("TPU is available! (Unexpected but great if true)")
            
    except ImportError as e:
        print(f"Failed to import tpu_engine: {e}")
        traceback.print_exc() # Print full traceback
    except Exception as e:
        print(f"Error testing engine: {e}")
        traceback.print_exc() # Print full traceback

def test_text_model():
    print("\n--- Testing Text Model (CPU Fallback) ---")
    try:
        from coral_tpu_mcp.server import get_text_model
        model = get_text_model()
        if model:
            print("Text model loaded successfully.")
            print("Running sample embedding...")
            embedding = model.encode("This is a test sentence.")
            print(f"Embedding generated. Shape: {embedding.shape}")
            print("Text model is WORKING.")
        else:
            print("Text model failed to load.")
    except Exception as e:
        print(f"Error testing text model: {e}")
        traceback.print_exc() # Print full traceback

if __name__ == "__main__":
    test_tpu_engine()
    test_text_model()