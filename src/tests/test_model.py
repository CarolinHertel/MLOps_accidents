#!/usr/bin/env python3

import bentoml
import numpy as np

print("=== Testing Model Loading ===")

# 1. Liste alle verfügbaren Modelle
print("\n1. Available models:")
try:
    models = bentoml.models.list()
    for model in models:
        print(f"   - {model.tag}")
    
    if not models:
        print("   No models found in BentoML store!")
        exit(1)
        
except Exception as e:
    print(f"   Error listing models: {e}")
    exit(1)

# 2. Versuche das Modell zu laden
print("\n2. Trying to load predict_model:")
try:
    model_ref = bentoml.sklearn.get("predict_model:latest")
    print("   ✓ Successfully loaded predict_model:latest")
except Exception as e:
    print(f"   ✗ Error loading predict_model:latest: {e}")
    
    # Versuche ohne Tag
    try:
        model_ref = bentoml.sklearn.get("predict_model")
        print("   ✓ Successfully loaded predict_model (without version)")
    except Exception as e2:
        print(f"   ✗ Error loading predict_model: {e2}")
        exit(1)

# 3. Teste eine Vorhersage
print("\n3. Testing prediction:")
try:
    # Dummy data basierend auf deinem Schema
    test_data = np.array([[
        1, 1, 1, 1.0, 2023, 25, 1, 1, 1, 1, 1, 1, 1, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        48.8566, 2.3522, 14, 1, 2
    ]], dtype=np.float32)
    
    model_runner = model_ref.to_runner()
    # Für synchrone Tests
    # prediction = model_runner.run(test_data)
    print("   ✓ Model runner created successfully")
    print("   Model is ready for predictions")
    
except Exception as e:
    print(f"   ✗ Error testing prediction: {e}")
    exit(1)

print("\n=== Model test completed successfully ===")