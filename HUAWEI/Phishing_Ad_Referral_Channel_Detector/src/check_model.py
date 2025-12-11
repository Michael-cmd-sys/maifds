"""
Diagnostic script to check if model files exist and are loadable
"""

import os
import sys

print("=" * 70)
print("MindSpore Model Diagnostic Tool")
print("=" * 70)

# Check directory structure
print("\n1. Checking directory structure...")
dirs_to_check = ['data', 'data/mindspore_models']

for dir_path in dirs_to_check:
    if os.path.exists(dir_path):
        print(f"   ✓ {dir_path} exists")
    else:
        print(f"   ✗ {dir_path} NOT FOUND")
        os.makedirs(dir_path, exist_ok=True)
        print(f"     Created {dir_path}")

# Check model files
print("\n2. Checking model files...")
models_dir = 'data/mindspore_models'

if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    print(f"   Found {len(files)} files in {models_dir}:")
    
    for f in files:
        file_path = os.path.join(models_dir, f)
        file_size = os.path.getsize(file_path)
        print(f"     • {f} ({file_size:,} bytes)")
    
    # Check for required files
    print("\n3. Checking required files...")
    
    feature_config = os.path.join(models_dir, 'feature_config.pkl')
    if os.path.exists(feature_config):
        print(f"   ✓ feature_config.pkl exists")
        
        # Try to load it
        try:
            import pickle
            with open(feature_config, 'rb') as f:
                config = pickle.load(f)
            print(f"     • Features: {len(config['feature_names'])}")
            print(f"     • Feature names: {config['feature_names'][:5]}...")
        except Exception as e:
            print(f"   ✗ Error loading feature_config.pkl: {e}")
    else:
        print(f"   ✗ feature_config.pkl NOT FOUND")
    
    # Check for checkpoint files
    ckpt_files = [f for f in files if f.endswith('.ckpt')]
    if ckpt_files:
        print(f"   ✓ Found {len(ckpt_files)} checkpoint file(s):")
        for ckpt in ckpt_files:
            print(f"     • {ckpt}")
    else:
        print(f"   ✗ No .ckpt files found")
else:
    print(f"   ✗ Models directory does not exist")

# Check config.json
print("\n4. Checking config.json...")
if os.path.exists('config.json'):
    print("   ✓ config.json exists")
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        print(f"     • Models directory: {config['storage']['models_dir']}")
        print(f"     • Checkpoint prefix: {config['storage']['checkpoint_prefix']}")
    except Exception as e:
        print(f"   ✗ Error loading config.json: {e}")
else:
    print("   ✗ config.json NOT FOUND")

# Check watchlist
print("\n5. Checking watchlist...")
watchlist_file = 'data/watchlist.json'
if os.path.exists(watchlist_file):
    print("   ✓ watchlist.json exists")
    try:
        import json
        with open(watchlist_file, 'r') as f:
            watchlist = json.load(f)
        print(f"     • Watchlist size: {len(watchlist)}")
    except Exception as e:
        print(f"   ✗ Error loading watchlist: {e}")
else:
    print("   ✗ watchlist.json NOT FOUND")
    print("     Creating empty watchlist...")
    with open(watchlist_file, 'w') as f:
        json.dump([], f)
    print("     ✓ Created")

# Try to load the detector
print("\n6. Testing detector initialization...")
try:
    sys.path.append('.')
    from mindspore_detector import MindSporePhishingDetector
    
    detector = MindSporePhishingDetector()
    
    print(f"   ✓ Detector initialized")
    print(f"     • Is trained: {detector.is_trained}")
    print(f"     • Feature count: {len(detector.feature_names) if detector.feature_names else 0}")
    print(f"     • Watchlist size: {len(detector.watchlist)}")
    
    if detector.is_trained:
        print("\n   ✅ MODEL IS LOADED AND READY!")
    else:
        print("\n   ⚠️  MODEL NOT LOADED")
        print("      Possible reasons:")
        print("      1. No checkpoint files found")
        print("      2. Feature config missing")
        print("      3. Training was not completed")
        print("\n      Solution: Run 'python train_mindspore.py'")
        
except Exception as e:
    print(f"   ✗ Error initializing detector: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Diagnostic Complete")
print("=" * 70)
print("\nRecommended actions:")
print("  1. If training completed successfully, check checkpoint files")
print("  2. If no checkpoints, run: python train_mindspore.py")
print("  3. After training, restart API: python api_mindspore.py")
print("=" * 70)