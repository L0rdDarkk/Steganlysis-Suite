#!/usr/bin/env python3
"""
Simple Dataset Setup Script for StegAnalysis Suite

This script copies 1000 images from your ALASKA dataset to the project structure
for training and testing.
"""

import os
import shutil
import random
from pathlib import Path

def simple_progress(iterable, desc="Processing"):
    """Simple progress indicator without external dependencies."""
    total = len(iterable)
    for i, item in enumerate(iterable, 1):
        if i % max(1, total // 10) == 0 or i == total:  # Show progress every 10%
            percent = (i / total) * 100
            print(f"{desc}: {i}/{total} ({percent:.1f}%)")
        yield item

def setup_directories():
    """Create the required directory structure."""
    print("📁 Creating directory structure...")
    
    directories = [
        "datasets/images/clean",
        "datasets/images/steganographic", 
        "datasets/images/test",
        "datasets/metadata",
        "models/trained",
        "logs",
        "reports/generated"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created!")

def copy_dataset_subset(alaska_path, num_images=250):
    """
    Copy a subset of ALASKA dataset to our project structure.
    
    Args:
        alaska_path: Path to your ALASKA dataset folder
        num_images: Number of images per category (default: 250 per category = 1000 total)
    """
    alaska_path = Path(alaska_path)
    
    if not alaska_path.exists():
        print(f"❌ Error: ALASKA dataset path not found: {alaska_path}")
        return False
    
    print(f"📂 Using ALASKA dataset from: {alaska_path}")
    
    # Check required directories
    required_dirs = {
        "Cover": "clean",
        "JMiPOD": "steganographic", 
        "JUNIWARD": "steganographic",
        "UERD": "steganographic"
    }
    
    for alaska_dir in required_dirs.keys():
        dir_path = alaska_path / alaska_dir
        if not dir_path.exists():
            print(f"❌ Error: Required directory not found: {dir_path}")
            return False
    
    print(f"🔄 Copying {num_images} images per category...")
    
    # Copy cover images
    print("📋 Copying cover images...")
    cover_dir = alaska_path / "Cover"
    cover_files = list(cover_dir.glob("*.jpg"))[:num_images]
    
    for i, src_file in enumerate(simple_progress(cover_files, "Cover images")):
        dst_file = Path("datasets/images/clean") / f"cover_{i+1:04d}.jpg"
        shutil.copy2(src_file, dst_file)
    
    # Copy steganographic images
    stego_categories = ["JMiPOD", "JUNIWARD", "UERD"]
    stego_counter = 1
    
    for category in stego_categories:
        print(f"📋 Copying {category} images...")
        category_dir = alaska_path / category
        category_files = list(category_dir.glob("*.jpg"))[:num_images]
        
        for src_file in simple_progress(category_files, f"{category} images"):
            dst_file = Path("datasets/images/steganographic") / f"stego_{stego_counter:04d}_{category.lower()}.jpg"
            shutil.copy2(src_file, dst_file)
            stego_counter += 1
    
    print(f"✅ Dataset setup complete!")
    print(f"   - Cover images: {len(cover_files)}")
    print(f"   - Steganographic images: {(stego_counter-1)}")
    print(f"   - Total images: {len(cover_files) + (stego_counter-1)}")
    
    return True

def create_simple_splits():
    """Create simple train/validation/test splits."""
    print("🔀 Creating train/validation/test splits...")
    
    # Get all images
    clean_files = list(Path("datasets/images/clean").glob("*.jpg"))
    stego_files = list(Path("datasets/images/steganographic").glob("*.jpg"))
    
    # Create splits (70% train, 15% val, 15% test)
    def split_files(files, train_ratio=0.7, val_ratio=0.15):
        random.shuffle(files)
        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        return {
            'train': files[:train_end],
            'val': files[train_end:val_end], 
            'test': files[val_end:]
        }
    
    clean_splits = split_files(clean_files)
    stego_splits = split_files(stego_files)
    
    # Copy test images to test directory
    test_dir = Path("datasets/images/test")
    for test_file in clean_splits['test'] + stego_splits['test']:
        dst_file = test_dir / test_file.name
        shutil.copy2(test_file, dst_file)
    
    # Save split information
    import json
    splits_info = {
        "train": {
            "clean": [f.name for f in clean_splits['train']],
            "steganographic": [f.name for f in stego_splits['train']]
        },
        "validation": {
            "clean": [f.name for f in clean_splits['val']],
            "steganographic": [f.name for f in stego_splits['val']]
        },
        "test": {
            "clean": [f.name for f in clean_splits['test']],
            "steganographic": [f.name for f in stego_splits['test']]
        }
    }
    
    with open("datasets/metadata/splits.json", "w") as f:
        json.dump(splits_info, f, indent=2)
    
    print("✅ Splits created!")
    print(f"   - Train: {len(clean_splits['train']) + len(stego_splits['train'])} images")
    print(f"   - Validation: {len(clean_splits['val']) + len(stego_splits['val'])} images") 
    print(f"   - Test: {len(clean_splits['test']) + len(stego_splits['test'])} images")

def check_alaska_structure(alaska_path):
    """Check if the ALASKA dataset has the correct structure."""
    alaska_path = Path(alaska_path)
    
    print(f"🔍 Checking ALASKA dataset structure at: {alaska_path}")
    
    if not alaska_path.exists():
        print(f"❌ Path does not exist: {alaska_path}")
        return False
    
    required_dirs = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    found_dirs = []
    
    for dir_name in required_dirs:
        dir_path = alaska_path / dir_name
        if dir_path.exists():
            jpg_count = len(list(dir_path.glob("*.jpg")))
            print(f"✅ Found {dir_name}/ - {jpg_count} .jpg files")
            found_dirs.append(dir_name)
        else:
            print(f"❌ Missing {dir_name}/ directory")
    
    if len(found_dirs) == len(required_dirs):
        print("✅ ALASKA dataset structure is correct!")
        return True
    else:
        print(f"❌ Missing directories: {set(required_dirs) - set(found_dirs)}")
        return False

def suggest_paths():
    """Suggest common ALASKA dataset paths based on OS."""
    import platform
    
    print("\n💡 Common ALASKA dataset locations:")
    
    if platform.system() == "Windows":
        suggestions = [
            "C:\\Users\\YourName\\Downloads\\ALASKA_v2.0",
            "D:\\Datasets\\ALASKA_v2.0",
            "C:\\Data\\ALASKA_v2.0"
        ]
    else:  # Linux/Mac
        suggestions = [
            "/home/$(whoami)/Downloads/ALASKA_v2.0",
            "/home/$(whoami)/datasets/ALASKA_v2.0",
            "/data/ALASKA_v2.0",
            "~/Downloads/ALASKA_v2.0"
        ]
    
    for i, path in enumerate(suggestions, 1):
        print(f"   {i}. {path}")
    
    print("\n📁 Or look for a folder containing: Cover/, JMiPOD/, JUNIWARD/, UERD/")

def main():
    """Main setup function."""
    print("🚀 StegAnalysis Suite - Dataset Setup")
    print("=" * 50)
    print("This will copy 1000 images from your ALASKA dataset for training.")
    print("We need: 250 Cover + 250 JMiPOD + 250 JUNIWARD + 250 UERD = 1000 total")
    print()
    
    # Pre-fill the known path for this user
    default_path = r"C:\Users\admin\Desktop\Juled\CIT MASTER 2025-2026\Scientific_Research\Scientific_Research\alaska2-image-steganalysis"
    
    print(f"🎯 Detected ALASKA dataset path:")
    print(f"   {default_path}")
    print()
    
    use_default = input("📂 Use this path? (y/n) [y]: ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        alaska_path = default_path
    else:
        # Suggest common paths
        suggest_paths()
        
        # Get ALASKA dataset path
        while True:
            print("\n" + "="*50)
            alaska_path = input("📂 Enter the full path to your ALASKA dataset folder: ").strip()
            
            if not alaska_path:
                print("❌ No path provided.")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Exiting...")
                    return
                continue
            
            # Remove quotes if user copied path with quotes
            alaska_path = alaska_path.strip('"').strip("'")
            break
    
    # Check if path is correct
    if not check_alaska_structure(alaska_path):
        print("\n❌ This doesn't look like a valid ALASKA dataset.")
        print("Make sure the path contains folders: Cover/, JMiPOD/, JUNIWARD/, UERD/")
        print("Please check your ALASKA dataset and try again.")
        return
    
    print(f"✅ Great! Using ALASKA dataset from: {alaska_path}")
    
    print("\n" + "="*50)
    print("🚀 Starting dataset setup...")
    
    # Setup directories
    setup_directories()
    
    # Copy dataset subset
    if copy_dataset_subset(alaska_path, num_images=250):
        # Create splits
        create_simple_splits()
        
        print("\n" + "="*50)
        print("🎉 SETUP COMPLETE! 🎉")
        print("\n📊 What was created:")
        print("   ├── 250 cover images → datasets/images/clean/")
        print("   ├── 750 stego images → datasets/images/steganographic/")
        print("   ├── 150 test images → datasets/images/test/")
        print("   └── metadata files → datasets/metadata/")
        
        print("\n🚀 NEXT STEPS:")
        print("1️⃣  Install requirements:")
        print("   pip install -r requirements.txt")
        print()
        print("2️⃣  Train the models:")
        print("   python tools/model_trainer.py")
        print()
        print("3️⃣  Start using it:")
        print("   python main.py --gui")
        print()
        print("🎯 Your stegananalysis system is ready for training!")
        
    else:
        print("\n❌ Setup failed. Please check your ALASKA dataset path and try again.")

if __name__ == "__main__":
    main()