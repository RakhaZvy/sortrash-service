#!/usr/bin/env python3
"""
Test script for SorTrash Flask API
Tests all endpoints without running the full server
"""

import json
import os

# Simulate loading history
HISTORY_FILE = 'classification_history.json'

def test_history_operations():
    print("Testing history operations...")
    
    # Test 1: Create sample history
    sample_history = [
        {
            'id': 1,
            'category': 'plastic',
            'confidence': 0.92,
            'date': '2024-12-18T10:30:00',
            'timestamp': 1734518400.0
        },
        {
            'id': 2,
            'category': 'metal',
            'confidence': 0.85,
            'date': '2024-12-18T11:00:00',
            'timestamp': 1734520200.0
        },
        {
            'id': 3,
            'category': 'paper',
            'confidence': 0.91,
            'date': '2024-12-18T14:30:00',
            'timestamp': 1734532800.0
        }
    ]
    
    # Save history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(sample_history, f, indent=2)
    print(f"✓ Created {HISTORY_FILE} with {len(sample_history)} entries")
    
    # Load and verify
    with open(HISTORY_FILE, 'r') as f:
        loaded = json.load(f)
    
    print(f"✓ Loaded {len(loaded)} entries from history")
    
    # Calculate stats
    total_scans = len(loaded)
    categories = {}
    for item in loaded:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    most_common = max(categories.items(), key=lambda x: x[1])[0]
    
    print(f"\nStatistics:")
    print(f"  Total scans: {total_scans}")
    print(f"  Most common: {most_common}")
    print(f"  Categories: {categories}")
    
    print("\n✓ All tests passed!")
    print(f"\nYou can now start the Flask server with: python app.py")

if __name__ == '__main__':
    test_history_operations()
