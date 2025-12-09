# Function: Merge prediction results from text and vision models
# Input: JSON files containing predictions from text and vision models
# Output: Unified multimodal prediction JSON file
# 
# Merge rules:
# 1. Match data from both files by image name
# 2. Only merge records with consistent true_label
# 3. Retain all fields from the text model and add vision prediction probabilities
# 
# Purpose: Prepare data for multimodal fusion analysis

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def merge_predictions_files(file_a_path: str, file_b_path: str, output_dir: str):
    """
    Merge prediction results from two JSON files
    
    Args:
        file_a_path: Path to file A (text model predictions)
        file_b_path: Path to file B (vision model predictions)
        output_dir: Output directory
    """
    
    print("=" * 80)
    print("Starting to merge predictions from text and vision models")
    print("=" * 80)
    
    # Check if files exist
    if not os.path.exists(file_a_path):
        print(f"❌ Error: File A does not exist: {file_a_path}")
        return
    
    if not os.path.exists(file_b_path):
        print(f"❌ Error: File B does not exist: {file_b_path}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load file A
        print(f"📁 Loading file A: {file_a_path}")
        with open(file_a_path, 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        
        # Load file B
        print(f"📁 Loading file B: {file_b_path}")
        with open(file_b_path, 'r', encoding='utf-8') as f:
            data_b = json.load(f)
        
        print(f"✓ Loaded file A: {len(data_a)} records")
        print(f"✓ Loaded file B: {len(data_b)} records")
        
        # Create a name-to-data mapping for quick lookup in file B
        b_name_to_data = {}
        for item in data_b:
            name = item.get('name')
            if name:
                b_name_to_data[name] = item
        
        print(f"✓ Created index mapping for file B with {len(b_name_to_data)} unique names")
        
        # Initialize mismatched data lists
        mismatched_names_a = []  # Names unmatched in file A
        mismatched_names_b = []  # Names unmatched in file B
        matched_count = 0
        mismatch_label_count = 0
        
        # Statistics
        stats = {
            "total_matched": 0,
            "total_unmatched_a": 0,
            "total_unmatched_b": 0,
            "mismatch_label_count": 0,
            "prediction_match_stats": {
                "match": 0,
                "mismatch": 0,
                "match_rate": 0.0
            },
            "model_accuracy_stats": {
                "text_correct": 0,
                "vision_correct": 0,
                "both_correct": 0,
                "text_accuracy": 0.0,
                "vision_accuracy": 0.0,
                "both_accuracy": 0.0
            },
            "class_distribution": defaultdict(int),
            "prediction_agreement_by_class": defaultdict(lambda: {"match": 0, "total": 0})
        }
        
        # Create merged data
        merged_data = []
        
        print("\n🔍 Starting to match and merge data...")
        
        # Process each record in file A
        for i, item_a in enumerate(data_a):
            name_a = item_a.get('name')
            
            if not name_a:
                print(f"⚠ Warning: Record {i+1} in file A lacks the 'name' field, skipping")
                mismatched_names_a.append(f"Record {i+1} (no name)")
                continue
            
            # Find matching item in file B
            if name_a in b_name_to_data:
                item_b = b_name_to_data[name_a]
                
                # Check if true_label matches
                true_label_a = item_a.get('true_label')
                true_label_b = item_b.get('true_label')
                
                if true_label_a == true_label_b:
                    # Get prediction probabilities
                    text_probs = item_a.get('text_prediction_probs', [])
                    vision_probs = item_b.get('vision_prediction_probs', [])
                    
                    # Calculate predicted labels (index of max probability)
                    text_predicted_label = text_probs.index(max(text_probs)) if text_probs else -1
                    vision_predicted_label = vision_probs.index(max(vision_probs)) if vision_probs else -1
                    
                    # Check if predictions match
                    prediction_match = "Yes" if text_predicted_label == vision_predicted_label else "No"
                    
                    # Check if predictions are correct
                    text_correct = 1 if text_predicted_label == true_label_a else 0
                    vision_correct = 1 if vision_predicted_label == true_label_a else 0
                    both_correct = 1 if (text_correct and vision_correct) else 0
                    
                    # Update statistics
                    stats["total_matched"] += 1
                    if prediction_match == "Yes":
                        stats["prediction_match_stats"]["match"] += 1
                    else:
                        stats["prediction_match_stats"]["mismatch"] += 1
                    
                    stats["model_accuracy_stats"]["text_correct"] += text_correct
                    stats["model_accuracy_stats"]["vision_correct"] += vision_correct
                    stats["model_accuracy_stats"]["both_correct"] += both_correct
                    
                    stats["class_distribution"][true_label_a] += 1
                    stats["prediction_agreement_by_class"][true_label_a]["total"] += 1
                    if prediction_match == "Yes":
                        stats["prediction_agreement_by_class"][true_label_a]["match"] += 1
                    
                    # Merge data, constructing the dictionary in the specified order
                    merged_item = {
                        'name': name_a,
                        'true_label': true_label_a,
                        'vision_prediction_probs': vision_probs,
                        'vision_predicted_label': vision_predicted_label,
                        'text_prediction_probs': text_probs,
                        'text_predicted_label': text_predicted_label,
                        # 'match': prediction_match
                    }
                    
                    merged_data.append(merged_item)
                    matched_count += 1
                    
                    # Remove matched item from the mapping
                    del b_name_to_data[name_a]
                else:
                    # true_label does not match
                    print(f"⚠ Warning: Image '{name_a}' has mismatched true_label (A:{true_label_a}, B:{true_label_b})")
                    mismatched_names_a.append(name_a)
                    mismatched_names_b.append(name_a)
                    mismatch_label_count += 1
                    stats["mismatch_label_count"] += 1
            else:
                # No match found in file B
                print(f"⚠ Warning: Image '{name_a}' not found in file B")
                mismatched_names_a.append(name_a)
                stats["total_unmatched_a"] += 1
        
        # Process remaining unmatched items in file B
        for name_b, item_b in b_name_to_data.items():
            print(f"⚠ Warning: Image '{name_b}' not found in file A")
            mismatched_names_b.append(name_b)
            stats["total_unmatched_b"] += 1
        
        # Calculate statistics ratios
        if stats["total_matched"] > 0:
            stats["prediction_match_stats"]["match_rate"] = stats["prediction_match_stats"]["match"] / stats["total_matched"]
            stats["model_accuracy_stats"]["text_accuracy"] = stats["model_accuracy_stats"]["text_correct"] / stats["total_matched"]
            stats["model_accuracy_stats"]["vision_accuracy"] = stats["model_accuracy_stats"]["vision_correct"] / stats["total_matched"]
            stats["model_accuracy_stats"]["both_accuracy"] = stats["model_accuracy_stats"]["both_correct"] / stats["total_matched"]
        
        # Generate output file path
        merged_file_path = os.path.join(output_dir, "multimodal_fusion_dataset.json")
        
        # Save merged data
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Merge complete!")
        print("=" * 60)
        print("Merge Statistics:")
        print(f"  Successfully matched and merged: {matched_count} records")
        print(f"  Mismatched true_label: {mismatch_label_count} records")
        print(f"  Unmatched in file A: {len(mismatched_names_a)} records")
        print(f"  Unmatched in file B: {len(mismatched_names_b)} records")
        print(f"  Total merged records: {len(merged_data)}")
        print(f"\n📁 Merged file saved to: {merged_file_path}")
        
        # Save mismatched data information
        if mismatched_names_a or mismatched_names_b:
            mismatch_info = {
                "mismatch_summary": {
                    "total_matched": matched_count,
                    "total_mismatch_label": mismatch_label_count,
                    "total_unmatched_a": len(mismatched_names_a),
                    "total_unmatched_b": len(mismatched_names_b)
                },
                "unmatched_in_file_a": mismatched_names_a,
                "unmatched_in_file_b": mismatched_names_b,
                "both_unmatched": list(set(mismatched_names_a) & set(mismatched_names_b))
            }
            
            # mismatch_file_path = os.path.join(output_dir, "mismatch_info.json")
            # with open(mismatch_file_path, 'w', encoding='utf-8') as f:
            #     json.dump(mismatch_info, f, ensure_ascii=False, indent=2)
            
            # print(f"📁 Mismatch information saved to: {mismatch_file_path}")
            
            # Print mismatch examples
            print("\n⚠ Mismatch examples:")
            if mismatched_names_a:
                print(f"  First 5 unmatched names in file A: {mismatched_names_a[:5]}")
            if mismatched_names_b:
                print(f"  First 5 unmatched names in file B: {mismatched_names_b[:5]}")
        
        # Generate detailed statistics file
        stats["total_samples_file_a"] = len(data_a)
        stats["total_samples_file_b"] = len(data_b)
        stats["mismatched_names_a"] = mismatched_names_a
        stats["mismatched_names_b"] = mismatched_names_b
        
        # Convert defaultdict to regular dict for JSON serialization
        stats["class_distribution"] = dict(stats["class_distribution"])
        stats["prediction_agreement_by_class"] = dict(stats["prediction_agreement_by_class"])
        
        # Calculate match rate for each class
        for class_label, agreement_data in stats["prediction_agreement_by_class"].items():
            if agreement_data["total"] > 0:
                agreement_data["match_rate"] = agreement_data["match"] / agreement_data["total"]
            else:
                agreement_data["match_rate"] = 0.0
        
        stats_file_path = os.path.join(output_dir, "fusion_statistics.json")
        with open(stats_file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Detailed statistics saved to: {stats_file_path}")
        
        # Print merged data structure example
        if merged_data:
            print(f"\n📋 Example of merged data structure (first record):")
            sample = merged_data[0]
            print(f"  Name: {sample.get('name')}")
            print(f"  True label: {sample.get('true_label')}")
            print(f"  Vision prediction probabilities: {sample.get('vision_prediction_probs')[:3] if sample.get('vision_prediction_probs') else 'N/A'}...")
            print(f"  Vision predicted label: {sample.get('vision_predicted_label')}")
            print(f"  Text prediction probabilities: {sample.get('text_prediction_probs')[:3] if sample.get('text_prediction_probs') else 'N/A'}...")
            print(f"  Text predicted label: {sample.get('text_predicted_label')}")
            print(f"  Prediction match: {sample.get('match')}")
            
            # Check all keys
            all_keys = list(sample.keys())
            print(f"  All key names: {all_keys}")
            print(f"  Field order: {all_keys}")
        
        # Print statistics summary
        print(f"\n📊 Statistics summary:")
        print(f"  Text-vision prediction match rate: {stats['prediction_match_stats']['match_rate']:.2%} ({stats['prediction_match_stats']['match']}/{stats['total_matched']})")
        print(f"  Text model accuracy: {stats['model_accuracy_stats']['text_accuracy']:.2%}")
        print(f"  Vision model accuracy: {stats['model_accuracy_stats']['vision_accuracy']:.2%}")
        print(f"  Both models correct rate: {stats['model_accuracy_stats']['both_accuracy']:.2%}")
        
        print(f"\n📈 Class distribution:")
        for class_label, count in sorted(stats["class_distribution"].items()):
            agreement = stats["prediction_agreement_by_class"].get(class_label, {"match": 0, "total": 0, "match_rate": 0.0})
            match_rate = agreement.get("match_rate", 0.0)
            print(f"  Class {class_label}: {count} samples, prediction match rate: {match_rate:.2%}")
        
        # Verify merged data
        print(f"\n🔍 Data verification:")
        print(f"  Merged file exists: {os.path.exists(merged_file_path)}")
        print(f"  Merged file size: {os.path.getsize(merged_file_path) / 1024:.2f} KB")
        
        # Reload for verification
        with open(merged_file_path, 'r', encoding='utf-8') as f:
            verify_data = json.load(f)
        
        print(f"  Reloaded record count: {len(verify_data)} (should match merged count)")
        
        # Check for vision_prediction_probs field
        has_vision_probs = 0
        has_text_probs = 0
        has_match_field = 0
        for item in verify_data[:10]:  # Check first 10 records
            if 'vision_prediction_probs' in item:
                has_vision_probs += 1
            if 'text_prediction_probs' in item:
                has_text_probs += 1
            if 'match' in item:
                has_match_field += 1
        
        print(f"  Records with vision prediction probabilities in first 10: {has_vision_probs}")
        print(f"  Records with text prediction probabilities in first 10: {has_text_probs}")
        print(f"  Records with match field in first 10: {has_match_field}")
        
        # Verify field order
        if verify_data:
            first_item_keys = list(verify_data[0].keys())
            print(f"  Field order of first record: {first_item_keys}")
            expected_order = ['name', 'true_label', 'vision_prediction_probs', 'vision_predicted_label', 
                              'text_prediction_probs', 'text_predicted_label', 'match']
            if first_item_keys == expected_order:
                print(f"  ✓ Field order is correct")
            else:
                print(f"  ⚠ Field order is incorrect, should be: {expected_order}")
        
        return merged_file_path, len(merged_data)
        
    except Exception as e:
        print(f"❌ Error occurred during merging: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def main():
    """Main function"""
    
    # Define file paths
    file_a_path = "/data/Desktop/BioMiner/Generative_model/text_grading_predictions.json"
    file_b_path = "/data/Desktop/BioMiner/Vision_grading_model/vision_grading_predictions.json"
    output_dir = "/data/Desktop/BioMiner/Multimodal_fusion"
    
    print("Multimodal Fusion Data Merge Tool")
    print("=" * 60)
    print(f"File A (text model): {file_a_path}")
    print(f"File B (vision model): {file_b_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Execute merge
    merged_file, record_count = merge_predictions_files(file_a_path, file_b_path, output_dir)
    
    if merged_file:
        print(f"\n🎉 Merge successful! Merged {record_count} records")
        print(f"Merged file: {merged_file}")
    else:
        print("\n❌ Merge failed!")
    
    print("\n" + "=" * 60)
    print("Multimodal data merge complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
