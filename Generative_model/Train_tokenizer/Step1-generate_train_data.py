import json
import re
import os
from glob import glob
from pathlib import Path

def merge_json_files_without_name(json_dir_path, output_json_path):
    """
    Merge all JSON files in the specified directory and remove any 'name' key-value pairs.
    
    Args:
        json_dir_path (str): Directory path containing JSON files.
        output_json_path (str): Output path for the merged JSON file.
    """
    print(f"Starting to merge JSON files in directory: {json_dir_path}")
    
    # Retrieve all JSON files in the directory
    json_pattern = os.path.join(json_dir_path, "*.json")
    json_files = glob(json_pattern)
    
    if not json_files:
        print(f"Warning: No JSON files found in directory {json_dir_path}")
        return 0
    
    print(f"Found {len(json_files)} JSON files:")
    for i, file_path in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # Merge all data
    all_data = []
    total_samples = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # Process each sample, removing the 'name' key if present
                processed_samples = []
                for item in data:
                    if isinstance(item, dict):
                        # Copy item excluding the 'name' key
                        processed_item = {k: v for k, v in item.items() if k != 'name'}
                        processed_samples.append(processed_item)
                
                all_data.extend(processed_samples)
                total_samples += len(processed_samples)
                print(f"  ✓ {os.path.basename(file_path)}: processed {len(processed_samples)} samples")
                
            elif isinstance(data, dict):
                # If it's a dictionary, try to locate the data list
                print(f"  Warning: {os.path.basename(file_path)} is a dictionary, searching for data...")
                
                # Try common key names
                possible_keys = ['data', 'samples', 'items', 'list', 'records']
                found_data = False
                
                for key in possible_keys:
                    if key in data and isinstance(data[key], list):
                        processed_samples = []
                        for item in data[key]:
                            if isinstance(item, dict):
                                processed_item = {k: v for k, v in item.items() if k != 'name'}
                                processed_samples.append(processed_item)
                        
                        all_data.extend(processed_samples)
                        total_samples += len(processed_samples)
                        print(f"  ✓ {os.path.basename(file_path)}: found and processed {len(processed_samples)} samples under key '{key}'")
                        found_data = True
                        break
                
                if not found_data:
                    print(f"  ✗ {os.path.basename(file_path)}: no valid data list found")
                    
        except Exception as e:
            print(f"  ✗ {os.path.basename(file_path)}: read failed - {str(e)}")
    
    if not all_data:
        print("Warning: No valid data found")
        return 0
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save merged data
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Merge completed!")
    print(f"  Total samples: {total_samples}")
    print(f"  Output file: {output_json_path}")
    
    # Validate output file
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Check for any remaining 'name' keys
        samples_with_name = 0
        for item in saved_data:
            if 'name' in item:
                samples_with_name += 1
        
        if samples_with_name > 0:
            print(f"  Warning: {samples_with_name} samples still contain the 'name' key")
        else:
            print(f"  ✓ Validation passed: 'name' key removed from all samples")
        
        # Inspect data format
        first_sample = saved_data[0] if saved_data else {}
        print(f"  Data format: keys per sample: {list(first_sample.keys())}")
        
    except Exception as e:
        print(f"  Warning: error validating output file - {str(e)}")
    
    return total_samples

def merge_medical_data_from_template_B(input_json_path, output_txt_path):
    """
    Extract and merge 'observation' and 'forecast' directly from a Template-B JSON file, saving as TXT.
    
    Args:
        input_json_path (str): Path to the Template-B JSON file.
        output_txt_path (str): Path for the output TXT file.
    """
    # Read JSON file
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Open output file
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        # Iterate over each item
        for item in data:
            # Extract observation and forecast
            observation = item.get('observation', '')
            forecast = item.get('forecast', '')

            # Method 1: direct merge
            # merged_line = f"{observation} {forecast}"
            
            # Method 2: extract original context and question from observation (closer to Template A)
            # Observation usually looks like: "Context: xxx Question: yyy"
            # We extract xxx and yyy, then merge with forecast
            merged_line = extract_and_merge_from_observation(observation, forecast)
            
            # Write to file
            txt_file.write(merged_line + '\n')

    print(f"Data merged from Template B and saved to: {output_txt_path}")
    return len(data)

def extract_and_merge_from_observation(observation, forecast):
    """
    Extract original context and question from observation, then merge with forecast.
    
    Args:
        observation (str): Observation string, format "Context: xxx Question: yyy"
        forecast (str): Forecast string
    
    Returns:
        str: Merged string
    """
    # Method 1: regex extraction
    # Pattern: "Context: [content] Question: [content]"
    pattern = r"Context:\s*(.*?)\s*Question:\s*(.*)"
    match = re.search(pattern, observation, re.DOTALL)
    
    if match:
        context = match.group(1).strip()
        question = match.group(2).strip()
        # Merge into single line identical to Template A
        return f"{context} {question} {forecast}"
    else:
        # Method 2: simple split if regex fails
        # Locate "Context:" and "Question:"
        if "Context:" in observation and "Question:" in observation:
            # Split
            parts = observation.split("Context:", 1)[1] if "Context:" in observation else observation
            if "Question:" in parts:
                context_part, question_part = parts.split("Question:", 1)
                return f"{context_part.strip()} {question_part.strip()} {forecast}"
        
        # Method 3: fallback – direct merge
        return f"{observation} {forecast}"

def merge_medical_data_simple_from_template_B(input_json_path, output_txt_path):
    """
    Simplified version: directly merge observation and forecast.
    
    Args:
        input_json_path (str): Path to the Template-B JSON file.
        output_txt_path (str): Path for the output TXT file.
    """
    # Read JSON file
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Open output file
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        # Iterate over each item
        for item in data:
            # Extract observation and forecast
            observation = item.get('observation', '')
            forecast = item.get('forecast', '')

            # Direct merge
            merged_line = f"{observation} {forecast}"
            
            # Write to file
            txt_file.write(merged_line + '\n')

    print(f"Data merged from Template B and saved to: {output_txt_path}")
    return len(data)

def compare_template_A_and_B_results(template_A_input, template_B_input, output_path_a, output_path_b):
    """
    Compare results generated from Template A and Template B.
    
    Args:
        template_A_input (str): Input file path for Template A.
        template_B_input (str): Input file path for Template B.
        output_path_a (str): Output file path generated from Template A.
        output_path_b (str): Output file path generated from Template B.
    """
    from merge_medical_data import merge_medical_data
    
    print("=" * 60)
    print("Comparing results from Template A and Template B")
    print("=" * 60)
    
    # Generate result from Template A
    print("\n1. Generating result from Template A:")
    try:
        merge_medical_data(template_A_input, output_path_a)
        with open(output_path_a, 'r', encoding='utf-8') as f:
            template_a_lines = f.readlines()
        print(f"   Generated {len(template_a_lines)} lines")
    except Exception as e:
        print(f"   Template A generation failed: {e}")
        template_a_lines = []
    
    # Generate result from Template B
    print("\n2. Generating result from Template B:")
    try:
        merge_medical_data_from_template_B(template_B_input, output_path_b)
        with open(output_path_b, 'r', encoding='utf-8') as f:
            template_b_lines = f.readlines()
        print(f"   Generated {len(template_b_lines)} lines")
    except Exception as e:
        print(f"   Template B generation failed: {e}")
        template_b_lines = []
    
    # Compare results
    if template_a_lines and template_b_lines:
        print("\n3. Result comparison:")
        
        if len(template_a_lines) == len(template_b_lines):
            print(f"   Same line count: {len(template_a_lines)} lines")
        else:
            print(f"   Different line counts - Template A: {len(template_a_lines)}, Template B: {len(template_b_lines)}")
        
        # Compare first few lines
        print("\n   First 3 lines comparison:")
        for i in range(min(3, len(template_a_lines), len(template_b_lines))):
            print(f"\n   Line {i+1}:")
            print(f"     Template A: {template_a_lines[i][:100]}..." if len(template_a_lines[i]) > 100 else f"     Template A: {template_a_lines[i].strip()}")
            print(f"     Template B: {template_b_lines[i][:100]}..." if len(template_b_lines[i]) > 100 else f"     Template B: {template_b_lines[i].strip()}")
            
            # Check identity
            if template_a_lines[i].strip() == template_b_lines[i].strip():
                print("     ✓ Identical content")
            else:
                print("     ✗ Different content")
    
    print("\n" + "=" * 60)
    print("Comparison complete")
    print("=" * 60)

# Main: merge JSON files and generate training data
if __name__ == "__main__":
    dataset_name = "LCs_corpus"
    
    # Define paths
    json_dir_path = "/data/Desktop/BioMiner/Generative_model/datasets/" + dataset_name + "/Three_components"
    merged_json_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/merged_corpus.json"
    output_txt_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_train_data.txt"
    
    print("=" * 60)
    print("JSON File Merge & Training Data Generation Tool")
    print("=" * 60)
    
    print(f"JSON directory: {json_dir_path}")
    print(f"Merged JSON file: {merged_json_path}")
    print(f"Output TXT file: {output_txt_path}")
    print("-" * 60)
    
    # Step 1: merge JSON files and strip 'name' key
    print("\nStep 1: Merge JSON files and remove 'name' key")
    print("-" * 40)
    
    # Check directory existence
    if not os.path.exists(json_dir_path):
        print(f"Error: JSON directory does not exist: {json_dir_path}")
        print("Please verify the directory path")
        exit(1)
    
    # Merge JSON files
    total_samples = merge_json_files_without_name(json_dir_path, merged_json_path)
    
    if total_samples == 0:
        print("Error: No data merged successfully, exiting")
        exit(1)
    
    print("-" * 40)
    
    # Step 2: generate training data from merged JSON
    print("\nStep 2: Generate training data from merged JSON")
    print("-" * 40)
    
    try:
        # Check merged file existence
        if not os.path.exists(merged_json_path):
            print(f"Error: Merged JSON file does not exist: {merged_json_path}")
            exit(1)
        
        # Load and validate file format
        with open(merged_json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        if not sample_data:
            print("Error: Merged JSON file is empty")
            exit(1)
        
        # Check for Template B format
        first_item = sample_data[0]
        if 'observation' in first_item and 'forecast' in first_item:
            print("✓ Confirmed merged file is in Template B format")
            print(f"  Sample data contains {len(sample_data)} records")
            
            # Verify absence of 'name' key
            has_name = any('name' in item for item in sample_data[:10])  # check first 10
            if has_name:
                print("  Warning: some samples still contain the 'name' key")
            else:
                print("  ✓ 'name' key removed from all samples")
        else:
            print("Error: Merged file is not in Template B format")
            print(f"  Keys in first sample: {list(first_item.keys())}")
            exit(1)
        
        # Generate training data
        print(f"\nGenerating training data...")
        count = merge_medical_data_from_template_B(merged_json_path, output_txt_path)
        
        print("\n" + "=" * 60)
        print(f"Training data generation complete!")
        print(f"Processed {count} records")
        print(f"Output file: {output_txt_path}")
        
        # Display first few lines as examples
        print("\nFirst 3 lines of output:")
        try:
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i in range(min(3, len(lines))):
                    line_preview = lines[i].strip()
                    if len(line_preview) > 150:
                        print(f"  Line {i+1}: {line_preview[:150]}...")
                    else:
                        print(f"  Line {i+1}: {line_preview}")
        except Exception as e:
            print(f"  Unable to read output examples: {e}")
        
        # Statistics
        print("\nStatistics:")
        print(f"  JSON directory: {json_dir_path}")
        print(f"  Merged JSON file: {merged_json_path}")
        print(f"  Training data file: {output_txt_path}")
        print(f"  Total samples: {count}")
        
        print("=" * 60)
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Please ensure the merged JSON file is valid")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
