import json
import re
import os
from glob import glob
from pathlib import Path

def merge_json_files_without_name(json_dir_path, output_json_path):
    """Documentation for this retained training component."""
    print(f"开始合并目录下的JSON文件: {json_dir_path}")
    
    # Ensure the output directory exists.
    json_pattern = os.path.join(json_dir_path, "*.json")
    json_files = glob(json_pattern)
    
    if not json_files:
        print(f"警告: 在目录 {json_dir_path} 中未找到JSON文件")
        return 0
    
    print(f"找到 {len(json_files)} 个JSON文件:")
    for i, file_path in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # Retain this implementation detail from the original training pipeline.
    all_data = []
    total_samples = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # Process the current value.
                processed_samples = []
                for item in data:
                    if isinstance(item, dict):
                        # Retain this implementation detail from the original training pipeline.
                        processed_item = {k: v for k, v in item.items() if k != 'name'}
                        processed_samples.append(processed_item)
                
                all_data.extend(processed_samples)
                total_samples += len(processed_samples)
                print(f"  ✓ {os.path.basename(file_path)}: 处理了 {len(processed_samples)} 个样本")
                
            elif isinstance(data, dict):
                # Retain this implementation detail from the original training pipeline.
                print(f"  警告: {os.path.basename(file_path)} 是字典格式，尝试查找数据...")
                
                # Retain this implementation detail from the original training pipeline.
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
                        print(f"  ✓ {os.path.basename(file_path)}: 在键 '{key}' 中找到并处理了 {len(processed_samples)} 个样本")
                        found_data = True
                        break
                
                if not found_data:
                    print(f"  ✗ {os.path.basename(file_path)}: 未找到有效的数据列表")
                    
        except Exception as e:
            print(f"  ✗ {os.path.basename(file_path)}: 读取失败 - {str(e)}")
    
    if not all_data:
        print("警告: 未找到任何有效数据")
        return 0
    
    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the generated artifact.
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 合并完成!")
    print(f"  总样本数: {total_samples}")
    print(f"  输出文件: {output_json_path}")
    
    # Run the validation step.
    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Validate the required condition.
        samples_with_name = 0
        for item in saved_data:
            if 'name' in item:
                samples_with_name += 1
        
        if samples_with_name > 0:
            print(f"  警告: 发现 {samples_with_name} 个样本仍然包含name键")
        else:
            print(f"  ✓ 验证通过: 所有样本都已删除name键")
        
        # Validate the required condition.
        first_sample = saved_data[0] if saved_data else {}
        print(f"  数据格式: 每个样本包含的键: {list(first_sample.keys())}")
        
    except Exception as e:
        print(f"  警告: 验证输出文件时出错 - {str(e)}")
    
    return total_samples

def merge_medical_data_from_template_B(input_json_path, output_txt_path):
    """Documentation for this retained training component."""
    # Read the input data.
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Prepare or report the output.
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        # Retain this implementation detail from the original training pipeline.
        for item in data:
            # Retain this implementation detail from the original training pipeline.
            observation = item.get('observation', '')
            forecast = item.get('forecast', '')

            # Retain this implementation detail from the original training pipeline.
            # merged_line = f"{observation} {forecast}"
            
            # Retain this implementation detail from the original training pipeline.
            merged_line = extract_and_merge_from_observation(observation, forecast)
            
            # Write the output data.
            txt_file.write(merged_line + '\n')

    print(f"数据已成功从模板B合并并保存到: {output_txt_path}")
    return len(data)

def extract_and_merge_from_observation(observation, forecast):
    """Documentation for this retained training component."""
    # Retain this implementation detail from the original training pipeline.
    # Retain this implementation detail from the original training pipeline.
    
    # Retain this implementation detail from the original training pipeline.
    if "Context:" in observation and "Question:" in observation:
        # Retain this implementation detail from the original training pipeline.
        pattern = r"Context:\s*(.*?)\s*Question:\s*(.*)"
        match = re.search(pattern, observation, re.DOTALL)
        
        if match:
            context = match.group(1).strip()
            question = match.group(2).strip()
            # Retain this implementation detail from the original training pipeline.
            return f"{context} {question} {forecast}"
        else:
            # Retain this implementation detail from the original training pipeline.
            parts = observation.split("Context:", 1)[1] if "Context:" in observation else observation
            if "Question:" in parts:
                context_part, question_part = parts.split("Question:", 1)
                return f"{context_part.strip()} {question_part.strip()} {forecast}"
    else:
        # Retain this implementation detail from the original training pipeline.
        # Prepare the model input.
        return f"{observation} {forecast}"

def merge_medical_data_simple_from_template_B(input_json_path, output_txt_path):
    """Documentation for this retained training component."""
    # Read the input data.
    with open(input_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Prepare or report the output.
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        # Retain this implementation detail from the original training pipeline.
        for item in data:
            # Retain this implementation detail from the original training pipeline.
            observation = item.get('observation', '')
            forecast = item.get('forecast', '')

            # Retain this implementation detail from the original training pipeline.
            merged_line = f"{observation} {forecast}"
            
            # Write the output data.
            txt_file.write(merged_line + '\n')

    print(f"数据已成功从模板B合并并保存到: {output_txt_path}")
    return len(data)

def compare_template_A_and_B_results(template_A_input, template_B_input, output_path_a, output_path_b):
    """Documentation for this retained training component."""
    print("=" * 60)
    print("比较模板A和模板B生成的结果")
    print("=" * 60)
    
    # Retain this implementation detail from the original training pipeline.
    print("\n1. 从模板A生成结果:")
    try:
        # Retain this implementation detail from the original training pipeline.
        # from merge_medical_data import merge_medical_data
        # merge_medical_data(template_A_input, output_path_a)
        with open(output_path_a, 'r', encoding='utf-8') as f:
            template_a_lines = f.readlines()
        print(f"   生成 {len(template_a_lines)} 行数据")
    except Exception as e:
        print(f"   从模板A生成失败: {e}")
        template_a_lines = []
    
    # Retain this implementation detail from the original training pipeline.
    print("\n2. 从模板B生成结果:")
    try:
        merge_medical_data_from_template_B(template_B_input, output_path_b)
        with open(output_path_b, 'r', encoding='utf-8') as f:
            template_b_lines = f.readlines()
        print(f"   生成 {len(template_b_lines)} 行数据")
    except Exception as e:
        print(f"   从模板B生成失败: {e}")
        template_b_lines = []
    
    # Retain this implementation detail from the original training pipeline.
    if template_a_lines and template_b_lines:
        print("\n3. 结果比较:")
        
        if len(template_a_lines) == len(template_b_lines):
            print(f"   行数相同: {len(template_a_lines)} 行")
        else:
            print(f"   行数不同 - 模板A: {len(template_a_lines)} 行, 模板B: {len(template_b_lines)} 行")
        
        # Retain this implementation detail from the original training pipeline.
        print("\n   前3行内容比较:")
        for i in range(min(3, len(template_a_lines), len(template_b_lines))):
            print(f"\n   第{i+1}行:")
            print(f"     模板A: {template_a_lines[i][:100]}..." if len(template_a_lines[i]) > 100 else f"     模板A: {template_a_lines[i].strip()}")
            print(f"     模板B: {template_b_lines[i][:100]}..." if len(template_b_lines[i]) > 100 else f"     模板B: {template_b_lines[i].strip()}")
            
            # Validate the required condition.
            if template_a_lines[i].strip() == template_b_lines[i].strip():
                print("     ✓ 内容相同")
            else:
                print("     ✗ 内容不同")
    
    print("\n" + "=" * 60)
    print("比较完成")
    print("=" * 60)

def validate_data_format(input_json_path):
    """Documentation for this retained training component."""
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("数据为空")
            return False
        
        first_item = data[0]
        observation = first_item.get('observation', '')
        
        # Validate the required condition.
        has_context_keyword = "Context:" in observation
        has_question_keyword = "Question:" in observation
        
        print(f"数据格式检查:")
        print(f"  observation长度: {len(observation)} 字符")
        print(f"  包含'Context:'关键词: {has_context_keyword}")
        print(f"  包含'Question:'关键词: {has_question_keyword}")
        
        if has_context_keyword and has_question_keyword:
            print("  → 数据格式: 旧格式 (包含Context:和Question:关键词)")
            return False
        else:
            print("  → 数据格式: 新格式 (不包含Context:和Question:关键词)")
            return True
            
    except Exception as e:
        print(f"验证数据格式时出错: {e}")
        return False

def process_observation_new_format(observation, forecast):
    """Documentation for this retained training component."""
    # Retain this implementation detail from the original training pipeline.
    # Process the current value.
    return f"{observation} {forecast}"

# Run the training step.
if __name__ == "__main__":
    dataset_name = "Both_corpus"
    
    # Resolve the required path.
    json_dir_path = "/data/Desktop/BioMiner/Generative_model/datasets/" + dataset_name + "/Three_components"
    merged_json_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/merged_corpus_CNs_LCs.json"
    output_txt_path = "/data/Desktop/BioMiner/Generative_model/Train_tokenizer/tokenizer_train_data_CNs_LCs.txt"
    
    print("=" * 60)
    print("JSON文件合并与训练数据生成工具")
    print("=" * 60)
    
    print(f"JSON文件目录: {json_dir_path}")
    print(f"合并后JSON文件: {merged_json_path}")
    print(f"输出TXT文件: {output_txt_path}")
    print("-" * 60)
    
    # Retain this implementation detail from the original training pipeline.
    print("\n步骤1: 合并JSON文件并删除name键")
    print("-" * 40)
    
    # Ensure the output directory exists.
    if not os.path.exists(json_dir_path):
        print(f"错误: JSON文件目录不存在: {json_dir_path}")
        print("请检查目录路径是否正确")
        exit(1)
    
    # Retain this implementation detail from the original training pipeline.
    total_samples = merge_json_files_without_name(json_dir_path, merged_json_path)
    
    if total_samples == 0:
        print("错误: 未成功合并任何数据，程序退出")
        exit(1)
    
    print("-" * 40)
    
    # Run the training step.
    print("\n步骤2: 从合并后的JSON文件生成训练数据")
    print("-" * 40)
    
    try:
        # Validate the required condition.
        if not os.path.exists(merged_json_path):
            print(f"错误: 合并后的JSON文件不存在: {merged_json_path}")
            exit(1)
        
        # Run the validation step.
        with open(merged_json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        if not sample_data:
            print("错误: 合并后的JSON文件为空")
            exit(1)
        
        # Validate the required condition.
        data_format_check = validate_data_format(merged_json_path)
        
        # Validate the required condition.
        first_item = sample_data[0]
        if 'observation' in first_item and 'forecast' in first_item:
            print("✓ 确认合并后的文件为模板B格式")
            print(f"  样本数据包含 {len(sample_data)} 条记录")
            
            # Run the validation step.
            has_name = any('name' in item for item in sample_data[:10])  # Validate the required condition.
            if has_name:
                print("  警告: 发现部分样本仍包含name键")
            else:
                print("  ✓ 所有样本都已删除name键")
        else:
            print("错误: 合并后的文件不是模板B格式")
            print(f"  第一个样本的键: {list(first_item.keys())}")
            exit(1)
        
        # Run the training step.
        print(f"\n开始生成训练数据...")
        
        # Process the current value.
        if data_format_check:
            print("  使用新格式处理方法")
            # Create the required object.
            with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                for item in sample_data:
                    observation = item.get('observation', '')
                    forecast = item.get('forecast', '')
                    merged_line = process_observation_new_format(observation, forecast)
                    txt_file.write(merged_line + '\n')
            
            count = len(sample_data)
            print(f"  已处理 {count} 条记录")
        else:
            print("  使用旧格式处理方法")
            count = merge_medical_data_from_template_B(merged_json_path, output_txt_path)
        
        print("\n" + "=" * 60)
        print(f"训练数据生成完成!")
        print(f"处理了 {count} 条记录")
        print(f"输出文件: {output_txt_path}")
        
        # Retain this implementation detail from the original training pipeline.
        print("\n输出文件前3行示例:")
        try:
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i in range(min(3, len(lines))):
                    line_preview = lines[i].strip()
                    if len(line_preview) > 150:
                        print(f"  第{i+1}行: {line_preview[:150]}...")
                    else:
                        print(f"  第{i+1}行: {line_preview}")
        except Exception as e:
            print(f"  无法读取输出文件示例: {e}")
        
        # Retain this implementation detail from the original training pipeline.
        print("\n统计信息:")
        print(f"  JSON目录: {json_dir_path}")
        print(f"  合并后的JSON文件: {merged_json_path}")
        print(f"  训练数据文件: {output_txt_path}")
        print(f"  总样本数: {count}")
        
        print("=" * 60)
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("请确保合并后的JSON文件是有效的JSON格式")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()