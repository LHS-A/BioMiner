import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict, Counter
import os
from pathlib import Path
from prettytable import PrettyTable
import sys

# Add path to import evaluation metric functions
sys.path.append(r"/data/Desktop/BioMiner")
from Vision_grading_model.utils.evaluation_metrics import get_metrix

def load_multimodal_data(json_path):
    """
    Load multimodal fusion dataset
    
    Args:
        json_path: JSON file path
        
    Returns:
        vision_probs: Visual prediction probabilities [n_samples, 4]
        text_probs: Text prediction probabilities [n_samples, 4]
        labels: True labels [n_samples]
        sample_names: List of sample names
        sample_data: Original sample data
    """
    print(f"Loading data: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Extract data
    vision_probs = []
    text_probs = []
    labels = []
    sample_names = []
    sample_data = []
    
    for i, item in enumerate(data):
        vision_probs.append(item['vision_prediction_probs'])
        text_probs.append(item['text_prediction_probs'])
        labels.append(item['true_label'])
        sample_names.append(item['name'])
        sample_data.append(item)
    
    vision_probs = np.array(vision_probs)
    text_probs = np.array(text_probs)
    labels = np.array(labels)
    
    print(f"Data shapes:")
    print(f"  Visual probabilities: {vision_probs.shape}")
    print(f"  Text probabilities: {text_probs.shape}")
    print(f"  True labels: {labels.shape}")
    
    # Count class distribution
    label_counts = Counter(labels)
    print(f"  Class distribution:")
    for label in sorted(label_counts.keys()):
        print(f"    Level {label}: {label_counts[label]} samples ({label_counts[label]/len(labels)*100:.1f}%)")
    
    return vision_probs, text_probs, labels, sample_names, sample_data, data

def calculate_metrics(predictions, true_labels, modality_name="Model"):
    """
    Calculate model evaluation metrics
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        modality_name: Model name (for display)
        
    Returns:
        Dictionary containing various metrics
    """
    # Use get_metrix function to calculate weighted metrics
    wacc, wse, wsp = get_metrix(predictions, true_labels)
    
    # Calculate other metrics
    accuracy = accuracy_score(true_labels, predictions)
    kappa = cohen_kappa_score(true_labels, predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate accuracy, sensitivity, specificity for each class
    n_classes = len(np.unique(true_labels))
    class_metrics = {}
    
    for i in range(n_classes):
        # True positive: predicted as i and actually i
        tp = np.sum((predictions == i) & (true_labels == i))
        # False positive: predicted as i but actually not i
        fp = np.sum((predictions == i) & (true_labels != i))
        # True negative: predicted not i and actually not i
        tn = np.sum((predictions != i) & (true_labels != i))
        # False negative: predicted not i but actually i
        fn = np.sum((predictions != i) & (true_labels == i))
        
        # Class-specific metrics
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
            
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0
            
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        class_metrics[i] = {
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'support': int(tp + fn)
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'weighted_accuracy': float(wacc[0]),
        'weighted_sensitivity': float(wse[0]),
        'weighted_specificity': float(wsp[0]),
        'class_accuracy': [float(x) for x in wacc[1]] if len(wacc) > 1 else [float(wacc[0])],
        'class_sensitivity': [float(x) for x in wse[1]] if len(wse) > 1 else [float(wse[0])],
        'class_specificity': [float(x) for x in wsp[1]] if len(wsp) > 1 else [float(wsp[0])],
        'confusion_matrix': cm.tolist(),
        'class_metrics': class_metrics,
        'predictions': predictions.tolist()
    }
    
    return metrics

def print_metrics_table(metrics_dict, title="Model Performance Evaluation"):
    """
    Print beautiful metrics tables following the format of reference code
    
    Args:
        metrics_dict: Dictionary containing metrics for different models
        title: Table title
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    # Create weighted metrics table
    weighted_table = PrettyTable()
    weighted_table.field_names = ["Model", "wAcc", "wSe", "wSp", "Accuracy", "Kappa"]
    weighted_table.align["Model"] = "l"
    
    for model_name, metrics in metrics_dict.items():
        weighted_table.add_row([
            model_name,
            f"{metrics['weighted_accuracy']:.4f}",
            f"{metrics['weighted_sensitivity']:.4f}",
            f"{metrics['weighted_specificity']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['kappa']:.4f}"
        ])
    
    print("Weighted Performance Metrics:")
    print(weighted_table)
    
    # Create class-specific metrics tables
    for model_name, metrics in metrics_dict.items():
        print(f"\n{model_name} Class-Specific Metrics:")
        class_table = PrettyTable()
        class_table.field_names = ["Level", "Count", "Accuracy", "Sensitivity", "Specificity", "F1-Score"]
        
        for class_idx in range(len(metrics['class_accuracy'])):
            class_metrics = metrics['class_metrics'].get(class_idx, {})
            class_table.add_row([
                f"Level {class_idx}",
                class_metrics.get('support', 0),
                f"{metrics['class_accuracy'][class_idx]:.4f}",
                f"{metrics['class_sensitivity'][class_idx]:.4f}",
                f"{metrics['class_specificity'][class_idx]:.4f}",
                f"{class_metrics.get('f1_score', 0):.4f}"
            ])
        
        # Add mean row
        class_table.add_row([
            "Mean/Total",
            np.sum([metrics['class_metrics'].get(i, {}).get('support', 0) for i in range(len(metrics['class_accuracy']))]),
            f"{metrics['weighted_accuracy']:.4f}",
            f"{metrics['weighted_sensitivity']:.4f}",
            f"{metrics['weighted_specificity']:.4f}",
            f"{np.mean([metrics['class_metrics'].get(i, {}).get('f1_score', 0) for i in range(len(metrics['class_accuracy']))]):.4f}"
        ])
        
        print(class_table)
    
    print(f"\n{'='*80}")

def logistic_regression_fusion(vision_features, text_features, labels, fusion_method='concat', test_size=0.2, random_state=42):
    """
    Logistic regression multimodal fusion classification
    
    Args:
        vision_features: Visual features [n_samples, 4]
        text_features: Textual features [n_samples, 4]  
        labels: Labels [n_samples]
        fusion_method: Feature fusion method ('concat', 'weighted_sum', 'average')
        test_size: Test set proportion
        random_state: Random seed
    """
    
    # Feature fusion
    if fusion_method == 'concat':
        # Direct feature concatenation
        fused_features = np.concatenate([vision_features, text_features], axis=1)
        
    elif fusion_method == 'weighted_sum':
        # Weighted sum (using simple average here, adjust weights as needed)
        fused_features = 0.5 * vision_features + 0.5 * text_features
        
    elif fusion_method == 'average':
        # Simple average
        fused_features = (vision_features + text_features) / 2
        
    else:
        raise ValueError("Fusion method must be 'concat', 'weighted_sum', or 'average'")
    
    print(f"Fused feature shape: {fused_features.shape}")
    
    # Data standardization
    scaler = StandardScaler()
    fused_features_scaled = scaler.fit_transform(fused_features)
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        fused_features_scaled, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Create and train logistic regression model
    lr_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,  # Maximum number of iterations
        multi_class='multinomial',  # Multiclass classification
        solver='lbfgs'  # Optimization algorithm suitable for multiclass classification
    )
    
    # Train model
    lr_model.fit(X_train, y_train)
    
    # Predict
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)
    
    return lr_model, scaler, X_test, y_test, y_pred, y_pred_proba

def predict_all_samples(model, scaler, vision_features, text_features, fusion_method='concat'):
    """
    Use trained logistic regression model for prediction on all samples
    
    Args:
        model: Trained logistic regression model
        scaler: Data standardizer
        vision_features: Visual features
        text_features: Textual features
        fusion_method: Feature fusion method
    """
    # Feature fusion
    if fusion_method == 'concat':
        fused_features = np.concatenate([vision_features, text_features], axis=1)
    elif fusion_method == 'weighted_sum':
        fused_features = 0.5 * vision_features + 0.5 * text_features
    elif fusion_method == 'average':
        fused_features = (vision_features + text_features) / 2
    
    # Standardization
    fused_features_scaled = scaler.transform(fused_features)
    
    # Predict
    predictions = model.predict(fused_features_scaled)
    probabilities = model.predict_proba(fused_features_scaled)
    
    return predictions, probabilities

def calculate_fusion_statistics(true_labels, vision_preds, text_preds, multimodal_preds, vision_probs, text_probs, multimodal_probs, sample_names):
    """
    Calculate statistics before and after fusion
    
    Returns:
        statistics: Dictionary containing all statistical information
    """
    statistics = {}
    
    # 1. Calculate detailed metrics for each modality
    vision_metrics = calculate_metrics(vision_preds, true_labels, "Visual Model")
    text_metrics = calculate_metrics(text_preds, true_labels, "Text Model")
    multimodal_metrics = calculate_metrics(multimodal_preds, true_labels, "Multimodal Fusion")
    
    statistics['metrics'] = {
        'vision': vision_metrics,
        'text': text_metrics,
        'multimodal': multimodal_metrics
    }
    
    # 2. Accuracy comparison
    statistics['accuracy_comparison'] = {
        'vision': float(vision_metrics['accuracy']),
        'text': float(text_metrics['accuracy']),
        'multimodal': float(multimodal_metrics['accuracy']),
        'vision_vs_multimodal': float(multimodal_metrics['accuracy'] - vision_metrics['accuracy']),
        'text_vs_multimodal': float(multimodal_metrics['accuracy'] - text_metrics['accuracy']),
        'best_modality': 'vision' if vision_metrics['accuracy'] > text_metrics['accuracy'] else 'text',
        'improvement_over_best': float(multimodal_metrics['accuracy'] - max(vision_metrics['accuracy'], text_metrics['accuracy']))
    }
    
    # 3. Weighted metrics comparison
    statistics['weighted_metrics_comparison'] = {
        'vision': {
            'wAcc': float(vision_metrics['weighted_accuracy']),
            'wSe': float(vision_metrics['weighted_sensitivity']),
            'wSp': float(vision_metrics['weighted_specificity'])
        },
        'text': {
            'wAcc': float(text_metrics['weighted_accuracy']),
            'wSe': float(text_metrics['weighted_sensitivity']),
            'wSp': float(text_metrics['weighted_specificity'])
        },
        'multimodal': {
            'wAcc': float(multimodal_metrics['weighted_accuracy']),
            'wSe': float(multimodal_metrics['weighted_sensitivity']),
            'wSp': float(multimodal_metrics['weighted_specificity'])
        }
    }
    
    # 4. Prediction confidence statistics
    statistics['confidence'] = {
        'vision_mean_confidence': float(np.mean(np.max(vision_probs, axis=1))),
        'text_mean_confidence': float(np.mean(np.max(text_probs, axis=1))),
        'multimodal_mean_confidence': float(np.mean(np.max(multimodal_probs, axis=1))),
        'vision_std_confidence': float(np.std(np.max(vision_probs, axis=1))),
        'text_std_confidence': float(np.std(np.max(text_probs, axis=1))),
        'multimodal_std_confidence': float(np.std(np.max(multimodal_probs, axis=1)))
    }
    
    # 5. Inconsistent sample analysis
    vision_text_disagree = vision_preds != text_preds
    multimodal_agrees_with_vision = multimodal_preds[vision_text_disagree] == vision_preds[vision_text_disagree]
    multimodal_agrees_with_text = multimodal_preds[vision_text_disagree] == text_preds[vision_text_disagree]
    multimodal_different = ~(multimodal_agrees_with_vision | multimodal_agrees_with_text)
    
    statistics['disagreement_analysis'] = {
        'total_disagreements': int(np.sum(vision_text_disagree)),
        'disagreement_rate': float(np.mean(vision_text_disagree)),
        'multimodal_agrees_with_vision': int(np.sum(multimodal_agrees_with_vision)),
        'multimodal_agrees_with_text': int(np.sum(multimodal_agrees_with_text)),
        'multimodal_different': int(np.sum(multimodal_different)),
        'vision_wins_rate': float(np.sum(multimodal_agrees_with_vision) / np.sum(vision_text_disagree)) if np.sum(vision_text_disagree) > 0 else 0.0,
        'text_wins_rate': float(np.sum(multimodal_agrees_with_text) / np.sum(vision_text_disagree)) if np.sum(vision_text_disagree) > 0 else 0.0
    }
    
    # 6. Detailed improvement sample analysis
    improvements = []
    
    for i in range(len(true_labels)):
        vision_correct = vision_preds[i] == true_labels[i]
        text_correct = text_preds[i] == true_labels[i]
        multimodal_correct = multimodal_preds[i] == true_labels[i]
        
        # Calculate confidence
        vision_conf = vision_probs[i][vision_preds[i]]
        text_conf = text_probs[i][text_preds[i]]
        multimodal_conf = multimodal_probs[i][multimodal_preds[i]]
        
        improvement = {
            'sample_name': sample_names[i],
            'true_label': int(true_labels[i]),
            'vision_pred': int(vision_preds[i]),
            'text_pred': int(text_preds[i]),
            'multimodal_pred': int(multimodal_preds[i]),
            'vision_correct': bool(vision_correct),
            'text_correct': bool(text_correct),
            'multimodal_correct': bool(multimodal_correct),
            'vision_confidence': float(vision_conf),
            'text_confidence': float(text_conf),
            'multimodal_confidence': float(multimodal_conf),
            'improvement_type': 'none'
        }
        
        # Determine improvement type
        if not vision_correct and not text_correct and multimodal_correct:
            improvement['improvement_type'] = 'both_wrong_to_correct'
        elif not vision_correct and text_correct and multimodal_correct:
            improvement['improvement_type'] = 'vision_wrong_fixed'
        elif vision_correct and not text_correct and multimodal_correct:
            improvement['improvement_type'] = 'text_wrong_fixed'
        elif vision_correct and text_correct and not multimodal_correct:
            improvement['improvement_type'] = 'regression'
        elif vision_correct != multimodal_correct or text_correct != multimodal_correct:
            improvement['improvement_type'] = 'change'
        
        improvements.append(improvement)
    
    statistics['detailed_improvements'] = improvements
    
    # 7. Improvement type statistics
    improvement_types = [imp['improvement_type'] for imp in improvements]
    improvement_counts = Counter(improvement_types)
    
    statistics['improvement_summary'] = {
        'total_samples': len(improvements),
        'both_wrong_to_correct': improvement_counts.get('both_wrong_to_correct', 0),
        'vision_wrong_fixed': improvement_counts.get('vision_wrong_fixed', 0),
        'text_wrong_fixed': improvement_counts.get('text_wrong_fixed', 0),
        'regression': improvement_counts.get('regression', 0),
        'change': improvement_counts.get('change', 0),
        'none': improvement_counts.get('none', 0),
        'net_improvement': (improvement_counts.get('both_wrong_to_correct', 0) + 
                           improvement_counts.get('vision_wrong_fixed', 0) + 
                           improvement_counts.get('text_wrong_fixed', 0) - 
                           improvement_counts.get('regression', 0))
    }
    
    # 8. Class-specific improvement analysis
    n_classes = len(np.unique(true_labels))
    class_improvements = {}
    
    for class_label in range(n_classes):
        class_mask = true_labels == class_label
        if np.sum(class_mask) > 0:
            class_vision_acc = np.mean(vision_preds[class_mask] == true_labels[class_mask])
            class_text_acc = np.mean(text_preds[class_mask] == true_labels[class_mask])
            class_multimodal_acc = np.mean(multimodal_preds[class_mask] == true_labels[class_mask])
            
            class_improvements[str(class_label)] = {
                'count': int(np.sum(class_mask)),
                'vision_accuracy': float(class_vision_acc),
                'text_accuracy': float(class_text_acc),
                'multimodal_accuracy': float(class_multimodal_acc),
                'improvement_over_vision': float(class_multimodal_acc - class_vision_acc),
                'improvement_over_text': float(class_multimodal_acc - class_text_acc),
                'best_single_modality': 'vision' if class_vision_acc > class_text_acc else 'text',
                'improvement_over_best': float(class_multimodal_acc - max(class_vision_acc, class_text_acc))
            }
    
    statistics['class_improvements'] = class_improvements
    
    return statistics

def print_detailed_fusion_report(statistics):
    """
    Print detailed fusion analysis report
    """
    print("\n" + "="*80)
    print("Multimodal Fusion Performance Detailed Analysis Report")
    print("="*80)
    
    # 1. Print performance metrics tables for each model
    metrics_dict = {
        'Visual Model': statistics['metrics']['vision'],
        'Text Model': statistics['metrics']['text'],
        'Multimodal Fusion': statistics['metrics']['multimodal']
    }
    
    print_metrics_table(metrics_dict, "Model Performance Comparison")
    
    # 2. Print improvement summary
    print("\n" + "="*80)
    print("Fusion Improvement Effect Summary")
    print("="*80)
    
    acc_comp = statistics['accuracy_comparison']
    improve = statistics['improvement_summary']
    weighted_comp = statistics['weighted_metrics_comparison']
    
    print(f"\n1. Accuracy Improvement:")
    print(f"   Visual Model: {acc_comp['vision']:.4f}")
    print(f"   Text Model: {acc_comp['text']:.4f}")
    print(f"   Multimodal Fusion: {acc_comp['multimodal']:.4f}")
    
    if acc_comp['improvement_over_best'] > 0:
        print(f"   ✓ Fusion model outperforms best single modality model({acc_comp['best_modality']}): +{acc_comp['improvement_over_best']:.4f}")
    else:
        print(f"   ⚠ Fusion model does not outperform best single modality model({acc_comp['best_modality']}): {acc_comp['improvement_over_best']:.4f}")
    
    print(f"\n2. Weighted Metrics Improvement:")
    print(f"   wAcc: Visual={weighted_comp['vision']['wAcc']:.4f}, Text={weighted_comp['text']['wAcc']:.4f}, Fusion={weighted_comp['multimodal']['wAcc']:.4f}")
    print(f"   wSe: Visual={weighted_comp['vision']['wSe']:.4f}, Text={weighted_comp['text']['wSe']:.4f}, Fusion={weighted_comp['multimodal']['wSe']:.4f}")
    print(f"   wSp: Visual={weighted_comp['vision']['wSp']:.4f}, Text={weighted_comp['text']['wSp']:.4f}, Fusion={weighted_comp['multimodal']['wSp']:.4f}")
    
    print(f"\n3. Improvement Sample Analysis:")
    print(f"   Total samples: {improve['total_samples']}")
    print(f"   Both wrong corrected (both vision & text wrong → fusion correct): {improve['both_wrong_to_correct']}")
    print(f"   Vision wrong fixed (vision wrong, text right → fusion correct): {improve['vision_wrong_fixed']}")
    print(f"   Text wrong fixed (text wrong, vision right → fusion correct): {improve['text_wrong_fixed']}")
    print(f"   Performance regression (both vision & text right → fusion wrong): {improve['regression']}")
    print(f"   Net improvement samples: {improve['net_improvement']}")
    
    print(f"\n4. Decision when modalities disagree:")
    disagree = statistics['disagreement_analysis']
    print(f"   Vision-text disagreement samples: {disagree['total_disagreements']} ({disagree['disagreement_rate']:.2%})")
    print(f"   Fusion supports vision: {disagree['multimodal_agrees_with_vision']} ({disagree['vision_wins_rate']:.2%})")
    print(f"   Fusion supports text: {disagree['multimodal_agrees_with_text']} ({disagree['text_wins_rate']:.2%})")
    print(f"   Fusion gives new result: {disagree['multimodal_different']}")
    
    print(f"\n5. Class-Specific Improvement Analysis:")
    class_table = PrettyTable()
    class_table.field_names = ["Disease Level", "Sample Count", "Visual Accuracy", "Text Accuracy", "Fusion Accuracy", "Improvement (vs. Best)"]
    
    for class_label, class_stat in statistics['class_improvements'].items():
        improvement_sign = "+" if class_stat['improvement_over_best'] > 0 else ""
        class_table.add_row([
            f"Level {class_label}",
            class_stat['count'],
            f"{class_stat['vision_accuracy']:.4f}",
            f"{class_stat['text_accuracy']:.4f}",
            f"{class_stat['multimodal_accuracy']:.4f}",
            f"{improvement_sign}{class_stat['improvement_over_best']:.4f}"
        ])
    
    print(class_table)
    
    print(f"\n6. Prediction Confidence Comparison:")
    conf = statistics['confidence']
    print(f"   Visual mean confidence: {conf['vision_mean_confidence']:.4f} (±{conf['vision_std_confidence']:.4f})")
    print(f"   Text mean confidence: {conf['text_mean_confidence']:.4f} (±{conf['text_std_confidence']:.4f})")
    print(f"   Fusion mean confidence: {conf['multimodal_mean_confidence']:.4f} (±{conf['multimodal_std_confidence']:.4f})")
    
    print("="*80)

def save_fusion_predictions(original_data, multimodal_preds, multimodal_probs, output_path):
    """
    Save fusion prediction results to new JSON file
    
    Args:
        original_data: Original data
        multimodal_preds: Fusion prediction labels
        multimodal_probs: Fusion prediction probabilities
        output_path: Output file path
    """
    print(f"\nSaving fusion prediction results to: {output_path}")
    
    # Create new data
    new_data = []
    
    for i, item in enumerate(original_data):
        new_item = item.copy()  # Copy original data
        new_item['multimodal_prediction_probs'] = multimodal_probs[i].tolist()
        new_item['multimodal_predicted_label'] = int(multimodal_preds[i])
        new_data.append(new_item)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(new_data)} sample fusion prediction results")

def save_statistics(statistics, output_path):
    """
    Save statistical information to JSON file
    
    Args:
        statistics: Statistical information dictionary
        output_path: Output file path
    """
    print(f"\nSaving statistical information to: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Statistical information saved")

def main():
    """
    Main function: Execute multimodal fusion analysis
    """
    print("="*80)
    print("Multimodal Fusion Analysis Tool")
    print("Using logistic regression to fuse visual and text prediction probabilities")
    print("="*80)
    
    # File paths
    input_json_path = "/data/Desktop/BioMiner/Multimodal_fusion/multimodal_fusion_dataset.json"
    predictions_output_path = "/data/Desktop/BioMiner/Multimodal_fusion/multimodal_fusion_predictions.json"
    statistics_output_path = "/data/Desktop/BioMiner/Multimodal_fusion/output_statistics.json"
    
    # 1. Load data
    vision_probs, text_probs, true_labels, sample_names, sample_data, original_data = load_multimodal_data(input_json_path)
    
    # 2. Get single-modality prediction results
    vision_preds = np.argmax(vision_probs, axis=1)
    text_preds = np.argmax(text_probs, axis=1)
    
    # 3. Train fusion model
    print("\n" + "="*80)
    print("Training Logistic Regression Fusion Model")
    print("="*80)
    
    fusion_method = 'concat'  # Can choose 'concat', 'weighted_sum', 'average'
    
    model, scaler, X_test, y_test, y_pred_test, y_pred_proba_test = logistic_regression_fusion(
        vision_features=vision_probs,
        text_features=text_probs,
        labels=true_labels,
        fusion_method=fusion_method,
        test_size=0.2,
        random_state=42
    )
    
    # 4. Predict on all samples
    print("\n" + "="*80)
    print("Performing Fusion Prediction on All Samples")
    print("="*80)
    
    multimodal_preds, multimodal_probs = predict_all_samples(
        model=model,
        scaler=scaler,
        vision_features=vision_probs,
        text_features=text_probs,
        fusion_method=fusion_method
    )
    
    print(f"Fusion prediction completed:")
    print(f"  Visual prediction distribution: {dict(Counter(vision_preds))}")
    print(f"  Text prediction distribution: {dict(Counter(text_preds))}")
    print(f"  Fusion prediction distribution: {dict(Counter(multimodal_preds))}")
    
    # 5. Calculate detailed statistical information
    print("\n" + "="*80)
    print("Calculating Fusion Performance Statistical Information")
    print("="*80)
    
    statistics = calculate_fusion_statistics(
        true_labels=true_labels,
        vision_preds=vision_preds,
        text_preds=text_preds,
        multimodal_preds=multimodal_preds,
        vision_probs=vision_probs,
        text_probs=text_probs,
        multimodal_probs=multimodal_probs,
        sample_names=sample_names
    )
    
    # 6. Print detailed fusion report
    print_detailed_fusion_report(statistics)
    
    # 7. Save result files
    save_fusion_predictions(original_data, multimodal_preds, multimodal_probs, predictions_output_path)
    save_statistics(statistics, statistics_output_path)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Generated output files:")
    print(f"  1. Fusion prediction file: {predictions_output_path}")
    print(f"  2. Statistical information file: {statistics_output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
