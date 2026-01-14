"""
Classification Performance Reporting Module

This module generates statistical reports to evaluate the accuracy of the rule-based image classifier.
It requires a labeled dataset (where filenames imply the ground truth).

Metrics generated:
- Confusion Matrix
- Precision, Recall, and F1-Score per class
- Summary statistics of misclassifications

This is primarily used during the development phase to tune the feature extraction parameters.
"""

import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime

def extract_true_label(filename):
    """Extract true label from filename (brand names using underscores)"""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.match(r'(.+?)_', name_without_ext)
    if match:
        return match.group(1)
    return 'unknown'

def generate_confusion_matrix_and_report(classified_folder, output_csv_folder=None):
    """Generate confusion matrix and classification report from classified images"""
    
    if output_csv_folder is None:
        output_csv_folder = classified_folder
    
    os.makedirs(output_csv_folder, exist_ok=True)
    
    print("=" * 120)
    print("GENERATING CLASSIFICATION REPORT")
    print("=" * 120)
    
    # Get categories from folder structure
    categories = [d for d in os.listdir(classified_folder) 
                  if os.path.isdir(os.path.join(classified_folder, d))]
    categories.sort()
    
    print(f"Found {len(categories)} categories: {', '.join(categories)}\n")
    
    # Initialize confusion data and predictions list
    confusion_data = defaultdict(lambda: defaultdict(int))
    all_predictions = []
    
    # Process each predicted category folder
    for predicted_label in categories:
        folder_path = os.path.join(classified_folder, predicted_label)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        for image_file in image_files:
            true_label = extract_true_label(image_file)
            all_predictions.append({
                'filename': image_file,
                'true_label': true_label,
                'predicted_label': predicted_label
            })
            confusion_data[true_label][predicted_label] += 1
    
    # Get all unique true labels
    all_true_labels = sorted(set([pred['true_label'] for pred in all_predictions]))
    all_predicted_labels = categories
    
    print(f"Total images analyzed: {len(all_predictions)}\n")
    
    # 1. Create Predictions DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df['correct'] = predictions_df['true_label'] == predictions_df['predicted_label']
    
    # 2. Create Confusion Matrix DataFrame
    confusion_matrix_data = []
    for true_label in all_true_labels:
        row = {'True Label': true_label}
        row_total = 0
        for pred_label in all_predicted_labels:
            count = confusion_data[true_label][pred_label]
            row[pred_label] = count
            row_total += count
        row['Total'] = row_total
        confusion_matrix_data.append(row)
    
    confusion_df = pd.DataFrame(confusion_matrix_data)
    
    # Add column totals
    totals_row = {'True Label': 'Total'}
    for pred_label in all_predicted_labels:
        totals_row[pred_label] = confusion_df[pred_label].sum()
    totals_row['Total'] = confusion_df['Total'].sum()
    confusion_df = pd.concat([confusion_df, pd.DataFrame([totals_row])], ignore_index=True)
    
    print("=" * 120)
    print("CONFUSION MATRIX")
    print("=" * 120)
    print(confusion_df.to_string(index=False))
    print("=" * 120)
    print()
    
    # 3. Calculate metrics
    metrics_data = []
    total_correct = 0
    total_samples = 0
    
    for category in all_true_labels:
        tp = confusion_data[category][category]
        fp = sum(confusion_data[other_cat][category] 
                for other_cat in all_true_labels if other_cat != category)
        fn = sum(confusion_data[category][other_pred] 
                for other_pred in all_predicted_labels if other_pred != category)
        tn = sum(confusion_data[other_true][other_pred] 
                for other_true in all_true_labels if other_true != category
                for other_pred in all_predicted_labels if other_pred != category)
        
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / support if support > 0 else 0.0
        
        metrics_data.append({
            'Category': category,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1_score, 4),
            'Accuracy': round(accuracy, 4),
            'Support': support
        })
        
        total_samples += support
        total_correct += tp
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Calculate averages
    macro_precision = metrics_df['Precision'].mean()
    macro_recall = metrics_df['Recall'].mean()
    macro_f1 = metrics_df['F1 Score'].mean()
    macro_accuracy = metrics_df['Accuracy'].mean()
    total_support = metrics_df['Support'].sum()
    
    weighted_precision = (metrics_df['Precision'] * metrics_df['Support']).sum() / total_support if total_support > 0 else 0.0
    weighted_recall = (metrics_df['Recall'] * metrics_df['Support']).sum() / total_support if total_support > 0 else 0.0
    weighted_f1 = (metrics_df['F1 Score'] * metrics_df['Support']).sum() / total_support if total_support > 0 else 0.0
    weighted_accuracy = (metrics_df['Accuracy'] * metrics_df['Support']).sum() / total_support if total_support > 0 else 0.0
    
    # Add summary rows
    macro_row = pd.DataFrame([{
        'Category': 'macro_avg',
        'TP': '',
        'FP': '',
        'FN': '',
        'TN': '',
        'Precision': round(macro_precision, 4),
        'Recall': round(macro_recall, 4),
        'F1 Score': round(macro_f1, 4),
        'Accuracy': round(macro_accuracy, 4),
        'Support': total_support
    }])
    
    weighted_row = pd.DataFrame([{
        'Category': 'weighted_avg',
        'TP': '',
        'FP': '',
        'FN': '',
        'TN': '',
        'Precision': round(weighted_precision, 4),
        'Recall': round(weighted_recall, 4),
        'F1 Score': round(weighted_f1, 4),
        'Accuracy': round(weighted_accuracy, 4),
        'Support': total_support
    }])
    
    metrics_df = pd.concat([metrics_df, macro_row, weighted_row], ignore_index=True)
    
    print("=" * 120)
    print("CLASSIFICATION METRICS")
    print("=" * 120)
    print(metrics_df.to_string(index=False))
    print("=" * 120)
    print()
    
    # 4. Misclassification Analysis DataFrame
    misclassifications = defaultdict(list)
    for pred in all_predictions:
        if pred['true_label'] != pred['predicted_label']:
            key = (pred['true_label'], pred['predicted_label'])
            misclassifications[key].append(pred['filename'])
    
    misclass_data = []
    if misclassifications:
        for (true_label, pred_label), files in misclassifications.items():
            misclass_data.append({
                'True Label': true_label,
                'Predicted As': pred_label,
                'Count': len(files),
                'Example 1': files[0] if len(files) > 0 else '',
                'Example 2': files[1] if len(files) > 1 else '',
                'Example 3': files[2] if len(files) > 2 else '',
                'All Files': ', '.join(files)
            })
    
    misclass_df = pd.DataFrame(misclass_data)
    
    if not misclass_df.empty:
        misclass_df = misclass_df.sort_values('Count', ascending=False)
        print("=" * 120)
        print("MISCLASSIFICATION ANALYSIS")
        print("=" * 120)
        print(misclass_df[['True Label', 'Predicted As', 'Count', 'Example 1', 'Example 2', 'Example 3']].to_string(index=False))
        print("=" * 120)
        print()
    
    # 5. Overall Accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 6. Summary Statistics DataFrame
    summary_data = {
        'Metric': [
            'Total Categories',
            'Total Images',
            'Correctly Classified',
            'Misclassified',
            'Overall Accuracy',
            'Macro Precision',
            'Macro Recall',
            'Macro F1 Score',
            'Macro Accuracy',
            'Weighted Precision',
            'Weighted Recall',
            'Weighted F1 Score',
            'Weighted Accuracy'
        ],
        'Value': [
            len(all_true_labels),
            len(all_predictions),
            total_correct,
            total_samples - total_correct,
            round(overall_accuracy, 4),
            round(macro_precision, 4),
            round(macro_recall, 4),
            round(macro_f1, 4),
            round(macro_accuracy, 4),
            round(weighted_precision, 4),
            round(weighted_recall, 4),
            round(weighted_f1, 4),
            round(weighted_accuracy, 4)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    print("=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print("=" * 120)
    print()
    
    # 7. Class-wise Accuracy DataFrame
    classwise_accuracy_df = metrics_df[~metrics_df['Category'].str.contains('avg')][['Category', 'Accuracy', 'Support']].copy()
    classwise_accuracy_df = classwise_accuracy_df.sort_values('Accuracy', ascending=False)
    
    print("=" * 120)
    print("CLASS-WISE ACCURACY (Sorted)")
    print("=" * 120)
    print(classwise_accuracy_df.to_string(index=False))
    print("=" * 120)
    print()
    
    # Save all DataFrames to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_files = {}
    
    # Save Confusion Matrix
    confusion_csv = os.path.join(output_csv_folder, f'confusion_matrix_{timestamp}.csv')
    confusion_df.to_csv(confusion_csv, index=False)
    csv_files['confusion_matrix'] = confusion_csv
    print(f"✓ Saved: {confusion_csv}")
    
    # Save Classification Metrics
    metrics_csv = os.path.join(output_csv_folder, f'classification_metrics_{timestamp}.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    csv_files['classification_metrics'] = metrics_csv
    print(f"✓ Saved: {metrics_csv}")
    
    # Save Summary Statistics
    summary_csv = os.path.join(output_csv_folder, f'summary_statistics_{timestamp}.csv')
    summary_df.to_csv(summary_csv, index=False)
    csv_files['summary_statistics'] = summary_csv
    print(f"✓ Saved: {summary_csv}")
    
    # Save Class-wise Accuracy
    classwise_csv = os.path.join(output_csv_folder, f'classwise_accuracy_{timestamp}.csv')
    classwise_accuracy_df.to_csv(classwise_csv, index=False)
    csv_files['classwise_accuracy'] = classwise_csv
    print(f"✓ Saved: {classwise_csv}")
    
    # Save All Predictions
    predictions_csv = os.path.join(output_csv_folder, f'all_predictions_{timestamp}.csv')
    predictions_df.to_csv(predictions_csv, index=False)
    csv_files['all_predictions'] = predictions_csv
    print(f"✓ Saved: {predictions_csv}")
    
    # Save Misclassifications (if any)
    if not misclass_df.empty:
        misclass_csv = os.path.join(output_csv_folder, f'misclassifications_{timestamp}.csv')
        misclass_df.to_csv(misclass_csv, index=False)
        csv_files['misclassifications'] = misclass_csv
        print(f"✓ Saved: {misclass_csv}")
    
    # Create a combined report in Excel (if openpyxl is available)
    try:
        excel_file = os.path.join(output_csv_folder, f'classification_report_{timestamp}.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            confusion_df.to_excel(writer, sheet_name='Confusion Matrix', index=False)
            classwise_accuracy_df.to_excel(writer, sheet_name='Classwise Accuracy', index=False)
            predictions_df.to_excel(writer, sheet_name='All Predictions', index=False)
            if not misclass_df.empty:
                misclass_df.to_excel(writer, sheet_name='Misclassifications', index=False)
        
        csv_files['excel_report'] = excel_file
        print(f"✓ Saved: {excel_file}")
    except ImportError:
        print("⚠ openpyxl not installed. Skipping Excel report. Install with: pip install openpyxl")
    except Exception as e:
        print(f"⚠ Could not create Excel file: {e}")
    
    print()
    
    return {
        'confusion_df': confusion_df,
        'metrics_df': metrics_df,
        'classwise_accuracy_df': classwise_accuracy_df,
        'predictions_df': predictions_df,
        'misclass_df': misclass_df if not misclass_df.empty else None,
        'overall_accuracy': overall_accuracy,
        'csv_files': csv_files
    }


# Main execution
if __name__ == "__main__":
    CLASSIFIED_FOLDER = r"E:\Master's\2nd Year\MyWork\Data\classified_images_final"
    OUTPUT_CSV_FOLDER = r"E:\Master's\2nd Year\MyWork\Data\classification_reports_final"
    
    print("=" * 120)
    print("IMAGE CLASSIFICATION EVALUATION REPORT (CSV Export)")
    print("=" * 120)
    
    results = generate_confusion_matrix_and_report(CLASSIFIED_FOLDER, OUTPUT_CSV_FOLDER)
    
    print("=" * 120)
    print("REPORT GENERATION COMPLETE!")
    print("=" * 120)
    print(f"All CSV files saved to: {OUTPUT_CSV_FOLDER}")
    print("=" * 120)