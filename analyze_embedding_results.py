#!/usr/bin/env python3
"""
Analyze and compare embedding model test results
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all test results from the results directory"""
    results = {}

    # Find all result JSON files
    for result_file in results_dir.glob("*_results.json"):
        parts = result_file.stem.split("_")
        # Extract model name and method from filename
        # Format: {model_name}_{method}_results.json
        method = parts[-2]  # search or recall
        model_name = "_".join(parts[:-2])

        if model_name not in results:
            results[model_name] = {}

        with open(result_file) as f:
            results[model_name][method] = json.load(f)

    return results


def calculate_metrics(test_results: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate metrics from test results"""
    total = len(test_results)
    if total == 0:
        return {}

    successful = sum(1 for r in test_results if r['success'])
    relevant = sum(1 for r in test_results if r.get('evaluation', {}).get('appears_relevant', False))

    avg_duration = sum(r['duration_ms'] for r in test_results) / total
    avg_match_ratio = sum(
        r.get('evaluation', {}).get('match_ratio', 0)
        for r in test_results
    ) / total

    # Category breakdown
    by_category = defaultdict(list)
    for r in test_results:
        by_category[r['category']].append(r)

    category_scores = {}
    for cat, cat_results in by_category.items():
        cat_relevant = sum(
            1 for r in cat_results
            if r.get('evaluation', {}).get('appears_relevant', False)
        )
        category_scores[cat] = {
            'total': len(cat_results),
            'relevant': cat_relevant,
            'accuracy': cat_relevant / len(cat_results) if cat_results else 0
        }

    return {
        'total_questions': total,
        'successful_queries': successful,
        'relevant_answers': relevant,
        'accuracy': relevant / total,
        'avg_duration_ms': avg_duration,
        'avg_match_ratio': avg_match_ratio,
        'category_breakdown': category_scores
    }


def compare_models(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate a comparison report across all models"""

    report = []
    report.append("=" * 80)
    report.append("EMBEDDING MODEL COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")

    # Calculate metrics for each model/method combination
    model_metrics = {}
    for model_name, methods in results.items():
        model_metrics[model_name] = {}
        for method, test_results in methods.items():
            metrics = calculate_metrics(test_results)
            model_metrics[model_name][method] = metrics

    # Overall accuracy comparison
    report.append("ACCURACY COMPARISON (% of relevant answers)")
    report.append("-" * 80)
    report.append(f"{'Model':<25} {'Search':<15} {'Recall':<15} {'Avg':<15}")
    report.append("-" * 80)

    accuracy_scores = []
    for model_name in sorted(model_metrics.keys()):
        search_acc = model_metrics[model_name].get('search', {}).get('accuracy', 0)
        recall_acc = model_metrics[model_name].get('recall', {}).get('accuracy', 0)
        avg_acc = (search_acc + recall_acc) / 2
        accuracy_scores.append((model_name, search_acc, recall_acc, avg_acc))

        report.append(
            f"{model_name:<25} {search_acc:>6.1%}         {recall_acc:>6.1%}         {avg_acc:>6.1%}"
        )

    # Find best performer
    best_model = max(accuracy_scores, key=lambda x: x[3])
    report.append("-" * 80)
    report.append(f"Best overall: {best_model[0]} ({best_model[3]:.1%})")
    report.append("")

    # Performance (speed) comparison
    report.append("PERFORMANCE COMPARISON (avg query time in ms)")
    report.append("-" * 80)
    report.append(f"{'Model':<25} {'Search':<15} {'Recall':<15} {'Avg':<15}")
    report.append("-" * 80)

    for model_name in sorted(model_metrics.keys()):
        search_time = model_metrics[model_name].get('search', {}).get('avg_duration_ms', 0)
        recall_time = model_metrics[model_name].get('recall', {}).get('avg_duration_ms', 0)
        avg_time = (search_time + recall_time) / 2

        report.append(
            f"{model_name:<25} {search_time:>6.0f}ms        {recall_time:>6.0f}ms        {avg_time:>6.0f}ms"
        )

    report.append("")

    # Detailed breakdown by category
    report.append("CATEGORY BREAKDOWN")
    report.append("=" * 80)

    for model_name in sorted(model_metrics.keys()):
        report.append("")
        report.append(f"{model_name}")
        report.append("-" * 80)

        for method in ['search', 'recall']:
            if method not in model_metrics[model_name]:
                continue

            metrics = model_metrics[model_name][method]
            cat_breakdown = metrics.get('category_breakdown', {})

            report.append(f"  {method.upper()}:")
            for cat, scores in sorted(cat_breakdown.items()):
                report.append(
                    f"    {cat:<20} {scores['relevant']}/{scores['total']} "
                    f"({scores['accuracy']:.1%})"
                )

    report.append("")
    report.append("=" * 80)

    # Quality metrics comparison
    report.append("MATCH RATIO COMPARISON (avg % of expected keywords found)")
    report.append("-" * 80)
    report.append(f"{'Model':<25} {'Search':<15} {'Recall':<15} {'Avg':<15}")
    report.append("-" * 80)

    for model_name in sorted(model_metrics.keys()):
        search_match = model_metrics[model_name].get('search', {}).get('avg_match_ratio', 0)
        recall_match = model_metrics[model_name].get('recall', {}).get('avg_match_ratio', 0)
        avg_match = (search_match + recall_match) / 2

        report.append(
            f"{model_name:<25} {search_match:>6.1%}         {recall_match:>6.1%}         {avg_match:>6.1%}"
        )

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_embedding_results.py <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    print("Loading results...")
    results = load_results(results_dir)

    if not results:
        print("No results found in directory")
        sys.exit(1)

    print(f"Found results for {len(results)} models\n")

    # Generate comparison report
    report = compare_models(results)
    print(report)

    # Save report to file
    report_file = results_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
