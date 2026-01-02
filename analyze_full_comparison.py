#!/usr/bin/env python3
"""
Analyze full embedding + TraceRank comparison results
Handles 7 models × 3 TraceRank configs = 21 configurations
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
        # Extract model name, tracerank config, and method
        # Format: {model}_{tracerank}_{method}_results.json
        method = parts[-2]  # search or recall

        # Model + tracerank config (everything except last 2 parts)
        config_name = "_".join(parts[:-2])

        if config_name not in results:
            results[config_name] = {}

        with open(result_file) as f:
            results[config_name][method] = json.load(f)

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


def parse_config_name(config_name: str) -> tuple:
    """Parse config name into model and tracerank variant"""
    parts = config_name.rsplit('_', 1)
    if len(parts) == 2:
        model, tracerank = parts
    else:
        model = config_name
        tracerank = "unknown"
    return model, tracerank


def generate_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive comparison report"""

    report = []
    report.append("=" * 100)
    report.append("FULL EMBEDDING + TRACERANK COMPARISON REPORT")
    report.append("=" * 100)
    report.append("")

    # Calculate metrics for all configurations
    all_metrics = {}
    for config_name, methods in results.items():
        all_metrics[config_name] = {}
        for method, test_results in methods.items():
            metrics = calculate_metrics(test_results)
            all_metrics[config_name][method] = metrics

    # Group by model
    by_model = defaultdict(dict)
    for config_name in all_metrics.keys():
        model, tracerank = parse_config_name(config_name)
        by_model[model][tracerank] = all_metrics[config_name]

    # ========================================
    # 1. OVERALL RANKINGS
    # ========================================
    report.append("1. OVERALL RANKINGS (by average accuracy)")
    report.append("=" * 100)
    report.append("")

    rankings = []
    for config_name, methods in all_metrics.items():
        search_acc = methods.get('search', {}).get('accuracy', 0)
        recall_acc = methods.get('recall', {}).get('accuracy', 0)
        avg_acc = (search_acc + recall_acc) / 2
        rankings.append((config_name, avg_acc, search_acc, recall_acc))

    rankings.sort(key=lambda x: x[1], reverse=True)

    report.append(f"{'Rank':<6} {'Configuration':<40} {'Avg':<10} {'Search':<10} {'Recall':<10}")
    report.append("-" * 100)
    for i, (config, avg_acc, search_acc, recall_acc) in enumerate(rankings[:10], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        report.append(f"{medal:<6} {config:<40} {avg_acc:>8.1%}   {search_acc:>8.1%}   {recall_acc:>8.1%}")

    report.append("")
    report.append("")

    # ========================================
    # 2. MODEL COMPARISON (Best config for each)
    # ========================================
    report.append("2. BEST CONFIGURATION PER MODEL")
    report.append("=" * 100)
    report.append("")

    model_best = []
    for model, tracerank_configs in sorted(by_model.items()):
        best_acc = 0
        best_config = None

        for tracerank, methods in tracerank_configs.items():
            search_acc = methods.get('search', {}).get('accuracy', 0)
            recall_acc = methods.get('recall', {}).get('accuracy', 0)
            avg_acc = (search_acc + recall_acc) / 2

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_config = tracerank

        model_best.append((model, best_config, best_acc))

    model_best.sort(key=lambda x: x[2], reverse=True)

    report.append(f"{'Model':<25} {'Best TraceRank Config':<25} {'Accuracy':<10}")
    report.append("-" * 100)
    for model, config, acc in model_best:
        report.append(f"{model:<25} {config:<25} {acc:>8.1%}")

    report.append("")
    report.append("")

    # ========================================
    # 3. TRACERANK IMPACT BY MODEL
    # ========================================
    report.append("3. TRACERANK IMPACT ANALYSIS")
    report.append("=" * 100)
    report.append("")

    for model in sorted(by_model.keys()):
        configs = by_model[model]

        report.append(f"{model}:")
        report.append("-" * 100)

        # Compare the three configs
        config_results = {}
        for tracerank in ['full', 'no-graph', 'off']:
            if tracerank in configs:
                methods = configs[tracerank]
                search_acc = methods.get('search', {}).get('accuracy', 0)
                recall_acc = methods.get('recall', {}).get('accuracy', 0)
                avg_acc = (search_acc + recall_acc) / 2
                config_results[tracerank] = avg_acc

        # Show comparison
        if 'full' in config_results and 'no-graph' in config_results:
            graph_impact = config_results['full'] - config_results['no-graph']
            graph_symbol = "📈" if graph_impact > 0 else "📉" if graph_impact < 0 else "➡️"
            report.append(f"  Full TraceRank:     {config_results.get('full', 0):>6.1%}")
            report.append(f"  No Graph Boost:     {config_results.get('no-graph', 0):>6.1%}  {graph_symbol} Graph impact: {graph_impact:+.1%}")

        if 'no-graph' in config_results and 'off' in config_results:
            tracerank_impact = config_results['no-graph'] - config_results['off']
            tr_symbol = "📈" if tracerank_impact > 0 else "📉" if tracerank_impact < 0 else "➡️"
            report.append(f"  No TraceRank:       {config_results.get('off', 0):>6.1%}  {tr_symbol} TraceRank impact: {tracerank_impact:+.1%}")

        report.append("")

    report.append("")

    # ========================================
    # 4. PERFORMANCE COMPARISON
    # ========================================
    report.append("4. PERFORMANCE (Speed) COMPARISON")
    report.append("=" * 100)
    report.append("")

    speeds = []
    for config_name, methods in all_metrics.items():
        search_time = methods.get('search', {}).get('avg_duration_ms', 0)
        recall_time = methods.get('recall', {}).get('avg_duration_ms', 0)
        avg_time = (search_time + recall_time) / 2
        speeds.append((config_name, avg_time))

    speeds.sort(key=lambda x: x[1])

    report.append(f"{'Configuration':<40} {'Avg Query Time':<15}")
    report.append("-" * 100)
    for config, avg_time in speeds[:10]:
        report.append(f"{config:<40} {avg_time:>10.0f}ms")

    report.append("")
    report.append("")

    # ========================================
    # 5. COMMERCIAL vs OPEN SOURCE
    # ========================================
    if 'ada-002' in by_model:
        report.append("5. COMMERCIAL vs OPEN SOURCE")
        report.append("=" * 100)
        report.append("")

        # Get best ada-002 config
        ada_configs = by_model['ada-002']
        ada_best_acc = 0
        ada_best_config = None

        for tracerank, methods in ada_configs.items():
            search_acc = methods.get('search', {}).get('accuracy', 0)
            recall_acc = methods.get('recall', {}).get('accuracy', 0)
            avg_acc = (search_acc + recall_acc) / 2
            if avg_acc > ada_best_acc:
                ada_best_acc = avg_acc
                ada_best_config = tracerank

        report.append(f"OpenAI ada-002 (best config: {ada_best_config}): {ada_best_acc:.1%}")
        report.append("")
        report.append("Open source models beating ada-002:")

        beaten_count = 0
        for model, best_config, acc in model_best:
            if model != 'ada-002' and acc > ada_best_acc:
                beaten_count += 1
                diff = acc - ada_best_acc
                report.append(f"  {model:<25} {best_config:<25} {acc:>6.1%} ({diff:+.1%})")

        if beaten_count == 0:
            report.append("  (None - ada-002 is best)")

        report.append("")

    report.append("=" * 100)

    return "\n".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_full_comparison.py <results_directory>")
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

    print(f"Found results for {len(results)} configurations\n")

    # Generate comparison report
    report = generate_report(results)
    print(report)

    # Save report to file
    report_file = results_dir / "full_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
