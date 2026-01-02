#!/usr/bin/env python3
"""
Test harness for evaluating vestig's Q&A performance
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import sys


class QATestHarness:
    def __init__(self, config_path: str, qa_file: str):
        self.config_path = config_path
        self.qa_file = qa_file
        self.results = []

    def load_qa_pairs(self) -> List[Dict[str, Any]]:
        """Load Q&A pairs from JSON file"""
        with open(self.qa_file, 'r') as f:
            return json.load(f)

    def query_vestig(self, question: str, method: str = "search", use_explain: bool = False) -> Dict[str, Any]:
        """
        Query vestig using either 'search' or 'recall' method
        Returns: {
            'output': str,
            'duration_ms': float,
            'success': bool,
            'error': str | None
        }
        """
        start_time = time.perf_counter()

        try:
            if method == "search":
                cmd = [
                    "vestig",
                    "--config", self.config_path,
                    "memory", "search",
                    question
                ]
            else:  # recall
                cmd = [
                    "vestig",
                    "--config", self.config_path,
                    "memory", "recall",
                    question
                ]
                if use_explain:
                    cmd.append("--explain")

            # Run from the directory containing the config file
            # This ensures relative db_path in config is resolved correctly
            config_dir = Path(self.config_path).parent.absolute()

            # Convert config path to absolute so it works from any cwd
            abs_config = Path(self.config_path).absolute()
            cmd[2] = str(abs_config)  # Update --config argument

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=config_dir
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            return {
                'output': result.stdout,
                'stderr': result.stderr,
                'duration_ms': duration_ms,
                'success': result.returncode == 0,
                'error': None if result.returncode == 0 else result.stderr
            }

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return {
                'output': '',
                'stderr': '',
                'duration_ms': duration_ms,
                'success': False,
                'error': 'Timeout after 60s'
            }
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return {
                'output': '',
                'stderr': '',
                'duration_ms': duration_ms,
                'success': False,
                'error': str(e)
            }

    def evaluate_answer(self, qa_pair: Dict, vestig_output: str) -> Dict[str, Any]:
        """
        Simple evaluation of whether the answer appears relevant
        This is a basic keyword/substring check - could be enhanced with LLM evaluation
        """
        expected = qa_pair['answer'].lower()
        output = vestig_output.lower()

        # Extract key terms from expected answer (simple heuristic)
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        expected_words = set(expected.split()) - stop_words

        # Count how many expected words appear in output
        matches = sum(1 for word in expected_words if word in output)
        match_ratio = matches / len(expected_words) if expected_words else 0

        return {
            'match_ratio': match_ratio,
            'matches': matches,
            'total_expected_words': len(expected_words),
            'appears_relevant': match_ratio > 0.3,  # If 30%+ of key words match
            'has_output': len(output.strip()) > 0
        }

    def run_test(self, qa_pair: Dict, method: str = "search", use_explain: bool = False) -> Dict[str, Any]:
        """Run a single test"""
        print(f"\n{'='*80}")
        print(f"Q{qa_pair['id']}: {qa_pair['question']}")
        print(f"Expected: {qa_pair['answer'][:100]}...")
        print(f"Method: {method}{' --explain' if use_explain else ''}")

        # Query vestig
        vestig_result = self.query_vestig(qa_pair['question'], method, use_explain)

        if not vestig_result['success']:
            print(f"❌ Query failed: {vestig_result['error']}")
            return {
                'qa_id': qa_pair['id'],
                'question': qa_pair['question'],
                'expected_answer': qa_pair['answer'],
                'category': qa_pair['category'],
                'project': qa_pair['project'],
                'method': method,
                'success': False,
                'error': vestig_result['error'],
                'duration_ms': vestig_result['duration_ms']
            }

        # Evaluate the answer
        evaluation = self.evaluate_answer(qa_pair, vestig_result['output'])

        print(f"Duration: {vestig_result['duration_ms']:.0f}ms")
        print(f"Match ratio: {evaluation['match_ratio']:.2%}")
        print(f"Relevant: {'✓' if evaluation['appears_relevant'] else '✗'}")
        print(f"\nVestig output preview:")
        print(vestig_result['output'][:500])

        return {
            'qa_id': qa_pair['id'],
            'question': qa_pair['question'],
            'expected_answer': qa_pair['answer'],
            'vestig_output': vestig_result['output'],
            'category': qa_pair['category'],
            'project': qa_pair['project'],
            'method': method,
            'success': True,
            'duration_ms': vestig_result['duration_ms'],
            'evaluation': evaluation
        }

    def run_all_tests(self, method: str = "search", use_explain: bool = False) -> List[Dict[str, Any]]:
        """Run all Q&A tests"""
        qa_pairs = self.load_qa_pairs()
        results = []

        print(f"\n{'='*80}")
        print(f"Running {len(qa_pairs)} tests using method: {method}{' --explain' if use_explain else ''}")
        print(f"Config: {self.config_path}")
        print(f"{'='*80}")

        for qa_pair in qa_pairs:
            result = self.run_test(qa_pair, method, use_explain)
            results.append(result)

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate a summary report"""
        if not self.results:
            return "No results to report"

        total = len(self.results)
        successful_queries = sum(1 for r in self.results if r['success'])
        relevant_answers = sum(1 for r in self.results if r.get('evaluation', {}).get('appears_relevant', False))

        avg_duration = sum(r['duration_ms'] for r in self.results) / total

        # Group by category
        by_category = {}
        for r in self.results:
            cat = r['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        report = f"\n{'='*80}\n"
        report += "TEST SUMMARY\n"
        report += f"{'='*80}\n\n"
        report += f"Total questions: {total}\n"
        report += f"Successful queries: {successful_queries}/{total} ({successful_queries/total:.1%})\n"
        report += f"Relevant answers: {relevant_answers}/{total} ({relevant_answers/total:.1%})\n"
        report += f"Average duration: {avg_duration:.0f}ms\n"
        report += f"\nBREAKDOWN BY CATEGORY:\n"

        for cat, results in sorted(by_category.items()):
            relevant = sum(1 for r in results if r.get('evaluation', {}).get('appears_relevant', False))
            report += f"\n  {cat}: {relevant}/{len(results)} relevant ({relevant/len(results):.1%})\n"
            for r in results:
                eval_data = r.get('evaluation', {})
                status = "✓" if eval_data.get('appears_relevant', False) else "✗"
                ratio = eval_data.get('match_ratio', 0)
                report += f"    {status} Q{r['qa_id']}: {ratio:.1%} match - {r['question'][:60]}...\n"

        # Performance breakdown
        report += f"\n{'='*80}\n"
        report += "PERFORMANCE ANALYSIS:\n"
        report += f"{'='*80}\n"

        fast = sum(1 for r in self.results if r['duration_ms'] < 1000)
        medium = sum(1 for r in self.results if 1000 <= r['duration_ms'] < 5000)
        slow = sum(1 for r in self.results if r['duration_ms'] >= 5000)

        report += f"\n  Fast (<1s):     {fast}/{total}\n"
        report += f"  Medium (1-5s):  {medium}/{total}\n"
        report += f"  Slow (>5s):     {slow}/{total}\n"

        # Slowest queries
        slowest = sorted(self.results, key=lambda x: x['duration_ms'], reverse=True)[:5]
        report += f"\n  Slowest queries:\n"
        for r in slowest:
            report += f"    {r['duration_ms']:.0f}ms - Q{r['qa_id']}: {r['question'][:60]}...\n"

        return report

    def save_results(self, output_file: str):
        """Save detailed results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_qa_harness.py <method> [--explain] [config_path] [qa_file]")
        print("  method: 'search' or 'recall'")
        print("  --explain: add explain flag (only works with recall)")
        print("  config_path: path to vestig config (default: ./test/config-cerebras2.yaml)")
        print("  qa_file: path to Q&A JSON (default: ./test/qa_matterbase_projects.json)")
        sys.exit(1)

    method = sys.argv[1]

    # Check for --explain flag
    use_explain = False
    arg_offset = 0
    if len(sys.argv) > 2 and sys.argv[2] == '--explain':
        use_explain = True
        arg_offset = 1

    config_path = sys.argv[2 + arg_offset] if len(sys.argv) > 2 + arg_offset else "./test/config-cerebras2.yaml"
    qa_file = sys.argv[3 + arg_offset] if len(sys.argv) > 3 + arg_offset else "./test/qa_matterbase_projects.json"

    if method not in ['search', 'recall']:
        print(f"Error: method must be 'search' or 'recall', got '{method}'")
        sys.exit(1)

    if use_explain and method != 'recall':
        print("Warning: --explain flag only works with 'recall' method")
        use_explain = False

    harness = QATestHarness(config_path, qa_file)

    # Run tests
    results = harness.run_all_tests(method, use_explain)

    # Generate and print report
    report = harness.generate_report()
    print(report)

    # Save detailed results
    # Determine output path based on current directory
    cwd = Path.cwd()
    if cwd.name == 'test':
        # Already in test directory
        output_file = f"qa_results_{method}_{int(time.time())}.json"
    else:
        # In parent directory
        output_file = f"./test/qa_results_{method}_{int(time.time())}.json"
    harness.save_results(output_file)


if __name__ == "__main__":
    main()
