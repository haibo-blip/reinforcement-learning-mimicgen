#!/usr/bin/env python3
"""
Run all RL pipeline tests.

Usage:
    python tests/rl_pipeline/run_all_tests.py
    python tests/rl_pipeline/run_all_tests.py --verbose
    python tests/rl_pipeline/run_all_tests.py --test rollout
    python tests/rl_pipeline/run_all_tests.py --test advantage
    python tests/rl_pipeline/run_all_tests.py --test policy
    python tests/rl_pipeline/run_all_tests.py --test integration
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_filter=None, verbose=False):
    """Run all or selected tests."""

    print("\n" + "=" * 80)
    print("MANIFLOW RL PIPELINE TEST SUITE")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print("=" * 80)

    # Import test modules
    from tests.rl_pipeline import test_rollout_generation
    from tests.rl_pipeline import test_advantage_calculation
    from tests.rl_pipeline import test_rl_policy
    from tests.rl_pipeline import test_pipeline_integration

    test_modules = {
        'rollout': ('Rollout Generation', test_rollout_generation),
        'advantage': ('Advantage Calculation', test_advantage_calculation),
        'policy': ('RL Policy', test_rl_policy),
        'integration': ('Pipeline Integration', test_pipeline_integration),
    }

    # Filter tests if requested
    if test_filter:
        if test_filter not in test_modules:
            print(f"Unknown test: {test_filter}")
            print(f"Available tests: {list(test_modules.keys())}")
            return False
        test_modules = {test_filter: test_modules[test_filter]}

    # Run tests
    results = {}
    for key, (name, module) in test_modules.items():
        print(f"\n{'=' * 80}")
        print(f"RUNNING: {name} Tests")
        print("=" * 80)

        try:
            success = module.run_all_tests()
            results[name] = success
        except Exception as e:
            print(f"Test module {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {name}")
        if not success:
            all_passed = False

    passed_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    print(f"\nOverall: {passed_count}/{total_count} test suites passed")

    if all_passed:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run RL pipeline tests")
    parser.add_argument('--test', '-t', type=str, default=None,
                       help='Run specific test (rollout, advantage, policy, integration)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    success = run_tests(test_filter=args.test, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
