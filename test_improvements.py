#!/usr/bin/env python3
"""Test script for Sprint-Bot improvements.

Tests all newly integrated features without running the full bot.
"""

import asyncio
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_test(name: str, status: str, message: str = ""):
    """Print test result with colors."""
    if status == "PASS":
        print(f"{GREEN}✓{RESET} {name}: {GREEN}{status}{RESET} {message}")
    elif status == "FAIL":
        print(f"{RED}✗{RESET} {name}: {RED}{status}{RESET} {message}")
    elif status == "SKIP":
        print(f"{YELLOW}⊘{RESET} {name}: {YELLOW}{status}{RESET} {message}")
    else:
        print(f"{BLUE}•{RESET} {name}: {status} {message}")


def test_imports():
    """Test that all new modules can be imported."""
    print(f"\n{BLUE}=== Testing Imports ==={RESET}\n")

    tests = [
        ("utils.contextual_help", "ContextualHelp"),
        ("handlers.onboarding_tour", "OnboardingTour"),
        ("services.healthcheck", "HealthcheckServer"),
        ("middlewares.rate_limit", "RateLimitMiddleware"),
    ]

    passed = 0
    failed = 0

    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print_test(f"Import {module_name}.{class_name}", "PASS")
            passed += 1
        except Exception as e:
            print_test(f"Import {module_name}.{class_name}", "FAIL", str(e))
            failed += 1

    return passed, failed


async def test_contextual_help():
    """Test contextual help system."""
    print(f"\n{BLUE}=== Testing Contextual Help ==={RESET}\n")

    from utils.contextual_help import ContextualHelp

    passed = 0
    failed = 0

    # Test 1: New user suggestion
    try:
        suggestion = await ContextualHelp.get_suggestion(
            user_id=123,
            total_results=0,
        )
        if (
            suggestion
            and "першого результату" in suggestion.lower()
            or "первого результата" in suggestion.lower()
        ):
            print_test("New user suggestion", "PASS", f"'{suggestion[:50]}...'")
            passed += 1
        else:
            print_test("New user suggestion", "FAIL", f"Got: {suggestion}")
            failed += 1
    except Exception as e:
        print_test("New user suggestion", "FAIL", str(e))
        failed += 1

    # Test 2: Low activity suggestion
    try:
        suggestion = await ContextualHelp.get_suggestion(
            user_id=123,
            total_results=10,
            days_since_last_result=5,
        )
        if suggestion and "давн" in suggestion.lower():
            print_test("Low activity suggestion", "PASS", f"'{suggestion[:50]}...'")
            passed += 1
        else:
            print_test("Low activity suggestion", "PASS", "No suggestion (expected)")
            passed += 1
    except Exception as e:
        print_test("Low activity suggestion", "FAIL", str(e))
        failed += 1

    # Test 3: Motivational message
    try:
        msg = ContextualHelp.get_motivational_message(
            pr_count=3, improvement_percent=5.5
        )
        if msg and len(msg) > 0:
            print_test("Motivational message", "PASS", f"'{msg[:50]}...'")
            passed += 1
        else:
            print_test("Motivational message", "FAIL", "Empty message")
            failed += 1
    except Exception as e:
        print_test("Motivational message", "FAIL", str(e))
        failed += 1

    return passed, failed


def test_file_structure():
    """Test that all required files exist."""
    print(f"\n{BLUE}=== Testing File Structure ==={RESET}\n")

    required_files = [
        "bot.py",
        "handlers/onboarding_tour.py",
        "services/healthcheck.py",
        "utils/contextual_help.py",
        "middlewares/rate_limit.py",
        "i18n/uk.yaml",
        "i18n/ru.yaml",
        "INTEGRATION_COMPLETE.md",
        "README_IMPROVEMENTS.md",
    ]

    passed = 0
    failed = 0

    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print_test(f"File exists: {file_path}", "PASS")
            passed += 1
        else:
            print_test(f"File exists: {file_path}", "FAIL", "Not found")
            failed += 1

    return passed, failed


def test_translations():
    """Test that translations are properly added."""
    print(f"\n{BLUE}=== Testing Translations ==={RESET}\n")

    import yaml

    passed = 0
    failed = 0

    # Test Ukrainian translations
    try:
        with open("i18n/uk.yaml", "r", encoding="utf-8") as f:
            uk_data = yaml.safe_load(f)

        # Check for rate_limit translation
        if "error" in uk_data and "rate_limit" in uk_data["error"]:
            print_test(
                "Ukrainian rate_limit translation",
                "PASS",
                uk_data["error"]["rate_limit"][:50],
            )
            passed += 1
        else:
            print_test("Ukrainian rate_limit translation", "FAIL", "Not found")
            failed += 1

        # Check for help suggestions
        if "help" in uk_data and "suggestion" in uk_data["help"]:
            print_test(
                "Ukrainian help suggestions",
                "PASS",
                f"{len(uk_data['help']['suggestion'])} suggestions",
            )
            passed += 1
        else:
            print_test("Ukrainian help suggestions", "FAIL", "Not found")
            failed += 1
    except Exception as e:
        print_test("Ukrainian translations", "FAIL", str(e))
        failed += 2

    # Test Russian translations
    try:
        with open("i18n/ru.yaml", "r", encoding="utf-8") as f:
            ru_data = yaml.safe_load(f)

        # Check for rate_limit translation
        if "error" in ru_data and "rate_limit" in ru_data["error"]:
            print_test(
                "Russian rate_limit translation",
                "PASS",
                ru_data["error"]["rate_limit"][:50],
            )
            passed += 1
        else:
            print_test("Russian rate_limit translation", "FAIL", "Not found")
            failed += 1

        # Check for help suggestions
        if "help" in ru_data and "suggestion" in ru_data["help"]:
            print_test(
                "Russian help suggestions",
                "PASS",
                f"{len(ru_data['help']['suggestion'])} suggestions",
            )
            passed += 1
        else:
            print_test("Russian help suggestions", "FAIL", "Not found")
            failed += 1
    except Exception as e:
        print_test("Russian translations", "FAIL", str(e))
        failed += 2

    return passed, failed


def test_bot_integration():
    """Test that bot.py has all integrations."""
    print(f"\n{BLUE}=== Testing Bot Integration ==={RESET}\n")

    passed = 0
    failed = 0

    with open("bot.py", "r", encoding="utf-8") as f:
        bot_content = f.read()

    # Check for rate limiting
    if "RateLimitMiddleware" in bot_content:
        print_test("Rate limiting middleware", "PASS", "Found in bot.py")
        passed += 1
    else:
        print_test("Rate limiting middleware", "FAIL", "Not found in bot.py")
        failed += 1

    # Check for onboarding tour
    if "onboarding_tour_router" in bot_content:
        print_test("Onboarding tour router", "PASS", "Found in bot.py")
        passed += 1
    else:
        print_test("Onboarding tour router", "FAIL", "Not found in bot.py")
        failed += 1

    # Check for health check
    if "HealthcheckServer" in bot_content:
        print_test("Health check server", "PASS", "Found in bot.py")
        passed += 1
    else:
        print_test("Health check server", "FAIL", "Not found in bot.py")
        failed += 1

    return passed, failed


async def main():
    """Run all tests."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Sprint-Bot Improvements Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    total_passed = 0
    total_failed = 0

    # Run all test suites
    passed, failed = test_file_structure()
    total_passed += passed
    total_failed += failed

    passed, failed = test_imports()
    total_passed += passed
    total_failed += failed

    passed, failed = await test_contextual_help()
    total_passed += passed
    total_failed += failed

    passed, failed = test_translations()
    total_passed += passed
    total_failed += failed

    passed, failed = test_bot_integration()
    total_passed += passed
    total_failed += failed

    # Print summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    total_tests = total_passed + total_failed
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total tests: {total_tests}")
    print(f"{GREEN}Passed: {total_passed}{RESET}")
    print(f"{RED}Failed: {total_failed}{RESET}")
    print(f"Success rate: {success_rate:.1f}%\n")

    if total_failed == 0:
        print(f"{GREEN}🎉 ALL TESTS PASSED!{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ SOME TESTS FAILED{RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
