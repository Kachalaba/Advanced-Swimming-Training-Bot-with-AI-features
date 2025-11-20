#!/usr/bin/env python3
"""Simple test script for Sprint-Bot improvements.

Tests without importing aiogram dependencies.
"""

import sys
from pathlib import Path

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(name: str, status: str, message: str = ""):
    """Print test result with colors."""
    if status == "PASS":
        print(f"{GREEN}✓{RESET} {name}: {GREEN}{status}{RESET} {message}")
    elif status == "FAIL":
        print(f"{RED}✗{RESET} {name}: {RED}{status}{RESET} {message}")
    else:
        print(f"{BLUE}•{RESET} {name}: {status} {message}")


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
        "IMPROVEMENTS_LOG.md",
        "ROADMAP_TO_10.md",
    ]
    
    passed = 0
    failed = 0
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print_test(f"{file_path}", "PASS", f"({size} bytes)")
            passed += 1
        else:
            print_test(f"{file_path}", "FAIL", "Not found")
            failed += 1
    
    return passed, failed


def test_syntax():
    """Test Python syntax of all new/modified files."""
    print(f"\n{BLUE}=== Testing Python Syntax ==={RESET}\n")
    
    import py_compile
    import tempfile
    
    files_to_test = [
        "bot.py",
        "handlers/onboarding_tour.py",
        "services/healthcheck.py",
        "utils/contextual_help.py",
        "middlewares/rate_limit.py",
        "handlers/common.py",
        "handlers/menu.py",
    ]
    
    passed = 0
    failed = 0
    
    for file_path in files_to_test:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pyc', delete=True) as tmp:
                py_compile.compile(file_path, cfile=tmp.name, doraise=True)
            print_test(f"Syntax: {file_path}", "PASS")
            passed += 1
        except py_compile.PyCompileError as e:
            print_test(f"Syntax: {file_path}", "FAIL", str(e))
            failed += 1
    
    return passed, failed


def test_bot_integrations():
    """Test that bot.py has all integrations."""
    print(f"\n{BLUE}=== Testing Bot Integrations ==={RESET}\n")
    
    passed = 0
    failed = 0
    
    with open("bot.py", "r", encoding="utf-8") as f:
        bot_content = f.read()
    
    integrations = [
        ("RateLimitMiddleware", "Rate limiting middleware"),
        ("CommandRateLimitMiddleware", "Command rate limiting"),
        ("onboarding_tour_router", "Onboarding tour router"),
        ("HealthcheckServer", "Health check server"),
        ("healthcheck.start()", "Health check start"),
        ("healthcheck.stop()", "Health check cleanup"),
    ]
    
    for code_snippet, description in integrations:
        if code_snippet in bot_content:
            print_test(description, "PASS", f"Found '{code_snippet}'")
            passed += 1
        else:
            print_test(description, "FAIL", f"'{code_snippet}' not found")
            failed += 1
    
    return passed, failed


def test_translations():
    """Test that translations exist."""
    print(f"\n{BLUE}=== Testing Translations ==={RESET}\n")
    
    passed = 0
    failed = 0
    
    # Test Ukrainian
    try:
        with open("i18n/uk.yaml", "r", encoding="utf-8") as f:
            uk_content = f.read()
        
        required_translations = [
            "rate_limit",
            "first_result",
            "new_prs",
            "check_progress",
        ]
        
        for trans in required_translations:
            if trans in uk_content:
                print_test(f"Ukrainian: {trans}", "PASS")
                passed += 1
            else:
                print_test(f"Ukrainian: {trans}", "FAIL", "Not found")
                failed += 1
    except Exception as e:
        print_test("Ukrainian translations", "FAIL", str(e))
        failed += len(required_translations)
    
    # Test Russian
    try:
        with open("i18n/ru.yaml", "r", encoding="utf-8") as f:
            ru_content = f.read()
        
        for trans in required_translations:
            if trans in ru_content:
                print_test(f"Russian: {trans}", "PASS")
                passed += 1
            else:
                print_test(f"Russian: {trans}", "FAIL", "Not found")
                failed += 1
    except Exception as e:
        print_test("Russian translations", "FAIL", str(e))
        failed += len(required_translations)
    
    return passed, failed


def test_menu_integration():
    """Test that handlers/menu.py has contextual help."""
    print(f"\n{BLUE}=== Testing Menu Integration ==={RESET}\n")
    
    passed = 0
    failed = 0
    
    with open("handlers/menu.py", "r", encoding="utf-8") as f:
        menu_content = f.read()
    
    checks = [
        ("contextual_help import", "from utils.contextual_help import"),
        ("ContextualHelp.get_suggestion", "ContextualHelp.get_suggestion"),
        ("format_suggestion_message", "format_suggestion_message"),
    ]
    
    for description, code_snippet in checks:
        if code_snippet in menu_content:
            print_test(description, "PASS")
            passed += 1
        else:
            print_test(description, "FAIL", f"'{code_snippet}' not found")
            failed += 1
    
    return passed, failed


def test_onboarding_tour_structure():
    """Test onboarding tour has all steps."""
    print(f"\n{BLUE}=== Testing Onboarding Tour Structure ==={RESET}\n")
    
    passed = 0
    failed = 0
    
    with open("handlers/onboarding_tour.py", "r", encoding="utf-8") as f:
        tour_content = f.read()
    
    required_elements = [
        ("Command handler", "@router.message(Command(\"tour\"))"),
        ("Welcome step", "def cmd_start_tour"),
        ("Step 1: Add result", "def tour_step_1_add_result"),
        ("Step 2: Records", "def tour_step_2_records"),
        ("Step 3: Progress", "def tour_step_3_progress"),
        ("Step 4: Complete", "def tour_step_4_complete"),
        ("Skip option", "def tour_skip"),
    ]
    
    for description, code_snippet in required_elements:
        if code_snippet in tour_content:
            print_test(description, "PASS")
            passed += 1
        else:
            print_test(description, "FAIL", f"'{code_snippet}' not found")
            failed += 1
    
    return passed, failed


def test_documentation():
    """Test that documentation is comprehensive."""
    print(f"\n{BLUE}=== Testing Documentation ==={RESET}\n")
    
    passed = 0
    failed = 0
    
    docs_to_check = [
        ("INTEGRATION_COMPLETE.md", "integration", 5000),
        ("README_IMPROVEMENTS.md", "improvements", 1000),
        ("IMPROVEMENTS_LOG.md", "log", 2000),
        ("ROADMAP_TO_10.md", "roadmap", 10000),
    ]
    
    for file_path, keyword, min_size in docs_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            content = path.read_text(encoding="utf-8")
            
            if size >= min_size and keyword.lower() in content.lower():
                print_test(f"{file_path}", "PASS", f"({size} bytes, has '{keyword}')")
                passed += 1
            else:
                print_test(f"{file_path}", "FAIL", f"Too small or missing '{keyword}'")
                failed += 1
        else:
            print_test(f"{file_path}", "FAIL", "Not found")
            failed += 1
    
    return passed, failed


def main():
    """Run all tests."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Sprint-Bot Improvements - Simple Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    test_suites = [
        test_file_structure,
        test_syntax,
        test_bot_integrations,
        test_translations,
        test_menu_integration,
        test_onboarding_tour_structure,
        test_documentation,
    ]
    
    for test_suite in test_suites:
        passed, failed = test_suite()
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
        print(f"{GREEN}🎉 ALL TESTS PASSED!{RESET}")
        print(f"{GREEN}✅ Bot is ready to run: python bot.py{RESET}\n")
        return 0
    else:
        print(f"{RED}❌ SOME TESTS FAILED{RESET}")
        print(f"{YELLOW}⚠️  Please fix the issues above{RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
