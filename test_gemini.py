#!/usr/bin/env python
"""
Test script for Gemini integration.
Run this to verify your Gemini API setup before running the full app.

Usage:
    python test_gemini.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from gemini_client import call_gemini


def test_basic_connection():
    """Test basic connection to Gemini API"""
    print("=" * 60)
    print("TEST 1: Basic Gemini API Connection")
    print("=" * 60)
    
    api_key = cfg.GEMINI_API_KEY
    model = cfg.GEMINI_MODEL
    
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("   Please set: set GEMINI_API_KEY=your-api-key-here")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-5:]}")
    print(f"✓ Model: {model}")
    
    prompt = "Trả về một JSON object có field 'test' với giá trị 'success'. Chỉ trả về JSON, không text khác."
    
    print("\nSending test request to Gemini...")
    result = call_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
    )
    
    if result:
        print("✓ Got response from Gemini!")
        print(f"  Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return True
    else:
        print("❌ No response from Gemini API")
        return False


def test_json_extraction():
    """Test JSON extraction and validation"""
    print("\n" + "=" * 60)
    print("TEST 2: JSON Parsing and Validation")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
    
    if not api_key:
        print("⚠ Skipping - API key not set")
        return True
    
    prompt = """Trả về một JSON object với chính xác 3 field sau (bắt buộc):
    - "explanation": một giải thích ngắn (string)
    - "recommendations": một array có 2 khuyến nghị (array of strings)
    - "nutrition_tips": một array có 2 lời khuyên (array of strings)
    
    Chỉ trả về JSON hợp lệ, không markdown hay text khác.
    """
    
    print("Sending validation request to Gemini...")
    result = call_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
    )
    
    required_keys = {"explanation", "recommendations", "nutrition_tips"}
    
    if not result:
        print("❌ No response from Gemini")
        return False
    
    missing_keys = required_keys - set(result.keys())
    
    if missing_keys:
        print(f"⚠ Response missing keys: {missing_keys}")
        print(f"  Got keys: {list(result.keys())}")
        print(f"  Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return False
    
    print("✓ All required keys present!")
    print(f"  - explanation: {type(result.get('explanation'))}")
    print(f"  - recommendations: {type(result.get('recommendations'))}")
    print(f"  - nutrition_tips: {type(result.get('nutrition_tips'))}")
    print(f"\nFull response:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    return True


def test_vietnamese_output():
    """Test Vietnamese language output"""
    print("\n" + "=" * 60)
    print("TEST 3: Vietnamese Language Output")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
    
    if not api_key:
        print("⚠ Skipping - API key not set")
        return True
    
    prompt = """Hãy viết một giải thích ngắn về bệnh tiểu đường bằng tiếng Việt.
    Trả về JSON object với field:
    - "explanation": giải thích (string)
    
    Chỉ trả về JSON hợp lệ."""
    
    print("Testing Vietnamese language support...")
    result = call_gemini(
        api_key=api_key,
        model=model,
        prompt=prompt,
    )
    
    if not result:
        print("❌ No response from Gemini")
        return False
    
    explanation = result.get("explanation", "")
    if explanation:
        # Check if response contains Vietnamese characters
        has_vietnamese = any(ord(c) > 255 for c in explanation)
        if has_vietnamese:
            print("✓ Vietnamese content detected!")
            print(f"  Sample: {explanation[:100]}...")
        else:
            print("⚠ Response doesn't contain Vietnamese characters")
            print(f"  Response: {explanation[:100]}...")
    
    return bool(explanation)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("GEMINI INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")
    
    enabled = os.environ.get("GEMINI_ENABLED", "1") == "1"
    if not enabled:
        print("⚠ GEMINI_ENABLED is set to 0 (disabled)")
        print("  Set GEMINI_ENABLED=1 to enable Gemini")
        return True
    
    results = []
    
    try:
        results.append(("API Connection", test_basic_connection()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("API Connection", False))
    
    try:
        results.append(("JSON Validation", test_json_extraction()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("JSON Validation", False))
    
    try:
        results.append(("Vietnamese Output", test_vietnamese_output()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Vietnamese Output", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Gemini integration is ready.")
        print("  You can now run the app with: python app.py")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        print("   Common issues:")
        print("   - Invalid API key")
        print("   - API quota exceeded")
        print("   - Network connection problem")
        print("   - Model not responding correctly")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
