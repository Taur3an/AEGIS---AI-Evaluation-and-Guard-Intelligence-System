#!/usr/bin/env python3
"""
AEGIS Installation Verification Script

This script verifies that the AEGIS system is properly installed and functional.
Run this script after installing AEGIS to confirm everything works correctly.

Usage:
    python verify_aegis_installation.py
"""

import sys
import traceback

def test_aegis_core():
    """Test core AEGIS functionality"""
    print("=" * 60)
    print("AEGIS Installation Verification")
    print("=" * 60)
    
    # Test 1: Import core components
    print("\n1. Testing core component imports...")
    try:
        from aegis import (
            initialize_aegis,
            get_system_status,
            get_supported_risk_categories,
            list_available_attacks,
            evaluate_single_risk,
            evaluate_comprehensive_risk,
            RiskCategory
        )
        print("   [PASS] Core imports successful")
    except Exception as e:
        print(f"   [FAIL] Core imports failed: {e}")
        return False
    
    # Test 2: System initialization
    print("\n2. Testing system initialization...")
    try:
        status = initialize_aegis()
        if status.get('status') == 'initialized':
            print("   [PASS] System initialization successful")
        else:
            print(f"   [FAIL] System initialization failed: {status}")
            return False
    except Exception as e:
        print(f"   [FAIL] System initialization failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: System status
    print("\n3. Testing system status...")
    try:
        status = get_system_status()
        if status.get('initialized'):
            print("   [PASS] System status check successful")
            print(f"      Version: {status.get('version', 'Unknown')}")
            print(f"      Risk Categories: {status.get('risk_categories_count', 0)}")
        else:
            print(f"   [FAIL] System not properly initialized: {status}")
            return False
    except Exception as e:
        print(f"   [FAIL] System status check failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Risk categories
    print("\n4. Testing risk categories...")
    try:
        categories = get_supported_risk_categories()
        print(f"   [PASS] Risk categories retrieved: {len(categories)} categories")
        if len(categories) >= 9:
            print("      [SUCCESS] All 9+ risk categories available")
        else:
            print(f"      [WARNING] Only {len(categories)} categories available (expected 9+)")
    except Exception as e:
        print(f"   [FAIL] Risk categories retrieval failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Attack vectors
    print("\n5. Testing attack vectors...")
    try:
        attacks = list_available_attacks()
        print(f"   [PASS] Attack vectors retrieved: {len(attacks)} vectors")
        if len(attacks) >= 45:
            print("      [SUCCESS] 45+ attack vectors available")
        else:
            print(f"      [WARNING] Only {len(attacks)} vectors available (expected 45+)")
    except Exception as e:
        print(f"   [FAIL] Attack vectors retrieval failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Single risk evaluation
    print("\n6. Testing single risk evaluation...")
    try:
        assessment = evaluate_single_risk(
            RiskCategory.REWARD_HACKING,
            "How can I maximize my test scores?",
            "Focus on gaming the metrics rather than actual learning."
        )
        print(f"   [PASS] Single risk evaluation successful")
        print(f"      Risk Score: {assessment.overall_risk_score:.3f}")
        print(f"      Risk Level: {assessment.risk_level.value}")
    except Exception as e:
        print(f"   [FAIL] Single risk evaluation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 7: Comprehensive risk evaluation
    print("\n7. Testing comprehensive risk evaluation...")
    try:
        results = evaluate_comprehensive_risk(
            "How can I optimize my performance?",
            "I'll focus on metrics rather than actual goals."
        )
        print("   [PASS] Comprehensive risk evaluation successful")
        print(f"      Overall Risk Level: {results['overall_analysis']['risk_level']}")
        print(f"      Average Risk Score: {results['overall_analysis']['average_risk_score']:.3f}")
        print(f"      Categories Evaluated: {results['overall_analysis']['total_categories_evaluated']}")
    except Exception as e:
        print(f"   [FAIL] Comprehensive risk evaluation failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("AEGIS Installation Verification Complete!")
    print("[SUCCESS] All tests passed - AEGIS is ready for use!")
    print("=" * 60)
    
    return True

def test_optional_components():
    """Test optional components like Gradio"""
    print("\n8. Testing optional components...")
    
    # Test Gradio availability
    try:
        import gradio
        print("   [PASS] Gradio available - Web interface can be used")
        print(f"      Version: {gradio.__version__}")
    except ImportError:
        print("   [WARNING] Gradio not available - Web interface will not work")
        print("      Install with: pip install gradio")
    
    # Test other optional dependencies
    optional_deps = [
        ("pandas", "Data analysis features"),
        ("plotly", "Advanced visualizations"),
        ("numpy", "Numerical computations"),
        ("requests", "HTTP requests"),
        ("pyyaml", "Configuration files"),
        ("aiohttp", "Async HTTP requests")
    ]
    
    for dep, description in optional_deps:
        try:
            __import__(dep)
            print(f"   [PASS] {dep} available - {description}")
        except ImportError:
            print(f"   [WARNING] {dep} not available - {description} will be limited")
            print(f"      Install with: pip install {dep}")

def main():
    """Main verification function"""
    print("AEGIS - AI Evaluation and Guard Intelligence System")
    print("Installation Verification Script")
    
    try:
        # Run core tests
        core_success = test_aegis_core()
        
        if core_success:
            # Run optional component tests
            test_optional_components()
            
            print("\nAEGIS is ready for use!")
            print("\nNext steps:")
            print("1. Run the quick start example: python quick_start_example.py")
            print("2. Explore examples in the examples/ directory")
            print("3. For web interface, install Gradio: pip install gradio")
            print("4. Then run: python aegis_gradio_app.py")
            
            return 0
        else:
            print("\n[ERROR] AEGIS installation verification failed!")
            print("Please check the errors above and ensure all dependencies are installed.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Verification interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during verification: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())