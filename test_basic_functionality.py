"""
Simple test script to verify AEGIS structure
"""

def test_basic_functionality():
    """Test basic AEGIS functionality"""
    print("Testing basic AEGIS functionality...")
    
    # Test importing core components
    try:
        from aegis import (
            initialize_aegis,
            get_system_status,
            get_supported_risk_categories,
            list_available_attacks
        )
        print("[PASS] Core imports successful")
    except Exception as e:
        print(f"[FAIL] Core imports failed: {e}")
        return False
    
    # Test system initialization
    try:
        status = initialize_aegis()
        print(f"[PASS] System initialization: {status['status']}")
    except Exception as e:
        print(f"[FAIL] System initialization failed: {e}")
        return False
    
    # Test getting system status
    try:
        status = get_system_status()
        print(f"[PASS] System status check: Initialized = {status['initialized']}")
    except Exception as e:
        print(f"[FAIL] System status check failed: {e}")
        return False
    
    # Test getting risk categories
    try:
        categories = get_supported_risk_categories()
        print(f"[PASS] Risk categories: {len(categories)} categories available")
    except Exception as e:
        print(f"[FAIL] Risk categories failed: {e}")
        return False
    
    # Test getting attack vectors
    try:
        attacks = list_available_attacks()
        print(f"[PASS] Attack vectors: {len(attacks)} vectors available")
    except Exception as e:
        print(f"[FAIL] Attack vectors failed: {e}")
        return False
    
    print("[SUCCESS] All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    test_basic_functionality()