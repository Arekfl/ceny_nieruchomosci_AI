#!/usr/bin/env python3
"""
Test script for the API
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

# ANSI colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def test_health_check():
    """Test health check endpoint"""
    print(f"\n{BLUE}Testing health check...{RESET}")
    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        print(f"{GREEN}✓ Health check passed{RESET}")
        return True
    except Exception as e:
        print(f"{RED}✗ Health check failed: {e}{RESET}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print(f"\n{BLUE}Testing model info endpoint...{RESET}")
    try:
        response = requests.get(f"{BASE_URL}/info")
        assert response.status_code == 200
        data = response.json()
        print(f"{GREEN}✓ Model info retrieved:{RESET}")
        print(f"  - Algorithm: {data['algorithm']}")
        print(f"  - Test R² Score: {data['test_r2_score']:.4f}")
        print(f"  - Test RMSE: {data['test_rmse']:,.2f} PLN")
        return True
    except Exception as e:
        print(f"{RED}✗ Model info test failed: {e}{RESET}")
        return False


def test_prediction(payload: Dict[str, Any]):
    """Test prediction endpoint"""
    print(f"\n{BLUE}Testing prediction with payload:{RESET}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"{RED}✗ Prediction failed with status {response.status_code}{RESET}")
            print(f"Error: {response.json()}")
            return False
        
        result = response.json()
        print(f"{GREEN}✓ Prediction successful:{RESET}")
        print(f"  - Predicted Price: {result['predicted_price']:,.2f} {result['currency']}")
        print(f"  - Confidence: {result['confidence']}")
        return True
    except Exception as e:
        print(f"{RED}✗ Prediction test failed: {e}{RESET}")
        return False


def test_filter():
    """Test filter endpoint"""
    print(f"\n{BLUE}Testing filter endpoint (voivodeship: mazowieckie)...{RESET}")
    try:
        response = requests.get(
            f"{BASE_URL}/filter",
            params={"voivodeship": "mazowieckie"}
        )
        
        if response.status_code != 200:
            print(f"{RED}✗ Filter failed with status {response.status_code}{RESET}")
            return False
        
        result = response.json()
        print(f"{GREEN}✓ Filter successful:{RESET}")
        print(f"  - Properties found: {result['count']}")
        print(f"  - Average price: {result['avg_price']:,.2f} PLN")
        print(f"  - Price range: {result['min_price']:,.2f} - {result['max_price']:,.2f} PLN")
        return True
    except Exception as e:
        print(f"{RED}✗ Filter test failed: {e}{RESET}")
        return False


def run_all_tests():
    """Run all tests"""
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}Starting API Tests{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    # Wait for server to be ready
    print(f"\n{YELLOW}Waiting for server to be ready...{RESET}")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"{GREEN}✓ Server is ready{RESET}")
                break
        except:
            if i < max_retries - 1:
                print(f"Waiting... ({i+1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"{RED}✗ Server did not respond within timeout{RESET}")
                return False
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Model Info", test_model_info()))
    
    # Test multiple predictions
    test_cases = [
        {
            "name": "Typical apartment (Warsaw, 3 rooms, 85 m²)",
            "payload": {
                "area": 85.0,
                "rooms": 3,
                "year_constructed": 2015,
                "heating": "gazowe",
                "building_material": "cegła",
                "building_type": "blok",
                "market": "wtórny",
                "voivodeship": "mazowieckie"
            }
        },
        {
            "name": "Modern house (Kraków, 4 rooms, 150 m²)",
            "payload": {
                "area": 150.0,
                "rooms": 4,
                "year_constructed": 2020,
                "heating": "pompa ciepła",
                "building_material": "pustak",
                "building_type": "dom wolnostojący",
                "market": "pierwotny",
                "voivodeship": "małopolskie"
            }
        },
        {
            "name": "Townhouse (Wrocław, 5 rooms, 130 m²)",
            "payload": {
                "area": 130.0,
                "rooms": 5,
                "year_constructed": 2010,
                "heating": "węglowe",
                "building_material": "cegła",
                "building_type": "szeregowiec",
                "market": "wtórny",
                "voivodeship": "dolnośląskie"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{YELLOW}Test: {test_case['name']}{RESET}")
        results.append((test_case['name'], test_prediction(test_case['payload'])))
    
    # Test filter
    results.append(("Filter Endpoint", test_filter()))
    
    # Summary
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}Test Summary{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"{name}: {status}")
    
    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
