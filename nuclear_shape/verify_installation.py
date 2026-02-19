#!/usr/bin/env python3
"""
Verification script to check that nuclear_shape package is installed correctly.

Run this after installation to verify everything works:
    python -m nuclear_shape.verify_installation
    # or if installed:
    verify-installation
"""

import sys
from pathlib import Path


def print_status(message, status="INFO"):
    """Print a status message."""
    symbols = {
        "INFO": "ℹ",
        "PASS": "✓",
        "FAIL": "✗",
        "WARN": "⚠"
    }
    symbol = symbols.get(status, "•")
    print(f"{symbol} {message}")


def test_imports():
    """Test that all imports work."""
    print_status("Testing imports...", "INFO")
    
    try:
        import nuclear_shape
        print_status("  ✓ nuclear_shape package imported", "PASS")
    except ImportError as e:
        print_status(f"  ✗ Failed to import nuclear_shape: {e}", "FAIL")
        return False
    
    try:
        from nuclear_shape import nuclear_shape as ns_class
        print_status("  ✓ nuclear_shape class imported", "PASS")
    except ImportError as e:
        print_status(f"  ✗ Failed to import nuclear_shape class: {e}", "FAIL")
        return False
    
    try:
        from nuclear_shape import plotting_results
        print_status("  ✓ plotting_results module imported", "PASS")
    except ImportError as e:
        print_status(f"  ✗ Failed to import plotting_results: {e}", "FAIL")
        return False
    
    # Test dependencies
    dependencies = [
        "numpy", "scipy", "sklearn", "tqdm",
        "pandas", "seaborn", "matplotlib", "jinja2", "trimesh", "cvxpy"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print_status(f"  ✓ {dep} available", "PASS")
        except ImportError:
            print_status(f"  ✗ {dep} not available", "FAIL")
            return False
    
    return True


def test_basic_functionality():
    """Test basic functionality with a simple example."""
    print_status("\nTesting basic functionality...", "INFO")
    
    try:
        from nuclear_shape import nuclear_shape
        import numpy as np
        import tempfile
        import xml.etree.ElementTree as ET
        
        # Create a simple test .cmm file
        np.random.seed(42)
        num_points = 50
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        x = 5 * np.sin(phi) * np.cos(theta)
        y = 3 * np.sin(phi) * np.sin(theta)
        z = 2 * np.cos(phi)
        
        root = ET.Element("marker_set")
        for i in range(num_points):
            marker = ET.SubElement(root, "marker")
            marker.set("id", str(i))
            marker.set("x", str(x[i]))
            marker.set("y", str(y[i]))
            marker.set("z", str(z[i]))
        
        fd, temp_path = tempfile.mkstemp(suffix='.cmm')
        tree = ET.ElementTree(root)
        tree.write(temp_path)
        
        # Test initialization
        shape = nuclear_shape(temp_path)
        print_status("  ✓ nuclear_shape initialized", "PASS")
        
        # Test ellipsoid fit
        shape.ellipsoid_fit()
        assert "ellipsoid" in shape.results
        print_status("  ✓ ellipsoid_fit() works", "PASS")
        
        # Test PCA
        shape.principal_components()
        assert "PCA" in shape.results
        print_status("  ✓ principal_components() works", "PASS")
        
        # Cleanup
        Path(temp_path).unlink()
        
        return True
        
    except Exception as e:
        print_status(f"  ✗ Basic functionality test failed: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


def test_cli_availability():
    """Test that CLI commands are available."""
    print_status("\nTesting CLI availability...", "INFO")
    
    import subprocess
    
    # Test if entry points are available
    commands = ["verify-installation"]
    available = []
    
    for cmd in commands:
        try:
            result = subprocess.run(
                [cmd, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode in [0, 2]:
                print_status(f"  ✓ {cmd} command available", "PASS")
                available.append(cmd)
            else:
                print_status(f"  ⚠ {cmd} command exists but returned error", "WARN")
        except FileNotFoundError:
            print_status(f"  ⚠ {cmd} command not found (may need to install package)", "WARN")
        except Exception as e:
            print_status(f"  ⚠ {cmd} check failed: {e}", "WARN")
    
    if not available:
        print_status("  ⚠ CLI command not available (reinstall package?)", "WARN")
    
    return True  # Don't fail if CLI not available


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("nuclear_shape Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("CLI Availability", test_cli_availability()))
    
    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print_status(f"{name}: {status}", status)
    
    print()
    if passed == total:
        print_status("All critical tests passed! ✓", "PASS")
        return 0
    else:
        print_status(f"{passed}/{total} test suites passed", "WARN")
        return 1


if __name__ == "__main__":
    sys.exit(main())
