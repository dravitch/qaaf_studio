#!/usr/bin/env python3
"""
MIF v4.0 Metric Certification Tool (FIXED)
===========================================

Fixes:
1. Confidence calculation now uses actual test results
2. Phase 2 parsing improved (counts pairs passed)
3. Degradation extraction from Phase 1
4. Better error handling

Author: QAAF Metrics Team
License: MIT
"""

import argparse
import sys
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import re


class MetricCertifier:
    """
    MIF v4.0 Certification Pipeline
    """
    
    def __init__(self, metric_name: str, verbose: bool = False):
        self.metric_name = metric_name
        self.verbose = verbose
        self.metric_dir = Path(f"metrics/{metric_name}/v1_0")
        self.results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log message."""
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️"
        }.get(level, "")
        
        print(f"{prefix} {message}")
    
    def check_prerequisites(self) -> bool:
        """Check if metric directory and files exist."""
        self.log(f"Checking prerequisites for {self.metric_name}...")
        
        if not self.metric_dir.exists():
            self.log(f"Metric directory not found: {self.metric_dir}", "ERROR")
            return False
        
        impl_file = self.metric_dir / "implementation.py"
        if not impl_file.exists():
            self.log(f"implementation.py not found", "ERROR")
            return False
        
        tests_dir = self.metric_dir / "tests"
        if not tests_dir.exists():
            self.log(f"tests/ directory not found", "ERROR")
            return False
        
        self.log("Prerequisites OK", "SUCCESS")
        return True
    
    def classify_metric(self) -> str:
        """Auto-detect metric domain."""
        self.log("Classifying metric domain...")
        
        impl_file = self.metric_dir / "implementation.py"
        code = impl_file.read_text()
        
        # Simple keyword-based classification
        if any(kw in code.lower() for kw in ["volatility", "std(", "var(", "drawdown", "variance"]):
            domain = "risk"
        elif any(kw in code.lower() for kw in ["return", "sharpe", "profit", "alpha"]):
            domain = "performance"
        elif any(kw in code.lower() for kw in ["regime", "hmm", "cluster", "markov"]):
            domain = "regime"
        elif any(kw in code.lower() for kw in ["autocorr", "entropy", "stability", "coherence", "bound"]):
            domain = "stability"
        else:
            domain = "unknown"
        
        self.log(f"Domain detected: {domain}", "SUCCESS")
        self.results["classification"] = {"domain": domain}
        return domain
    
    def run_phase0(self) -> Dict[str, Any]:
        """Run Phase 0: Isolation tests."""
        self.log("Running Phase 0: Isolation tests...")
        
        test_file = self.metric_dir / "tests" / "test_phase0.py"
        
        try:
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.metric_dir.parent.parent.parent
            )
            
            # Parse output more carefully
            output = result.stdout
            
            # Count PASSED and FAILED
            passed_count = output.count("PASSED")
            failed_count = output.count("FAILED")
            skipped_count = output.count("SKIPPED")
            
            # Total tests (excluding summary test)
            total_tests = passed_count + failed_count + skipped_count
            
            # Adjust for summary test (if present)
            if "test_summary" in output and "PASSED" in output:
                total_tests = max(6, total_tests - 1)  # Phase 0 has 6 core tests
                passed_count = max(0, passed_count - 1)
            
            success = failed_count == 0 and passed_count > 0
            
            phase0_result = {
                "status": "PASSED" if success else "FAILED",
                "tests_passed": passed_count,
                "tests_total": total_tests,
                "tests_skipped": skipped_count,
                "output": output if self.verbose else ""
            }
            
            if success:
                self.log(f"Phase 0: {passed_count}/{total_tests} tests passed ({skipped_count} skipped)", "SUCCESS")
            else:
                self.log(f"Phase 0: {passed_count}/{total_tests} tests passed", "ERROR")
            
            self.results["phase_0"] = phase0_result
            return phase0_result
            
        except Exception as e:
            self.log(f"Phase 0 failed: {e}", "ERROR")
            self.results["phase_0"] = {"status": "ERROR", "error": str(e)}
            return {"status": "ERROR"}
    
    def run_phase1(self) -> Dict[str, Any]:
        """Run Phase 1: OOS Generalization."""
        self.log("Running Phase 1: OOS Generalization...")
        
        test_file = self.metric_dir / "tests" / "test_phase1.py"
        
        try:
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.metric_dir.parent.parent.parent
            )
            
            output = result.stdout
            passed = result.returncode == 0
            
            # Extract degradation from output
            degradation = None
            for line in output.split('\n'):
                # Look for patterns like "8.9%" or "degradation: 0.089"
                match = re.search(r'(?:degradation|Degradation).*?(\d+\.?\d*)%', line, re.IGNORECASE)
                if match:
                    degradation = float(match.group(1))
                    break
                
                # Alternative: look for decimal format
                match = re.search(r'(?:degradation|Degradation).*?(\d\.\d+)', line, re.IGNORECASE)
                if match:
                    degradation = float(match.group(1)) * 100  # Convert to percentage
                    break
            
            phase1_result = {
                "status": "PASSED" if passed else "FAILED",
                "degradation_pct": degradation,
                "output": output if self.verbose else ""
            }
            
            if passed:
                deg_str = f"(degradation: {degradation}%)" if degradation else ""
                self.log(f"Phase 1: PASSED {deg_str}", "SUCCESS")
            else:
                self.log(f"Phase 1: FAILED", "ERROR")
            
            self.results["phase_1"] = phase1_result
            return phase1_result
            
        except Exception as e:
            self.log(f"Phase 1 failed: {e}", "ERROR")
            self.results["phase_1"] = {"status": "ERROR", "error": str(e)}
            return {"status": "ERROR"}
    
    def run_phase2(self) -> Dict[str, Any]:
        """Run Phase 2: Multi-Asset Transfer."""
        self.log("Running Phase 2: Multi-Asset Transfer...")
        self.log("Note: This may take 1-2 minutes (downloading market data)")
        
        test_file = self.metric_dir / "tests" / "test_phase2.py"
        
        try:
            result = subprocess.run(
                ["pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.metric_dir.parent.parent.parent,
                timeout=300
            )
            
            output = result.stdout
            passed = result.returncode == 0
            
            # Count pairs passed - look for "PASSED" in test_pair_* tests
            pair_tests = re.findall(r'test_pair_\w+\s+(PASSED|FAILED|SKIPPED)', output)
            pairs_passed = sum(1 for status in pair_tests if status == "PASSED")
            pairs_total = len(pair_tests)
            
            # If no individual tests found, look for summary
            if pairs_total == 0:
                # Look for patterns like "4/4 pairs" or "3/4 pairs passed"
                match = re.search(r'(\d+)/(\d+)\s+pairs', output)
                if match:
                    pairs_passed = int(match.group(1))
                    pairs_total = int(match.group(2))
                else:
                    pairs_total = 4  # Standard
                    pairs_passed = 4 if passed else 0
            
            phase2_result = {
                "status": "PASSED" if passed else "FAILED",
                "pairs_passed": pairs_passed,
                "pairs_total": pairs_total,
                "output": output if self.verbose else ""
            }
            
            if passed:
                self.log(f"Phase 2: {pairs_passed}/{pairs_total} pairs passed", "SUCCESS")
            else:
                self.log(f"Phase 2: {pairs_passed}/{pairs_total} pairs passed", "ERROR")
            
            self.results["phase_2"] = phase2_result
            return phase2_result
            
        except subprocess.TimeoutExpired:
            self.log("Phase 2 timeout (>5 min)", "ERROR")
            self.results["phase_2"] = {"status": "TIMEOUT", "pairs_passed": 0, "pairs_total": 4}
            return {"status": "TIMEOUT"}
            
        except Exception as e:
            self.log(f"Phase 2 failed: {e}", "ERROR")
            self.results["phase_2"] = {"status": "ERROR", "error": str(e), "pairs_passed": 0, "pairs_total": 4}
            return {"status": "ERROR"}
    
    def generate_certification_yaml(self) -> bool:
        """Generate or update certification.yaml."""
        self.log("Generating certification.yaml...")
        
        cert_file = self.metric_dir / "certification.yaml"
        
        # Load existing or create new
        if cert_file.exists():
            with open(cert_file, 'r') as f:
                cert_data = yaml.safe_load(f) or {}
        else:
            cert_data = {}
        
        # Update with results
        cert_data["certification"] = {
            "overall_status": self._determine_overall_status(),
            "certification_date": datetime.now().isoformat(),
            "certified_by": "MIF v4.0 Pipeline",
            "confidence": self._calculate_confidence(),
            "scores": {
                "phase_0": self.results.get("phase_0", {}).get("status", "UNKNOWN"),
                "phase_1": self.results.get("phase_1", {}).get("status", "UNKNOWN"),
                "phase_2": self.results.get("phase_2", {}).get("status", "UNKNOWN")
            }
        }
        
        # Write back
        try:
            with open(cert_file, 'w') as f:
                yaml.dump(cert_data, f, default_flow_style=False, sort_keys=False)
            
            self.log(f"certification.yaml saved", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to save certification.yaml: {e}", "ERROR")
            return False
    
    def _determine_overall_status(self) -> str:
        """Determine overall certification status."""
        phase0 = self.results.get("phase_0", {}).get("status", "")
        phase1 = self.results.get("phase_1", {}).get("status", "")
        phase2 = self.results.get("phase_2", {}).get("status", "")
        
        if all(s == "PASSED" for s in [phase0, phase1, phase2]):
            return "✅ CERTIFIED"
        elif phase0 == "PASSED" and phase1 == "PASSED":
            return "⚠️ CONDITIONAL (Phase 2 incomplete)"
        else:
            return "❌ NOT CERTIFIED"
    
    def _calculate_confidence(self) -> float:
        """
        Calculate certification confidence score.
        
        FIXED: Now uses actual test results, not just boolean pass/fail.
        """
        score = 0.0
        
        # Phase 0: Weight by test pass rate (30% max)
        phase0 = self.results.get("phase_0", {})
        if phase0.get("status") == "PASSED":
            tests_passed = phase0.get("tests_passed", 0)
            tests_total = phase0.get("tests_total", 6)
            # Exclude skipped from total
            tests_skipped = phase0.get("tests_skipped", 0)
            tests_total_effective = tests_total - tests_skipped
            
            if tests_total_effective > 0:
                pass_rate = tests_passed / tests_total_effective
                score += 0.30 * pass_rate
        
        # Phase 1: Weight by degradation quality (40% max)
        phase1 = self.results.get("phase_1", {})
        if phase1.get("status") == "PASSED":
            degradation = phase1.get("degradation_pct")
            if degradation is not None:
                # Score based on how far below 40% threshold
                # degradation = 0% → score = 0.40
                # degradation = 40% → score = 0.0
                degradation_score = max(0, 1 - (degradation / 40))
                score += 0.40 * degradation_score
            else:
                # If no degradation found, assume decent (0.30 out of 0.40)
                score += 0.30
        
        # Phase 2: Weight by pairs pass rate (30% max)
        phase2 = self.results.get("phase_2", {})
        if phase2.get("status") in ["PASSED", "FAILED"]:
            pairs_passed = phase2.get("pairs_passed", 0)
            pairs_total = phase2.get("pairs_total", 4)
            if pairs_total > 0:
                pass_rate = pairs_passed / pairs_total
                score += 0.30 * pass_rate
        
        return round(score, 2)
    
    def update_registry(self) -> bool:
        """Update metrics_registry.json."""
        self.log("Updating metrics_registry.json...")
        
        registry_file = Path("metrics") / "metrics_registry.json"
        
        # Load existing
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"metrics": []}
        
        # Update or add metric
        metric_entry = {
            "name": self.metric_name,
            "version": "1.0",
            "domain": self.results.get("classification", {}).get("domain", "unknown"),
            "status": self._determine_overall_status(),
            "certification_date": datetime.now().isoformat(),
            "confidence": self._calculate_confidence()
        }
        
        # Find and update or append
        updated = False
        for i, m in enumerate(registry["metrics"]):
            if m["name"] == self.metric_name:
                registry["metrics"][i] = metric_entry
                updated = True
                break
        
        if not updated:
            registry["metrics"].append(metric_entry)
        
        # Save
        try:
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            
            self.log("metrics_registry.json updated", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to update registry: {e}", "ERROR")
            return False
    
    def generate_report(self):
        """Print certification report."""
        print("\n" + "="*60)
        print(f"CERTIFICATION REPORT: {self.metric_name}")
        print("="*60)
        
        # Classification
        domain = self.results.get("classification", {}).get("domain", "unknown")
        print(f"\n📊 Domain: {domain}")
        
        # Phase 0
        phase0 = self.results.get("phase_0", {})
        if phase0.get("status") == "PASSED":
            passed = phase0.get('tests_passed', 0)
            total = phase0.get('tests_total', 0)
            skipped = phase0.get('tests_skipped', 0)
            print(f"\n✅ Phase 0 (Isolation): {passed}/{total} tests ({skipped} skipped)")
        else:
            print(f"\n❌ Phase 0 (Isolation): FAILED")
        
        # Phase 1
        phase1 = self.results.get("phase_1", {})
        if phase1.get("status") == "PASSED":
            deg = phase1.get("degradation_pct")
            deg_str = f" ({deg}% degradation)" if deg else ""
            print(f"✅ Phase 1 (OOS): PASSED{deg_str}")
        else:
            print(f"❌ Phase 1 (OOS): FAILED")
        
        # Phase 2
        phase2 = self.results.get("phase_2", {})
        if phase2.get("status") == "PASSED":
            pairs_passed = phase2.get('pairs_passed', 0)
            pairs_total = phase2.get('pairs_total', 4)
            print(f"✅ Phase 2 (Multi-Asset): {pairs_passed}/{pairs_total} pairs")
        elif phase2.get("status") == "TIMEOUT":
            print(f"⏱️  Phase 2 (Multi-Asset): TIMEOUT")
        else:
            pairs_passed = phase2.get('pairs_passed', 0)
            pairs_total = phase2.get('pairs_total', 4)
            print(f"❌ Phase 2 (Multi-Asset): {pairs_passed}/{pairs_total} pairs")
        
        # Overall
        overall = self._determine_overall_status()
        confidence = self._calculate_confidence()
        
        print(f"\n{'='*60}")
        print(f"Overall Status: {overall}")
        print(f"Confidence: {confidence:.0%}")
        print("="*60)
        
        # Next steps
        if "✅" in overall:
            print("\n🎉 CERTIFICATION COMPLETE!")
            print("\nNext steps:")
            print("  1. Review certification.yaml")
            print("  2. Integrate metric into strategy")
            print("  3. Enable Phase 3 monitoring")
        elif "⚠️" in overall:
            print("\n⚠️  CONDITIONAL CERTIFICATION")
            print("\nNext steps:")
            print("  1. Complete Phase 2 (multi-asset)")
            print("  2. Or document limitations for specific pairs")
        else:
            print("\n❌ CERTIFICATION FAILED")
            print("\nNext steps:")
            print("  1. Review test failures")
            print("  2. Fix issues")
            print("  3. Re-run certification")
    
    def certify(self, skip_phase2: bool = False) -> bool:
        """Run complete certification pipeline."""
        print("\n" + "="*60)
        print(f"MIF v4.0 CERTIFICATION PIPELINE")
        print(f"Metric: {self.metric_name}")
        print("="*60)
        
        # Prerequisites
        if not self.check_prerequisites():
            return False
        
        # Classify
        self.classify_metric()
        
        # Phase 0
        phase0_result = self.run_phase0()
        if phase0_result.get("status") != "PASSED":
            self.log("Phase 0 failed - stopping certification", "ERROR")
            self.generate_report()
            return False
        
        # Phase 1
        phase1_result = self.run_phase1()
        if phase1_result.get("status") != "PASSED":
            self.log("Phase 1 failed - stopping certification", "ERROR")
            self.generate_report()
            return False
        
        # Phase 2 (optional)
        if not skip_phase2:
            phase2_result = self.run_phase2()
            if phase2_result.get("status") not in ["PASSED", "TIMEOUT"]:
                self.log("Phase 2 failed - conditional certification", "WARNING")
        else:
            self.log("Phase 2 skipped by user", "WARNING")
        
        # Generate outputs
        self.generate_certification_yaml()
        self.update_registry()
        
        # Report
        self.generate_report()
        
        overall = self._determine_overall_status()
        return "✅" in overall


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MIF v4.0 Metric Certification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full certification
  python tools/certify_metric.py vol_ratio
  
  # Skip Phase 2 (multi-asset)
  python tools/certify_metric.py vol_ratio --skip-phase2
  
  # Verbose output
  python tools/certify_metric.py vol_ratio --verbose
  
  # List all metrics
  python tools/certify_metric.py --list
        """
    )
    
    parser.add_argument(
        "metric",
        nargs="?",
        help="Metric name (e.g., vol_ratio)"
    )
    
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (multi-asset transfer) tests"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all metrics in registry"
    )
    
    args = parser.parse_args()
    
    # List metrics
    if args.list:
        registry_file = Path("metrics") / "metrics_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            print("\n📊 METRICS REGISTRY")
            print("="*60)
            for m in registry.get("metrics", []):
                status_icon = "✅" if "✅" in m.get("status", "") else "❌"
                conf = m.get("confidence", 0)
                print(f"{status_icon} {m['name']:<20} {m.get('domain', 'unknown'):<15} {conf:.0%} - {m.get('status', 'unknown')}")
            print("="*60)
        else:
            print("⚠️  No metrics registry found")
        return 0
    
    # Check metric provided
    if not args.metric:
        parser.print_help()
        return 1
    
    # Run certification
    certifier = MetricCertifier(args.metric, verbose=args.verbose)
    success = certifier.certify(skip_phase2=args.skip_phase2)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
