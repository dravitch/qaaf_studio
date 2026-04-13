"""
MIF v4.0 - Fancy Progress Module
=================================

Beautiful terminal output for certification tests.
Zero dependency, pure ANSI escape codes.

Features:
    - Color-coded test results
    - Progress bars
    - Visual verdicts
    - Minimal code footprint (< 100 lines)

Usage:
    from tools.fancy_progress import FancyProgress
    
    fp = FancyProgress()
    fp.test_result("Variance", 0.156, threshold=0.05, passed=True)
    fp.verdict(passed=True, message="Phase 0 CERTIFIED")

Author: QAAF Metrics Team
License: MIT
"""

import sys
from typing import Optional


class FancyProgress:
    """
    Fancy terminal output with colors and progress bars.
    
    Zero external dependencies, pure ANSI codes.
    """
    
    # ANSI Color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    # Unicode symbols
    CHECK = "✅"
    CROSS = "❌"
    WARN = "⚠️"
    INFO = "ℹ️"
    ARROW = "→"
    BAR_FULL = "█"
    BAR_EMPTY = "░"
    
    # Progress bar settings
    BAR_LENGTH = 30
    
    def __init__(self, use_color: bool = True):
        """
        Initialize fancy progress.
        
        Args:
            use_color: Enable color output (default: True)
        """
        self.use_color = use_color and sys.stdout.isatty()
    
    def _colorize(self, text: str, color: str) -> str:
        """Colorize text if colors enabled."""
        if self.use_color:
            return f"{color}{text}{self.RESET}"
        return text
    
    def _progress_bar(self, value: float, width: int = None) -> str:
        """
        Generate progress bar.
        
        Args:
            value: Value between 0.0 and 1.0
            width: Bar width (default: BAR_LENGTH)
        
        Returns:
            str: Progress bar string
        """
        if width is None:
            width = self.BAR_LENGTH
        
        filled = int(round(value * width))
        return self.BAR_FULL * filled + self.BAR_EMPTY * (width - filled)
    
    def header(self, title: str):
        """Print section header."""
        line = "=" * 60
        print(f"\n{self._colorize(line, self.CYAN)}")
        print(f"{self._colorize(title.center(60), self.BOLD + self.CYAN)}")
        print(f"{self._colorize(line, self.CYAN)}\n")
    
    def subheader(self, title: str):
        """Print subsection header."""
        print(f"\n{self._colorize(title, self.BOLD + self.BLUE)}")
        print(self._colorize("-" * len(title), self.BLUE))
    
    def test_result(self, 
                   name: str, 
                   value: float, 
                   threshold: float,
                   passed: bool,
                   invert: bool = False,
                   unit: str = "",
                   bar: bool = True):
        """
        Print test result with color and optional progress bar.
        
        Args:
            name: Test name
            value: Test value
            threshold: Threshold for pass/fail
            passed: Whether test passed
            invert: Lower is better (default: False, higher is better)
            unit: Unit string (e.g., "%", "x")
            bar: Show progress bar (default: True)
        """
        # Format value
        value_str = f"{value:.4f}{unit}"
        threshold_str = f"{threshold:.4f}{unit}"
        
        # Status symbol
        symbol = self.CHECK if passed else self.CROSS
        
        # Color
        color = self.GREEN if passed else self.RED
        
        # Progress bar (normalize to 0-1 range)
        if bar and not invert:
            # Higher is better
            normalized = min(1.0, value / (threshold * 2))  # Scale to 2x threshold
            bar_str = self._colorize(self._progress_bar(normalized), color)
        elif bar and invert:
            # Lower is better
            normalized = 1.0 - min(1.0, value / (threshold * 2))
            bar_str = self._colorize(self._progress_bar(normalized), color)
        else:
            bar_str = ""
        
        # Comparison
        compare = "<" if invert else ">"
        comp_str = f"({compare} {threshold_str})"
        
        # Print
        if bar:
            print(f"{symbol} {name:20s} : {bar_str} {self._colorize(value_str, color):>12s} {comp_str}")
        else:
            print(f"{symbol} {name:20s} : {self._colorize(value_str, color):>12s} {comp_str}")
    
    def info(self, message: str):
        """Print info message."""
        print(f"{self.INFO}  {message}")
    
    def warning(self, message: str):
        """Print warning message."""
        print(f"{self.WARN}  {self._colorize(message, self.YELLOW)}")
    
    def error(self, message: str):
        """Print error message."""
        print(f"{self.CROSS} {self._colorize(message, self.RED)}")
    
    def success(self, message: str):
        """Print success message."""
        print(f"{self.CHECK} {self._colorize(message, self.GREEN)}")
    
    def verdict(self, passed: bool, message: str, confidence: Optional[float] = None):
        """
        Print final verdict with box.
        
        Args:
            passed: Whether certified
            message: Verdict message
            confidence: Confidence score 0-1 (optional)
        """
        symbol = self.CHECK if passed else self.CROSS
        color = self.GREEN if passed else self.RED
        
        print("\n" + self._colorize("="*60, color))
        
        verdict_text = f"{symbol} {message}"
        print(self._colorize(verdict_text.center(60), self.BOLD + color))
        
        if confidence is not None:
            conf_text = f"Confidence: {confidence:.0%}"
            print(self._colorize(conf_text.center(60), color))
        
        print(self._colorize("="*60, color) + "\n")
    
    def phase_summary(self, phase: str, tests_passed: int, tests_total: int):
        """
        Print phase summary.
        
        Args:
            phase: Phase name (e.g., "Phase 0")
            tests_passed: Number of tests passed
            tests_total: Total number of tests
        """
        pass_rate = tests_passed / tests_total if tests_total > 0 else 0
        passed = tests_passed == tests_total
        
        color = self.GREEN if passed else self.YELLOW if pass_rate >= 0.75 else self.RED
        symbol = self.CHECK if passed else self.WARN if pass_rate >= 0.75 else self.CROSS
        
        print(f"\n{symbol} {self._colorize(phase, self.BOLD)}: {tests_passed}/{tests_total} tests passed")
        
        # Progress bar
        bar = self._progress_bar(pass_rate)
        print(f"   {self._colorize(bar, color)} {pass_rate:.0%}")
    
    def degradation_gauge(self, degradation: float, threshold: float = 0.40):
        """
        Show degradation gauge (Phase 1 specific).
        
        Args:
            degradation: Degradation percentage (0.0 - 1.0)
            threshold: Pass threshold (default: 0.40 = 40%)
        """
        passed = degradation < threshold
        color = self.GREEN if passed else self.RED
        
        # Inverse bar (lower is better)
        normalized = 1.0 - min(1.0, degradation / threshold)
        bar = self._progress_bar(normalized)
        
        print(f"\n📊 Degradation: {self._colorize(bar, color)} {degradation:.1%}")
        print(f"   Threshold: {threshold:.0%} {'✅ PASS' if passed else '❌ FAIL'}")
    
    def pair_result(self, pair_name: str, mean_ratio: float, passed: bool):
        """
        Show pair test result (Phase 2 specific).
        
        Args:
            pair_name: Name of pair (e.g., "BTC/PAXG")
            mean_ratio: Mean volatility ratio
            passed: Whether test passed
        """
        symbol = self.CHECK if passed else self.CROSS
        color = self.GREEN if passed else self.RED
        
        print(f"{symbol} {pair_name:15s} : {self._colorize(f'{mean_ratio:.2f}x', color)}")


# ========== CONVENIENCE FUNCTIONS ==========

def test_result(name: str, value: float, threshold: float, 
                passed: bool, **kwargs):
    """Quick test result without creating instance."""
    fp = FancyProgress()
    fp.test_result(name, value, threshold, passed, **kwargs)


def verdict(passed: bool, message: str, confidence: Optional[float] = None):
    """Quick verdict without creating instance."""
    fp = FancyProgress()
    fp.verdict(passed, message, confidence)


def header(title: str):
    """Quick header without creating instance."""
    fp = FancyProgress()
    fp.header(title)


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Demo
    fp = FancyProgress()
    
    fp.header("MIF v4.0 Certification Demo")
    
    fp.subheader("Phase 0: Isolation Tests")
    fp.test_result("Variance", 0.156, 0.05, True, unit="")
    fp.test_result("Discrimination", 0.087, 0.05, True, unit="")
    fp.test_result("R² Forward", 0.025, 0.01, True, unit="")
    fp.test_result("Orthogonality", 0.314, 0.5, True, unit="")
    fp.test_result("Lookahead", 0.02, 0.05, True, invert=True, unit="%")
    fp.test_result("Persistence", 0.92, 0.3, True, unit="")
    
    fp.phase_summary("Phase 0", 6, 6)
    
    fp.subheader("Phase 1: OOS Generalization")
    fp.degradation_gauge(0.066, 0.40)
    
    fp.subheader("Phase 2: Multi-Asset Transfer")
    fp.pair_result("BTC/PAXG", 2.1, True)
    fp.pair_result("SPY/TLT", 4.5, True)
    fp.pair_result("SPY/GLD", 0.9, True)
    fp.pair_result("QQQ/IEF", 3.4, True)
    
    fp.verdict(True, "CERTIFIED", confidence=0.87)
    
    fp.info("Certification completed successfully")
    fp.warning("Remember to enable Phase 3 monitoring")
    fp.success("Ready for production deployment")
