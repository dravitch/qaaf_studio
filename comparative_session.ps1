$base = "qaaf_studio"

$items = @(
    "sessions/comparative_001/__init__.py",
    "sessions/comparative_001/signals.py",
    "sessions/comparative_001/run_comparative.py",
    "sessions/comparative_001/test_comparative_signals.py",
    "sessions/comparative_001/results/checkpoints/",
    "sessions/h9_ema60j/certify_h9_ema60j.py",
    "sessions/h9_ema60j/checkpoints/",
    "sessions/h9_ema60j/kb_h9_ema60j.yaml",
    "sessions/session_template.py",
    "tests/conftest.py",
    "tests/test_layer1_metrics.py",
    "tests/test_layer1_backtester.py",
    "tests/test_layer2_mif.py",
    "tests/test_layer2_paf.py",
    "tests/test_layer3_metis.py",
    "tests/test_layer4_dsig.py"
)

foreach ($item in $items) {
    $full = Join-Path $base $item

    if ($item.EndsWith("/")) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
    } else {
        $dir = Split-Path $full
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        New-Item -ItemType File -Path $full -Force | Out-Null
    }
}

Write-Host "Structure créée."
