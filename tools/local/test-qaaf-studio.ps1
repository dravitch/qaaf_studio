# tools/local/test-qaaf-studio.ps1
# QAAF Studio -- Script de synchronisation et tests complets
# Usage : .\tools\local\test-qaaf-studio.ps1 [-SkipSync] [-Fast] [-Layer <1|2|3|all>]
#
# Options :
#   -SkipSync   : ne pas faire git pull (si deja a jour)
#   -Fast       : skip tests slow (adversarial)
#   -Layer      : "1", "2", "3", ou "all" (defaut : all)

param(
    [switch]$SkipSync,
    [switch]$Fast,
    [string]$Layer = "all"
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "QAAF Studio - Tests"

# ----------------------------------------------------------------------------
# Couleurs
# ----------------------------------------------------------------------------
function Write-Header  { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Step    { param($msg) Write-Host "  >> $msg" -ForegroundColor White }
function Write-OK      { param($msg) Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn    { param($msg) Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Write-Fail    { param($msg) Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Summary {
    param($pass, $fail, $skip)
    $total = $pass + $fail + $skip
    if ($fail -eq 0) {
        Write-Host "`n  [OK] $pass/$total PASS" -ForegroundColor Green
    } else {
        Write-Host "`n  [FAIL] $fail/$total FAIL  ($pass pass, $skip skip)" -ForegroundColor Red
    }
}

$StartTime  = Get-Date
$TotalPass  = 0
$TotalFail  = 0

# ----------------------------------------------------------------------------
# 1. Detection de l'environnement
# ----------------------------------------------------------------------------
Write-Header "Environnement"

# Git
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) { Write-Fail "git non trouve -- installer Git for Windows"; exit 1 }
Write-OK "git : $(git --version)"

# uv vs pip
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    Write-OK "uv detecte -- utilisation de uv"
    $UseUv = $true
} else {
    Write-Warn "uv non trouve -- fallback sur pip"
    $UseUv = $false
}

# Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Write-Fail "python non trouve"; exit 1 }
$pyVersion = python --version 2>&1
Write-OK "python : $pyVersion"

# ----------------------------------------------------------------------------
# 2. Synchronisation Git
# ----------------------------------------------------------------------------
Write-Header "Synchronisation Git"

if ($SkipSync) {
    Write-Warn "SkipSync active -- pas de git pull"
} else {
    Write-Step "git fetch origin..."
    $fetchProc = Start-Process "git" `
        -ArgumentList "fetch","origin" `
        -Wait -PassThru -WindowStyle Hidden -ErrorAction Stop
    if ($fetchProc.ExitCode -ne 0) {
        Write-Fail "git fetch a echoue (exit $($fetchProc.ExitCode))"
        exit 1
    }

    Write-Step "git pull origin main..."
    $pullProc = Start-Process "git" `
        -ArgumentList "pull","origin","main" `
        -Wait -PassThru -WindowStyle Hidden -ErrorAction Stop
    if ($pullProc.ExitCode -ne 0) {
        Write-Fail "git pull a echoue (exit $($pullProc.ExitCode))"
        exit 1
    }

    $status = & git status --short 2>$null
    if ($status) {
        Write-Warn "Fichiers non commites :`n$status"
    } else {
        Write-OK "Repo propre et a jour"
    }

    $lastCommit = & git log -1 --pretty=format:"%h %s (%cr)" 2>$null
    Write-OK "Dernier commit : $lastCommit"
}

# ----------------------------------------------------------------------------
# 3. Environnement virtuel
# ----------------------------------------------------------------------------
Write-Header "Environnement virtuel"

$VenvPath     = ".venv"
$VenvActivate = "$VenvPath\Scripts\Activate.ps1"

if (-not (Test-Path $VenvActivate)) {
    Write-Step "Creation du venv..."
    if ($UseUv) {
        uv venv $VenvPath
    } else {
        python -m venv $VenvPath
    }
    if ($LASTEXITCODE -ne 0) { Write-Fail "Creation venv echouee"; exit 1 }
    Write-OK "venv cree dans $VenvPath"
} else {
    Write-OK "venv existant trouve dans $VenvPath"
}

Write-Step "Activation du venv..."
& $VenvActivate
Write-OK "venv active"

# ----------------------------------------------------------------------------
# 4. Installation des dependances
# ----------------------------------------------------------------------------
Write-Header "Dependances"

$RequirementsFile = "requirements.txt"
if (-not (Test-Path $RequirementsFile)) {
    Write-Fail "requirements.txt introuvable -- verifier le repertoire courant"
    exit 1
}

Write-Step "Installation requirements.txt..."
if ($UseUv) {
    uv pip install -r $RequirementsFile --quiet
} else {
    python -m pip install -r $RequirementsFile --quiet --upgrade
}
if ($LASTEXITCODE -ne 0) { Write-Fail "Installation des dependances echouee"; exit 1 }
Write-OK "Dependances installees"

Write-Step "Installation du projet (editable)..."
if (Test-Path "pyproject.toml") {
    if ($UseUv) { uv pip install -e . --quiet }
    else { python -m pip install -e . --quiet }
    Write-OK "pyproject.toml trouve -- install editable OK"
} elseif (Test-Path "setup.py") {
    if ($UseUv) { uv pip install -e . --quiet }
    else { python -m pip install -e . --quiet }
    Write-OK "setup.py trouve -- install editable OK"
} else {
    Write-Warn "Pas de pyproject.toml ni setup.py -- skip install editable"
}

Write-Step "Verification package mif-dqf (optionnel)..."
$dqfProc = Start-Process "python" `
    -ArgumentList "-c","import mif_dqf; print('OK')" `
    -Wait -PassThru -WindowStyle Hidden -ErrorAction SilentlyContinue
if ($null -ne $dqfProc -and $dqfProc.ExitCode -eq 0) {
    Write-OK "mif-dqf installe -- mode DQF complet actif"
} else {
    Write-Warn "mif-dqf non installe -- stub DQF actif (normal en developpement)"
}

# ----------------------------------------------------------------------------
# 5. Verification des imports critiques
# ----------------------------------------------------------------------------
Write-Header "Imports critiques"

$imports = @(
    # Layer 1
    "from layer1_engine import MetricsEngine, compute_cnsr",
    "from layer1_engine.backtester import Backtester",
    "from layer1_engine.benchmark_factory import BenchmarkFactory",
    # Layer 2
    "from layer2_qualification.paf.paf_runner import run_paf, load_paf_bundle",
    "from layer2_qualification.paf.paf_d1_hierarchy import run_d1",
    "from layer2_qualification.paf.paf_d2_attribution import run_d2",
    "from layer2_qualification.paf.paf_d3_source import run_d3",
    # Layer 3
    "from layer3_validation import METISRunner, METISReport",
    "from layer3_validation.metis_q1_walkforward import run_q1, Q1Result",
    "from layer3_validation.metis_q2_permutation import run_q2, Q2Result",
    "from layer3_validation.metis_q3_ema_stability import run_q3, Q3Result",
    "from layer3_validation.metis_q4_dsr import run_q4, Q4Result"
)

# IMPORTANT : appel direct "python -c $imp" (pas Start-Process) pour heriter du venv.
$importFail = 0
foreach ($imp in $imports) {
    $modName = ($imp -split " ")[1]
    try {
        $result = python -c $imp 2>&1
        $ok     = ($LASTEXITCODE -eq 0)
    } catch {
        $result = $_.Exception.Message
        $ok     = $false
    }
    if ($ok) {
        Write-OK $modName
    } else {
        Write-Fail "Import echoue : $imp"
        $lines = ($result | Out-String) -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 3
        $lines | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
        $importFail++
    }
}

if ($importFail -gt 0) {
    Write-Fail "$importFail import(s) echoue(s) -- corriger avant de lancer les tests"
    exit 1
}

# ----------------------------------------------------------------------------
# 6. Tests par couche
# ----------------------------------------------------------------------------

function Invoke-PytestSuite {
    param(
        [string]$SuiteName,
        [string]$TestPath,
        [string]$ExtraArgs     = "",
        [string]$ExpectedCount = "?"
    )

    Write-Step "$SuiteName (attendu : $ExpectedCount tests)..."

    $cmdArgs = "pytest $TestPath -v --tb=short $ExtraArgs"
    $output  = Invoke-Expression $cmdArgs 2>&1
    $exitCode = $LASTEXITCODE

    $summaryLine = $output | Where-Object { $_ -match "passed|failed|error" } | Select-Object -Last 1
    if ($summaryLine) {
        $passed     = if ($summaryLine -match "(\d+) passed")  { [int]$Matches[1] } else { 0 }
        $failed     = if ($summaryLine -match "(\d+) failed")  { [int]$Matches[1] } else { 0 }
        $errors     = if ($summaryLine -match "(\d+) error")   { [int]$Matches[1] } else { 0 }
        $skipped    = if ($summaryLine -match "(\d+) skipped") { [int]$Matches[1] } else { 0 }
        $total_fail = $failed + $errors

        $script:TotalPass += $passed
        $script:TotalFail += $total_fail

        if ($total_fail -eq 0) {
            $skipNote = if ($skipped -gt 0) { " ($skipped skip)" } else { "" }
            Write-OK "$SuiteName : $passed PASS$skipNote"
        } else {
            Write-Fail "$SuiteName : $total_fail FAIL / $passed PASS"
            $output | Where-Object { $_ -match "FAILED|ERROR|assert" } | ForEach-Object {
                Write-Host "    $_" -ForegroundColor Red
            }
        }
    } else {
        if ($exitCode -ne 0) {
            Write-Fail "$SuiteName : erreur d'execution"
            $output | Select-Object -Last 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
            $script:TotalFail++
        } else {
            Write-Warn "$SuiteName : 0 tests collectes (verifier le chemin)"
        }
    }

    return $exitCode
}

# -- Gate 0 : calibrage benchmarks -------------------------------------------
if ($Layer -eq "all" -or $Layer -eq "1") {
    Write-Header "Gate 0 -- Calibrage Benchmarks (physique du marche)"
    Write-Warn "Ces tests necessitent un acces reseau (yfinance). Skip si hors ligne."

    $gate0Exit = Invoke-PytestSuite `
        -SuiteName     "Calibrage Benchmarks" `
        -TestPath      "tests/test_benchmark_calibration.py" `
        -ExpectedCount "11"

    if ($gate0Exit -ne 0) {
        Write-Fail "GATE 0 ECHOUEE -- le moteur produit des resultats physiquement incoherents"
        Write-Fail "NE PAS lancer de session comparative tant que cette gate echoue"
        Write-Fail "Consulter KB_Benchmarks_Calibration_Avril2026.md pour le diagnostic"
    } else {
        Write-OK "GATE 0 VERTE -- benchmarks B_5050=1.343+-0.15, B_BTC=1.244+-0.15"
    }
}

# -- Layer 1 : moteur unifie -------------------------------------------------
if ($Layer -eq "all" -or $Layer -eq "1") {
    Write-Header "Layer 1 -- Moteur Unifie"

    Invoke-PytestSuite `
        -SuiteName     "Metriques (CNSR, DSR)" `
        -TestPath      "tests/test_layer1_metrics.py" `
        -ExpectedCount "10" | Out-Null

    Invoke-PytestSuite `
        -SuiteName     "Backtester + DQF Stub" `
        -TestPath      "tests/test_layer1_backtester.py" `
        -ExpectedCount "16" | Out-Null
}

# -- Layer 2 : PAF -----------------------------------------------------------
if ($Layer -eq "all" -or $Layer -eq "2") {
    Write-Header "Layer 2 -- PAF (Pair Adequacy Framework)"

    Invoke-PytestSuite `
        -SuiteName     "PAF fonctionnel (D1/D2/D3)" `
        -TestPath      "tests/test_layer2_paf.py" `
        -ExpectedCount "8" | Out-Null

    if ($Fast) {
        Write-Warn "Mode Fast -- tests adverses slow ignores"
        Invoke-PytestSuite `
            -SuiteName     "PAF adversarial (not slow)" `
            -TestPath      "tests/test_layer2_paf_adversarial.py" `
            -ExtraArgs     "-m 'not slow'" `
            -ExpectedCount "4" | Out-Null
    } else {
        Invoke-PytestSuite `
            -SuiteName     "PAF adversarial (tous)" `
            -TestPath      "tests/test_layer2_paf_adversarial.py" `
            -ExpectedCount "6" | Out-Null
    }
}

# -- Layer 3 : METIS ---------------------------------------------------------
if ($Layer -eq "all" -or $Layer -eq "3") {
    Write-Header "Layer 3 -- METIS (Walk-forward / Permutation / EMA Stability / DSR)"

    Invoke-PytestSuite `
        -SuiteName     "METIS Q1-Q4" `
        -TestPath      "tests/test_layer3_metis.py" `
        -ExpectedCount "11" | Out-Null
}

# -- Sessions ----------------------------------------------------------------
if ($Layer -eq "all") {
    Write-Header "Sessions -- Tests signaux"

    if (Test-Path "sessions/comparative_001/test_comparative_signals.py") {
        Invoke-PytestSuite `
            -SuiteName     "Signaux comparative_001" `
            -TestPath      "sessions/comparative_001/test_comparative_signals.py" `
            -ExpectedCount "3" | Out-Null
    } else {
        Write-Warn "Pas de tests pour comparative_001 (normal si session non initialisee)"
    }
}

# ----------------------------------------------------------------------------
# 7. Resume final
# ----------------------------------------------------------------------------
$Duration    = (Get-Date) - $StartTime
$DurationStr = "{0:mm}m{0:ss}s" -f $Duration

Write-Header "Resume Final"
Write-Host ""
Write-Host "  Duree totale  : $DurationStr" -ForegroundColor White
Write-Host "  Tests passes  : $TotalPass"   -ForegroundColor $(if ($TotalFail -eq 0) { "Green" } else { "White" })
Write-Host "  Tests echoues : $TotalFail"   -ForegroundColor $(if ($TotalFail -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($TotalFail -eq 0) {
    Write-Host "  [OK] Toutes les suites sont vertes." -ForegroundColor Green
    Write-Host "  [OK] Pret pour la prochaine session Claude Code." -ForegroundColor Green
} else {
    Write-Host "  [FAIL] $TotalFail test(s) en echec -- corriger avant de continuer." -ForegroundColor Red
    Write-Host "  Conseil : relancer avec -Layer 1, -Layer 2 ou -Layer 3 pour isoler." -ForegroundColor Yellow
}

Write-Host ""

exit $(if ($TotalFail -eq 0) { 0 } else { 1 })
