import pytest
from layer4_decision.dsig.mapper import DSIGMapper, score_to_label


def test_gate1_b5050_score_range():
    """
    Gate 1 — CNSR(B_5050) ≈ 1.343 doit mapper vers score ∈ [72, 78].
    Hors plage → normalisation du mapper incorrecte.
    """
    mapper = DSIGMapper()
    signal = mapper.map(
        cnsr_usd_fed=1.343, sortino=1.60, calmar=0.80,
        max_dd_pct=15.0, walk_forward_score=1.0,
        paf_d1_verdict="N_A", dsr=None,
        source_id="qaaf-studio::b5050",
    )
    assert 72 <= signal.score <= 78, (
        f"Gate 1 FAIL — score(B_5050) = {signal.score}, attendu ∈ [72, 78]."
    )
    assert signal.label == "GOOD"
    assert signal.color == "YELLOW"


def test_dsig_label_color_coherence():
    """label et color doivent être cohérents avec score."""
    mapper = DSIGMapper()
    for cnsr in [1.76, -0.8, 2.5, 0.0]:
        s = mapper.map(
            cnsr_usd_fed=cnsr, sortino=max(cnsr, 0), calmar=max(cnsr / 2, 0),
            max_dd_pct=20, walk_forward_score=0.8,
            paf_d1_verdict="HIERARCHIE_CONFIRMEE",
        )
        assert s.label == score_to_label(s.score)


def test_dsig_dsr_cap():
    """DSR < 0.80 plafonne le score à 59."""
    mapper = DSIGMapper()
    s = mapper.map(
        cnsr_usd_fed=1.76, sortino=2.1, calmar=0.94,
        max_dd_pct=14.5, walk_forward_score=0.8,
        paf_d1_verdict="HIERARCHIE_CONFIRMEE", dsr=0.75,
    )
    assert s.score <= 59
    assert s.label in ["DEGRADED", "CRITICAL"]


def test_dsig_paf_stop_cap():
    """PAF STOP plafonne le score à 20."""
    mapper = DSIGMapper()
    s = mapper.map(
        cnsr_usd_fed=-0.8, sortino=-0.5, calmar=0,
        max_dd_pct=45, walk_forward_score=0,
        paf_d1_verdict="STOP",
    )
    assert s.score <= 20


def test_kb_manager_load(tmp_path):
    import shutil
    from layer4_decision.kb_manager import KBManager
    inv = tmp_path / "lentilles_inventory.yaml"
    shutil.copy("layer4_decision/lentilles_inventory.yaml", inv)
    kb = KBManager(inv)
    active = kb.get_active()
    assert len(active) == 1
    assert active[0]["nom"] == "H9+EMA60j"


def test_kb_manager_stale(tmp_path):
    import yaml
    from layer4_decision.kb_manager import KBManager
    data = {"lentilles": {"active": [{
        "nom": "test", "status": "EN_COURS",
        "last_verified": "2020-01-01", "ttl_days": 30,
    }]}}
    p = tmp_path / "inv.yaml"
    p.write_text(yaml.safe_dump(data))
    assert KBManager(p).is_stale(KBManager(p).get_active()[0]) is True


def test_n_trials_tracker(tmp_path):
    from layer4_decision.n_trials_tracker import NTrialsTracker
    t = NTrialsTracker(tmp_path / "state.yaml")
    assert t.register("EMA_span_variants") == 1
    assert t.register("EMA_span_variants") == 2
    t2 = NTrialsTracker(tmp_path / "state.yaml")
    assert t2.get("EMA_span_variants") == 2
