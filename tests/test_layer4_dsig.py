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
    """KBManager charge et sauvegarde proprement — indépendant du contenu réel."""
    import yaml
    from layer4_decision.kb_manager import KBManager

    kb_file = tmp_path / "kb_test.yaml"
    inv_file = tmp_path / "inventory.yaml"
    kb_file.write_text(
        "hypothese:\n  nom: TEST\n  statut: en_cours\n", encoding="utf-8"
    )
    inv_file.write_text(
        "lentilles:\n  active: []\n  archivees: []\n  queue: []\n", encoding="utf-8"
    )

    kb = KBManager(kb_file, inv_file)
    kb.record_verdict("TEST", "ARCHIVE_FAIL_Q2", metrics={"cnsr_usd_fed": 1.2})

    data = kb._load_hyp()
    assert data["verdict"] == "ARCHIVE_FAIL_Q2"
    assert data["metrics"]["cnsr_usd_fed"] == 1.2


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


def test_load_hyp_multiblock_yaml(tmp_path):
    """_load_hyp() sur un YAML multi-blocs doit charger le premier et émettre un warning."""
    import yaml
    from layer4_decision.kb_manager import KBManager

    kb_file = tmp_path / "test_kb.yaml"
    inv_file = tmp_path / "inv.yaml"
    kb_file.write_text("a: 1\n---\nb: 2\n", encoding="utf-8")
    inv_file.write_text(
        "lentilles:\n  active: []\n  archivees: []\n", encoding="utf-8"
    )
    kb = KBManager(kb_file, inv_file)
    with pytest.warns(UserWarning, match="multi-blocs"):
        data = kb._load_hyp()
    assert data == {"a": 1}


def test_n_trials_tracker(tmp_path):
    """NTrialsTracker incrémente correctement."""
    from layer4_decision.n_trials_tracker import NTrialsTracker
    tracker = NTrialsTracker(tmp_path / "tracker.yaml")
    tracker.register("H9+EMA60j")
    tracker.register("H9+EMA60j")
    assert tracker.get("H9+EMA60j") == 2


def test_n_trials_tracker_get_family(tmp_path):
    """get_family_n_trials lit N_trials_famille depuis le YAML si présent."""
    import yaml
    from layer4_decision.n_trials_tracker import NTrialsTracker

    kb_file = tmp_path / "kb.yaml"
    kb_file.write_text("N_trials_famille: 101\nhypothese:\n  nom: TEST\n", encoding="utf-8")
    tracker = NTrialsTracker(kb_file)
    assert tracker.get_family_n_trials("EMA_span_variants") == 101


def test_n_trials_tracker_get_family_fallback(tmp_path):
    """get_family_n_trials tombe en fallback sur le compteur si N_trials_famille absent."""
    from layer4_decision.n_trials_tracker import NTrialsTracker
    tracker = NTrialsTracker(tmp_path / "state.yaml")
    tracker.register("EMA_span_variants")
    tracker.register("EMA_span_variants")
    assert tracker.get_family_n_trials("EMA_span_variants") == 2
