from scripts.validate_retrieval import validation_passes


def test_grounded_validation_allows_additional_context_when_expected_evidence_is_present() -> None:
    assert validation_passes(
        "grounded",
        expected_titles={"Payment Retry Policy"},
        admitted_titles=["Calypso Timeout Runbook", "Payment Retry Policy"],
    )


def test_grounded_validation_fails_when_expected_evidence_is_missing() -> None:
    assert not validation_passes(
        "grounded",
        expected_titles={"Payment Retry Policy"},
        admitted_titles=["Calypso Timeout Runbook"],
    )


def test_unanswerable_and_out_of_domain_validation_require_no_admitted_evidence() -> None:
    assert validation_passes(
        "in_domain_unanswerable",
        expected_titles=set(),
        admitted_titles=[],
    )
    assert validation_passes(
        "out_of_domain",
        expected_titles=set(),
        admitted_titles=[],
    )
    assert not validation_passes(
        "in_domain_unanswerable",
        expected_titles=set(),
        admitted_titles=["Payment Retry Policy"],
    )
    assert not validation_passes(
        "out_of_domain",
        expected_titles=set(),
        admitted_titles=["Payment Retry Policy"],
    )
