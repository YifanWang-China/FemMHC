from scripts.aggregate_mcphases_nested_loso_all13 import bootstrap_p_value


def test_bootstrap_p_value_is_two_sided_with_finite_sample_correction() -> None:
    assert bootstrap_p_value(1.0, 2000) == 2.0 / 2001.0
    assert bootstrap_p_value(0.5, 2000) == 1.0
    assert bootstrap_p_value(0.975, 2000) == 102.0 / 2001.0
