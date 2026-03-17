import pytest
from genedesign.checkers.g_quadruplex_checker import GQuadruplexChecker

def test_g4_detection():
    checker = GQuadruplexChecker()

    # 1. Positive Case: A classic G-Quadruplex motif
    # (GGGCGGGAGGGAGGG) - 4 runs of 3Gs with 1bp loops
    g4_sequence = "ATGCGGGCGGGAGGGAGGGTTTAAA"
    assert checker.run(g4_sequence) is True

    # 2. Negative Case: Disconnected Guanines
    # (Not enough runs to form the 'knot')
    clean_sequence = "ATGCGGATGCGGATGCGGATGCGGTTAAA"
    assert checker.run(clean_sequence) is False

    # 3. Negative Case: Random sequence with no G-clusters
    random_seq = "ATCGTACGTACGATCGTACG"
    assert checker.run(random_seq) is False

    # 4. Edge Case: Empty string
    assert checker.run("") is False

    # 5. Case Sensitivity: Should work with lowercase
    lower_g4 = "gggcgggagggaggg"
    assert checker.run(lower_g4) is True
