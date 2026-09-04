from hw_autoresearch_nts07.scripts.ttx_temporal_pair_layout_model import model


def test_ttx_temporal_pair_fits_one_64_bit_word():
    result = model(timesteps=2, spatial_tokens=81, lanes=32, tensors=2, bus_bits=64)
    assert result["word_bits"]["packed_temporal_pair"] == 64
    assert result["serial"]["requests"] == 324
    assert result["packed"]["requests"] == 162
    assert result["deltas"]["request_reduction_vs_uncoalesced"] == 0.5
    assert result["deltas"]["request_reduction_vs_already_coalesced"] == 0.0
    assert result["deltas"]["logical_storage_reduction"] == 0.0
