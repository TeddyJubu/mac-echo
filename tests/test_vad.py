from macecho.vad.vad import VadConfig, VadProcessor


def test_vad_session_is_single_threaded():
    # ONNX Runtime's default per-core pool spin-waits between frames: ~106% of a core while idle.
    opts = VadProcessor(VadConfig()).session.get_session_options()
    assert opts.intra_op_num_threads == 1 and opts.inter_op_num_threads == 1
