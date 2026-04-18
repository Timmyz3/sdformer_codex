def compute_throughput_stats(sample_count, elapsed_sec):
    sample_count = int(sample_count)
    elapsed_sec = float(elapsed_sec)

    if sample_count <= 0 or elapsed_sec <= 0.0:
        return 0.0, 0.0

    step_time_sec = elapsed_sec / sample_count
    samples_per_sec = sample_count / elapsed_sec
    return step_time_sec, samples_per_sec
