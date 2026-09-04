import unittest

from scripts.model_local5_frontier_retirement import (
    allocate_terms,
    closed_form_retirement_events,
    frontier_geometry,
    retirement_events,
    simulate_frontier,
    simulate_plane_serial_frontier,
    simulate_plane_serial_stripe,
    simulate_plane_serial_two_phase,
    simulate_two_phase,
    source_consumers,
    storage_bits,
    stripe_retirement_events,
)


class Local5FrontierRetirementTest(unittest.TestCase):
    def test_center_source_retires_at_south_consumer(self):
        consumers = source_consumers(9, 9, 4, 4)
        self.assertEqual(max(consumers), 5 * 9 + 4)

    def test_all_sources_retire_exactly_once(self):
        events = retirement_events(9, 9, time_planes=2)
        retired = [source for event in events for source in event]
        self.assertEqual(sorted(retired), list(range(162)))

    def test_closed_form_retirement_matches_consumer_enumeration(self):
        for size in (3, 9, 15):
            self.assertEqual(
                closed_form_retirement_events(size, size),
                retirement_events(size, size),
            )

    def test_stripe_baseline_retires_rows_at_barrier(self):
        events = stripe_retirement_events(9, 9)
        self.assertEqual(len(events[2 * 9 - 1]), 9)
        self.assertEqual(len(events[9 * 9 - 1]), 18)
        self.assertEqual(sum(len(event) for event in events), 162)

    def test_fixed_stencil_retire_burst_is_three(self):
        for size in (9, 15):
            geometry = frontier_geometry(size, size)
            self.assertEqual(geometry["max_retire_burst"], 3)
            self.assertEqual(geometry["gate_ring_rows"], 3)

    def test_serial_retire_accounts_bottom_burst(self):
        events = retirement_events(9, 9)
        work = [1] * 162
        serial = simulate_frontier(
            events,
            work,
            fifo_depth=8,
            ready_percent=100,
            retire_width=1,
        )
        wide = simulate_frontier(
            events,
            work,
            fifo_depth=8,
            ready_percent=100,
            retire_width=3,
        )
        self.assertGreater(serial["producer_stalls"], wide["producer_stalls"])
        self.assertGreaterEqual(serial["cycles"], wide["cycles"])

    def test_term_allocations_conserve_work(self):
        events = retirement_events(9, 9)
        for scenario in ("uniform", "front_loaded", "tail_loaded"):
            work = allocate_terms(257, events, scenario)
            self.assertEqual(sum(work), 257)
            self.assertLessEqual(max(work), 160)

    def test_frontier_is_bitcount_conservative(self):
        events = retirement_events(9, 9)
        work = allocate_terms(126, events, "uniform")
        result = simulate_frontier(
            events,
            work,
            fifo_depth=8,
            ready_percent=90,
        )
        self.assertEqual(result["terms"], 126)
        self.assertLessEqual(result["max_fifo_sources"], 8)

    def test_overlap_not_worse_for_uniform_trace(self):
        events = retirement_events(9, 9)
        work = allocate_terms(126, events, "uniform")
        frontier = simulate_frontier(
            events,
            work,
            fifo_depth=8,
            ready_percent=100,
        )
        baseline = simulate_two_phase(
            tokens=162,
            total_terms=126,
            ready_percent=100,
        )
        self.assertLess(frontier["cycles"], baseline["cycles"])

    def test_storage_reduction_exceeds_half(self):
        for size in (9, 15):
            result = storage_bits(size, size, fifo_depth=8)
            self.assertGreater(result["reduction"], 0.5)

    def test_plane_serial_frontier_drains_between_time_planes(self):
        events = retirement_events(3, 3, time_planes=2)
        work = [1] * 18
        result = simulate_plane_serial_frontier(
            events,
            work,
            plane_tokens=9,
            fifo_depth=3,
            ready_percent=75,
        )
        self.assertEqual(result["terms"], 18)
        self.assertGreaterEqual(result["cycles"], 18)

    def test_plane_serial_two_phase_preserves_work_and_score_cycles(self):
        work = [1, 2, 0, 1, 1, 0, 1, 0]
        destination_cycles = [1, 2, 1, 2, 2, 1, 2, 1]
        result = simulate_plane_serial_two_phase(
            source_work=work,
            plane_tokens=4,
            ready_percent=90,
            destination_cycles=destination_cycles,
        )
        self.assertEqual(result["terms"], sum(work))
        self.assertEqual(
            result["score_cycles"],
            sum(destination_cycles),
        )

    def test_nonblocking_stripe_preserves_service_and_uses_two_rows(self):
        work = [1] * 18
        result = simulate_plane_serial_stripe(
            work,
            height=3,
            width=3,
            ready_percent=100,
            destination_cycles=[1] * 18,
            row_buffer_slots=2,
        )
        self.assertEqual(result["terms"], 18)
        self.assertLessEqual(result["max_fifo_sources"], 6)
        self.assertLessEqual(result["max_stripe_owned_rows"], 2)
        self.assertGreaterEqual(result["producer_work_cycles"], 18)


if __name__ == "__main__":
    unittest.main()
