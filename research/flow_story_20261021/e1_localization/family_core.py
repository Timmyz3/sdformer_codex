"""Inference-only composition of frozen R27 Q6 and R28 Q4 integer operators.

No learned policy. All unselected complete tiles remain Q6; every true tail Q8.
Old modules are imported by the caller and their source hashes are recorded.
"""
import contextlib

MODES = ("Q6", "patch_Q4", "fc1_Q4", "fc2_Q4", "mlp_Q4")


def selected(name, mode, q6):
    if mode not in MODES:
        raise ValueError(mode)
    return ((mode == "patch_Q4" and name in q6.CONVS)
            or (mode == "fc1_Q4" and name.endswith(".fc1"))
            or (mode == "fc2_Q4" and name.endswith(".fc2"))
            or (mode == "mlp_Q4" and name in q6.LINEARS))


class FamilyCore:
    def __init__(self, q6, q4):
        self.q6, self.q4 = q6, q4
        self.TARGETS = q6.TARGETS
        if q6.TARGETS != q4.TARGETS:
            raise ValueError("Exactly the same 28 target operators required")

    @contextlib.contextmanager
    def installed(self, model, mode, *, ste=False, reference=False):
        if ste:
            raise ValueError("E1 is inference only, never weight recovery")
        original = {n: model.get_submodule(n) for n in self.TARGETS}
        wrappers = {}
        try:
            for n, source in original.items():
                low = selected(n, mode, self.q6)
                c = self.q4 if low else self.q6
                w = c.STEAffine(source, "linear" if n in c.LINEARS else "conv",
                               "Q4" if low else "Q6", ste=False, reference=reference)
                parent, attr = n.rsplit(".", 1)
                setattr(model.get_submodule(parent), attr, w)
                wrappers[n] = w
            yield wrappers
        finally:
            for n, source in original.items():
                parent, attr = n.rsplit(".", 1)
                setattr(model.get_submodule(parent), attr, source)
            assert all(model.get_submodule(n) is v for n, v in original.items())
