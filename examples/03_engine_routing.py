"""Example 03: inspect how choose_engine routes across the (dim x stiff x need_grad) cube.

No Problem instance needed here — ``choose_engine`` is a pure function of three
salient workload axes (state dimension, stiffness, whether reverse-mode gradients are
needed), so this script just calls it directly and prints the routing table.
``gradsolve.dispatch.DECISION_MAP`` is the table backing the same policy (one row per
corner, each with the evidence that justifies it); a test in this repo asserts
``choose_engine`` agrees with it on every row.

Run with:
    python examples/03_engine_routing.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

from gradsolve.dispatch import choose_engine, DECISION_MAP, NVAR_CEILING, CUDA_NVAR_CEILING

# A handful of illustrative (dim, stiff, need_grad) points spanning the cube.
cases = [
    (3,    False, False),
    (3,    False, True),
    (3,    True,  False),
    (3,    True,  True),
    (32,   False, False),
    (1000, False, False),
    (1000, True,  True),
]

print(f"Dimension limits: dim<={CUDA_NVAR_CEILING} -> cuda_tsit5 eligible (nonstiff+forward); "
      f"dim<={NVAR_CEILING} -> \"low NVAR\" (fused-engine range).")
print()
print(f"{'dim':>6}  {'stiff':>5}  {'grad':>4}  ->  engine")
print("-" * 42)
for dim, stiff, grad in cases:
    engine = choose_engine(dim=dim, stiff=stiff, need_grad=grad)
    print(f"{dim:>6}  {str(stiff):>5}  {str(grad):>4}  ->  {engine}")

assert choose_engine(3, False, True) == "warp_replay", (
    f"expected warp_replay, got {choose_engine(3, False, True)!r}"
)

# The full DECISION_MAP: same policy, plus the first sentence of the evidence recorded
# for each row (the per-corner rationale).
print()
print("Full DECISION_MAP (nvar side, stiff, need_grad -> selected engine):")
print("-" * 70)
for row in DECISION_MAP:
    engine = choose_engine(dim=row["dim"], stiff=row["stiff"], need_grad=row["need_grad"])
    assert engine == row["engine"], (
        f"DECISION_MAP row {row!r} disagrees with choose_engine ({engine!r})"
    )
    gated = f"  (gated: {row['gated_engine']})" if row["gated_engine"] else ""
    evidence_headline = row["evidence"].split(".")[0] + "."
    print(f"  {row['nvar']:>4}-nvar  stiff={row['stiff']!s:5}  grad={row['need_grad']!s:5}  "
          f"-> {row['engine']}{gated}")
    print(f"           {evidence_headline}")

print()
print("OK — choose_engine agrees with DECISION_MAP on every row")
