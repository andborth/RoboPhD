# Slotted harnesses for `--frozen-harness`

`--frozen-harness` evolves an agent's module-level model slots and constants and
nothing else. A model the code names inline, at a call site, would be one no
config could change, so `frozen_harness.py` refuses an agent that names any model
outside a slot.

Each directory here is a copy of such an agent with one change: every
`model_registry` handle named in the code is hoisted to a module-level slot, one
slot per call site, and the call site names the slot. Nothing else differs — no
constant, prompt or line of logic — so the slotted copy behaves as the original
does and its seed config is the original's own choice of models.

| Slotted copy | Made from | Slots | Constants |
| --- | --- | --- | --- |
| `iter16_feedcast_slotted` | `example_runs/robophd/asta_paper_finder/v0_1_0_cap_0_063_fable51/agents/iter16_feedcast` (official 0.399 @ $0.058) | 8 | 35 |

The rule for what is evolvable is unchanged: literals inside functions stay
frozen, as they do for a harness that needed no copy. Comments are frozen too, so
one that names a model (`# papers re-judged per-criterion by GPT-5.4`) describes
the seed config, not whatever a config puts in that slot.

`unit_tests/test_frozen_harness.py` holds each copy to this: undoing the hoist
must reproduce the original agent byte for byte, and the harness must accept the
copy. To add a copy, hoist the handles under a one-line comment
directly after the `model_registry` import and register the pair in that test's
`SLOTTED` table.

`iter16_feedcast` was evolved under a $0.063 cap. The copy exists to ask the
opposite question of the one in the example README: how far does a harness built
for the cheap end go when only its settings are re-chosen for the price of the
board's top entry, $0.279?

```bash
python examples/asta_paper_finder/main.py --eval-test-set \
  --frozen-harness examples/asta_paper_finder/frozen_harnesses/iter16_feedcast_slotted \
  --cost-threshold 0.279 \
  --engine-config '{"evolution_model": "fable-5.2"}'
```
