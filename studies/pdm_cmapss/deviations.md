1. **P4 is scored with unpaired intervals.** The protocol words P4 as "paired CI excludes zero", but the completion stage stores the pooled NMSE of each variant and not the paired mask / no-mask difference. P4 is therefore scored by non-overlapping unpaired 95 % intervals, which is stricter than a paired test: a CONFIRMED is credible, a REFUTED may only mean that the intervals overlap.

Notes on how pre-registered rules behaved (no change was made):

2. **The `quantize` family has integer resolution.** The ADC-bits family jumps from a harmless 3 bits to a damaging 2 bits, so the calibration found no severity in between: L25, L50 and L100 are one and the same condition for that family (the reference degrades by the same factor at all three levels). Its L50 column is therefore not a x1.5 condition.
3. **`dropout`, `stuck` and `combined` overshoot their targets.** Severity maps to a whole number of channels (`ceil(s * 14)`), so the achieved degradation is the first step at or above the target (for example x1.41 for the x1.25 target). The achieved multiples are in the table of calibrated severities.
4. **Grid edges.** Several selections sit at the edge of the pre-registered grid (see the flags in the tuning tables). They were not extended, as the protocol requires.
