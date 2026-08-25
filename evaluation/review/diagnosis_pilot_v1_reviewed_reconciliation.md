# Diagnosis Pilot v1 — Human-Review Reconciliation

The original provisional JSONL remains unchanged. This reconciliation records the
independent mathematical adjudication in `diagnosis_pilot_v1_reviewed.jsonl`.
The cases remain synthetic; human review improves annotation reliability, not
ecological validity.

## Outcome

- Reviewed cases: 50
- Approved without mathematical label changes: 42
- Modified: 8
- Status distribution: {'correct_but_inefficient': 1, 'fully_correct': 16, 'incomplete': 4, 'incorrect': 27, 'indeterminate': 2}
- Diagnosis schema: `1.1`
- Benchmark schema: `diagnosis_pilot.reviewed.v1`

## Modified cases

- `pilot.14.expand-fourth-power` — Repeated multiplication is an ordinary fully valid expansion method.
- `pilot.15.gcd-by-divisor-lists` — Divisor listing is relevant and valid but less efficient than Euclid's algorithm.
- `pilot.16.three-coins-enumeration` — Complete enumeration of eight outcomes is standard and proportionate.
- `pilot.17.derivative-from-definition` — First-principles differentiation is a standard valid method.
- `pilot.22.transpose-sign` — The localized error is arithmetic: 8 / (-2) has the wrong sign.
- `pilot.36.pythagorean-on-any-triangle` — The first issue is the missing perpendicular assumption; Step 2 separately misnames the area formula as the Pythagorean theorem.
- `pilot.40.zero-derivative-at-point` — The overgeneralization already occurs in Step 1; Step 2 depends on it.
- `pilot.44.gcd-lucky-claim` — Both primality premises are false: 35=5×7 and 64=2^6; the final gcd is lucky.

## Schema reconciliation

- Valid inefficiency uses `valid_but_inefficient` plus `efficiency_note`; it is
  not a mathematical issue and does not populate `first_issue`.
- Locally coherent consequences use `dependent_on_previous_error` plus
  `depends_on_step_ids`, preserving the root error for future minimal-repair work.
- Independent later errors may remain `invalid` while also naming an earlier
  dependency, as in the Pythagorean/area terminology case.

## Final dispositions

- `pilot.01.linear-system-elimination` — **APPROVED**
- `pilot.02.divisible-by-six` — **APPROVED**
- `pilot.03.isosceles-base-angles` — **APPROVED**
- `pilot.04.choose-committee` — **APPROVED**
- `pilot.05.square-nonnegative` — **APPROVED**
- `pilot.06.compose-functions` — **APPROVED**
- `pilot.07.at-least-one-six` — **APPROVED**
- `pilot.08.quadratic-vieta` — **APPROVED**
- `pilot.09.odd-square-mod-eight` — **APPROVED**
- `pilot.10.triangle-area-coordinates` — **APPROVED**
- `pilot.11.strings-with-a` — **APPROVED**
- `pilot.12.positive-product-inequality` — **APPROVED**
- `pilot.13.inverse-by-reflection` — **APPROVED**
- `pilot.14.expand-fourth-power` — **MODIFIED**
- `pilot.15.gcd-by-divisor-lists` — **MODIFIED**
- `pilot.16.three-coins-enumeration` — **MODIFIED**
- `pilot.17.derivative-from-definition` — **MODIFIED**
- `pilot.18.discriminant-slip` — **APPROVED**
- `pilot.19.trapezoid-product-slip` — **APPROVED**
- `pilot.20.sum-count-slip` — **APPROVED**
- `pilot.21.power-of-power` — **APPROVED**
- `pilot.22.transpose-sign` — **MODIFIED**
- `pilot.23.add-rational-functions` — **APPROVED**
- `pilot.24.cancel-across-sum` — **APPROVED**
- `pilot.25.divide-by-variable` — **APPROVED**
- `pilot.26.cancel-square-roots` — **APPROVED**
- `pilot.27.even-square-converse` — **APPROVED**
- `pilot.28.induction-no-step` — **APPROVED**
- `pilot.29.union-count-overlap` — **APPROVED**
- `pilot.30.parallel-from-picture` — **APPROVED**
- `pilot.31.large-number-prime` — **APPROVED**
- `pilot.32.monotone-two-points` — **APPROVED**
- `pilot.33.squared-equation-check` — **APPROVED**
- `pilot.34.multiply-inequality-sign` — **APPROVED**
- `pilot.35.inverse-without-injective` — **APPROVED**
- `pilot.36.pythagorean-on-any-triangle` — **MODIFIED**
- `pilot.37.fermat-composite-modulus` — **APPROVED**
- `pilot.38.lhopital-non-indeterminate` — **APPROVED**
- `pilot.39.exclusive-implies-independent` — **APPROVED**
- `pilot.40.zero-derivative-at-point` — **MODIFIED**
- `pilot.41.repeated-letter-permutations` — **APPROVED**
- `pilot.42.zero-division-right-root` — **APPROVED**
- `pilot.43.triangle-area-lucky` — **APPROVED**
- `pilot.44.gcd-lucky-claim` — **MODIFIED**
- `pilot.45.factor-cubic-prefix` — **APPROVED**
- `pilot.46.sqrt-two-prefix` — **APPROVED**
- `pilot.47.circle-tangent-prefix` — **APPROVED**
- `pilot.48.binary-no-consecutive-prefix` — **APPROVED**
- `pilot.49.diagram-dependent-angle` — **APPROVED**
- `pilot.50.custom-operation-notation` — **APPROVED**
