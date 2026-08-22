# Diagnosis Pilot v1 — Human Review Worksheet

> All annotations below are Codex proposals with `human_review_status=pending`.
> They are not mathematical ground truth or validated research labels.

Review each case using `evaluation/annotation_guidelines.md`. Replace the disposition
placeholder with exactly one of: **APPROVE**, **MODIFY**, **REJECT**, **AMBIGUOUS**.
Record concise corrections under reviewer notes; do not edit the generated proposal
in place until the review is reconciled into a separately versioned reviewed dataset.

## pilot.01.linear-system-elimination

**Domain / category:** algebra / `fully_correct_conventional`

**Problem:** Solve the system 2x + y = 7 and x - y = 2.

**Student solution:**

1. Add the equations: 3x = 9.
2. Hence x = 3.
3. Substituting gives y = 1.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_ddcdc825d927bcf6`): “3. Substituting gives y = 1.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The proposed label accepts each reversible algebraic step.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.02.divisible-by-six

**Domain / category:** number_theory / `fully_correct_conventional`

**Problem:** Prove that n(n + 1)(n + 2) is divisible by 6 for every integer n.

**Student solution:**

1. Among three consecutive integers, one is divisible by 3.
2. At least one of them is even.
3. Their product therefore has factors 2 and 3, so it is divisible by 6.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_8dc978a1ef96acf4`): “3. Their product therefore has factors 2 and 3, so it is divisible by 6.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The proposal treats the divisibility argument as complete for integers.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.03.isosceles-base-angles

**Domain / category:** geometry / `fully_correct_conventional`

**Problem:** In triangle ABC, AB = AC. Prove that angles B and C are equal.

**Student solution:**

1. Compare triangles ABC and ACB.
2. AB = AC, AC = AB, and BC = CB.
3. The triangles are congruent by SSS, so corresponding base angles B and C are equal.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_34ca9ad2eaa4d7e4`): “3. The triangles are congruent by SSS, so corresponding base angles B and C are equal.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The correspondence and congruence justification are proposed as valid.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.04.choose-committee

**Domain / category:** combinatorics / `fully_correct_conventional`

**Problem:** How many 3-person committees can be selected from 8 people?

**Student solution:**

1. Order does not matter.
2. The count is C(8,3) = 8!/(3!5!) = 56.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 2 (`step_a479de6b7dc0d157`): “2. The count is C(8,3) = 8!/(3!5!) = 56.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The proposal accepts the unordered-selection model and computation.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.05.square-nonnegative

**Domain / category:** inequalities / `fully_correct_conventional`

**Problem:** Prove that a^2 + b^2 is at least 2ab for all real a,b.

**Student solution:**

1. For real a,b, (a-b)^2 is nonnegative.
2. Expanding gives a^2 - 2ab + b^2 >= 0.
3. Rearranging yields a^2 + b^2 >= 2ab.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_2a52c42d3015afab`): “3. Rearranging yields a^2 + b^2 >= 2ab.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** Every implication is proposed as valid over the stated real domain.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.06.compose-functions

**Domain / category:** functions / `fully_correct_conventional`

**Problem:** Let f(x)=2x+1 and g(x)=x^2. Find (g composed with f)(3).

**Student solution:**

1. f(3) = 7.
2. Then g(f(3)) = g(7) = 49.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 2 (`step_e9e48d8c2969160e`): “2. Then g(f(3)) = g(7) = 49.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The proposed label follows the definition of composition.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.07.at-least-one-six

**Domain / category:** probability / `fully_correct_conventional`

**Problem:** Two fair dice are rolled. Find the probability that at least one die is a six.

**Student solution:**

1. The probability of no six is (5/6)^2 = 25/36.
2. Therefore the probability of at least one six is 1 - 25/36 = 11/36.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 2 (`step_fd5b2d64f885438c`): “2. Therefore the probability of at least one six is 1 - 25/36 = 11/36.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The complement calculation is proposed as correct under independence.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.08.quadratic-vieta

**Domain / category:** algebra / `fully_correct_unconventional`

**Problem:** Solve x^2 - 7x + 12 = 0.

**Student solution:**

1. Two numbers with sum 7 and product 12 are 3 and 4.
2. Therefore the roots are x=3 and x=4.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 2 (`step_a1c84f115e30be3d`): “2. Therefore the roots are x=3 and x=4.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The proposal recognizes a Vieta-style root identification without requiring displayed factorization.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.09.odd-square-mod-eight

**Domain / category:** number_theory / `fully_correct_unconventional`

**Problem:** Prove that the square of every odd integer is congruent to 1 modulo 8.

**Student solution:**

1. Write an odd integer as 2k+1.
2. Its square is 4k(k+1)+1.
3. One of k,k+1 is even, so 4k(k+1) is divisible by 8; the remainder is 1.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_d4a0450bfdce31b1`): “3. One of k,k+1 is even, so 4k(k+1) is divisible by 8; the remainder is 1.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The algebraic parity proof is proposed as a valid alternative to residue enumeration.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.10.triangle-area-coordinates

**Domain / category:** geometry / `fully_correct_unconventional`

**Problem:** Find the area of the triangle with vertices (0,0), (4,0), and (1,3).

**Student solution:**

1. The determinant of vectors (4,0) and (1,3) is 12.
2. Half its absolute value is 6, so the area is 6.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 2 (`step_dc5285ed36283745`): “2. Half its absolute value is 6, so the area is 6.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The determinant method is proposed as equivalent to base times height.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.11.strings-with-a

**Domain / category:** combinatorics / `fully_correct_unconventional`

**Problem:** How many length-5 binary strings contain at least one 1?

**Student solution:**

1. There are 2^5 total binary strings.
2. Only 00000 contains no 1.
3. Thus the answer is 32-1=31.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_62330c1b420f8472`): “3. Thus the answer is 32-1=31.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The complement count is proposed as a valid alternative to binomial summation.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.12.positive-product-inequality

**Domain / category:** inequalities / `fully_correct_unconventional`

**Problem:** For positive x, prove x + 1/x is at least 2.

**Student solution:**

1. Since x>0, (x-1)^2/x is nonnegative.
2. Expanding that quotient gives x - 2 + 1/x >= 0.
3. Hence x + 1/x >= 2.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_85ea1377ae0c610f`): “3. Hence x + 1/x >= 2.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The nonnegative-square proof is proposed as valid and domain-aware.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.13.inverse-by-reflection

**Domain / category:** functions / `fully_correct_unconventional`

**Problem:** Find the inverse of f(x)=3x-5.

**Student solution:**

1. Reflect the line y=3x-5 across y=x, giving x=3y-5.
2. Solving this relation for y gives y=(x+5)/3.
3. Therefore f inverse of x is (x+5)/3.

**Proposed status:** `fully_correct`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** Step 3 (`step_9d79830c2499792c`): “3. Therefore f inverse of x is (x+5)/3.”

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The geometric reflection framing is proposed as equivalent to the standard symbolic method.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.14.expand-fourth-power

**Domain / category:** algebra / `correct_but_inefficient`

**Problem:** Expand (x+1)^4.

**Student solution:**

1. Multiply (x+1)(x+1) to get x^2+2x+1.
2. Multiply that polynomial by x+1 to get x^3+3x^2+3x+1.
3. Multiply once more by x+1 to obtain x^4+4x^3+6x^2+4x+1.

**Proposed status:** `correct_but_inefficient`

**Proposed first issue:** Step 1 (`step_5675621e99933353`), “1. Multiply (x+1)(x+1) to get x^2+2x+1.” — The method is valid but unnecessarily laborious for this expansion.

**Proposed issue category:** `relevance.irrelevant`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The repeated multiplication is proposed as correct though substantially longer than the binomial theorem.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.15.gcd-by-divisor-lists

**Domain / category:** number_theory / `correct_but_inefficient`

**Problem:** Find gcd(84,126).

**Student solution:**

1. List divisors of 84: 1,2,3,4,6,7,12,14,21,28,42,84.
2. List divisors of 126: 1,2,3,6,7,9,14,18,21,42,63,126.
3. The greatest common divisor is 42.

**Proposed status:** `correct_but_inefficient`

**Proposed first issue:** Step 1 (`step_682dd4e58fd99a6d`), “1. List divisors of 84: 1,2,3,4,6,7,12,14,21,28,42,84.” — Listing every divisor is avoidable when the Euclidean algorithm is available.

**Proposed issue category:** `relevance.irrelevant`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The exhaustive lists and selected gcd are proposed as correct but inefficient.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.16.three-coins-enumeration

**Domain / category:** probability / `correct_but_inefficient`

**Problem:** Three fair coins are tossed. Find the probability of exactly two heads.

**Student solution:**

1. List HHH,HHT,HTH,HTT,THH,THT,TTH,TTT.
2. HHT, HTH, and THH have exactly two heads.
3. The probability is 3/8.

**Proposed status:** `correct_but_inefficient`

**Proposed first issue:** Step 1 (`step_efb2cfbcdcd66050`), “1. List HHH,HHT,HTH,HTT,THH,THT,TTH,TTT.” — Full enumeration is valid but inefficient for this small binomial count.

**Proposed issue category:** `relevance.irrelevant`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The complete sample-space enumeration is proposed as correct but longer than needed.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.17.derivative-from-definition

**Domain / category:** calculus, functions / `correct_but_inefficient`

**Problem:** Differentiate f(x)=x^2+3x.

**Student solution:**

1. Form ((x+h)^2+3(x+h)-x^2-3x)/h.
2. Simplify to (2xh+h^2+3h)/h.
3. For h nonzero this is 2x+h+3, whose limit is 2x+3.

**Proposed status:** `correct_but_inefficient`

**Proposed first issue:** Step 1 (`step_b4d0cae5f5eafb08`), “1. Form ((x+h)^2+3(x+h)-x^2-3x)/h.” — Using first principles is valid but inefficient in this procedural context.

**Proposed issue category:** `relevance.irrelevant`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `WAIT` / `0`

**Short rationale:** The limit-definition calculation is proposed as correct but more work than standard rules require.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.18.discriminant-slip

**Domain / category:** algebra / `arithmetic_slip`

**Problem:** Solve 2x^2-5x-3=0 using the quadratic formula.

**Student solution:**

1. Here a=2, b=-5, c=-3.
2. The discriminant is 25-4(2)(-3)=25+24=47.
3. Thus x=(5 plus or minus sqrt(47))/4.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_baa4115d832f87d1`), “2. The discriminant is 25-4(2)(-3)=25+24=47.” — The sum 25+24 equals 49, not 47.

**Proposed issue category:** `computation.arithmetic`

**Proposed reusable prefix end:** Step 1 (`step_4d43963f95639558`): “1. Here a=2, b=-5, c=-3.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes the first arithmetic slip in evaluating the discriminant.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.19.trapezoid-product-slip

**Domain / category:** geometry / `arithmetic_slip`

**Problem:** A trapezoid has bases 7 and 11 and height 4. Find its area.

**Student solution:**

1. Use A=(1/2)(b1+b2)h.
2. A=(1/2)(18)(4)=32.
3. So the area is 32 square units.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_902c7f42fc9b0eee`), “2. A=(1/2)(18)(4)=32.” — Half of 18 times 4 is 36, not 32.

**Proposed issue category:** `computation.arithmetic`

**Proposed reusable prefix end:** Step 1 (`step_26746c59ed4d38d7`): “1. Use A=(1/2)(b1+b2)h.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The formula is proposed as correct, with the first error in multiplication.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.20.sum-count-slip

**Domain / category:** combinatorics / `arithmetic_slip`

**Problem:** How many integers from 1 through 100 are divisible by 4 or by 6?

**Student solution:**

1. There are 25 multiples of 4 and 16 multiples of 6.
2. There are 8 multiples of 12.
3. Inclusion-exclusion gives 25+16-8=32.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 3 (`step_6d2fcb49000b811c`), “3. Inclusion-exclusion gives 25+16-8=32.” — The displayed expression evaluates to 33, not 32.

**Proposed issue category:** `computation.arithmetic`

**Proposed reusable prefix end:** Step 2 (`step_1967bcb2bf537214`): “2. There are 8 multiples of 12.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The counting setup is proposed as valid and the last numerical sum as the first error.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.21.power-of-power

**Domain / category:** algebra / `algebraic_error`

**Problem:** Simplify (x^3)^4.

**Student solution:**

1. A power raised to a power means add the exponents.
2. Therefore (x^3)^4=x^7.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_d9310ae16249c96f`), “1. A power raised to a power means add the exponents.” — The exponents should be multiplied, not added.

**Proposed issue category:** `computation.algebraic`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies the misstated exponent law before the resulting expression.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.22.transpose-sign

**Domain / category:** algebra / `algebraic_error`

**Problem:** Solve 5-2x=13.

**Student solution:**

1. Subtract 5 from both sides: -2x=8.
2. Divide by -2: x=4.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_639906f3a2c9607f`), “2. Divide by -2: x=4.” — Dividing 8 by -2 gives -4, not 4.

**Proposed issue category:** `computation.algebraic`

**Proposed reusable prefix end:** Step 1 (`step_8bd0ac2e48a29390`): “1. Subtract 5 from both sides: -2x=8.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal treats the first line as valid and localizes the sign error at division.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.23.add-rational-functions

**Domain / category:** algebra, functions / `algebraic_error`

**Problem:** Simplify 1/x + 1/(x+1).

**Student solution:**

1. Add numerators and denominators separately.
2. This gives 2/(2x+1).

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_043edb166f963854`), “1. Add numerators and denominators separately.” — Fractions cannot be added by adding their denominators.

**Proposed issue category:** `computation.algebraic`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies the invalid addition rule at its statement.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.24.cancel-across-sum

**Domain / category:** algebra / `invalid_cancellation_division`

**Problem:** Simplify (x+6)/x for x nonzero.

**Student solution:**

1. Cancel x from the numerator and denominator.
2. The expression becomes 1+6=7.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_58e7d565299cb0b4`), “1. Cancel x from the numerator and denominator.” — Cancellation applies to common factors, not one term of a sum.

**Proposed issue category:** `reasoning.invalid_inference`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes cancellation across addition as the first invalid move.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.25.divide-by-variable

**Domain / category:** algebra / `invalid_cancellation_division`

**Problem:** Solve x(x-5)=0 over the real numbers.

**Student solution:**

1. Divide both sides by x to get x-5=0.
2. Therefore x=5 is the solution.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_6113e15f0371be40`), “1. Divide both sides by x to get x-5=0.” — Dividing by x assumes x is nonzero and discards the solution x=0.

**Proposed issue category:** `condition.missing`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal flags division by a potentially zero quantity and the lost solution.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.26.cancel-square-roots

**Domain / category:** algebra / `invalid_cancellation_division`

**Problem:** Simplify sqrt(a+b) when a,b are nonnegative.

**Student solution:**

1. Distribute the square root over addition.
2. So sqrt(a+b)=sqrt(a)+sqrt(b).

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_1e87cd82ecba1124`), “1. Distribute the square root over addition.” — Square roots do not generally distribute over addition.

**Proposed issue category:** `reasoning.invalid_inference`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal treats the distribution claim as the first invalid identity.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.27.even-square-converse

**Domain / category:** number_theory / `logical_gap`

**Problem:** Prove that if n^2 is even, then n is even.

**Student solution:**

1. If n is even, then n^2 is even.
2. Therefore, if n^2 is even, n must be even.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_5985633628211baf`), “2. Therefore, if n^2 is even, n must be even.” — The converse does not follow from the implication proved in step 1.

**Proposed issue category:** `reasoning.logical_gap`

**Proposed reusable prefix end:** Step 1 (`step_e745c1ff9f5188f3`): “1. If n is even, then n^2 is even.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies reversal of the implication rather than disputing the conclusion.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.28.induction-no-step

**Domain / category:** number_theory / `logical_gap`

**Problem:** Prove by induction that 1+3+...+(2n-1)=n^2 for positive integers n.

**Student solution:**

1. For n=1, both sides equal 1.
2. Assume the formula is true for n=k.
3. Therefore it is true for every positive integer n.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 3 (`step_9772d8996128f6cc`), “3. Therefore it is true for every positive integer n.” — No argument establishes the case k+1 from the induction hypothesis.

**Proposed issue category:** `reasoning.logical_gap`

**Proposed reusable prefix end:** Step 2 (`step_f919565bb18ce920`): “2. Assume the formula is true for n=k.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes the omitted induction step before the universal conclusion.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.29.union-count-overlap

**Domain / category:** combinatorics / `logical_gap`

**Problem:** In a class, 18 study French, 14 study Spanish, and 6 study both. How many study at least one?

**Student solution:**

1. Add the French and Spanish counts: 18+14=32.
2. Therefore 32 students study at least one language.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_b37dc570533717bc`), “1. Add the French and Spanish counts: 18+14=32.” — Students in both groups are counted twice in the sum.

**Proposed issue category:** `reasoning.logical_gap`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal marks the failure to account for overlap at the addition step.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.30.parallel-from-picture

**Domain / category:** geometry / `unsupported_claim`

**Problem:** Given a quadrilateral ABCD with no additional assumptions, prove or disprove that AB is parallel to CD.

**Student solution:**

1. In the sketch, AB and CD look horizontal.
2. Hence AB is parallel to CD.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_96e727ec1e6ae516`), “1. In the sketch, AB and CD look horizontal.” — An unmarked sketch is not proof that the segments are parallel.

**Proposed issue category:** `justification.unjustified_claim`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal treats visual appearance as unsupported evidence.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.31.large-number-prime

**Domain / category:** number_theory / `unsupported_claim`

**Problem:** Determine whether 221 is prime.

**Student solution:**

1. 221 is not divisible by 2, 3, or 5.
2. Therefore 221 is prime.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_92c453c9dccfa6d5`), “2. Therefore 221 is prime.” — Checking only 2, 3, and 5 does not eliminate all possible prime divisors.

**Proposed issue category:** `justification.unjustified_claim`

**Proposed reusable prefix end:** Step 1 (`step_44721b9d5618fc8e`): “1. 221 is not divisible by 2, 3, or 5.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies the incomplete divisibility check as insufficient support.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.32.monotone-two-points

**Domain / category:** functions / `unsupported_claim`

**Problem:** A function satisfies f(0)=1 and f(1)=3. Must it be increasing on [0,1]?

**Student solution:**

1. Since f(1)>f(0), the output rose as x rose.
2. Therefore f is increasing on the whole interval.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_f9c07fe5747e6775`), “2. Therefore f is increasing on the whole interval.” — Endpoint values alone do not establish monotonicity at every pair of points.

**Proposed issue category:** `justification.unjustified_claim`

**Proposed reusable prefix end:** Step 1 (`step_cdeec936a45d9f2d`): “1. Since f(1)>f(0), the output rose as x rose.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal rejects inference of an interval property from only endpoints.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.33.squared-equation-check

**Domain / category:** algebra / `missing_condition`

**Problem:** Solve sqrt(x+2)=x over the reals.

**Student solution:**

1. Square both sides: x+2=x^2.
2. Rearrange: x^2-x-2=0, so x=2 or x=-1.
3. Thus both 2 and -1 solve the original equation.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 3 (`step_3e3150f07dcaf651`), “3. Thus both 2 and -1 solve the original equation.” — Squaring can introduce extraneous roots, so each candidate must be checked.

**Proposed issue category:** `condition.missing`

**Proposed reusable prefix end:** Step 2 (`step_aa6a7c150a305b55`): “2. Rearrange: x^2-x-2=0, so x=2 or x=-1.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes failure to check candidates after a non-reversible operation.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.34.multiply-inequality-sign

**Domain / category:** inequalities / `missing_condition`

**Problem:** If ax<ay and a is a nonzero real number, what can be concluded about x and y?

**Student solution:**

1. Divide both sides by a.
2. Therefore x<y.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_5f93ded26de1c882`), “1. Divide both sides by a.” — The inequality direction depends on whether a is positive or negative.

**Proposed issue category:** `condition.missing`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies the unstated sign condition on the divisor.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.35.inverse-without-injective

**Domain / category:** functions / `missing_condition`

**Problem:** Let f be a function from R to R. Does f necessarily have an inverse function?

**Student solution:**

1. Swap x and y in y=f(x).
2. This always defines f inverse, so every function has an inverse.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_90c4a3dac4e57549`), “2. This always defines f inverse, so every function has an inverse.” — Not every function is injective, so swapping symbols need not define a function inverse.

**Proposed issue category:** `condition.missing`

**Proposed reusable prefix end:** Step 1 (`step_242b47f4a28a24a3`): “1. Swap x and y in y=f(x).”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes the missing one-to-one condition in the universal claim.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.36.pythagorean-on-any-triangle

**Domain / category:** geometry / `theorem_misuse`

**Problem:** A triangle has side lengths 4,5,6. Find its area.

**Student solution:**

1. Treat 4 and 5 as perpendicular legs.
2. By the Pythagorean theorem, the area is (1/2)(4)(5)=10.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_f16d27cdb2f0894a`), “1. Treat 4 and 5 as perpendicular legs.” — No right angle is given, and 4^2+5^2 is not 6^2.

**Proposed issue category:** `theorem.misused`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal flags application of a right-triangle formula without a right-angle condition.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.37.fermat-composite-modulus

**Domain / category:** number_theory / `theorem_misuse`

**Problem:** Compute 2^14 modulo 15.

**Student solution:**

1. Fermat's little theorem gives 2^14 congruent to 1 modulo 15.
2. Therefore the remainder is 1.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_5a3425f265ac842a`), “1. Fermat's little theorem gives 2^14 congruent to 1 modulo 15.” — Fermat's little theorem in this form requires a prime modulus.

**Proposed issue category:** `theorem.misused`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposed first issue is invoking Fermat's theorem with a composite modulus.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.38.lhopital-non-indeterminate

**Domain / category:** calculus / `theorem_misuse`

**Problem:** Evaluate lim as x approaches 0 of (1+x)/x.

**Student solution:**

1. Differentiate numerator and denominator by L'Hopital's rule.
2. The resulting limit is 1/1=1.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_5a00820f900d25da`), “1. Differentiate numerator and denominator by L'Hopital's rule.” — The original quotient is not of form 0/0 or infinity/infinity.

**Proposed issue category:** `theorem.misused`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes use of L'Hopital outside its indeterminate-form hypotheses.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.39.exclusive-implies-independent

**Domain / category:** probability / `concept_misunderstood`

**Problem:** Events A and B are mutually exclusive and each has positive probability. Are they independent?

**Student solution:**

1. Mutually exclusive events do not affect each other.
2. Therefore A and B are independent.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_6498ce8bc7b425de`), “1. Mutually exclusive events do not affect each other.” — Positive-probability mutually exclusive events cannot be independent.

**Proposed issue category:** `concept.misunderstood`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies confusion between mutual exclusivity and independence.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.40.zero-derivative-at-point

**Domain / category:** calculus, functions / `concept_misunderstood`

**Problem:** If a differentiable function has f'(2)=0, must it be constant?

**Student solution:**

1. A derivative of zero means the function does not change.
2. Since f'(2)=0, f is constant everywhere.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_5e32a63b6d412071`), “2. Since f'(2)=0, f is constant everywhere.” — Zero derivative at a single point does not imply zero derivative everywhere.

**Proposed issue category:** `concept.misunderstood`

**Proposed reusable prefix end:** Step 1 (`step_74d25c1c53494810`): “1. A derivative of zero means the function does not change.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal localizes the leap from one derivative value to a global property.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.41.repeated-letter-permutations

**Domain / category:** combinatorics / `concept_misunderstood`

**Problem:** How many distinct arrangements of the letters in LEVEL are there?

**Student solution:**

1. There are five letter positions.
2. The number of arrangements is 5!=120.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 2 (`step_6fb17e38c1275b08`), “2. The number of arrangements is 5!=120.” — Permuting identical Ls or identical Es does not create a new arrangement.

**Proposed issue category:** `concept.misunderstood`

**Proposed reusable prefix end:** Step 1 (`step_1e4d8f26cf6f79cc`): “1. There are five letter positions.”

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The proposal identifies failure to account for indistinguishable repeated letters.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.42.zero-division-right-root

**Domain / category:** algebra / `correct_final_invalid_reasoning`

**Problem:** Solve (x-2)^2=0.

**Student solution:**

1. Divide both sides by x-2 to obtain x-2=0.
2. Hence x=2.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_463efe2d17b17076`), “1. Divide both sides by x-2 to obtain x-2=0.” — Division by x-2 is invalid at the solution where x-2=0.

**Proposed issue category:** `reasoning.invalid_inference`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** Although the final root is right, the proposal flags division by an expression not known to be nonzero.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.43.triangle-area-lucky

**Domain / category:** geometry / `correct_final_invalid_reasoning`

**Problem:** Find the area of a triangle with base 8 and height 5.

**Student solution:**

1. A triangle's area equals base times height.
2. Compute 8 times 5 = 40.
3. Divide by 2 because 40 is even, giving area 20.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_d7fed1d9a258ca1b`), “1. A triangle's area equals base times height.” — Base times height is the parallelogram area, not the triangle formula.

**Proposed issue category:** `concept.misunderstood`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The numerical result is right, but the proposal localizes the false area formula before the lucky correction.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.44.gcd-lucky-claim

**Domain / category:** number_theory / `correct_final_invalid_reasoning`

**Problem:** Find gcd(35,64).

**Student solution:**

1. Both 35 and 64 are prime.
2. Distinct primes have gcd 1.
3. Therefore gcd(35,64)=1.

**Proposed status:** `incorrect`

**Proposed first issue:** Step 1 (`step_68e4fc24d2ae4d87`), “1. Both 35 and 64 are prime.” — The number 35 is not prime.

**Proposed issue category:** `concept.misunderstood`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `LIGHT_HINT` / `1`

**Short rationale:** The final gcd is proposed as correct, but the first premise is false because 35 is composite.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.45.factor-cubic-prefix

**Domain / category:** algebra / `incomplete_valid`

**Problem:** Factor x^3-4x completely.

**Student solution:**

1. Factor out x to get x(x^2-4).
2. Recognize x^2-4 as a difference of squares.

**Proposed status:** `incomplete`

**Proposed first issue:** Completion gap after step 2 (`step_298e9c09545888e4`): Apply the difference-of-squares factorization to the remaining quadratic.

**Proposed issue category:** `completion.incomplete`

**Proposed reusable prefix end:** Step 2 (`step_298e9c09545888e4`): “2. Recognize x^2-4 as a difference of squares.”

**Proposed tutor action / maximum reveal:** `ASK_STUDENT` / `0`

**Short rationale:** The proposal treats both written steps as valid progress with one factoring step remaining.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.46.sqrt-two-prefix

**Domain / category:** number_theory / `incomplete_valid`

**Problem:** Prove that sqrt(2) is irrational.

**Student solution:**

1. Assume sqrt(2)=p/q in lowest terms.
2. Squaring gives p^2=2q^2, so p is even.

**Proposed status:** `incomplete`

**Proposed first issue:** Completion gap after step 2 (`step_9d82bd18a213b53a`): Use the evenness of p to deduce evenness of q and reach the contradiction.

**Proposed issue category:** `completion.incomplete`

**Proposed reusable prefix end:** Step 2 (`step_9d82bd18a213b53a`): “2. Squaring gives p^2=2q^2, so p is even.”

**Proposed tutor action / maximum reveal:** `ASK_STUDENT` / `0`

**Short rationale:** The contradiction proof is proposed as valid so far but unfinished.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.47.circle-tangent-prefix

**Domain / category:** geometry / `incomplete_valid`

**Problem:** Find the tangent line to x^2+y^2=25 at (3,4).

**Student solution:**

1. The radius from (0,0) to (3,4) has slope 4/3.
2. A tangent at the point is perpendicular to that radius.

**Proposed status:** `incomplete`

**Proposed first issue:** Completion gap after step 2 (`step_3c80a8bf30dfab4a`): Determine the perpendicular slope and write the line through the given point.

**Proposed issue category:** `completion.incomplete`

**Proposed reusable prefix end:** Step 2 (`step_3c80a8bf30dfab4a`): “2. A tangent at the point is perpendicular to that radius.”

**Proposed tutor action / maximum reveal:** `ASK_STUDENT` / `0`

**Short rationale:** The geometric facts are proposed as correct, with the line equation still missing.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.48.binary-no-consecutive-prefix

**Domain / category:** combinatorics / `incomplete_valid`

**Problem:** Count length-n binary strings with no consecutive 1s.

**Student solution:**

1. Let a_n denote the desired count.
2. Strings ending in 0 contribute a_(n-1), and strings ending in 1 contribute a_(n-2).
3. Thus a_n=a_(n-1)+a_(n-2).

**Proposed status:** `incomplete`

**Proposed first issue:** Completion gap after step 3 (`step_6eadb4a2f6ecf9ba`): State base cases and solve or identify the recurrence.

**Proposed issue category:** `completion.incomplete`

**Proposed reusable prefix end:** Step 3 (`step_6eadb4a2f6ecf9ba`): “3. Thus a_n=a_(n-1)+a_(n-2).”

**Proposed tutor action / maximum reveal:** `ASK_STUDENT` / `0`

**Short rationale:** The recurrence is proposed as valid, while initial conditions and a final count are absent.

**Ambiguity note:** None proposed.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.49.diagram-dependent-angle

**Domain / category:** geometry / `ambiguous_difficult`

**Problem:** In the accompanying diagram, determine angle x.

**Student solution:**

The two marked arcs look equal, so I think x=40 degrees.

**Proposed status:** `indeterminate`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `VERIFY_STEP` / `0`

**Short rationale:** The proposal abstains because the referenced diagram and markings are unavailable.

**Ambiguity note:** A reviewer must decide whether the source package should include a diagram; without it, neither the claim nor localization is adjudicable.

**Disposition:** `PENDING`

**Reviewer notes:**

---

## pilot.50.custom-operation-notation

**Domain / category:** algebra, functions / `ambiguous_difficult`

**Problem:** For real a and b, simplify a star b star 2 under the operation defined earlier.

**Student solution:**

Using associativity, a star b star 2 = a star (b star 2), which simplifies to 2a+b.

**Proposed status:** `indeterminate`

**Proposed first issue:** None

**Proposed issue category:** `none`

**Proposed reusable prefix end:** None

**Proposed tutor action / maximum reveal:** `VERIFY_STEP` / `0`

**Short rationale:** The proposal abstains because the operation definition and grouping convention are missing.

**Ambiguity note:** A reviewer needs the omitted definition of star and its associativity status before evaluating the step.

**Disposition:** `PENDING`

**Reviewer notes:**

---
