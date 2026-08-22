"""Deterministic source definitions for the provisional diagnosis pilot.

The annotations are Codex proposals designed to reduce, not replace, expert
review.  They must not be described as mathematical ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from math_feedback_ai.domain import (
    CompletionGap,
    DiagnosisV1,
    Issue,
    IssueCode,
    OverallStatus,
    Problem,
    ReferenceRelation,
    ReferenceSolution,
    RevealLevel,
    StepAssessment,
    StepStatus,
    TutorAction,
    TutorDecision,
)
from math_feedback_ai.evaluation.benchmark import BenchmarkFormatError
from math_feedback_ai.evaluation.pilot import (
    DiagnosisPilotCaseV1,
    HumanReviewStatus,
    PilotProvenance,
    PilotValidationReport,
    validate_pilot_cases,
)
from math_feedback_ai.parsing import parse_student_attempt

PILOT_VERSION = "diagnosis_pilot_v1"
EXPECTED_PILOT_CASE_COUNT = 50
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PILOT_PATH = REPOSITORY_ROOT / "evaluation" / "benchmarks" / f"{PILOT_VERSION}.jsonl"
DEFAULT_REVIEW_PATH = REPOSITORY_ROOT / "evaluation" / "review" / f"{PILOT_VERSION}_review.md"


@dataclass(frozen=True, slots=True)
class _PilotSpec:
    slug: str
    category: str
    domains: tuple[str, ...]
    characteristics: tuple[str, ...]
    problem: str
    student_solution: str
    reference_solution: str | None
    status: OverallStatus
    explanation: str
    issue_code: IssueCode | None = None
    issue_step: int | None = None
    issue_explanation: str | None = None
    completion_gap: str | None = None
    reference_relation: ReferenceRelation = ReferenceRelation.NOT_USED
    ambiguity_notes: str | None = None


def _specs() -> tuple[_PilotSpec, ...]:
    return (
        # Fully correct, conventional solutions (7).
        _PilotSpec(
            "linear-system-elimination",
            "fully_correct_conventional",
            ("algebra",),
            ("procedural",),
            "Solve the system 2x + y = 7 and x - y = 2.",
            "1. Add the equations: 3x = 9.\n2. Hence x = 3.\n3. Substituting gives y = 1.",
            "Adding the equations eliminates y; then substitute x = 3.",
            OverallStatus.FULLY_CORRECT,
            "The proposed label accepts each reversible algebraic step.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        _PilotSpec(
            "divisible-by-six",
            "fully_correct_conventional",
            ("number_theory",),
            ("proof",),
            "Prove that n(n + 1)(n + 2) is divisible by 6 for every integer n.",
            "1. Among three consecutive integers, one is divisible by 3.\n2. At least one of them is even.\n3. Their product therefore has factors 2 and 3, so it is divisible by 6.",
            "Use residues modulo 2 and 3 among three consecutive integers.",
            OverallStatus.FULLY_CORRECT,
            "The proposal treats the divisibility argument as complete for integers.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        _PilotSpec(
            "isosceles-base-angles",
            "fully_correct_conventional",
            ("geometry",),
            ("proof",),
            "In triangle ABC, AB = AC. Prove that angles B and C are equal.",
            "1. Compare triangles ABC and ACB.\n2. AB = AC, AC = AB, and BC = CB.\n3. The triangles are congruent by SSS, so corresponding base angles B and C are equal.",
            "Use the standard SSS self-congruence argument with the vertices swapped.",
            OverallStatus.FULLY_CORRECT,
            "The correspondence and congruence justification are proposed as valid.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        _PilotSpec(
            "choose-committee",
            "fully_correct_conventional",
            ("combinatorics",),
            ("counting",),
            "How many 3-person committees can be selected from 8 people?",
            "1. Order does not matter.\n2. The count is C(8,3) = 8!/(3!5!) = 56.",
            "Apply the binomial coefficient for choosing 3 of 8 people.",
            OverallStatus.FULLY_CORRECT,
            "The proposal accepts the unordered-selection model and computation.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        _PilotSpec(
            "square-nonnegative",
            "fully_correct_conventional",
            ("inequalities",),
            ("proof",),
            "Prove that a^2 + b^2 is at least 2ab for all real a,b.",
            "1. For real a,b, (a-b)^2 is nonnegative.\n2. Expanding gives a^2 - 2ab + b^2 >= 0.\n3. Rearranging yields a^2 + b^2 >= 2ab.",
            "Expand the nonnegative square (a-b)^2.",
            OverallStatus.FULLY_CORRECT,
            "Every implication is proposed as valid over the stated real domain.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        _PilotSpec(
            "compose-functions",
            "fully_correct_conventional",
            ("functions",),
            ("procedural",),
            "Let f(x)=2x+1 and g(x)=x^2. Find (g composed with f)(3).",
            "1. f(3) = 7.\n2. Then g(f(3)) = g(7) = 49.",
            "Evaluate the inner function first and square the result.",
            OverallStatus.FULLY_CORRECT,
            "The proposed label follows the definition of composition.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        _PilotSpec(
            "at-least-one-six",
            "fully_correct_conventional",
            ("probability",),
            ("finite_probability",),
            "Two fair dice are rolled. Find the probability that at least one die is a six.",
            "1. The probability of no six is (5/6)^2 = 25/36.\n2. Therefore the probability of at least one six is 1 - 25/36 = 11/36.",
            "Use the complement event that neither die is six.",
            OverallStatus.FULLY_CORRECT,
            "The complement calculation is proposed as correct under independence.",
            reference_relation=ReferenceRelation.SAME_METHOD,
        ),
        # Fully correct alternatives that materially differ from references (6).
        _PilotSpec(
            "quadratic-vieta",
            "fully_correct_unconventional",
            ("algebra",),
            ("alternative_valid",),
            "Solve x^2 - 7x + 12 = 0.",
            "1. Two numbers with sum 7 and product 12 are 3 and 4.\n2. Therefore the roots are x=3 and x=4.",
            "Factor the polynomial as (x-3)(x-4), then use the zero-product property.",
            OverallStatus.FULLY_CORRECT,
            "The proposal recognizes a Vieta-style root identification without requiring displayed factorization.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "odd-square-mod-eight",
            "fully_correct_unconventional",
            ("number_theory",),
            ("alternative_valid", "proof"),
            "Prove that the square of every odd integer is congruent to 1 modulo 8.",
            "1. Write an odd integer as 2k+1.\n2. Its square is 4k(k+1)+1.\n3. One of k,k+1 is even, so 4k(k+1) is divisible by 8; the remainder is 1.",
            "Check the four odd residue classes modulo 8 and square each one.",
            OverallStatus.FULLY_CORRECT,
            "The algebraic parity proof is proposed as a valid alternative to residue enumeration.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "triangle-area-coordinates",
            "fully_correct_unconventional",
            ("geometry",),
            ("alternative_valid",),
            "Find the area of the triangle with vertices (0,0), (4,0), and (1,3).",
            "1. The determinant of vectors (4,0) and (1,3) is 12.\n2. Half its absolute value is 6, so the area is 6.",
            "Use base 4 on the x-axis and perpendicular height 3.",
            OverallStatus.FULLY_CORRECT,
            "The determinant method is proposed as equivalent to base times height.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "strings-with-a",
            "fully_correct_unconventional",
            ("combinatorics",),
            ("alternative_valid",),
            "How many length-5 binary strings contain at least one 1?",
            "1. There are 2^5 total binary strings.\n2. Only 00000 contains no 1.\n3. Thus the answer is 32-1=31.",
            "Sum C(5,k) for k from 1 through 5.",
            OverallStatus.FULLY_CORRECT,
            "The complement count is proposed as a valid alternative to binomial summation.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "positive-product-inequality",
            "fully_correct_unconventional",
            ("inequalities",),
            ("alternative_valid", "proof"),
            "For positive x, prove x + 1/x is at least 2.",
            "1. Since x>0, (x-1)^2/x is nonnegative.\n2. Expanding that quotient gives x - 2 + 1/x >= 0.\n3. Hence x + 1/x >= 2.",
            "Apply AM-GM to the positive numbers x and 1/x.",
            OverallStatus.FULLY_CORRECT,
            "The nonnegative-square proof is proposed as valid and domain-aware.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "inverse-by-reflection",
            "fully_correct_unconventional",
            ("functions",),
            ("alternative_valid",),
            "Find the inverse of f(x)=3x-5.",
            "1. Reflect the line y=3x-5 across y=x, giving x=3y-5.\n2. Solving this relation for y gives y=(x+5)/3.\n3. Therefore f inverse of x is (x+5)/3.",
            "Set y=3x-5, interchange x and y, then isolate y.",
            OverallStatus.FULLY_CORRECT,
            "The geometric reflection framing is proposed as equivalent to the standard symbolic method.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        # Correct but inefficient (4).
        _PilotSpec(
            "expand-fourth-power",
            "correct_but_inefficient",
            ("algebra",),
            ("inefficient",),
            "Expand (x+1)^4.",
            "1. Multiply (x+1)(x+1) to get x^2+2x+1.\n2. Multiply that polynomial by x+1 to get x^3+3x^2+3x+1.\n3. Multiply once more by x+1 to obtain x^4+4x^3+6x^2+4x+1.",
            "Read the coefficients from the fourth row of Pascal's triangle.",
            OverallStatus.CORRECT_BUT_INEFFICIENT,
            "The repeated multiplication is proposed as correct though substantially longer than the binomial theorem.",
            IssueCode.RELEVANCE_IRRELEVANT,
            0,
            "The method is valid but unnecessarily laborious for this expansion.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "gcd-by-divisor-lists",
            "correct_but_inefficient",
            ("number_theory",),
            ("inefficient",),
            "Find gcd(84,126).",
            "1. List divisors of 84: 1,2,3,4,6,7,12,14,21,28,42,84.\n2. List divisors of 126: 1,2,3,6,7,9,14,18,21,42,63,126.\n3. The greatest common divisor is 42.",
            "Use the Euclidean algorithm: 126 mod 84 and then 84 mod 42.",
            OverallStatus.CORRECT_BUT_INEFFICIENT,
            "The exhaustive lists and selected gcd are proposed as correct but inefficient.",
            IssueCode.RELEVANCE_IRRELEVANT,
            0,
            "Listing every divisor is avoidable when the Euclidean algorithm is available.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "three-coins-enumeration",
            "correct_but_inefficient",
            ("probability",),
            ("inefficient",),
            "Three fair coins are tossed. Find the probability of exactly two heads.",
            "1. List HHH,HHT,HTH,HTT,THH,THT,TTH,TTT.\n2. HHT, HTH, and THH have exactly two heads.\n3. The probability is 3/8.",
            "Choose which two of the three tosses are heads, giving C(3,2)/8.",
            OverallStatus.CORRECT_BUT_INEFFICIENT,
            "The complete sample-space enumeration is proposed as correct but longer than needed.",
            IssueCode.RELEVANCE_IRRELEVANT,
            0,
            "Full enumeration is valid but inefficient for this small binomial count.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        _PilotSpec(
            "derivative-from-definition",
            "correct_but_inefficient",
            ("calculus", "functions"),
            ("inefficient",),
            "Differentiate f(x)=x^2+3x.",
            "1. Form ((x+h)^2+3(x+h)-x^2-3x)/h.\n2. Simplify to (2xh+h^2+3h)/h.\n3. For h nonzero this is 2x+h+3, whose limit is 2x+3.",
            "Differentiate term by term with the power and linearity rules.",
            OverallStatus.CORRECT_BUT_INEFFICIENT,
            "The limit-definition calculation is proposed as correct but more work than standard rules require.",
            IssueCode.RELEVANCE_IRRELEVANT,
            0,
            "Using first principles is valid but inefficient in this procedural context.",
            reference_relation=ReferenceRelation.ALTERNATIVE_VERIFIED,
        ),
        # Arithmetic slips (3).
        _PilotSpec(
            "discriminant-slip",
            "arithmetic_slip",
            ("algebra",),
            ("procedural",),
            "Solve 2x^2-5x-3=0 using the quadratic formula.",
            "1. Here a=2, b=-5, c=-3.\n2. The discriminant is 25-4(2)(-3)=25+24=47.\n3. Thus x=(5 plus or minus sqrt(47))/4.",
            "The discriminant is 49, after which the two rational roots follow.",
            OverallStatus.INCORRECT,
            "The proposal localizes the first arithmetic slip in evaluating the discriminant.",
            IssueCode.COMPUTATION_ARITHMETIC,
            1,
            "The sum 25+24 equals 49, not 47.",
        ),
        _PilotSpec(
            "trapezoid-product-slip",
            "arithmetic_slip",
            ("geometry",),
            ("procedural",),
            "A trapezoid has bases 7 and 11 and height 4. Find its area.",
            "1. Use A=(1/2)(b1+b2)h.\n2. A=(1/2)(18)(4)=32.\n3. So the area is 32 square units.",
            "Average the parallel sides and multiply by the height.",
            OverallStatus.INCORRECT,
            "The formula is proposed as correct, with the first error in multiplication.",
            IssueCode.COMPUTATION_ARITHMETIC,
            1,
            "Half of 18 times 4 is 36, not 32.",
        ),
        _PilotSpec(
            "sum-count-slip",
            "arithmetic_slip",
            ("combinatorics",),
            ("counting",),
            "How many integers from 1 through 100 are divisible by 4 or by 6?",
            "1. There are 25 multiples of 4 and 16 multiples of 6.\n2. There are 8 multiples of 12.\n3. Inclusion-exclusion gives 25+16-8=32.",
            "Apply inclusion-exclusion to the multiples of 4 and 6.",
            OverallStatus.INCORRECT,
            "The counting setup is proposed as valid and the last numerical sum as the first error.",
            IssueCode.COMPUTATION_ARITHMETIC,
            2,
            "The displayed expression evaluates to 33, not 32.",
        ),
        # Algebraic manipulation errors (3).
        _PilotSpec(
            "power-of-power",
            "algebraic_error",
            ("algebra",),
            ("procedural",),
            "Simplify (x^3)^4.",
            "1. A power raised to a power means add the exponents.\n2. Therefore (x^3)^4=x^7.",
            "Multiply the exponents in the power-of-a-power rule.",
            OverallStatus.INCORRECT,
            "The proposal identifies the misstated exponent law before the resulting expression.",
            IssueCode.COMPUTATION_ALGEBRAIC,
            0,
            "The exponents should be multiplied, not added.",
        ),
        _PilotSpec(
            "transpose-sign",
            "algebraic_error",
            ("algebra",),
            ("procedural",),
            "Solve 5-2x=13.",
            "1. Subtract 5 from both sides: -2x=8.\n2. Divide by -2: x=4.",
            "After obtaining -2x=8, divide by -2 and retain the sign.",
            OverallStatus.INCORRECT,
            "The proposal treats the first line as valid and localizes the sign error at division.",
            IssueCode.COMPUTATION_ALGEBRAIC,
            1,
            "Dividing 8 by -2 gives -4, not 4.",
        ),
        _PilotSpec(
            "add-rational-functions",
            "algebraic_error",
            ("algebra", "functions"),
            ("procedural",),
            "Simplify 1/x + 1/(x+1).",
            "1. Add numerators and denominators separately.\n2. This gives 2/(2x+1).",
            "Use the common denominator x(x+1), with the original domain restrictions.",
            OverallStatus.INCORRECT,
            "The proposal identifies the invalid addition rule at its statement.",
            IssueCode.COMPUTATION_ALGEBRAIC,
            0,
            "Fractions cannot be added by adding their denominators.",
        ),
        # Invalid cancellation or division (3).
        _PilotSpec(
            "cancel-across-sum",
            "invalid_cancellation_division",
            ("algebra",),
            ("procedural",),
            "Simplify (x+6)/x for x nonzero.",
            "1. Cancel x from the numerator and denominator.\n2. The expression becomes 1+6=7.",
            "Split the quotient into x/x plus 6/x.",
            OverallStatus.INCORRECT,
            "The proposal localizes cancellation across addition as the first invalid move.",
            IssueCode.REASONING_INVALID_INFERENCE,
            0,
            "Cancellation applies to common factors, not one term of a sum.",
        ),
        _PilotSpec(
            "divide-by-variable",
            "invalid_cancellation_division",
            ("algebra",),
            ("missing_case",),
            "Solve x(x-5)=0 over the real numbers.",
            "1. Divide both sides by x to get x-5=0.\n2. Therefore x=5 is the solution.",
            "Use the zero-product property to retain both possible factors.",
            OverallStatus.INCORRECT,
            "The proposal flags division by a potentially zero quantity and the lost solution.",
            IssueCode.CONDITION_MISSING,
            0,
            "Dividing by x assumes x is nonzero and discards the solution x=0.",
        ),
        _PilotSpec(
            "cancel-square-roots",
            "invalid_cancellation_division",
            ("algebra",),
            ("conceptual",),
            "Simplify sqrt(a+b) when a,b are nonnegative.",
            "1. Distribute the square root over addition.\n2. So sqrt(a+b)=sqrt(a)+sqrt(b).",
            "In general there is no additive square-root rule; the expression stays as written.",
            OverallStatus.INCORRECT,
            "The proposal treats the distribution claim as the first invalid identity.",
            IssueCode.REASONING_INVALID_INFERENCE,
            0,
            "Square roots do not generally distribute over addition.",
        ),
        # Logical gaps (3).
        _PilotSpec(
            "even-square-converse",
            "logical_gap",
            ("number_theory",),
            ("proof",),
            "Prove that if n^2 is even, then n is even.",
            "1. If n is even, then n^2 is even.\n2. Therefore, if n^2 is even, n must be even.",
            "Prove the contrapositive: an odd n has an odd square.",
            OverallStatus.INCORRECT,
            "The proposal identifies reversal of the implication rather than disputing the conclusion.",
            IssueCode.REASONING_LOGICAL_GAP,
            1,
            "The converse does not follow from the implication proved in step 1.",
        ),
        _PilotSpec(
            "induction-no-step",
            "logical_gap",
            ("number_theory",),
            ("proof",),
            "Prove by induction that 1+3+...+(2n-1)=n^2 for positive integers n.",
            "1. For n=1, both sides equal 1.\n2. Assume the formula is true for n=k.\n3. Therefore it is true for every positive integer n.",
            "After the hypothesis, add 2k+1 and derive (k+1)^2.",
            OverallStatus.INCORRECT,
            "The proposal localizes the omitted induction step before the universal conclusion.",
            IssueCode.REASONING_LOGICAL_GAP,
            2,
            "No argument establishes the case k+1 from the induction hypothesis.",
        ),
        _PilotSpec(
            "union-count-overlap",
            "logical_gap",
            ("combinatorics",),
            ("counting",),
            "In a class, 18 study French, 14 study Spanish, and 6 study both. How many study at least one?",
            "1. Add the French and Spanish counts: 18+14=32.\n2. Therefore 32 students study at least one language.",
            "Subtract the six students counted in both groups once.",
            OverallStatus.INCORRECT,
            "The proposal marks the failure to account for overlap at the addition step.",
            IssueCode.REASONING_LOGICAL_GAP,
            0,
            "Students in both groups are counted twice in the sum.",
        ),
        # Unsupported claims (3).
        _PilotSpec(
            "parallel-from-picture",
            "unsupported_claim",
            ("geometry",),
            ("proof",),
            "Given a quadrilateral ABCD with no additional assumptions, prove or disprove that AB is parallel to CD.",
            "1. In the sketch, AB and CD look horizontal.\n2. Hence AB is parallel to CD.",
            "A diagram alone cannot establish parallelism; coordinates or marked conditions are needed.",
            OverallStatus.INCORRECT,
            "The proposal treats visual appearance as unsupported evidence.",
            IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,
            0,
            "An unmarked sketch is not proof that the segments are parallel.",
        ),
        _PilotSpec(
            "large-number-prime",
            "unsupported_claim",
            ("number_theory",),
            ("proof",),
            "Determine whether 221 is prime.",
            "1. 221 is not divisible by 2, 3, or 5.\n2. Therefore 221 is prime.",
            "Test prime divisors through sqrt(221); one of them yields a factorization.",
            OverallStatus.INCORRECT,
            "The proposal identifies the incomplete divisibility check as insufficient support.",
            IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,
            1,
            "Checking only 2, 3, and 5 does not eliminate all possible prime divisors.",
        ),
        _PilotSpec(
            "monotone-two-points",
            "unsupported_claim",
            ("functions",),
            ("conceptual",),
            "A function satisfies f(0)=1 and f(1)=3. Must it be increasing on [0,1]?",
            "1. Since f(1)>f(0), the output rose as x rose.\n2. Therefore f is increasing on the whole interval.",
            "Two endpoint values do not control behavior between them; give a counterexample.",
            OverallStatus.INCORRECT,
            "The proposal rejects inference of an interval property from only endpoints.",
            IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,
            1,
            "Endpoint values alone do not establish monotonicity at every pair of points.",
        ),
        # Missing conditions (3).
        _PilotSpec(
            "squared-equation-check",
            "missing_condition",
            ("algebra",),
            ("procedural",),
            "Solve sqrt(x+2)=x over the reals.",
            "1. Square both sides: x+2=x^2.\n2. Rearrange: x^2-x-2=0, so x=2 or x=-1.\n3. Thus both 2 and -1 solve the original equation.",
            "Solve the quadratic and substitute candidates into the original radical equation.",
            OverallStatus.INCORRECT,
            "The proposal localizes failure to check candidates after a non-reversible operation.",
            IssueCode.CONDITION_MISSING,
            2,
            "Squaring can introduce extraneous roots, so each candidate must be checked.",
        ),
        _PilotSpec(
            "multiply-inequality-sign",
            "missing_condition",
            ("inequalities",),
            ("procedural",),
            "If ax<ay and a is a nonzero real number, what can be concluded about x and y?",
            "1. Divide both sides by a.\n2. Therefore x<y.",
            "Split into the cases a positive and a negative, reversing the inequality in the latter.",
            OverallStatus.INCORRECT,
            "The proposal identifies the unstated sign condition on the divisor.",
            IssueCode.CONDITION_MISSING,
            0,
            "The inequality direction depends on whether a is positive or negative.",
        ),
        _PilotSpec(
            "inverse-without-injective",
            "missing_condition",
            ("functions",),
            ("conceptual",),
            "Let f be a function from R to R. Does f necessarily have an inverse function?",
            "1. Swap x and y in y=f(x).\n2. This always defines f inverse, so every function has an inverse.",
            "An inverse function requires suitable injectivity and surjectivity conditions.",
            OverallStatus.INCORRECT,
            "The proposal localizes the missing one-to-one condition in the universal claim.",
            IssueCode.CONDITION_MISSING,
            1,
            "Not every function is injective, so swapping symbols need not define a function inverse.",
        ),
        # Theorem misuse (3).
        _PilotSpec(
            "pythagorean-on-any-triangle",
            "theorem_misuse",
            ("geometry",),
            ("conceptual",),
            "A triangle has side lengths 4,5,6. Find its area.",
            "1. Treat 4 and 5 as perpendicular legs.\n2. By the Pythagorean theorem, the area is (1/2)(4)(5)=10.",
            "Use Heron's formula because the supplied side lengths do not form a right triangle.",
            OverallStatus.INCORRECT,
            "The proposal flags application of a right-triangle formula without a right-angle condition.",
            IssueCode.THEOREM_MISUSED,
            0,
            "No right angle is given, and 4^2+5^2 is not 6^2.",
        ),
        _PilotSpec(
            "fermat-composite-modulus",
            "theorem_misuse",
            ("number_theory",),
            ("proof",),
            "Compute 2^14 modulo 15.",
            "1. Fermat's little theorem gives 2^14 congruent to 1 modulo 15.\n2. Therefore the remainder is 1.",
            "Use repeated squaring or Euler's theorem with the required coprimality conditions.",
            OverallStatus.INCORRECT,
            "The proposed first issue is invoking Fermat's theorem with a composite modulus.",
            IssueCode.THEOREM_MISUSED,
            0,
            "Fermat's little theorem in this form requires a prime modulus.",
        ),
        _PilotSpec(
            "lhopital-non-indeterminate",
            "theorem_misuse",
            ("calculus",),
            ("procedural",),
            "Evaluate lim as x approaches 0 of (1+x)/x.",
            "1. Differentiate numerator and denominator by L'Hopital's rule.\n2. The resulting limit is 1/1=1.",
            "Recognize that the quotient is not an indeterminate 0/0 form and diverges by side.",
            OverallStatus.INCORRECT,
            "The proposal localizes use of L'Hopital outside its indeterminate-form hypotheses.",
            IssueCode.THEOREM_MISUSED,
            0,
            "The original quotient is not of form 0/0 or infinity/infinity.",
        ),
        # Misunderstood concepts (3).
        _PilotSpec(
            "exclusive-implies-independent",
            "concept_misunderstood",
            ("probability",),
            ("conceptual",),
            "Events A and B are mutually exclusive and each has positive probability. Are they independent?",
            "1. Mutually exclusive events do not affect each other.\n2. Therefore A and B are independent.",
            "Compare P(A intersection B)=0 with the positive product P(A)P(B).",
            OverallStatus.INCORRECT,
            "The proposal identifies confusion between mutual exclusivity and independence.",
            IssueCode.CONCEPT_MISUNDERSTOOD,
            0,
            "Positive-probability mutually exclusive events cannot be independent.",
        ),
        _PilotSpec(
            "zero-derivative-at-point",
            "concept_misunderstood",
            ("calculus", "functions"),
            ("conceptual",),
            "If a differentiable function has f'(2)=0, must it be constant?",
            "1. A derivative of zero means the function does not change.\n2. Since f'(2)=0, f is constant everywhere.",
            "A derivative at one point is local information; give a nonconstant function with a stationary point.",
            OverallStatus.INCORRECT,
            "The proposal localizes the leap from one derivative value to a global property.",
            IssueCode.CONCEPT_MISUNDERSTOOD,
            1,
            "Zero derivative at a single point does not imply zero derivative everywhere.",
        ),
        _PilotSpec(
            "repeated-letter-permutations",
            "concept_misunderstood",
            ("combinatorics",),
            ("counting",),
            "How many distinct arrangements of the letters in LEVEL are there?",
            "1. There are five letter positions.\n2. The number of arrangements is 5!=120.",
            "Divide 5! by 2! for the repeated L and by 2! for the repeated E.",
            OverallStatus.INCORRECT,
            "The proposal identifies failure to account for indistinguishable repeated letters.",
            IssueCode.CONCEPT_MISUNDERSTOOD,
            1,
            "Permuting identical Ls or identical Es does not create a new arrangement.",
        ),
        # Correct final answer reached through invalid reasoning (3).
        _PilotSpec(
            "zero-division-right-root",
            "correct_final_invalid_reasoning",
            ("algebra",),
            ("right_answer_wrong_reason",),
            "Solve (x-2)^2=0.",
            "1. Divide both sides by x-2 to obtain x-2=0.\n2. Hence x=2.",
            "Use that a real square equals zero exactly when its base equals zero.",
            OverallStatus.INCORRECT,
            "Although the final root is right, the proposal flags division by an expression not known to be nonzero.",
            IssueCode.REASONING_INVALID_INFERENCE,
            0,
            "Division by x-2 is invalid at the solution where x-2=0.",
        ),
        _PilotSpec(
            "triangle-area-lucky",
            "correct_final_invalid_reasoning",
            ("geometry",),
            ("right_answer_wrong_reason",),
            "Find the area of a triangle with base 8 and height 5.",
            "1. A triangle's area equals base times height.\n2. Compute 8 times 5 = 40.\n3. Divide by 2 because 40 is even, giving area 20.",
            "Apply one half times base times perpendicular height directly.",
            OverallStatus.INCORRECT,
            "The numerical result is right, but the proposal localizes the false area formula before the lucky correction.",
            IssueCode.CONCEPT_MISUNDERSTOOD,
            0,
            "Base times height is the parallelogram area, not the triangle formula.",
        ),
        _PilotSpec(
            "gcd-lucky-claim",
            "correct_final_invalid_reasoning",
            ("number_theory",),
            ("right_answer_wrong_reason",),
            "Find gcd(35,64).",
            "1. Both 35 and 64 are prime.\n2. Distinct primes have gcd 1.\n3. Therefore gcd(35,64)=1.",
            "Use the Euclidean algorithm or note that neither 5 nor 7 divides 64.",
            OverallStatus.INCORRECT,
            "The final gcd is proposed as correct, but the first premise is false because 35 is composite.",
            IssueCode.CONCEPT_MISUNDERSTOOD,
            0,
            "The number 35 is not prime.",
        ),
        # Incomplete but valid progress (4).
        _PilotSpec(
            "factor-cubic-prefix",
            "incomplete_valid",
            ("algebra",),
            ("incomplete",),
            "Factor x^3-4x completely.",
            "1. Factor out x to get x(x^2-4).\n2. Recognize x^2-4 as a difference of squares.",
            "Continue to x(x-2)(x+2).",
            OverallStatus.INCOMPLETE,
            "The proposal treats both written steps as valid progress with one factoring step remaining.",
            completion_gap="Apply the difference-of-squares factorization to the remaining quadratic.",
        ),
        _PilotSpec(
            "sqrt-two-prefix",
            "incomplete_valid",
            ("number_theory",),
            ("incomplete", "proof"),
            "Prove that sqrt(2) is irrational.",
            "1. Assume sqrt(2)=p/q in lowest terms.\n2. Squaring gives p^2=2q^2, so p is even.",
            "Write p=2k, infer q is even, and contradict lowest terms.",
            OverallStatus.INCOMPLETE,
            "The contradiction proof is proposed as valid so far but unfinished.",
            completion_gap="Use the evenness of p to deduce evenness of q and reach the contradiction.",
        ),
        _PilotSpec(
            "circle-tangent-prefix",
            "incomplete_valid",
            ("geometry",),
            ("incomplete",),
            "Find the tangent line to x^2+y^2=25 at (3,4).",
            "1. The radius from (0,0) to (3,4) has slope 4/3.\n2. A tangent at the point is perpendicular to that radius.",
            "Use tangent slope -3/4 and the point-slope equation through (3,4).",
            OverallStatus.INCOMPLETE,
            "The geometric facts are proposed as correct, with the line equation still missing.",
            completion_gap="Determine the perpendicular slope and write the line through the given point.",
        ),
        _PilotSpec(
            "binary-no-consecutive-prefix",
            "incomplete_valid",
            ("combinatorics",),
            ("incomplete",),
            "Count length-n binary strings with no consecutive 1s.",
            "1. Let a_n denote the desired count.\n2. Strings ending in 0 contribute a_(n-1), and strings ending in 1 contribute a_(n-2).\n3. Thus a_n=a_(n-1)+a_(n-2).",
            "Supply the initial values and, if requested, identify the corresponding Fibonacci term.",
            OverallStatus.INCOMPLETE,
            "The recurrence is proposed as valid, while initial conditions and a final count are absent.",
            completion_gap="State base cases and solve or identify the recurrence.",
        ),
        # Genuinely difficult/ambiguous inputs for human adjudication (2).
        _PilotSpec(
            "diagram-dependent-angle",
            "ambiguous_difficult",
            ("geometry",),
            ("ambiguous", "missing_context"),
            "In the accompanying diagram, determine angle x.",
            "The two marked arcs look equal, so I think x=40 degrees.",
            None,
            OverallStatus.INDETERMINATE,
            "The proposal abstains because the referenced diagram and markings are unavailable.",
            ambiguity_notes="A reviewer must decide whether the source package should include a diagram; without it, neither the claim nor localization is adjudicable.",
        ),
        _PilotSpec(
            "custom-operation-notation",
            "ambiguous_difficult",
            ("algebra", "functions"),
            ("ambiguous", "undefined_notation"),
            "For real a and b, simplify a star b star 2 under the operation defined earlier.",
            "Using associativity, a star b star 2 = a star (b star 2), which simplifies to 2a+b.",
            None,
            OverallStatus.INDETERMINATE,
            "The proposal abstains because the operation definition and grouping convention are missing.",
            ambiguity_notes="A reviewer needs the omitted definition of star and its associativity status before evaluating the step.",
        ),
    )


def _build_diagnosis(spec: _PilotSpec, attempt_id: str, step_ids: tuple[str, ...]) -> DiagnosisV1:
    if spec.status is OverallStatus.INDETERMINATE:
        assessments = tuple(
            StepAssessment(
                step_id=step_id,
                status=StepStatus.AMBIGUOUS,
                issue_codes=(IssueCode.UNKNOWN_INSUFFICIENT_CONFIDENCE,),
                explanation=spec.ambiguity_notes or "Human adjudication is required.",
                confidence=0.15,
            )
            for step_id in step_ids
        )
        return DiagnosisV1(
            attempt_id=attempt_id,
            overall_status=OverallStatus.INDETERMINATE,
            step_assessments=assessments,
            earlier_reasoning_usable=False,
            confidence=0.15,
            confidence_reasons=("Required context is absent from the case record.",),
        )

    if spec.status is OverallStatus.INCOMPLETE:
        assessments = tuple(
            StepAssessment(step_id=step_id, status=StepStatus.VALID, confidence=0.82)
            for step_id in step_ids
        )
        return DiagnosisV1(
            attempt_id=attempt_id,
            overall_status=spec.status,
            step_assessments=assessments,
            completion_gap=CompletionGap(
                after_step_id=step_ids[-1],
                description=spec.completion_gap or "The solution stops before completion.",
                confidence=0.82,
            ),
            reusable_prefix_end_step_id=step_ids[-1],
            earlier_reasoning_usable=True,
            reference_relation=spec.reference_relation,
            confidence=0.82,
            confidence_reasons=("Codex-proposed annotation; pending human review.",),
        )

    if spec.status is OverallStatus.FULLY_CORRECT:
        assessments = tuple(
            StepAssessment(step_id=step_id, status=StepStatus.VALID, confidence=0.82)
            for step_id in step_ids
        )
        return DiagnosisV1(
            attempt_id=attempt_id,
            overall_status=spec.status,
            step_assessments=assessments,
            reusable_prefix_end_step_id=step_ids[-1],
            earlier_reasoning_usable=True,
            reference_relation=spec.reference_relation,
            confidence=0.82,
            confidence_reasons=("Codex-proposed annotation; pending human review.",),
        )

    if spec.issue_code is None or spec.issue_step is None or spec.issue_explanation is None:
        raise ValueError(f"case {spec.slug!r} needs a proposed issue")
    issue_step_id = step_ids[spec.issue_step]
    if spec.status is OverallStatus.CORRECT_BUT_INEFFICIENT:
        questionable_status = StepStatus.VALID
    else:
        questionable_status = (
            StepStatus.UNSUPPORTED
            if spec.issue_code
            in {
                IssueCode.JUSTIFICATION_UNJUSTIFIED_CLAIM,
                IssueCode.REASONING_LOGICAL_GAP,
                IssueCode.CONDITION_MISSING,
            }
            else StepStatus.INVALID
        )
    assessments = tuple(
        StepAssessment(
            step_id=step_id,
            status=questionable_status if index == spec.issue_step else StepStatus.VALID,
            issue_codes=(spec.issue_code,) if index == spec.issue_step else (),
            explanation=spec.issue_explanation if index == spec.issue_step else None,
            confidence=0.82,
        )
        for index, step_id in enumerate(step_ids)
    )
    prefix_id = step_ids[spec.issue_step - 1] if spec.issue_step > 0 else None
    return DiagnosisV1(
        attempt_id=attempt_id,
        overall_status=spec.status,
        step_assessments=assessments,
        first_issue=Issue(
            step_id=issue_step_id,
            code=spec.issue_code,
            explanation=spec.issue_explanation,
            confidence=0.82,
        ),
        reusable_prefix_end_step_id=prefix_id,
        earlier_reasoning_usable=prefix_id is not None
        or spec.status is OverallStatus.CORRECT_BUT_INEFFICIENT,
        reference_relation=spec.reference_relation,
        confidence=0.82,
        confidence_reasons=("Codex-proposed annotation; pending human review.",),
    )


def _decision(spec: _PilotSpec, diagnosis: DiagnosisV1) -> TutorDecision:
    if spec.status in {OverallStatus.FULLY_CORRECT, OverallStatus.CORRECT_BUT_INEFFICIENT}:
        action, level, target = TutorAction.WAIT, RevealLevel.NONE, None
    elif spec.status is OverallStatus.INCOMPLETE:
        action, level, target = TutorAction.ASK_STUDENT, RevealLevel.NONE, None
    elif spec.status is OverallStatus.INDETERMINATE:
        action = TutorAction.VERIFY_STEP
        level = RevealLevel.NONE
        target = diagnosis.step_assessments[0].step_id
    else:
        assert diagnosis.first_issue is not None
        action = TutorAction.LIGHT_HINT
        level = RevealLevel.LIGHT_DIRECTION
        target = diagnosis.first_issue.step_id
    return TutorDecision(
        action=action,
        max_reveal_level=level,
        target_step_id=target,
        rationale_code=f"pilot.proposed.{action.value.lower()}",
        rationale="Codex-proposed tutoring label; pending independent human review.",
    )


def build_diagnosis_pilot_cases() -> tuple[DiagnosisPilotCaseV1, ...]:
    """Build all proposed pilot cases in deterministic review order."""

    cases: list[DiagnosisPilotCaseV1] = []
    for index, spec in enumerate(_specs(), start=1):
        problem_id = f"pilot.problem.{spec.slug}"
        attempt = parse_student_attempt(
            problem_id,
            spec.student_solution,
            attempt_id=f"pilot.attempt.{spec.slug}",
        )
        references = (
            (
                ReferenceSolution(
                    reference_id=f"pilot.reference.{spec.slug}",
                    text=spec.reference_solution,
                    method_label="non-exhaustive proposed reference",
                ),
            )
            if spec.reference_solution is not None
            else ()
        )
        problem = Problem(
            problem_id=problem_id,
            statement=spec.problem,
            reference_solutions=references,
        )
        diagnosis = _build_diagnosis(
            spec,
            attempt.attempt_id,
            tuple(step.step_id for step in attempt.steps),
        )
        cases.append(
            DiagnosisPilotCaseV1(
                case_id=f"pilot.{index:02d}.{spec.slug}",
                category=spec.category,
                mathematical_domains=spec.domains,
                characteristics=spec.characteristics,
                provenance=PilotProvenance(
                    source=f"{PILOT_VERSION} deterministic Codex proposal",
                    notes=(
                        "Synthetic case and Codex-proposed annotations; not human-reviewed; "
                        "not mathematical ground truth and not valid for research claims."
                    ),
                ),
                human_review_status=HumanReviewStatus.PENDING,
                problem=problem,
                student_attempt=attempt,
                proposed_diagnosis=diagnosis,
                proposed_decision=_decision(spec, diagnosis),
                annotation_explanation=spec.explanation,
                ambiguity_notes=spec.ambiguity_notes,
            )
        )
    if len(cases) != EXPECTED_PILOT_CASE_COUNT:
        raise AssertionError(f"expected {EXPECTED_PILOT_CASE_COUNT} pilot cases, got {len(cases)}")
    return tuple(cases)


def render_diagnosis_pilot_jsonl(
    cases: tuple[DiagnosisPilotCaseV1, ...] | None = None,
) -> str:
    """Render canonical compact JSONL with one record per line."""

    selected = cases if cases is not None else build_diagnosis_pilot_cases()
    return "".join(f"{case.model_dump_json()}\n" for case in selected)


def render_diagnosis_pilot_review(
    cases: tuple[DiagnosisPilotCaseV1, ...] | None = None,
) -> str:
    """Render a compact human adjudication worksheet."""

    selected = cases if cases is not None else build_diagnosis_pilot_cases()
    lines = [
        "# Diagnosis Pilot v1 — Human Review Worksheet",
        "",
        "> All annotations below are Codex proposals with `human_review_status=pending`.",
        "> They are not mathematical ground truth or validated research labels.",
        "",
        "Review each case using `evaluation/annotation_guidelines.md`. Replace the disposition",
        "placeholder with exactly one of: **APPROVE**, **MODIFY**, **REJECT**, **AMBIGUOUS**.",
        "Record concise corrections under reviewer notes; do not edit the generated proposal",
        "in place until the review is reconciled into a separately versioned reviewed dataset.",
        "",
    ]
    for case in selected:
        diagnosis = case.proposed_diagnosis
        issue = diagnosis.first_issue
        gap = diagnosis.completion_gap
        steps = {step.step_id: step for step in case.student_attempt.steps}
        first_issue = "None"
        if issue is not None:
            issue_step = steps[issue.step_id]
            first_issue = (
                f"Step {issue_step.position + 1} (`{issue.step_id}`), "
                f"“{issue_step.text}” — {issue.explanation}"
            )
        if gap is not None:
            gap_location = "before any written step"
            if gap.after_step_id is not None:
                gap_step = steps[gap.after_step_id]
                gap_location = f"after step {gap_step.position + 1} (`{gap.after_step_id}`)"
            first_issue = f"Completion gap {gap_location}: {gap.description}"
        reusable = "None"
        if diagnosis.reusable_prefix_end_step_id is not None:
            prefix_step = steps[diagnosis.reusable_prefix_end_step_id]
            reusable = (
                f"Step {prefix_step.position + 1} "
                f"(`{diagnosis.reusable_prefix_end_step_id}`): “{prefix_step.text}”"
            )
        category = (
            issue.code.value
            if issue is not None
            else (IssueCode.COMPLETION_INCOMPLETE.value if gap is not None else "none")
        )
        lines.extend(
            [
                f"## {case.case_id}",
                "",
                f"**Domain / category:** {', '.join(case.mathematical_domains)} / `{case.category}`",
                "",
                f"**Problem:** {case.problem.statement}",
                "",
                "**Student solution:**",
                "",
                case.student_attempt.raw_text,
                "",
                f"**Proposed status:** `{diagnosis.overall_status.value}`",
                "",
                f"**Proposed first issue:** {first_issue}",
                "",
                f"**Proposed issue category:** `{category}`",
                "",
                f"**Proposed reusable prefix end:** {reusable}",
                "",
                (
                    "**Proposed tutor action / maximum reveal:** "
                    f"`{case.proposed_decision.action.value}` / "
                    f"`{int(case.proposed_decision.max_reveal_level)}`"
                ),
                "",
                f"**Short rationale:** {case.annotation_explanation}",
                "",
                f"**Ambiguity note:** {case.ambiguity_notes or 'None proposed.'}",
                "",
                "**Disposition:** `PENDING`",
                "",
                "**Reviewer notes:**",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def diagnosis_pilot_sha256() -> str:
    return sha256(render_diagnosis_pilot_jsonl().encode()).hexdigest()


def diagnosis_pilot_review_sha256() -> str:
    return sha256(render_diagnosis_pilot_review().encode()).hexdigest()


def load_diagnosis_pilot(path: Path | None = None) -> tuple[DiagnosisPilotCaseV1, ...]:
    """Load an explicit file or the repository/wheel provisional pilot.

    The deterministic source definitions provide an installed-wheel fallback;
    an explicitly requested missing path still fails closed.
    """

    selected_path = path if path is not None else DEFAULT_PILOT_PATH
    if path is None and not selected_path.is_file():
        return build_diagnosis_pilot_cases()

    try:
        lines = selected_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkFormatError(f"cannot read diagnosis pilot {selected_path}: {exc}") from exc
    cases: list[DiagnosisPilotCaseV1] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BenchmarkFormatError(f"blank pilot line at {line_number}")
        try:
            cases.append(DiagnosisPilotCaseV1.model_validate_json(line))
        except ValidationError as exc:
            raise BenchmarkFormatError(f"invalid pilot case at line {line_number}: {exc}") from exc
    if not cases:
        raise BenchmarkFormatError("diagnosis pilot contains no cases")
    report = validate_pilot_cases(cases)
    if not report.passed:
        summary = "; ".join(f"{issue.code}: {','.join(issue.case_ids)}" for issue in report.issues)
        raise BenchmarkFormatError(f"diagnosis pilot failed structural validation: {summary}")
    return tuple(cases)


def validate_built_diagnosis_pilot() -> PilotValidationReport:
    """Validate pilot structure and isolation from the development split."""

    from math_feedback_ai.evaluation.development_builder import build_development_examples

    return validate_pilot_cases(
        build_diagnosis_pilot_cases(),
        comparison_examples=build_development_examples(),
    )


def committed_diagnosis_pilot_is_current(
    jsonl_path: Path = DEFAULT_PILOT_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
) -> bool:
    try:
        jsonl = jsonl_path.read_text(encoding="utf-8")
        review = review_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return jsonl == render_diagnosis_pilot_jsonl() and review == render_diagnosis_pilot_review()


def write_diagnosis_pilot(
    jsonl_path: Path = DEFAULT_PILOT_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
) -> tuple[Path, Path]:
    """Write deterministic JSONL and its human review worksheet."""

    report = validate_built_diagnosis_pilot()
    if not report.passed:
        raise ValueError(f"refusing to write structurally invalid pilot: {report.issues}")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(render_diagnosis_pilot_jsonl(), encoding="utf-8", newline="\n")
    review_path.write_text(render_diagnosis_pilot_review(), encoding="utf-8", newline="\n")
    return jsonl_path, review_path


__all__ = [
    "DEFAULT_PILOT_PATH",
    "DEFAULT_REVIEW_PATH",
    "EXPECTED_PILOT_CASE_COUNT",
    "PILOT_VERSION",
    "build_diagnosis_pilot_cases",
    "committed_diagnosis_pilot_is_current",
    "diagnosis_pilot_review_sha256",
    "diagnosis_pilot_sha256",
    "load_diagnosis_pilot",
    "render_diagnosis_pilot_jsonl",
    "render_diagnosis_pilot_review",
    "validate_built_diagnosis_pilot",
    "write_diagnosis_pilot",
]
