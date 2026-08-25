"""Deterministic reconciliation of the independently reviewed diagnosis pilot."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from math_feedback_ai.domain import (
    DiagnosisV1,
    Issue,
    IssueCode,
    OverallStatus,
    RevealLevel,
    StepAssessment,
    StepStatus,
    TutorAction,
    TutorDecision,
)
from math_feedback_ai.evaluation.benchmark import BenchmarkFormatError
from math_feedback_ai.evaluation.diagnosis_pilot_builder import (
    EXPECTED_PILOT_CASE_COUNT,
    build_diagnosis_pilot_cases,
)
from math_feedback_ai.evaluation.pilot import (
    DiagnosisPilotReviewedCaseV1,
    PilotValidationReport,
    ReviewedPilotProvenance,
    validate_pilot_cases,
)

REVIEWED_PILOT_VERSION = "diagnosis_pilot_v1_reviewed"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REVIEWED_PILOT_PATH = (
    REPOSITORY_ROOT / "evaluation" / "benchmarks" / f"{REVIEWED_PILOT_VERSION}.jsonl"
)
DEFAULT_RECONCILIATION_PATH = (
    REPOSITORY_ROOT / "evaluation" / "review" / f"{REVIEWED_PILOT_VERSION}_reconciliation.md"
)

_MODIFIED_CASES = {
    "pilot.14.expand-fourth-power",
    "pilot.15.gcd-by-divisor-lists",
    "pilot.16.three-coins-enumeration",
    "pilot.17.derivative-from-definition",
    "pilot.22.transpose-sign",
    "pilot.36.pythagorean-on-any-triangle",
    "pilot.40.zero-derivative-at-point",
    "pilot.44.gcd-lucky-claim",
}

_CATEGORY_OVERRIDES = {
    "pilot.14.expand-fourth-power": "fully_correct_conventional",
    "pilot.16.three-coins-enumeration": "fully_correct_conventional",
    "pilot.17.derivative-from-definition": "fully_correct_conventional",
    "pilot.22.transpose-sign": "arithmetic_slip",
    "pilot.36.pythagorean-on-any-triangle": "missing_condition",
}

_CHARACTERISTIC_OVERRIDES = {
    "pilot.14.expand-fourth-power": ("procedural", "alternative_valid"),
    "pilot.16.three-coins-enumeration": ("counting", "alternative_valid"),
    "pilot.17.derivative-from-definition": ("procedural", "alternative_valid"),
}

_DEPENDENT_STEPS: dict[str, tuple[int, ...]] = {
    "pilot.18.discriminant-slip": (2,),
    "pilot.19.trapezoid-product-slip": (2,),
    "pilot.21.power-of-power": (1,),
    "pilot.23.add-rational-functions": (1,),
    "pilot.24.cancel-across-sum": (1,),
    "pilot.25.divide-by-variable": (1,),
    "pilot.26.cancel-square-roots": (1,),
    "pilot.29.union-count-overlap": (1,),
    "pilot.30.parallel-from-picture": (1,),
    "pilot.34.multiply-inequality-sign": (1,),
    "pilot.37.fermat-composite-modulus": (1,),
    "pilot.38.lhopital-non-indeterminate": (1,),
    "pilot.39.exclusive-implies-independent": (1,),
    "pilot.40.zero-derivative-at-point": (1,),
    "pilot.42.zero-division-right-root": (1,),
    "pilot.43.triangle-area-lucky": (1, 2),
    "pilot.44.gcd-lucky-claim": (1, 2),
}

_MODIFICATION_REASONS = {
    "pilot.14.expand-fourth-power": (
        "Repeated multiplication is an ordinary fully valid expansion method."
    ),
    "pilot.15.gcd-by-divisor-lists": (
        "Divisor listing is relevant and valid but less efficient than Euclid's algorithm."
    ),
    "pilot.16.three-coins-enumeration": (
        "Complete enumeration of eight outcomes is standard and proportionate."
    ),
    "pilot.17.derivative-from-definition": (
        "First-principles differentiation is a standard valid method."
    ),
    "pilot.22.transpose-sign": "The localized error is arithmetic: 8 / (-2) has the wrong sign.",
    "pilot.36.pythagorean-on-any-triangle": (
        "The first issue is the missing perpendicular assumption; Step 2 separately misnames "
        "the area formula as the Pythagorean theorem."
    ),
    "pilot.40.zero-derivative-at-point": (
        "The overgeneralization already occurs in Step 1; Step 2 depends on it."
    ),
    "pilot.44.gcd-lucky-claim": (
        "Both primality premises are false: 35=5×7 and 64=2^6; the final gcd is lucky."
    ),
}


def _reviewed_diagnosis(case_id: str, provisional: DiagnosisV1) -> DiagnosisV1:
    assessments = list(provisional.step_assessments)
    status = provisional.overall_status
    first_issue = provisional.first_issue

    if case_id in {
        "pilot.14.expand-fourth-power",
        "pilot.16.three-coins-enumeration",
        "pilot.17.derivative-from-definition",
    }:
        status = OverallStatus.FULLY_CORRECT
        first_issue = None
        assessments = [
            StepAssessment(step_id=item.step_id, status=StepStatus.VALID, confidence=0.95)
            for item in assessments
        ]
    elif case_id == "pilot.15.gcd-by-divisor-lists":
        first_issue = None
        note = "Correct divisor listing, but Euclid's algorithm is substantially more direct."
        assessments = [
            StepAssessment(
                step_id=item.step_id,
                status=(StepStatus.VALID_BUT_INEFFICIENT if index < 2 else StepStatus.VALID),
                efficiency_note=note if index < 2 else None,
                confidence=0.95,
            )
            for index, item in enumerate(assessments)
        ]
    elif status is OverallStatus.INCORRECT:
        root_index = next(
            index
            for index, item in enumerate(assessments)
            if first_issue is not None and item.step_id == first_issue.step_id
        )
        root_code = first_issue.code if first_issue is not None else None
        root_explanation = first_issue.explanation if first_issue is not None else None
        root_status = assessments[root_index].status

        if case_id == "pilot.22.transpose-sign":
            root_code = IssueCode.COMPUTATION_ARITHMETIC
            root_explanation = "Dividing 8 by -2 gives -4, not 4."
        elif case_id == "pilot.36.pythagorean-on-any-triangle":
            root_code = IssueCode.CONDITION_MISSING
            root_explanation = (
                "The side lengths 4 and 5 are treated as perpendicular without a right-angle "
                "condition or other justification."
            )
            root_status = StepStatus.UNSUPPORTED
        elif case_id == "pilot.40.zero-derivative-at-point":
            root_index = 0
            root_code = IssueCode.CONCEPT_MISUNDERSTOOD
            root_explanation = (
                "A zero derivative at one point gives zero instantaneous rate there; it does not "
                "mean the function is globally constant."
            )
            root_status = StepStatus.INVALID
        elif case_id == "pilot.44.gcd-lucky-claim":
            root_explanation = "Neither number is prime: 35=5×7 and 64=2^6."

        if root_code is None or root_explanation is None:
            raise AssertionError(f"reviewed incorrect case lacks a root issue: {case_id}")
        first_issue = Issue(
            step_id=assessments[root_index].step_id,
            code=root_code,
            explanation=root_explanation,
            confidence=0.95,
        )
        reviewed_assessments: list[StepAssessment] = []
        dependency_indices = set(_DEPENDENT_STEPS.get(case_id, ()))
        for index, item in enumerate(assessments):
            if index == root_index:
                reviewed_assessments.append(
                    StepAssessment(
                        step_id=item.step_id,
                        status=root_status,
                        issue_codes=(root_code,),
                        explanation=root_explanation,
                        confidence=0.95,
                    )
                )
            elif case_id == "pilot.36.pythagorean-on-any-triangle" and index == 1:
                reviewed_assessments.append(
                    StepAssessment(
                        step_id=item.step_id,
                        status=StepStatus.INVALID,
                        issue_codes=(IssueCode.THEOREM_MISUSED,),
                        explanation=(
                            "The triangle-area formula is not the Pythagorean theorem, and its "
                            "use here also depends on the unsupported perpendicular assumption."
                        ),
                        depends_on_step_ids=(assessments[root_index].step_id,),
                        confidence=0.95,
                    )
                )
            elif index in dependency_indices:
                prior_dependency = reviewed_assessments[index - 1]
                dependency_id = (
                    prior_dependency.step_id
                    if prior_dependency.status
                    in {
                        StepStatus.INVALID,
                        StepStatus.UNSUPPORTED,
                        StepStatus.DEPENDENT_ON_PREVIOUS_ERROR,
                    }
                    else assessments[root_index].step_id
                )
                reviewed_assessments.append(
                    StepAssessment(
                        step_id=item.step_id,
                        status=StepStatus.DEPENDENT_ON_PREVIOUS_ERROR,
                        explanation=(
                            "This step is locally coherent only under an earlier incorrect or "
                            "unsupported result."
                        ),
                        depends_on_step_ids=(dependency_id,),
                        confidence=0.95,
                    )
                )
            else:
                reviewed_assessments.append(item.model_copy(update={"confidence": 0.95}))
        assessments = reviewed_assessments

    reusable_prefix = provisional.reusable_prefix_end_step_id
    if status in {OverallStatus.FULLY_CORRECT, OverallStatus.CORRECT_BUT_INEFFICIENT}:
        reusable_prefix = assessments[-1].step_id
    elif status is OverallStatus.INCORRECT and first_issue is not None:
        root_index = next(
            index for index, item in enumerate(assessments) if item.step_id == first_issue.step_id
        )
        reusable_prefix = assessments[root_index - 1].step_id if root_index else None

    return DiagnosisV1(
        schema_version="1.1",
        attempt_id=provisional.attempt_id,
        overall_status=status,
        step_assessments=tuple(assessments),
        first_issue=first_issue,
        completion_gap=provisional.completion_gap,
        reusable_prefix_end_step_id=reusable_prefix,
        earlier_reasoning_usable=(
            status in {OverallStatus.FULLY_CORRECT, OverallStatus.CORRECT_BUT_INEFFICIENT}
            or reusable_prefix is not None
        ),
        reference_relation=provisional.reference_relation,
        confidence=0.95 if status is not OverallStatus.INDETERMINATE else provisional.confidence,
        confidence_reasons=(
            "Independently reviewed mathematical annotation; synthetic case provenance retained.",
        ),
    )


def _reviewed_decision(diagnosis: DiagnosisV1) -> TutorDecision:
    if diagnosis.overall_status in {
        OverallStatus.FULLY_CORRECT,
        OverallStatus.CORRECT_BUT_INEFFICIENT,
    }:
        action, level, target = TutorAction.WAIT, RevealLevel.NONE, None
    elif diagnosis.overall_status is OverallStatus.INCOMPLETE:
        action, level, target = TutorAction.ASK_STUDENT, RevealLevel.NONE, None
    elif diagnosis.overall_status is OverallStatus.INDETERMINATE:
        action = TutorAction.VERIFY_STEP
        level = RevealLevel.NONE
        target = diagnosis.step_assessments[0].step_id
    else:
        if diagnosis.first_issue is None:
            raise AssertionError("reviewed incorrect diagnosis requires first_issue")
        action = TutorAction.LIGHT_HINT
        level = RevealLevel.LIGHT_DIRECTION
        target = diagnosis.first_issue.step_id
    return TutorDecision(
        action=action,
        max_reveal_level=level,
        target_step_id=target,
        rationale_code=f"pilot.reviewed.{action.value.lower()}",
        rationale="Tutor action reconciled against the human-reviewed mathematical annotation.",
    )


def build_reviewed_diagnosis_pilot_cases() -> tuple[DiagnosisPilotReviewedCaseV1, ...]:
    """Build all 50 reviewed cases without mutating the provisional source artifact."""

    reviewed: list[DiagnosisPilotReviewedCaseV1] = []
    for case in build_diagnosis_pilot_cases():
        diagnosis = _reviewed_diagnosis(case.case_id, case.proposed_diagnosis)
        reviewed.append(
            DiagnosisPilotReviewedCaseV1(
                case_id=case.case_id,
                category=_CATEGORY_OVERRIDES.get(case.case_id, case.category),
                mathematical_domains=case.mathematical_domains,
                characteristics=_CHARACTERISTIC_OVERRIDES.get(case.case_id, case.characteristics),
                provenance=ReviewedPilotProvenance(
                    source="Human mathematical adjudication supplied for diagnosis pilot v1",
                    notes=(
                        "Human-reviewed annotations over synthetic cases; review improves label "
                        "reliability but does not establish ecological validity."
                    ),
                ),
                review_disposition=("modified" if case.case_id in _MODIFIED_CASES else "approved"),
                problem=case.problem,
                student_attempt=case.student_attempt,
                reviewed_diagnosis=diagnosis,
                reviewed_decision=_reviewed_decision(diagnosis),
                annotation_explanation=_MODIFICATION_REASONS.get(
                    case.case_id, case.annotation_explanation
                ),
                ambiguity_notes=case.ambiguity_notes,
            )
        )
    if len(reviewed) != EXPECTED_PILOT_CASE_COUNT:
        raise AssertionError(f"expected {EXPECTED_PILOT_CASE_COUNT} reviewed cases")
    return tuple(reviewed)


def render_reviewed_diagnosis_pilot_jsonl(
    cases: tuple[DiagnosisPilotReviewedCaseV1, ...] | None = None,
) -> str:
    selected = cases if cases is not None else build_reviewed_diagnosis_pilot_cases()
    return "".join(f"{case.model_dump_json()}\n" for case in selected)


def render_reviewed_reconciliation(
    cases: tuple[DiagnosisPilotReviewedCaseV1, ...] | None = None,
) -> str:
    selected = cases if cases is not None else build_reviewed_diagnosis_pilot_cases()
    status_counts = Counter(case.reviewed_diagnosis.overall_status.value for case in selected)
    lines = [
        "# Diagnosis Pilot v1 — Human-Review Reconciliation",
        "",
        "The original provisional JSONL remains unchanged. This reconciliation records the",
        "independent mathematical adjudication in `diagnosis_pilot_v1_reviewed.jsonl`.",
        "The cases remain synthetic; human review improves annotation reliability, not",
        "ecological validity.",
        "",
        "## Outcome",
        "",
        f"- Reviewed cases: {len(selected)}",
        f"- Approved without mathematical label changes: {len(selected) - len(_MODIFIED_CASES)}",
        f"- Modified: {len(_MODIFIED_CASES)}",
        f"- Status distribution: {dict(sorted(status_counts.items()))}",
        "- Diagnosis schema: `1.1`",
        "- Benchmark schema: `diagnosis_pilot.reviewed.v1`",
        "",
        "## Modified cases",
        "",
    ]
    for case_id in sorted(_MODIFIED_CASES):
        lines.append(f"- `{case_id}` — {_MODIFICATION_REASONS[case_id]}")
    lines.extend(
        [
            "",
            "## Schema reconciliation",
            "",
            "- Valid inefficiency uses `valid_but_inefficient` plus `efficiency_note`; it is",
            "  not a mathematical issue and does not populate `first_issue`.",
            "- Locally coherent consequences use `dependent_on_previous_error` plus",
            "  `depends_on_step_ids`, preserving the root error for future minimal-repair work.",
            "- Independent later errors may remain `invalid` while also naming an earlier",
            "  dependency, as in the Pythagorean/area terminology case.",
            "",
            "## Final dispositions",
            "",
        ]
    )
    for case in selected:
        lines.append(f"- `{case.case_id}` — **{case.review_disposition.upper()}**")
    lines.append("")
    return "\n".join(lines)


def reviewed_diagnosis_pilot_sha256() -> str:
    return sha256(render_reviewed_diagnosis_pilot_jsonl().encode()).hexdigest()


def reviewed_reconciliation_sha256() -> str:
    return sha256(render_reviewed_reconciliation().encode()).hexdigest()


def validate_built_reviewed_diagnosis_pilot() -> PilotValidationReport:
    from math_feedback_ai.evaluation.development_builder import build_development_examples

    return validate_pilot_cases(
        build_reviewed_diagnosis_pilot_cases(),
        comparison_examples=build_development_examples(),
    )


def load_reviewed_diagnosis_pilot(
    path: Path | None = None,
) -> tuple[DiagnosisPilotReviewedCaseV1, ...]:
    selected_path = path if path is not None else DEFAULT_REVIEWED_PILOT_PATH
    if path is None and not selected_path.is_file():
        return build_reviewed_diagnosis_pilot_cases()
    try:
        lines = selected_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkFormatError(f"cannot read reviewed pilot {selected_path}: {exc}") from exc
    cases: list[DiagnosisPilotReviewedCaseV1] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BenchmarkFormatError(f"blank reviewed pilot line at {line_number}")
        try:
            cases.append(DiagnosisPilotReviewedCaseV1.model_validate_json(line))
        except ValidationError as exc:
            raise BenchmarkFormatError(
                f"invalid reviewed pilot case at line {line_number}: {exc}"
            ) from exc
    if not cases:
        raise BenchmarkFormatError("reviewed diagnosis pilot contains no cases")
    report = validate_pilot_cases(cases)
    if not report.passed:
        summary = "; ".join(f"{item.code}: {','.join(item.case_ids)}" for item in report.issues)
        raise BenchmarkFormatError(f"reviewed pilot failed structural validation: {summary}")
    return tuple(cases)


def committed_reviewed_diagnosis_pilot_is_current(
    jsonl_path: Path = DEFAULT_REVIEWED_PILOT_PATH,
    reconciliation_path: Path = DEFAULT_RECONCILIATION_PATH,
) -> bool:
    try:
        jsonl = jsonl_path.read_text(encoding="utf-8")
        reconciliation = reconciliation_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        jsonl == render_reviewed_diagnosis_pilot_jsonl()
        and reconciliation == render_reviewed_reconciliation()
    )


def write_reviewed_diagnosis_pilot(
    jsonl_path: Path = DEFAULT_REVIEWED_PILOT_PATH,
    reconciliation_path: Path = DEFAULT_RECONCILIATION_PATH,
) -> tuple[Path, Path]:
    report = validate_built_reviewed_diagnosis_pilot()
    if not report.passed:
        raise ValueError(f"refusing to write invalid reviewed pilot: {report.issues}")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    reconciliation_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(render_reviewed_diagnosis_pilot_jsonl(), encoding="utf-8", newline="\n")
    reconciliation_path.write_text(render_reviewed_reconciliation(), encoding="utf-8", newline="\n")
    return jsonl_path, reconciliation_path


__all__ = [
    "DEFAULT_RECONCILIATION_PATH",
    "DEFAULT_REVIEWED_PILOT_PATH",
    "REVIEWED_PILOT_VERSION",
    "build_reviewed_diagnosis_pilot_cases",
    "committed_reviewed_diagnosis_pilot_is_current",
    "load_reviewed_diagnosis_pilot",
    "render_reviewed_diagnosis_pilot_jsonl",
    "render_reviewed_reconciliation",
    "reviewed_diagnosis_pilot_sha256",
    "reviewed_reconciliation_sha256",
    "validate_built_reviewed_diagnosis_pilot",
    "write_reviewed_diagnosis_pilot",
]
