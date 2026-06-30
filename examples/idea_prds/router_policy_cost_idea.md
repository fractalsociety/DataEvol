# Idea PRD: Router Policy Cost Reduction

## Observation
Accepted low-risk router tasks sometimes used more expensive models than needed.

## Hypothesis
A cost-aware router policy can route low-risk verified tasks to cheaper models while preserving correctness.

## Affected Component
router

## Baseline
Current router policy chooses the strongest safe model for many low-risk tasks.

## Variant
Prefer cheap/free workers for low-risk tasks when verifier coverage is available.

## Benchmark
Frozen router policy benchmark from prior Coordinate and Router/BioLatent traces.

## Primary Metric
cost_per_verified_task

## Non-Regression Metrics
correctness, verification_pass_rate

## Safety Checks
Safety score must not decline and safety regressions must equal 0.

## Reproducibility Requirement
2

## Rollback Plan
Restore the previous router policy snapshot recorded before promotion.

## Promotion Rule
Promote only if cost_per_verified_task improves, correctness and verification pass rate do not decline, safety passes, reproducibility is >= 2, and rollback snapshot exists.

## Rejection Rule
Reject if any promotion condition fails.
