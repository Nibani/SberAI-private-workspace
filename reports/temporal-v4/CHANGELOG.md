# Export compatibility correction before successful registration

The first attempt completed16 tune and64 evaluation seeds but failed before
summary/real export with `KeyError: level_center is not a file in the archive`.
It is preserved at `artifacts/temporal-v4-export-failed-20261009` with its original
registration, selected policy, all checkpoints and CSVs. No successful status
was assigned to that attempt.

The pinned historical v12 model predates serialization of its contemporaneous
monthly log-total median. Only the legacy projection exporter now reconstructs
that reference deterministically from the same pinned complete cohort. A new
author test reproduces all48384 saved monthly labels exactly and confirms the
48 historical constant-profile graph mismatches become zero corrected flags.
The five-coordinate real CP calibration still fits only2023.

Old benchmark SHA256:
`a3d3522fd8c975a3902192351629ac214e6b68d59506d6c6c8059f688d28248f`

Corrected benchmark SHA256:
`07e247692574ae7c45879d8ff3c3510dbf0ac47fa9b6ece14afb2b86793ffb0b`

Core/scorer SHA remains:
`f914ca705494e9978b34d780ad19bb084de4d290d6a77e7dfb8f8b9fcaa3c9cc`

Root required a full repeat into a new output, with fresh code registration,
unchanged frozen seeds/grid/thresholds and selection rule, rather than a
finalize-only hash amendment. No positive outcome was used for calibration.
After completion, `repeat-equivalence.json` records comparison of synthetic
CSVs/checkpoints and selected policy against the preserved first attempt.
