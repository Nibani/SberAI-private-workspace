# Execution decision before tune/evaluation

Recorded 2026-10-09 00:35 UTC, after timing-only calibration and before any tune or evaluation quality outcomes. The lead gave GO for diagnostics, calibration and the duration-selected registered plan. Slot label: teamlead-science-20261009-go-0032.

Decision: execute the ORIGINAL FULL registered-v1 plan unchanged, with 8 tuning seeds 11001–11008 and 32 independent evaluation seeds 22001–22032. Preserve both generators, all six original graph families, both evaluation topology permutations, all dynamic/stress regimes, all fixed/research controls, n_init50, monthly free KMeans n_init20, iteration/sweep caps, application targets, four C1–C4 contrasts, bootstrap33001 with10000 draws and Holm correction. No reduced-budget plan is needed.

Evidence for duration: calibration completed420 optimizer calls and10400 randomized starts on seeds44001/44002 in34.0409775 seconds. It did not compute quality metrics or make quality-based selections. All16 forecast branches are retained. Mean-rate forecasts range408.209–525.112 seconds; observed-max-rate forecasts range459.228–581.930 seconds. They omit scoring, output compression, checkpoint writing, bootstrap and future resource pauses. Observed maximum is not an upper bound for unseen seeds. Allocation is60 minutes from about00:35 UTC, targeting completion by01:35 UTC; the user-imposed final deadline remains02:22:44 UTC. The roughly10-minute worst observed-rate fit forecast leaves substantial room for the unmeasured overhead.

Disk pressure was near100% during diagnostics but fell below95% before calibration operations; the472 recorded gate waits sum to0.0014394 seconds. This is a current observation, not a promise that later disk pressure cannot return. The original one-CPU, BelowNormal and resource-monitor guards remain unchanged. No extra resource-policy patch is applied.

Diagnostic r0=48/2016 is an already permitted mechanistic projection measurement. It does not select seeds, methods, weights or sample size and is retained regardless of subsequent comparative outcomes. Comparative quality at the time of this decision: NOT_RUN.

Outputs: scratch/science/full-run plus original diagnostics and timing-only-v1. The frozen runner refuses existing output folders and has no resume support. Any forced termination leaves an explicitly partial run; no seed or unsuccessful result will be removed. Original registration and plans remain unmodified in LIVE.
