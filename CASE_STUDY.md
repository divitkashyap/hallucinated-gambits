# Eleven submission versions: decisions and lessons

## The problem

Build a chess-playing program that selects legal moves under a strict wall-clock budget. The final entry combined Python/Numba search, classical evaluation, adaptive timing and a limited opening book. Development used AI coding assistance and human direction; it should not be presented as unaided implementation or as a trained neural engine.

## Release evolution

Platform version numbers identify submissions, not eleven separate architectures. V2 failed validation, and v3 corrected its packaging. Local experiment names sometimes used different numbers; later v12/v13 research folders were not platform releases.

| Platform version | What changed and why | Evidence or lesson |
|---|---|---|
| v1 | Classical baseline to establish a working legal, clock-aware entry | First validated champion; establish a baseline before complexity |
| v2 | Numba-compiled board/search to fit more work into each turn | Local candidate scored 15W/5D/4L vs v1; platform wrapper failed because it assumed a local folder layout |
| v3 | Corrected packaging of the compiled candidate | Passed platform validation; local success is not deployment success |
| v4 | Fixed-size transposition table to reuse position results | 9W/12D/3L vs v3 across 24 paired games; less repeated search |
| v5 | Improved killer/history and tactical ordering, with search/generation efficiency work | 16W/5D/3L vs v3, not v4; comparator attribution matters |
| v6 | Conservative late-move reductions: search later quiet moves less deeply, verify promising results | Separate 18-game comparison vs v5: 9W/5D/4L; full-clock pair 1W/1D |
| v7 | Bounded check extensions, terminal fixes and dead-material recognition after a rated tactical loss | Combined candidate found the diagnostic mate; 3W/2D/1L vs v6 and two full-clock wins; small samples |
| v8 | Lazy ordering avoids sorting moves the search never reaches | Same sampled moves/scores/nodes; about 1.66× faster locally; 11W/6D/7L vs v7 |
| v9 | Tactical generation and reusable per-depth buffers reduce allocation work | 32.3% less benchmark time, with sampled parity; 6W/3D/3L vs v8 at full clock |
| v10 | Scaled late-move reductions direct effort away from late quiet moves | 20W/15D/13L vs v9 in 48 full-clock games; favourable but confidence interval crossed 50% |
| v11 | King safety, adaptive time, faster legality checks and a conservative human opening book | Combined candidate: 31W/14D/3L vs v10 in 48 full-clock games; final validated entry |

Local matches were against specific frozen opponents and often small samples. They do not establish eleven monotonic rating gains. Faster nodes do not imply proportionally stronger chess.

## Why v11 combined these features

King safety addressed exposed kings and weakening pawn moves. Adaptive timing allocated more effort when root choices or scores changed. The legality shortcut skipped unnecessary make/unmake work only for safe cases, keeping the full path for ambiguous moves. Opening preparation used exact positions and transpositions because rated games began from curated openings.

Component screens suggested king safety and timing were promising. Book-only scored 50% in its small covered-position screen; its individual contribution was not established. The combined configuration earned its release through the stronger paired result. We cannot assign its gain equally to four features.

## Did reviewing competitors help?

A sample of 33 distinct games from the top three teams supplied 31 starting positions and realistic early-game tests. Two teams often replied almost instantly early; the highest-rated team did not in that sample. This suggested different strategies, not proof of their implementations. Their moves were not copied into the runtime book.

The book instead used 236,955 filtered unique human games from the November 2025 Lichess Elite archive. It covered only 6/31 sampled leader starts. Preparation helped define a useful component, but did not explain or reproduce the leaders' overall advantage.

## What did not work

- Neural evaluation experiments had both inference-cost and position-quality problems. No neural weights shipped in the final entry.
- A later linear move-ranking pilot improved pair accuracy (56.79% to 59.88%) while worsening mean development regret (68.64cp to 105.44cp), especially in sparse endings. It was rejected. Its baseline was restricted candidate ordering, not a full search comparison.
- In the six-loss review, 14/18 reviewed moves reproduced locally at recorded remaining clocks. All six games ended in checkmate with time remaining; no opening-book replies were involved.
- Removing the middlegame pawn-advancement reward improved two diagnostic choices but introduced a 132cp regression and left the worst endgame error unchanged. The candidate failed its predefined gate; longer release matches were not run and v11 was retained.

## Engineering lessons

1. A working package is part of correctness: v2's import failure was a real loss even though local chess tests passed.
2. Profile before adding complexity: preserving search results while reducing allocation or sorting cost produced useful measured gains.
3. Compare under the same clock, swap colours and freeze opponents. A better prediction metric or tactical example is not sufficient evidence of strength.
4. Keep data partitions and reference scores coherent. Complete same-depth reference snapshots prevent misleading comparisons; whole-game splits reduce leakage risk.
5. Report negative results. Rejection gates limited last-minute changes that looked plausible but were not reliable improvements.

## Outcome

The supplied September 14 dashboard shows rank 211/465 (top 45%), rating 1660, peak 1704 and 49W/16D/48L. The result was mid-field, not a podium finish. The strongest portfolio contribution is the design, testing and evidence-led iteration behind a functioning competition engine. The screenshot does not establish a final Swiss rank or finalist status.
