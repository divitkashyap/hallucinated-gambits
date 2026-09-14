# Provenance and attribution

- Final competition build: platform v11, observed active after validation. Exact submitted archive SHA256: `36af8789dfdbda8d250edd035a32691932aeca59960cfb1f41b8689264ae8f4a`.
- The six engine/data files here match the retained final package source byte for byte. `SOURCE_MANIFEST.json` records their hashes. This folder also contains presentation material and is not itself the original ZIP.
- Project started from [AI Chessathon starter](https://github.com/advitrocks9/aichessathon-starter). Its MIT license and Advit Arora copyright notice are preserved in LICENSE. Original project search work was developed with AI coding assistance; no claim of unaided authorship is made.
- Runtime dependencies are installed separately, not bundled. Consult their respective licenses. The demo uses chess, NumPy and Numba; it does not require Torch or an offline reference engine.
- Opening data: [November 2025 Lichess Elite archive](https://database.nikonoel.fr/), derived from [Lichess database exports](https://database.lichess.org/). Lichess states its database exports are CC0. This curated folder includes only aggregate position/move/count records, not the raw game archive or player identities. Attribution is retained to both sources.
- The book was generated from 236,955 eligible unique human games with BOT-title exclusion, deduplication and a source-game split. It contains 56,333 positions. Runtime lookup checks legality, popularity, move number, repetition and per-game use limits. No reference evaluations are stored in this book.
- Stockfish was used outside the submission for analysis and training labels only. No Stockfish source, binary or neural weights are included here.
- Screenshot: user-supplied AI Chessathon dashboard capture on 14 September 2026. Site branding and badge artwork belong to their respective owners and are shown as competition evidence, not original project artwork or an endorsement.

## Evidence sources in the development archive

The case study was checked against the development roadmap, v6–v8 release manifests, buffer-v9 release result, v10 release report, Royal v11 result, first-five-rated-v11 report, corrected ML follow-up and final-day six-loss review. Those private research documents are intentionally omitted from this curated edition. Match numbers are historical reported measurements, not a fresh benchmark run for this export.
