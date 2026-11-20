# 2027 Material Track Preview Plan

**Goal:** Produce a versioned, inspectable 104 m MuJoCo scene and a terrain-only preview video without overwriting old artifacts.

**Spec:** `docs/superpowers/specs/2026-11-13-robocon-material-track-design.md`.

**Constraints:** No repository Git metadata; do not claim commits. Preview is not policy evaluation and does not establish traversability. The crawl height remains provisional until geometry-envelope measurements. Model and decoder integration, collision/sensor parity and training are separate gates.

1. Add tests for scene length, named 0.12 m steps, hollow tire construction, brick/bark section, bridge and overhead frame; verify missing implementation fails.
2. Generate a standalone MuJoCo XML with explicit primitive geometry and fixed deterministic seed. Preserve names, dimensions and source/scene hashes in a manifest; reject nonempty output directories.
3. Compile the scene, verify geometry tests and render a terrain-only fly-through to a new MP4; verify it using ffprobe and sample frames. Record limits, especially provisional clearance and absence of robot/agent integration.
4. Present video and disclose preview boundaries; do not initiate training.
