> Sanitized publication copy. Private identities, labels, machine paths and task prompts are omitted. Original hashes below identify local originals, not these edited copies. Some linked raw artifacts are intentionally local-only.

# Instrumented training runtime replay

This was an explicit replay using training and validation only, to fill the original missing end-to-end timing. It did not replace the frozen model or evaluate test/challenge observations.

- Train-only normalization fitting: 35.600 s.
- Verified training/validation loading: 43.143 s.
- PCA, optimizer, validation stopping and checkpoint I/O: 1.896 s.
- End-to-end measured replay: 83.892 wall s; 69.812 CPU s.
- Same selected epoch 22 and stopping epoch 52; maximum saved-array difference 0.

Measured replay cost on this CPU/two-thread environment, including integrity checks, repeated disk reads, normalization fit, PCA, optimizer, validation stopping and checkpoint I/O. Excludes initial dataset generation, readiness audit, half-data diagnostic and dependency import startup. Disk/cache/system load affect timing. Neural model used in comparison remains the original frozen checkpoint.
