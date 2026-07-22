# Evaluation guide

Lux S3 randomizes maps and several hidden mechanics at the game level. A single attractive replay is therefore not evidence of a stronger policy.

## Minimum comparison protocol

1. Run at least 20 fixed seeds.
2. Alternate which agent is `player_0`; starting corners are symmetric, but this catches integration asymmetries.
3. Keep `luxai-s3==0.2.1` and the same opponent revision.
4. Report wins, losses, draws, and the exact seed list.
5. Separate crashes/timeouts from gameplay losses.
6. Save a replay for every unexpected regression.

The bundled command implements items 2–4:

```bash
uv run cortex-arena benchmark --games 20 --seed 1000 --output reports/baseline.json
```

The bundled starter opponent is intended for installation checks and regression testing. Meaningful strategic claims require additional fixed opponents and preferably a round-robin tournament.

## Useful ablations

Disable one subsystem at a time and compare on the same seeds:

- symmetry propagation;
- hidden scoring-tile inference;
- risk costs in A*;
- combat target reliability filter;
- role quotas.

This reveals whether a feature contributes across maps rather than merely making the code more elaborate.
