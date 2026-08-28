# Environment and resource reporting

## Full `m=8` completion environment

```text
Date (UTC): 2026-08-07
Linux: 6.18.35 x86_64, little-endian
Compiler: GNU g++ 14.2.0
Python: 3.13.5
Boost: 1.83.0
GMP/GMPXX: 6.3.0
CPU: AMD EPYC 9V74; 5 exposed cores
Memory available to container: approximately 5.9 GiB
JOBS: 5
CHUNK_SIZE: 100000
```

The final `m=8` positive-hierarchy extension took `6:47.62` wall time and
reported peak RSS `737424` KiB. The subsequent kernel audit, complete
floor-cell scan, proposal, coverage check, and two exact replays took
`1:20:27` wall time and reported peak RSS `592508` KiB. Detailed
`/usr/bin/time -v` output and per-stage commands are preserved in
`logs/n8/m8_positive_enumeration.err` and
`logs/n8/full_m8_regeneration_console.log`.

The run produced and checked:

- 9,105,190 canonical positive-dual square kernels;
- 1,049,187 hard-cell records;
- 1,049,177 fixed and 10 adaptive certificates;
- zero failure records;
- signed-rational/GMP agreement on 122,280,434 saturation exclusions,
  6,543,104 open-floor checks, and 4,430,958 exact price LPs;
- exact minimum margins `1/1008` and `1/11`.

## Public-log path normalization

Before public distribution, container-local working paths in the retained text
logs were mechanically replaced with descriptive placeholders such as
`<artifact-root>` and `<regeneration-output>`. The commands' mathematical
parameters, program output, counters, timings, and resource measurements were
not changed. This removes nonportable build-directory details while preserving
the execution evidence.
