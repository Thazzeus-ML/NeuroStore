"""Domain adapters. A domain turns raw data into a DomainData (windows (n, T, C) z-scored, targets, groups); everything else in
nsharness (features, backends, corruption, OOD, benchmarks) is domain-neutral and sees only those arrays."""
