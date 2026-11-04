from hypothesis import settings, HealthCheck

# deterministic property tests: fixed example sequence, no wall-clock deadline
settings.register_profile("ns", derandomize=True, deadline=None, max_examples=40,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("ns")
