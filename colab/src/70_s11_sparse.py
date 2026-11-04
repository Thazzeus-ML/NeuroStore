# %% [markdown] id=s11-md
# ## 11. The NeuroStore prototype's core read kernel on the accelerator
#
# The Python prototype in this repository reduces every read to one primitive: **sparse accumulate**. Take the `k` active
# units, read their out-edge rows from the synapse lattice (CSR; each edge decays lazily as `w · (1-λ)^(epoch − edge_epoch)`),
# and add them into a score array. The same primitive scores assemblies (unit → assembly inverted index) and spreads
# activation. On a CPU/NumPy the CSR layout beat a dict-of-dicts by ~28× (see `REPORT.md`); this section asks the accelerator
# question: **sparse scatter-add over a padded CSR vs one dense matrix product**, as the number of units `N` grows.
#
# * `scatter_add`: gather `post`, `w`, `epoch` rows of the active units, apply the decay table, `at[].add` into `(B, N)`.
# * `segment_sum`: the same updates through `jax.ops.segment_sum`.
# * `dense_matmul`: materialise the effective weight matrix (`N × N`) once and multiply a multi-hot activation matrix. It
#   becomes impossible when `N²` floats exceed device memory, and its work grows as `B·N²` against `B·k·R` for the sparse form.
#
# All three are checked against a NumPy reference. The k-WTA selection step of the encoder (`top_k` vs `approx_max_k`) is timed too.

# %% id=s11-code
#@title 11. Sparse accumulate (padded CSR) vs dense; k-WTA selection
clear("sparse_accumulate"); clear("kwta")


def make_graph(N, R, rng, epochs=30):
    """R distinct targets per unit (arithmetic progression with an odd stride, which is coprime with power-of-two N)."""
    start = rng.integers(0, N, size=(N, 1)).astype(np.int64)
    step = rng.integers(0, max(N // 2, 1), size=(N, 1)).astype(np.int64) * 2 + 1
    post = ((start + step * np.arange(R)[None, :]) % N).astype(np.int32)
    return post, rng.random((N, R)).astype(np.float32), rng.integers(0, epochs, (N, R)).astype(np.int32)


def make_actives(N, k, B, rng):
    act = np.stack([rng.choice(N, size=k, replace=False) for _ in range(B)]).astype(np.int32)
    return act, rng.random((B, k)).astype(np.float32) + 0.1


@partial(jax.jit, static_argnames=("N",))
def acc_scatter(post, w, ep, dtab, epoch, act, x, N):
    p = post[act]                                                       # (B, k, R)
    contrib = w[act] * dtab[epoch - ep[act]] * x[:, :, None]
    return jnp.zeros((act.shape[0], N), jnp.float32).at[jnp.arange(act.shape[0])[:, None, None], p].add(contrib)


@partial(jax.jit, static_argnames=("N",))
def acc_segment(post, w, ep, dtab, epoch, act, x, N):
    B = act.shape[0]
    p = post[act] + (jnp.arange(B, dtype=jnp.int32) * N)[:, None, None]
    contrib = w[act] * dtab[epoch - ep[act]] * x[:, :, None]
    return jax.ops.segment_sum(contrib.reshape(-1), p.reshape(-1), num_segments=B * N).reshape(B, N)


@partial(jax.jit, static_argnames=("N",))
def dense_weights(post, w, ep, dtab, epoch, N):
    eff = w * dtab[epoch - ep]
    return jnp.zeros((N, N), jnp.float32).at[jnp.arange(N)[:, None], post].add(eff)


@partial(jax.jit, static_argnames=("N",))
def acc_dense(Wd, act, x, N):
    A = jnp.zeros((act.shape[0], N), jnp.float32).at[jnp.arange(act.shape[0])[:, None], act].add(x)
    return jnp.matmul(A, Wd, precision=HI)


def np_reference(post, w, ep, dtab, epoch, act, x, N, rows=4):
    out = np.zeros((rows, N), np.float64)
    for b in range(rows):
        for j, u in enumerate(act[b]):
            np.add.at(out[b], post[u], w[u].astype(np.float64) * dtab[epoch - ep[u]] * x[b, j])
    return out


if RUN_SPARSE:
    R_, K_, EPOCH, LAMBDA = 128, 32, 30, 0.03
    dtab = (1.0 - LAMBDA) ** np.arange(EPOCH + 2, dtype=np.float32)
    bud = Budget(P["budget_s"])
    verified = []
    for N in P["sparse_N"]:
        if bud.expired():
            print(f"sparse sweep stopped before N={N}: budget used")
            break
        R = min(R_, N // 2)
        B = min(P["sparse_B"], max(1, int(q_cap(N, frac=4))))
        rng = rng_for(11, 0, N)
        post, w, ep = make_graph(N, R, rng, EPOCH); act, x = make_actives(N, min(K_, N // 4), B, rng)
        dpost, dw, dep, ddt, dact, dx = put(post), put(w), put(ep), put(dtab), put(act), put(x)
        sparse_bytes = N * R * (4 + 4 + 4)
        calls = {"scatter_add": lambda: acc_scatter(dpost, dw, dep, ddt, EPOCH, dact, dx, N), "segment_sum": lambda: acc_segment(dpost, dw, dep, ddt, EPOCH, dact, dx, N)}
        if fits(N * N, 4 * 2.2):                                       # dense matrix + one temporary
            Wd = dense_weights(dpost, dw, dep, ddt, EPOCH, N)
            calls["dense_matmul"] = lambda: acc_dense(Wd, dact, dx, N)
        ref = np_reference(post, w, ep, dtab, EPOCH, act, x, N)
        for name, call in calls.items():
            got = np.asarray(call())
            ok = bool(np.allclose(got[: ref.shape[0]], ref, rtol=2e-4, atol=2e-4 * max(1.0, float(np.abs(ref).max()))))
            verified.append(ok)
            st = bench(lambda call=call: call(), batch=B)
            flops = 2.0 * B * N * N if name == "dense_matmul" else float(B * min(K_, N // 4) * R)
            record("sparse_accumulate", name, n=N, d=D, query_batch=B, out_degree=R, active_units=min(K_, N // 4), matches_numpy_reference=ok,
                   weight_bytes=(N * N * 4) if name == "dense_matmul" else sparse_bytes, work_units=flops, **st)
        del calls
        gc.collect()
    sp = results_df("sparse_accumulate")
    check("S11 JAX sparse-accumulate kernels match the NumPy reference", bool(verified) and all(verified), f"{len(verified)} (kernel, N) cells")
    show("sparse_accumulate", ["method", "n", "query_batch", "out_degree", "matches_numpy_reference", "lat_p50_ms", "pipe_p50_ms", "us_per_query", "weight_bytes"], sort=["n", "method"])
    both = sp.pivot_table(index="n", columns="method", values="pipe_p50_ms")
    if {"scatter_add", "dense_matmul"} <= set(both.columns):
        both["dense_over_sparse"] = both["dense_matmul"] / both["scatter_add"]
        print("dense / sparse time ratio (>1: sparse wins; missing: dense matrix does not fit):")
        display(both.round(4))

    # k-WTA selection step of the encoder
    for N in P["sparse_N"][:4]:
        B = min(P["sparse_B"], max(1, int(q_cap(N, frac=4))))
        sc = put(rng_for(11, 1, N).random((B, N)).astype(np.float32))
        f_ex = jax.jit(lambda s: lax.top_k(s, 32)[1])
        f_ap = jax.jit(lambda s: lax.approx_max_k(s, 32, recall_target=0.95)[1])
        ex = np.asarray(f_ex(sc))
        for name, fn in (("top_k_exact", f_ex), ("approx_max_k_r0.95", f_ap)):
            try:
                got = np.asarray(fn(sc)); ov = float(np.mean([len(set(a) & set(b)) / 32 for a, b in zip(got, ex)]))
                record("kwta", name, n=N, d=D, query_batch=B, set_overlap_with_exact=ov, **bench(fn, sc, batch=B))
            except Exception as exc:
                print(f"{name} unavailable at N={N}: {type(exc).__name__}")
    show("kwta", ["method", "n", "query_batch", "set_overlap_with_exact", "lat_p50_ms", "pipe_p50_ms", "us_per_query"], sort=["n", "method"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in ("scatter_add", "segment_sum", "dense_matmul"):
        g = sp[sp.method == name].sort_values("n")
        if len(g):
            axes[0].plot(g.n, g.pipe_p50_ms, "o-", label=name); axes[1].plot(g.n, g.weight_bytes / 1e6, "o-", label=name)
    axes[0].set(xscale="log", yscale="log", xlabel="units N", ylabel="pipelined ms / batch", title="sparse accumulate"); axes[1].set(xscale="log", yscale="log", xlabel="units N", ylabel="weight storage (MB)", title="storage")
    for a in axes:
        a.grid(alpha=.25, which="both"); a.legend(fontsize=8)
    plt.tight_layout(); savefig(fig, "s11_sparse"); plt.show()
else:
    print("Sparse-accumulate study disabled.")
