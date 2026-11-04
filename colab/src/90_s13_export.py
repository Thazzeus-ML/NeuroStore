# %% [markdown] id=s13-md
# ## 13. Export and cross-hardware comparison
#
# Exports `results.csv` / `results.json` (every row carries run id, profile, base seed, config hash, backend, device kind
# and JAX version), `hardware_metadata.json`, `config.json`, `checks.csv`, `findings.txt`, raw latency samples and all plots into one
# ZIP. Colab's filesystem is ephemeral: **download the ZIP before disconnecting**.
#
# The comparison cell merges several runs, refuses (with a warning) to compare runs whose profile or configuration hash differ,
# and reports speed-ups relative to a chosen reference device plus the cross-device agreement of the accuracy metrics (identical
# seeds should give identical recall on every backend apart from exact-score ties).

# %% id=s13-code
#@title 13a. Export this runtime's result bundle
AUTO_DOWNLOAD = False #@param {type:"boolean"}

rdf = results_df()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


rdf.to_csv(OUTPUT_DIR / "results.csv", index=False)
(OUTPUT_DIR / "results.json").write_text(rdf.to_json(orient="records", indent=1), encoding="utf-8")
(OUTPUT_DIR / "hardware_metadata.json").write_text(json.dumps(METADATA, indent=1, default=_jsonable), encoding="utf-8")
(OUTPUT_DIR / "config.json").write_text(json.dumps(CONFIG, indent=1, default=_jsonable), encoding="utf-8")
pd.DataFrame(CHECKS).assign(**{k: META_ROW[k] for k in ("run_id", "backend", "device_kind", "profile", "config_hash")}).to_csv(OUTPUT_DIR / "checks.csv", index=False)
pd.DataFrame(LEDGER).to_csv(OUTPUT_DIR / "claims_ledger.csv", index=False)
if LAT_SAMPLES:
    np.savez_compressed(OUTPUT_DIR / "latency_samples_ms.npz", **{f"{k[0]}__{k[1]}": v for k, v in LAT_SAMPLES.items()})
LIMITS = """NeuroStore accelerator benchmark lab v2: independent proxy implementation, not a reproduction of the original report's code.
One JAX device is benchmarked per section unless Section 10 says otherwise. Device-resident timings exclude transfers and compilation
(first_call_ms reports compilation). Batch throughput is not single-request latency. Percentiles are NaN unless enough samples exist.
FlyHash/VSA/SDM are algorithmic proxies; SDM is fixed-K. Fault injection is functional, not a calibrated device model.
Energy, endurance, ADC/DAC power, crossbar area and FeNAND speed-ups are NOT measured.
"""
(OUTPUT_DIR / "findings.txt").write_text(LIMITS + "\nFindings\n" + "\n".join("- " + f for f in FINDINGS) + "\n\nSelf-checks\n" +
                                         "\n".join(f"{'PASS' if c['ok'] else 'FAIL'} {c['name']} | {c['detail']}" for c in CHECKS) + "\n", encoding="utf-8")
ZIP_PATH = shutil.make_archive(str(OUTPUT_DIR.parent / f"NeuroStore_v2_{RUN_ID}_{BACKEND}_{PROFILE}"), "zip", root_dir=OUTPUT_DIR)
print(f"{len(rdf)} rows, {len(rdf.columns)} columns -> {OUTPUT_DIR}\nZIP: {ZIP_PATH}")
if AUTO_DOWNLOAD:
    from google.colab import files
    files.download(ZIP_PATH)
else:
    print("Set AUTO_DOWNLOAD=True and re-run this cell to download.")

# %% id=s13-compare
#@title 13b. Optional multi-runtime comparison
COMPARE_UPLOADS = False #@param {type:"boolean"}
COMPARE_GLOB = "" #@param {type:"string"}
REFERENCE_DEVICE = "" #@param {type:"string"}

if COMPARE_UPLOADS or COMPARE_GLOB:
    import glob, io
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)         # wide, sparse result tables
    frames, check_frames = [], []
    sources = []
    if COMPARE_UPLOADS:
        from google.colab import files
        sources += [(fname, io.BytesIO(data)) for fname, data in files.upload().items() if fname.lower().endswith(".csv")]
    for fname in sorted(glob.glob(COMPARE_GLOB)) if COMPARE_GLOB else []:
        sources.append((fname, fname))
        sib = Path(fname).with_name("checks.csv")                      # checks.csv sits next to results.csv in the ZIP
        if sib.exists():
            sources.append((str(sib), str(sib)))
    for fname, src in sources:
        (check_frames if "check" in Path(fname).name.lower() else frames).append(pd.read_csv(src).assign(source_file=fname))
    if check_frames:
        ck = pd.concat(check_frames, ignore_index=True)
        if "device_kind" in ck:
            ck["device"] = ck.backend.astype(str) + ":" + ck.device_kind.astype(str)
            ck["status"] = np.where(ck.ok, "PASS", "FAIL")
            print("self-checks by device (FAIL = an invariant was violated on that runtime):")
            display(ck.pivot_table(index="name", columns="device", values="status", aggfunc="first"))
    if not frames:
        print("no CSV files found")
    else:
        comb = pd.concat(frames, ignore_index=True)
        comb["device"] = comb.backend.astype(str) + ":" + comb.device_kind.astype(str)
        for col in ("profile", "config_hash", "jax_version"):
            if col in comb and comb[col].nunique() > 1:
                print(f"WARNING: runs differ in {col}: {sorted(comb[col].astype(str).unique())}; timing comparisons may not be like for like.")
            if col not in comb:
                print(f"WARNING: {col} column missing (results from the v1 notebook?); comparability cannot be verified.")
        print("devices:", sorted(comb.device.unique()))
        ref = REFERENCE_DEVICE or sorted(comb.device.unique())[0]
        out = []
        kv = comb[comb.experiment == "kernel_variants"]
        if len(kv):
            nn_ = kv.groupby("device").n.max().min()
            t = kv[kv.n == nn_].pivot_table(index="method", columns="device", values="pipe_p50_ms")
            sp = (t[ref].values[:, None] / t) if ref in t else None
            print(f"\nkernel_variants at N={int(nn_)}: pipelined ms per call"); display(t.round(4))
            if sp is not None:
                print(f"speed-up relative to {ref} (>1 faster):"); display(sp.round(2))
            out.append(t)
        sc_ = comb[comb.experiment == "scaling_N"]
        if len(sc_):
            print("\nachieved GFLOP/s (max over N) by method and device"); display(sc_.pivot_table(index="method", columns="device", values="gflops_effective", aggfunc="max").round(1))
        ld = comb[(comb.experiment == "latency_dist") & (comb.path == "device_resident")] if "path" in comb else pd.DataFrame()
        if len(ld):
            print("\nsingle-request device-resident latency (us)"); display((ld.pivot_table(index="method", columns="device", values=["lat_p50_ms", "lat_p99_ms"]) * 1e3).round(1))
        nr_ = comb[(comb.experiment == "noisy_recall")]
        if len(nr_):
            acc_t = nr_.pivot_table(index=["method", "n", "noise"], columns="device", values="recall_at_1")
            spread = (acc_t.max(axis=1) - acc_t.min(axis=1)).max()
            print(f"\nmax cross-device spread of recall@1 over identical seeds: {spread:.4f} (0 expected apart from exact-score ties)")
        comb.to_csv("neurostore_cross_hardware_combined.csv", index=False)
        print("saved neurostore_cross_hardware_combined.csv")
else:
    print("To compare runtimes: download each ZIP, then set COMPARE_UPLOADS=True (Colab upload dialog) or COMPARE_GLOB='path/*/results.csv'.")
