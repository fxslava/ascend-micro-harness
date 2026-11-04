# Native CAModel timeline

`trace.json` is the unmodified Chrome-trace JSON exported by CANN `msprof op
simulator` for `turboquant_mm_fused_decode_opt_kv4fp8_half`. Load it in a
Chrome-trace-compatible timeline viewer, such as Perfetto.

* SoC: Ascend950PR_9599, CANN 9.2.0-beta.2.
* Benchmark: `--impl fused-opt --single --context 384 --page-tokens 128
  --heads 1 --rows 16 --requests 1 --splitk 1`.
* One dispatch, six 64-token tiles, three-slot ring reused, two score windows.
* One Cube and both Vector cores; 12 Cube compute events.
* 67,431 native events; 16,018,888 bytes.
* Benchmark CPU reference passed; profiler exited successfully.
* Times are simulator values, not measured hardware latencies.

`trace-validation.json` records event/lane counts and the SHA-256 of the exact
export. `command.txt`, `application.sh`, `environment.txt`, `config/`,
`profile.log` and `results.csv` preserve capture provenance.

Reproduce using a completed benchmark build:

```sh
bash scripts/capture_fused_decode_opt_camodel_json.sh /tmp/ascend-fused-opt-bench.yOuivK
```
