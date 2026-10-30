Synthetic reporting fixture only; these are not hardware timings.

| Tier    | Model             | Context (S) | Batch (B) | Tokens Ingested | TQ Ingest BF16-Cube (μs) | Native V5 Ingest (μs) | Speedup vs V5   | Throughput / Note                                                                                             |
| :------ | :---------------- | :---------: | :-------: | :-------------: | :----------------------: | :-------------------: | :-------------: | :------------------------------------------------------------------------------------------------------------ |
| Short   | Qwen3.5-9B        | 8,192       | 2         | 2,048           | 10.000 (20.000 / 8.000)  | 40.000                | 4.000x (2.000x) | 20.5 GB/s effective cache write; warm; recycled; +100.0% throughput gain vs BF16-vec; crosses native V5 speed |
| Mid     | Model, "A"        | 65,536      | 2         | 2,048           | 10.000 (20.000 / 8.000)  | N/A                   | N/A             | 20.5 GB/s effective cache write; warm; recycled; +100.0% throughput gain vs BF16-vec; native unavailable      |
| Long    | DeepSeek-V4-Flash | 262,144     | 2         | 2,048           | 10.000 (20.000 / 8.000)  | 40.000                | 4.000x (2.000x) | 20.5 GB/s effective cache write; warm; recycled; +100.0% throughput gain vs BF16-vec; crosses native V5 speed |
| Extreme | GLM-5.2-744B      | 1,048,576   | 2         | 2,048           | 10.000 (20.000 / 8.000)  | 40.000                | 4.000x (2.000x) | 20.5 GB/s effective cache write; warm; recycled; +100.0% throughput gain vs BF16-vec; crosses native V5 speed |
PASS ingest audit: tiers, formulas, missing/invalid timings, sweep CLI, Markdown and CSV append/escaping
