# Browser/WASM runtime evidence

The checked report measures the pinned q8 ONNX classifier through Transformers.js with the
WASM backend in Chrome 151 on macOS. It is browser JavaScript evidence, not native ONNX or
cloud-serving latency.

| Tokenizer length | Warm p50 | Warm p95 | Queries |
|---:|---:|---:|---:|
| 10 | 41.00 ms | 41.70 ms | 30 |
| 34 | 118.00 ms | 118.85 ms | 30 |
| 130 | 439.75 ms | 446.67 ms | 30 |
| 258 | 894.45 ms | 897.32 ms | 30 |

The first pipeline initialization took 13,609.6 ms and reported 127,639,953 model/tokenizer
bytes. Browser-managed caches were not cleared, so this is the observed first initialization
for that page origin—not a guaranteed network-cold download. JavaScript heap use increased by
2,570,376 bytes from immediately before to immediately after the warm benchmark. The raw report
contains every latency, browser/hardware details, failures, resource timing, model revision,
and the benchmark page's SHA-256.

To reproduce, serve `docs/` over HTTP, open the page in Chrome, and choose **Run runtime
benchmark**. The page runs three warmups and 30 sequential measured queries at each length.
Verify that the report's `source_document_sha256` matches `docs/index.html` before comparing
results. Do not combine these numbers with the separate native ONNX measurements.
