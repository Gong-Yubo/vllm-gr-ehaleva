# Using catalog to constrain recommendation results

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting the Catalog](#getting-the-catalog)
- [Invoking the Server](#invoking-the-server)
- [Invoking the Client](#invoking-the-client)

## Overview
This example demonstrates using an items catalog to eliminate non-existing items given by the recommendation system.
Limiting the available items increases accuracy and enables reducing the search space to get the same accuracy but with less latency.

## Installation
- Refer to [quickstart.md](../../../docs/quickstart.md) for installation.

## Getting the Catalog
- Follow [Open One Rec Benchmark Dataset](../../../docs/benchmarks/README.md#open-one-rec-benchmark-dataset) for One Rec Benchmark Dataset retrieval.

- Use `convert_catalog.py` script to convert OpenOneRec catalog to vllm-gr catalog format.
- OpenOneRec catalog is in `<OpenOneRecFolder>/benchmark_data/sid2pid.json`
```bash
python convert_catalog.py ${OpenOneRecFolder}/benchmark_data/sid2pid.json ${MyFolder}/catalog.json
```


## Invoking the Server
- Add the catalog attribute to server invocation
```bash
 vllm serve --gr \
    --seed 123 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --catalog-path ${MyFolder}/catalog.json \
     OpenOneRec/OneRec-1.7B
```
## Invoking the Client
- Client execution does not change
- see beam_search example [README.md](../beam_search/README.md#run-client-on-another-terminal) for client single prompt execution.
- see benchmark documentation [README.md](../../../docs/benchmarks/README.md) for available benchmarks for online serving.

## Result verification

A utility was added to check if client results are all within the catalog
```bash
bash examples/online_serving/beam_search/client.sh > result.json
python examples/online_serving/catalog/verify_catalog.py result.json ${MyFolder}/catalog.json
```

Expected results:
```
Loading results from ../vllm/trc/test/result.json...
Loading catalog from /home/yoni/OpenOneRec/benchmark_data/catalog.json...
Catalog loaded with 1519078 items.
First 10 items in catalog: ['<|sid_begin|><s_a_7060><s_b_18><s_c_1748><|sid_end|>', '<|sid_begin|><s_a_522><s_b_2833><s_c_5018><|sid_end|>', '<|sid_begin|><s_a_5368><s_b_7001><s_c_5852><|sid_end|>', '<|sid_begin|><s_a_6118><s_b_1302><s_c_4114><|sid_end|>', '<|sid_begin|><s_a_4990><s_b_5835><s_c_5629><|sid_end|>', '<|sid_begin|><s_a_3492><s_b_2549><s_c_384><|sid_end|>', '<|sid_begin|><s_a_1280><s_b_6846><s_c_7175><|sid_end|>', '<|sid_begin|><s_a_1848><s_b_3131><s_c_3721><|sid_end|>', '<|sid_begin|><s_a_5001><s_b_2547><s_c_7816><|sid_end|>', '<|sid_begin|><s_a_4616><s_b_4751><s_c_2473><|sid_end|>']
Verifying 512 generated choices...

SUCCESS: All generated items exist in the catalog.
```
