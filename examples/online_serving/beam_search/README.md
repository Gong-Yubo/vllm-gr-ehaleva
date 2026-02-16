# Installation
refer to [quickstart.md](../../../docs/quickstart.md) for installation

# running example
- This example demonstrates using vllm-gr for executing generative recommendation system based OneRec-1.7B model

- Run vllm-gr server
```bash
vllm serve --gr \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 2048 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    OpenOneRec/OneRec-1.7B
```

run client (on another terminal)
- the client execute a predefined 120 items prompt and ask for item recommendations.
- the server would reply with **beam_width** suggested items.

```bash
./client.sh [-b beam_width ]


```
# Expected result
- sorted recommendations in json format with fields:
    - content: recommended token
- total execution time

```bash
./client.sh -b 512  
```

Server without --gr flag

```json
{"id":"chatcmpl-8b52e35966c0f8a1","object":"chat.completion","created":1770557950,"model":"OpenOneRec/OneRec-1.7B","choices":[{"index":0,"message":{"role":"assistant","content":"<|sid_begin|><s_a_6240><s_b_101><s_c_3136><|sid_end|>","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}...{"index":511...}}
total time:16.198407s
```

Server with --gr flag

```json
{"id":"chatcmpl-a779679c38e57685","object":"chat.completion","created":1770297709,"model":"OpenOneRec/OneRec-1.7B","choices":
{"index":0,"message":{"role":"assistant","content":"<|sid_begin|><s_a_4113><s_b_7330><s_c_2009><|sid_end|>","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}...{"index":511...}}
total time: 1.610658s
```

- Example was demonstrated on:
    - Host: Intel(R) Xeon(R) Gold 5320T CPU @ 2.30GHz
    - Device: Nvidia RTX A6000 48GB
