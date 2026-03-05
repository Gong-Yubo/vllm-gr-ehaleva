
bw=512
while getopts "b:" opt; do
  case $opt in
    b) bw=$OPTARG ;;
  esac
done

prompt=\"$(cat ../../../tests/resources/single_one_rec_prompt.txt)\"

curl -w " total time:%{time_total}s\n" -X  POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "OpenOneRec/OneRec-1.7B",
		"messages": [
			{
				"role": "user",
				"content": '"$prompt"'
			}
		],
        "use_beam_search": true,
         "n": '"$bw"',
        "temperature": 0.0,
        "max_tokens": 5,
        "ignore_eos": false ,
		    "vllm_xargs": {
            "begin_token": "<|sid_begin|>",
            "end_token": "<|sid_end|>"
        }
	}'
