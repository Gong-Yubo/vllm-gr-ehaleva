import random

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest, TokenizerLike

from .open_one_rec_loader import DataLoaderWrapper

# -----------------------------------------------------------------------------
# OpenOneRec Dataset Implementation
# -----------------------------------------------------------------------------


class OneRecDataset(BenchmarkDataset):
    """
    Custom dataset for OneRec benchmark with multiple task types.
    Loads data using data_loader.load_data() with task_name, split, and sample_size.
    """

    def __init__(self, task_types: list[str] = None, model_path: str = None, **kwargs) -> None:
        self.task_types = task_types or [
            "rec_reason",
            "item_understand",
            "ad",
            "product",
            "label_cond",
            "video",
            "interactive",
            "label_pred",
        ]

        self.tokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        self.data_loader = DataLoaderWrapper(
            model_path=model_path,
            benchmark_version="v1.0",
            data_dir=self.dataset_path,
            enable_thinking=None,
        )

        self.load_data()

    def load_data(self) -> None:
        self.data = []

        # Import or instantiate your data_loader here
        for task_type in self.task_types:
            # load the data
            test_data = self.data_loader.load_data(
                task_name=task_type,
                split="test",
                sample_size=getattr(self, "sample_size", None),
            )

            # Convert to expected format
            for id, data in test_data.items():
                output_len = data.get(
                    "expected_output_len", 5
                )  # Default to 5 for short GR generations

                self.data.append(
                    {
                        "prompt": data["prompt"],
                        "expected_output_len": output_len,
                        "ground_truth": data["ground_truth"],
                        "task_type": task_type,
                        "id": id,
                    }
                )

        random.seed(self.random_seed)
        if not getattr(self, "disable_shuffle", False):
            random.shuffle(self.data)

    def sample(self, tokenizer: TokenizerLike, num_requests: int, **kwargs) -> list:
        sampled_requests = []
        for i, item in enumerate(self.data[:num_requests]):
            prompt = item["prompt"]
            prompt_len = len(tokenizer(prompt).input_ids)
            expected_output_len = item["expected_output_len"]
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=expected_output_len,
                    request_id=f"{item['task_type']}_{item['id']}",
                )
            )
        return sampled_requests
