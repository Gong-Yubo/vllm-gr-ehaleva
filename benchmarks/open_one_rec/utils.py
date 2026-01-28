# Imported from: https://https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)
"""
Label Prediction Task Utilities

Functions for label extraction, probability processing, and AUC/wuAUC computation.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_after_think(text: str) -> str:
    """Extract text after the last </think> tag if present."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_label_from_answer(answer: str) -> int:
    """
    Extract binary label from answer string

    Args:
        answer: Answer string (e.g., "是<|im_end|>" or "否<|im_end|>")

    Returns:
        1 if positive ("是"), 0 if negative ("否"), -1 if unrecognized

    Examples:
        >>> extract_label_from_answer("是<|im_end|>")
        1
        >>> extract_label_from_answer("否")
        0
    """
    if "是" in answer:
        return 1
    elif "否" in answer:
        return 0
    else:
        return -1


def extract_probability_from_logprobs(
    generations: List[str],
    positive_token: str = "是",
    negative_token: str = "否",
    sample_id: str = None,
) -> Dict[str, Any]:
    """
    Extract probability for positive class from generations list containing reasoning and JSON probabilities.
    Applies softmax normalization to ensure probabilities sum to 1.

    Args:
        generations: List of strings, each containing reasoning text followed by JSON probabilities
        positive_token: Token representing positive class (default "是")
        negative_token: Token representing negative class (default "否")
        sample_id: Optional sample ID for error messages

    Returns:
        Dictionary containing:
        - parsed_probs: Original parsed probabilities before normalization
        - normalized_probs: Softmax normalized probabilities
        - score: Final positive class probability (after normalization)

    Raises:
        ValueError: If JSON parsing fails or required tokens are missing

    Examples:
        >>> generations = ['</think>\\n{"是": 0.7, "否": 0.3}']
        >>> result = extract_probability_from_logprobs(generations)
        >>> result['score']
        0.7
        >>> result['normalized_probs']
        {'是': 0.7, '否': 0.3}
    """
    parsed_list = []
    normalized_list = []
    scores = []

    for idx, generation in enumerate(generations):
        # Extract JSON part: check for </think> tag first
        if "</think>" in generation:
            # Extract content after </think>
            json_str = generation.split("</think>")[-1].strip()
        else:
            # No </think> tag, try to parse the entire string
            json_str = generation.strip()

        # Parse JSON and extract probability
        try:
            probs_dict = json.loads(json_str)

            # Validate that it's a dict and contains required tokens
            if not isinstance(probs_dict, dict):
                raise ValueError(f"Parsed JSON is not a dictionary: {type(probs_dict)}")

            if positive_token not in probs_dict:
                raise ValueError(
                    f"Positive token '{positive_token}' not found in probabilities: {probs_dict}"
                )

            if negative_token not in probs_dict:
                raise ValueError(
                    f"Negative token '{negative_token}' not found in probabilities: {probs_dict}"
                )

            # Extract probabilities
            p_pos = float(probs_dict[positive_token])
            p_neg = float(probs_dict[negative_token])

            # Apply softmax normalization (ensure probabilities sum to 1)
            total = p_pos + p_neg
            if total <= 0:
                raise ValueError(f"Sum of probabilities is non-positive: {total}")

            p_pos_normalized = p_pos / total
            p_neg_normalized = p_neg / total

            # Store results
            parsed_list.append({positive_token: p_pos, negative_token: p_neg})
            normalized_list.append(
                {
                    positive_token: p_pos_normalized,
                    negative_token: p_neg_normalized,
                }
            )
            scores.append(p_pos_normalized)

        except (
            json.JSONDecodeError,
            TypeError,
            AttributeError,
            ValueError,
            KeyError,
        ) as e:
            # Raise detailed exception
            error_msg = "Failed to parse generation"
            if sample_id:
                error_msg += f" for sample_id '{sample_id}'"
            error_msg += f" at index {idx}:\n"
            error_msg += (
                f"  Generation: {generation[:200]}..."
                if len(generation) > 200
                else f"  Generation: {generation}\n"
            )
            error_msg += f"\n  Error: {str(e)}"
            raise ValueError(error_msg)

    # If no valid probabilities were found (empty list), raise error
    if not scores:
        error_msg = "No valid probabilities found in generations"
        if sample_id:
            error_msg += f" for sample_id '{sample_id}'"
        raise ValueError(error_msg)

    # Average across all valid elements (usually just one element)
    if len(scores) == 1:
        return {
            "parsed_probs": parsed_list[0],
            "normalized_probs": normalized_list[0],
            "score": scores[0],
        }
    else:
        # Average the probabilities for each token
        avg_parsed = {
            positive_token: sum(p[positive_token] for p in parsed_list) / len(parsed_list),
            negative_token: sum(p[negative_token] for p in parsed_list) / len(parsed_list),
        }
        avg_normalized = {
            positive_token: sum(p[positive_token] for p in normalized_list) / len(normalized_list),
            negative_token: sum(p[negative_token] for p in normalized_list) / len(normalized_list),
        }
        avg_score = sum(scores) / len(scores)

        return {
            "parsed_probs": avg_parsed,
            "normalized_probs": avg_normalized,
            "score": avg_score,
        }


def calculate_auc(predictions: Dict[str, float], labels: Dict[str, int]) -> float:
    """
    Calculate AUC (Area Under ROC Curve) using sklearn

    Args:
        predictions: Predicted probabilities, format: {sample_id: probability}
        labels: Ground truth labels (0 or 1), format: {sample_id: label}

    Returns:
        AUC value (float between 0 and 1)
    """
    if not predictions or not labels:
        logger.info("[red]✗ Predictions or labels are empty[/red]")
        return 0.0

    # Align predictions and labels
    sample_ids = sorted(set(predictions.keys()) & set(labels.keys()))

    if len(sample_ids) == 0:
        logger.info("[red]✗ No overlapping samples between predictions and labels[/red]")
        return 0.0

    y_true = np.array([labels[id] for id in sample_ids])
    y_scores = np.array([predictions[id] for id in sample_ids])

    # Check if we have both positive and negative samples
    if len(np.unique(y_true)) < 2:
        logger.info("[yellow]⚠ Only one class present in labels, AUC is not defined[/yellow]")
        return 0.5

    try:
        # Calculate AUC using sklearn
        auc = roc_auc_score(y_true, y_scores)
        return float(auc)
    except ValueError as e:
        logger.info(f"[red]✗ Error calculating AUC: {e}[/red]")
        return 0.5


def get_debug_info(
    sample_id: str,
    logprobs_dict: Dict[str, float],
    predicted_prob: float,
    ground_truth: str,
    label: int,
    user_id: str = "",
) -> Dict[str, Any]:
    """
    Prepare debug information for a sample

    Args:
        sample_id: Sample ID
        logprobs_dict: Dictionary of token probabilities
        predicted_prob: Predicted probability for positive class
        ground_truth: Ground truth answer string
        label: Ground truth label (0 or 1)
        user_id: User ID (optional)

    Returns:
        Debug information dictionary
    """
    debug_item = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "label": label,
        "predicted_prob": predicted_prob,
        "logprobs": logprobs_dict,
    }

    if user_id:
        debug_item["user_id"] = user_id

    return debug_item


def extract_refined_reasoning(text: str) -> str:
    """
    Extract the refined reasoning section from the full text.

    Finds the last occurrence of "精炼推理" and extracts the text after it.

    Args:
        text: Full text containing the reasoning

    Returns:
        Extracted refined reasoning text, or original text if pattern not found
    """
    if not text:
        return ""

    # Find the last occurrence of "精炼推理"
    keyword = "精炼推理"
    last_pos = text.rfind(keyword)

    if last_pos != -1:
        # Extract text after "精炼推理"
        after_keyword = text[last_pos + len(keyword) :]
        # Remove leading punctuation, whitespace, and markdown symbols
        after_keyword = re.sub(r"^[\s\*#：:\n]+", "", after_keyword)
        if after_keyword.strip():
            return after_keyword.strip()

    # If "精炼推理" not found, return original text
    return text.strip()


def get_cache_path(save_dir: str, model_name: str) -> str:
    """Get the path for evaluation results cache file."""
    return os.path.join(save_dir, f"llm_eval_{model_name}.json")


def load_eval_cache(cache_path: str) -> Optional[Dict[str, Dict]]:
    """
    Load evaluation results from cache.

    Args:
        cache_path: Path to cache file

    Returns:
        Dict of {sample_id: evaluation_result} or None if not found
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract evaluation results
        eval_results = {}
        for sample_id, sample_data in data.items():
            if "eval_result" in sample_data and sample_data["eval_result"]:
                eval_results[sample_id] = sample_data["eval_result"]

        logger.info(f"Loaded {len(eval_results)} cached evaluation results")
        return eval_results

    except Exception as e:
        logger.info(f"Failed to load evaluation cache: {e}")
        return None


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response.

    Args:
        response: LLM response text

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    if not response:
        return None

    try:
        response = response.strip()
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        return json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.info(f"Failed to parse JSON: {response[:200]}...")
        return None


def evaluate_single(
    gt_reasoning: str, model_reasoning: str, llm_client
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Evaluate a single sample using LLM.

    Args:
        gt_reasoning: Ground truth refined reasoning
        model_reasoning: Model-generated refined reasoning
        llm_client: LLM client instance

    Returns:
        Tuple of (evaluation_result, error_message)
    """

    EVALUATION_PROMPT = """你是一位专业的推荐系统评估专家。你的任务是评估一个AI模型生成的"推荐理由"与"标准答案"的匹配程度。

        ### 评估任务
        请对模型生成的推荐理由进行综合评分（1-5分）。

        **核心评估原则：**
        请严格按照以下步骤进行思考和评估：
        1.  **核心要素提取**：
            - 从【标准答案】中提取：推荐的核心动机（用户为什么看）+ 推荐的内容类型（看的是什么）。
            - 从【模型生成】中提取：推荐的核心动机 + 推荐的内容类型。
        2.  **噪音过滤（关键步骤）**：
            - 忽略具体的措辞差异（同义词替换）。
            - **忽略与推荐逻辑无关的用户画像细节**（例如：具体的年龄数字、与推理逻辑和视频内容无关的兴趣等）。
        3.  **匹配度分析**：
            - 对比核心动机：是否抓住了相同的推荐的核心动机？
            - 对比内容方向：推荐的视频类别/主题是否一致？
        4.  **评分**：基于评分标准给出最终得分。

        **评分标准：**
        - 5分：核心逻辑与内容方向完全一致。即使表达方式不同，但语义内核完全相同。
        - 4分：核心逻辑正确，内容方向正确。可能遗漏了标准答案中极次要的补充信息，或包含了无伤大雅的冗余信息。
        - 3分：大方向（如视频类型）正确，但对“用户为什么喜欢”的归因不够准确，或遗漏了关键的转化动机。
        - 2分：推荐逻辑有明显误读，或者推荐的内容类型与标准答案有偏差（例如：把“学习教程”理解成了“娱乐搞笑”）。
        - 1分：逻辑和内容完全错误，或生成了风马牛不相及的内容。

        ### 输入

        **[标准答案]**
        {}

        **[模型生成]**
        {}

        ### 输出格式
        你的输出必须是【纯粹的 JSON 格式】，可以被 `json.loads` 直接解析。

        ```json
        {{
        "llm_score": <1-5的整数>,
        "llm_reason": "<简短的打分理由，不超过50字>"
        }}
        ```

        你的评估结果 (请严格按照上述要求返回一个格式规整的 JSON，可以被 json.loads 直接解析。请不要在 JSON 数据前后添加任何额外的解释性文字或代码块标记): """

    prompt = EVALUATION_PROMPT.format(gt_reasoning, model_reasoning)

    try:
        response = llm_client.generate(prompt)
        result = extract_json_from_response(response)

        if result is not None and "llm_score" in result:
            # Ensure score is in valid range
            score = result["llm_score"]
            if not isinstance(score, (int, float)) or score < 1 or score > 5:
                result["llm_score"] = 3  # Default to middle score if invalid
            return result, None

        return None, f"Failed to parse JSON or missing 'llm_score': {response[:100]}"

    except Exception as e:
        return None, f"API error: {str(e)}"


def evaluate_batch(
    gt_reasonings: Dict[str, str],
    model_reasonings: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    desc: str = "Evaluating reasoning",
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Evaluate multiple samples in parallel.

    Args:
        gt_reasonings: Dict of {sample_id: gt_reasoning}
        model_reasonings: Dict of {sample_id: model_reasoning}
        llm_client: LLM client instance
        max_workers: Number of concurrent workers
        desc: Progress bar description

    Returns:
        Tuple of (results, errors)
    """
    results = {}
    errors = {}

    # Only evaluate samples that have both GT and model reasoning
    common_ids = set(gt_reasonings.keys()) & set(model_reasonings.keys())
    common_ids = {id for id in common_ids if gt_reasonings[id] and model_reasonings[id]}

    def process_single(sample_id: str):
        gt = gt_reasonings[sample_id]
        model = model_reasonings[sample_id]
        result, error = evaluate_single(gt, model, llm_client)
        return sample_id, result, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, sid): sid for sid in common_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sample_id, result, error = future.result()
            if result is not None:
                results[sample_id] = result
            if error is not None:
                errors[sample_id] = error

    # Statistics
    total_attempted = len(common_ids)
    total_success = len(results)

    logger.info(f"{desc} statistics: {total_attempted} attempted, {total_success} successful")

    return results, errors


def calculate_metrics(eval_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Calculate micro and macro metrics from evaluation results.

    Args:
        eval_results: Dict of {sample_id: evaluation_result}

    Returns:
        Dict with micro/macro scores
    """
    if not eval_results:
        return {}

    # Collect scores
    scores = []
    for sample_id, result in eval_results.items():
        if "llm_score" in result:
            score = result["llm_score"]
            if isinstance(score, (int, float)) and 1 <= score <= 5:
                scores.append(score)

    metrics = {}

    if scores:
        # micro and macro are the same for single score
        avg_score = sum(scores) / len(scores)
        metrics["micro_llm_score"] = avg_score
        metrics["macro_llm_score"] = avg_score
        metrics["llm_score"] = avg_score

    metrics["llm_eval_num_samples"] = len(eval_results)

    return metrics


def get_per_sample_metrics(eval_results: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-sample metrics from evaluation results.

    Args:
        eval_results: Dict of {sample_id: evaluation_result}

    Returns:
        Dict of {sample_id: {llm_score, llm_reason}}
    """
    per_sample = {}

    for sample_id, result in eval_results.items():
        sample_metrics = {}

        if "llm_score" in result:
            sample_metrics["llm_score"] = result["llm_score"]

        if "llm_reason" in result:
            sample_metrics["llm_reason"] = result["llm_reason"]

        per_sample[sample_id] = sample_metrics

    return per_sample


def save_eval_results(
    save_dir: str,
    sample_ids: List[str],
    gt_reasonings: Dict[str, str],
    model_reasonings: Dict[str, str],
    eval_results: Dict[str, Dict],
    model_name: str,
):
    """
    Save evaluation results to file.

    Args:
        save_dir: Directory to save the file
        sample_ids: List of sample IDs
        gt_reasonings: Dict of {sample_id: gt_reasoning}
        model_reasonings: Dict of {sample_id: model_reasoning}
        eval_results: Dict of {sample_id: evaluation_result}
        model_name: Model name for filename
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = get_cache_path(save_dir, model_name)

    detailed_results = {}

    for sample_id in sample_ids:
        detailed_results[sample_id] = {
            "gt_reasoning": gt_reasonings.get(sample_id, ""),
            "model_reasoning": model_reasonings.get(sample_id, ""),
            "eval_result": eval_results.get(sample_id, {}),
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation results saved to {save_path}")


def evaluate_reasoning(
    predictions: Dict[str, str],
    references: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    max_samples: Optional[int] = None,
    model_name: str = "unknown",
    save_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Main evaluation function for recommendation reasoning.

    Args:
        predictions: Dict of {sample_id: prediction_text}
        references: Dict of {sample_id: reference_text}
        llm_client: LLM client instance for evaluation
        max_workers: Number of concurrent workers
        max_samples: Maximum samples to evaluate (None for all)
        model_name: Model name for cache file naming
        save_dir: Directory to save results

    Returns:
        Tuple of (metrics, per_sample_metrics)
    """
    # Select samples
    all_sample_ids = sorted(set(predictions.keys()) & set(references.keys()))

    if max_samples is not None and max_samples < len(all_sample_ids):
        sample_ids = all_sample_ids[:max_samples]
        logger.info(f"Selected {len(sample_ids)} samples for LLM evaluation")
    else:
        sample_ids = all_sample_ids
        logger.info(f"Evaluating all {len(sample_ids)} samples")

    # Extract refined reasoning from both GT and model outputs
    logger.info("Extracting refined reasoning...")
    gt_reasonings = {}
    model_reasonings = {}

    for sample_id in sample_ids:
        # Extract from reference (GT)
        gt_text = references.get(sample_id, "")
        gt_reasonings[sample_id] = extract_refined_reasoning(gt_text)

        # Extract from prediction (model output)
        pred_text = predictions.get(sample_id, "")
        pred_text = extract_after_think(pred_text)  # Remove <think> tags first
        model_reasonings[sample_id] = extract_refined_reasoning(pred_text)

    # Load cached results if available
    eval_results = {}
    if save_dir:
        cache_path = get_cache_path(save_dir, model_name)
        cached_results = load_eval_cache(cache_path)
        if cached_results:
            eval_results = {k: v for k, v in cached_results.items() if k in sample_ids}

    # Find samples that need evaluation
    missing_ids = set(sample_ids) - set(eval_results.keys())
    missing_ids = {id for id in missing_ids if gt_reasonings.get(id) and model_reasonings.get(id)}

    if missing_ids:
        logger.info(f"Evaluating {len(missing_ids)} samples with LLM...")
        missing_gt = {id: gt_reasonings[id] for id in missing_ids}
        missing_model = {id: model_reasonings[id] for id in missing_ids}

        new_results, errors = evaluate_batch(missing_gt, missing_model, llm_client, max_workers)
        eval_results.update(new_results)

        if errors:
            logger.error(f"Evaluation errors: {len(errors)} samples")
    else:
        logger.info(f"All {len(sample_ids)} samples already evaluated (from cache)")

    # Calculate metrics
    metrics = calculate_metrics(eval_results)
    per_sample_metrics = get_per_sample_metrics(eval_results)

    # Save results
    if save_dir:
        save_eval_results(
            save_dir, sample_ids, gt_reasonings, model_reasonings, eval_results, model_name
        )

    # Print summary
    logger.info(f"LLM evaluation completed: {metrics.get('llm_eval_num_samples', 0)} samples")
    logger.info(f" LLM Score: {metrics.get('llm_score', 0):.4f}")

    return metrics, per_sample_metrics
