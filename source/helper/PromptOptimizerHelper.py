import logging
import pickle
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from vllm import LLM, SamplingParams

from source.helper.Helper import Helper

logging.basicConfig(level=logging.INFO)


@dataclass
class PromptOptTask:
    """Represents a single prompt optimization evaluation task"""
    prompt_text: str
    target_label: str
    target_description: str
    text_label_pairs: str


class PromptOptimizerHelper(Helper):
    """
    Optimized helper for prompt optimization using vLLM.

    Features:
    - Efficient batching with configurable batch sizes
    - Synchronous vLLM inference (no async needed)
    - Iterative prompt improvement based on cosine similarity
    - Progress tracking with tqdm
    """

    def __init__(self, params):
        super(PromptOptimizerHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        logging.info(f"Loaded {len(self.samples)} samples")
        logging.info(f"Initializing vLLM with model: {self.params.llm.prompt_opt.model}")

        # Initialize vLLM Engine
        self.llm = LLM(
            model=self.params.llm.prompt_opt.model,
            tensor_parallel_size=self.params.llm.label_desc.tensor_parallel_size,
            gpu_memory_utilization=self.params.llm.label_desc.gpu_memory_utilization,
            trust_remote_code=self.params.llm.label_desc.trust_remote_code,
            max_num_batched_tokens=self.params.llm.label_desc.max_num_batched_tokens,
            max_num_seqs=self.params.llm.label_desc.max_num_seqs,
        )

        # Define sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.params.llm.prompt_opt.temperature,
            top_p=self.params.llm.prompt_opt.top_p,
            max_tokens=self.params.llm.prompt_opt.max_gen_len,
            ignore_eos=False,
        )

        # Meta-prompt sampling parameters (for generating new prompts)
        self.meta_sampling_params = SamplingParams(
            temperature=self.params.llm.prompt_opt.temperature,
            top_p=self.params.llm.prompt_opt.top_p,
            max_tokens=self.params.llm.prompt_opt.max_gen_len,
            stop=["</prompt>"],
            ignore_eos=False,
        )

        self.batch_size = self.params.llm.prompt_opt.batch_size

    def _format_label(self, label: str) -> str:
        """Format label by removing NA suffix if present"""
        splited_label = label.split("->")
        if splited_label[-1] == "NA":
            return splited_label[0]
        return label

    def _format_labels(self, labels: List[str]) -> str:
        """Format multiple labels as semicolon-separated string"""
        return '; '.join([self._format_label(label) for label in labels])

    def _get_candidates(self) -> Dict[str, List[int]]:
        """Build mapping from label to sample indices"""
        candidates = {}
        for sample in tqdm(self.samples, desc="Finding candidates"):
            for label in sample["labels"]:
                if label not in candidates:
                    candidates[label] = []
                candidates[label].append(sample["idx"])
        return candidates

    def _get_text_label_pairs(self, select_ids: List[int]) -> str:
        """Format text-label pairs for few-shot prompting"""
        text_label_pairs = ""
        for sample_idx in select_ids:
            text_label_pairs += f"    text: {' '.join(self.samples[sample_idx]['text'].split()[:128])}\n"
            text_label_pairs += f"    labels: {self._format_labels(self.samples[sample_idx]['labels'])}\n\n"
        return text_label_pairs

    def _get_prompt_samples(
            self,
            target_descriptions: Dict[str, str],
            candidates: Dict[str, List[int]]
    ) -> List[Dict]:
        """Generate evaluation samples for prompt optimization"""
        prompt_samples = []
        for target_label, target_description in target_descriptions.items():
            for _ in range(self.params.llm.prompt_opt.num_samples_per_target_label):
                selected_ids = random.choices(candidates[target_label], k=5)  # 5 examples per sample
                prompt_samples.append({
                    "text_label_pairs": self._get_text_label_pairs(selected_ids),
                    "target_label": self._format_label(target_label),
                    "target_description": target_description
                })
        return prompt_samples

    def _extract_prompt(self, response: str) -> str:
        """Extract prompt from XML tags in response"""
        pattern = r'<prompt>(.*?)<\/prompt>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return ""

    def _is_valid_prompt(self, prompt: str, required_vars: List[str]) -> bool:
        """Check if prompt contains all required variables"""
        if not prompt:
            return False
        missing_vars = [var for var in required_vars if var not in prompt]
        return len(missing_vars) == 0

    def _generate_evaluation_queue(
            self,
            description_prompt: str,
            prompt_samples: List[Dict]
    ) -> deque:
        """Generate queue of evaluation tasks for a candidate prompt"""
        eval_queue = deque()

        for sample in prompt_samples:
            prompt_text = description_prompt.format(
                text_label_pairs=sample["text_label_pairs"],
                target_label=sample["target_label"]
            )

            eval_queue.append(
                PromptOptTask(
                    prompt_text=prompt_text,
                    target_label=sample["target_label"],
                    target_description=sample["target_description"],
                    text_label_pairs=sample["text_label_pairs"]
                )
            )

        return eval_queue

    def _process_evaluation_batch(
            self,
            batch_tasks: List[PromptOptTask],
            progress_bar: Optional[tqdm] = None
    ) -> tuple[List[str], List[str]]:
        """
        Process a batch of evaluation tasks with vLLM.

        Returns:
            pred_descriptions: List of predicted descriptions
            true_descriptions: List of target descriptions
        """
        prompts = [task.prompt_text for task in batch_tasks]

        try:
            # Run batch inference
            outputs = self.llm.generate(prompts, self.sampling_params)

            pred_descriptions = []
            true_descriptions = []

            # Process outputs
            for i, output in enumerate(outputs):
                task = batch_tasks[i]

                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text.strip()
                    pred_descriptions.append(generated_text)
                    true_descriptions.append(task.target_description)
                else:
                    # If generation failed, use empty string
                    pred_descriptions.append("")
                    true_descriptions.append(task.target_description)

            if progress_bar:
                progress_bar.update(len(batch_tasks))

            return pred_descriptions, true_descriptions

        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            # Return empty predictions
            return (
                [""] * len(batch_tasks),
                [task.target_description for task in batch_tasks]
            )

    def _evaluate_prompt(
            self,
            description_prompt: str,
            prompt_samples: List[Dict]
    ) -> float:
        """
        Evaluate a candidate prompt by computing cosine similarity
        between predicted and target descriptions.

        Args:
            description_prompt: The prompt template to evaluate
            prompt_samples: List of evaluation samples

        Returns:
            Mean cosine similarity score
        """
        eval_queue = self._generate_evaluation_queue(description_prompt, prompt_samples)
        total_tasks = len(eval_queue)

        all_pred_descriptions = []
        all_true_descriptions = []

        logging.info(f"Evaluating prompt with {total_tasks} tasks")

        with tqdm(total=total_tasks, desc="Evaluating prompt") as pbar:
            while eval_queue:
                # Dequeue batch
                batch_size = min(self.batch_size, len(eval_queue))
                batch_tasks = [eval_queue.popleft() for _ in range(batch_size)]

                # Process batch
                pred_descs, true_descs = self._process_evaluation_batch(batch_tasks, pbar)

                all_pred_descriptions.extend(pred_descs)
                all_true_descriptions.extend(true_descs)

        # Compute effectiveness
        score = self._compute_effectiveness(all_true_descriptions, all_pred_descriptions)
        logging.info(f"Prompt effectiveness: {score:.4f}")

        return score

    def _compute_effectiveness(
            self,
            true_descriptions: List[str],
            pred_descriptions: List[str]
    ) -> float:
        """Compute mean cosine similarity between true and predicted descriptions"""
        embeddings1 = self.embedding_model.encode(true_descriptions, convert_to_numpy=True)
        embeddings2 = self.embedding_model.encode(pred_descriptions, convert_to_numpy=True)
        similarities = np.diag(cosine_similarity(embeddings1, embeddings2))
        return np.mean(similarities)

    def _generate_new_prompt(
            self,
            meta_prompt: str,
            prompts_history: List[tuple]
    ) -> str:
        """
        Generate a new candidate prompt using the meta-prompt.

        Args:
            meta_prompt: Meta-prompt template
            prompts_history: List of (prompt, score) tuples

        Returns:
            New candidate prompt
        """
        # Get current best prompt
        current_best_prompt = max(prompts_history, key=lambda x: x[1])[0]

        # Format top prompts with scores
        top_prompts = sorted(prompts_history, key=lambda x: x[1], reverse=True)[:2]
        prompts_scores = "\n\n".join(
            [f"Prompt:\n<prompt>\n{prompt}\n</prompt>\nScore: {score:.4f}"
             for prompt, score in top_prompts]
        )

        # Format meta-prompt
        formatted_meta_prompt = meta_prompt.format(
            prompts_scores=prompts_scores,
            description_prompt=current_best_prompt,
            prompt_samples="" # We don't include samples in the meta-prompt
        )

        # Generate new prompt
        logging.info("Generating new candidate prompt...")
        outputs = self.llm.generate([formatted_meta_prompt], self.meta_sampling_params)

        if outputs and outputs[0].outputs:
            response = outputs[0].outputs[0].text.strip() + "</prompt>"
            new_prompt = self._extract_prompt(response)

            # Validate prompt
            max_retries = 3
            retries = 0
            while not self._is_valid_prompt(new_prompt, ["{text_label_pairs}", "{target_label}"]) and retries < max_retries:
                logging.warning(f"Generated invalid prompt (attempt {retries + 1}/{max_retries}), retrying...")
                outputs = self.llm.generate([formatted_meta_prompt], self.meta_sampling_params)
                if outputs and outputs[0].outputs:
                    response = outputs[0].outputs[0].text.strip() + "</prompt>"
                    new_prompt = self._extract_prompt(response)
                retries += 1

            if not self._is_valid_prompt(new_prompt, ["{text_label_pairs}", "{target_label}"]):
                logging.error("Failed to generate valid prompt, using current best")
                return current_best_prompt

            return new_prompt
        else:
            logging.error("Failed to generate new prompt, using current best")
            return current_best_prompt

    def _optimize_prompt(
            self,
            meta_prompt: str,
            seed_prompt: str,
            prompt_samples: List[Dict]
    ) -> List[tuple]:
        """
        Optimize prompt through iterative refinement.

        Args:
            meta_prompt: Meta-prompt for generating new prompts
            seed_prompt: Initial prompt template
            prompt_samples: Evaluation samples

        Returns:
            List of (prompt, score) tuples
        """
        # Evaluate seed prompt
        logging.info("Evaluating seed prompt...")
        seed_score = self._evaluate_prompt(seed_prompt, prompt_samples)
        prompts_history = [(seed_prompt, seed_score)]

        logging.info(f"Seed prompt score: {seed_score:.4f}")

        # Iterative optimization
        for epoch in range(self.params.llm.prompt_opt.num_epochs):
            logging.info(f"\n{'='*70}")
            logging.info(f"OPTIMIZATION EPOCH {epoch + 1}/{self.params.llm.prompt_opt.num_epochs}")
            logging.info(f"{'='*70}")

            # Generate new candidate prompt
            new_prompt = self._generate_new_prompt(meta_prompt, prompts_history)

            # Evaluate new prompt
            new_score = self._evaluate_prompt(new_prompt, prompt_samples)

            # Add to history
            prompts_history.append((new_prompt, new_score))

            # Log progress
            best_score = max(prompts_history, key=lambda x: x[1])[1]
            logging.info(f"New prompt score: {new_score:.4f}")
            logging.info(f"Best score so far: {best_score:.4f}")

        return prompts_history

    def run(self):
        """Main execution method"""
        logging.info(
            f"Optimizing prompt for {self.params.data.name}\n"
            f"Params:\n{OmegaConf.to_yaml(self.params.llm.prompt_opt)}"
        )

        # Load prompts
        seed_prompt = self._load_prompt("seed_prompt")
        meta_prompt = self._load_prompt("meta_prompt")
        target_descriptions = self._load_target_descriptions()

        # Get candidates and prompt samples
        candidates = self._get_candidates()
        prompt_samples = self._get_prompt_samples(target_descriptions, candidates)

        logging.info(f"Generated {len(prompt_samples)} evaluation samples")

        # Optimize prompt
        prompts_history = self._optimize_prompt(meta_prompt, seed_prompt, prompt_samples)

        # Get best prompt
        best_prompt = max(prompts_history, key=lambda x: x[1])[0]
        best_score = max(prompts_history, key=lambda x: x[1])[1]


        logging.info(f"Best score: {best_score:.4f}")
        logging.info(f"\nBest prompt:\n{best_prompt}")

        # Save optimized prompt
        self._checkpoint_prompt(best_prompt, "optimized_prompt")
