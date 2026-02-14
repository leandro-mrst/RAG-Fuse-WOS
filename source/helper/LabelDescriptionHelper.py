import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from vllm import LLM, SamplingParams

from source.helper.Helper import Helper

logging.basicConfig(level=logging.INFO)


@dataclass
class PromptTask:
    """Represents a single prompt generation task"""
    label_idx: int
    label: str
    prompt: str


class LabelDescriptionHelper(Helper):
    """
    Optimized helper for generating label descriptions using vLLM.

    Features:
    - Efficient batching with configurable batch sizes
    - Incremental checkpointing to HDF5 for fault tolerance
    - Automatic retry mechanism for failed generations
    - Progress tracking and resumption from checkpoints
    """

    def __init__(self, params):
        super(LabelDescriptionHelper, self).__init__()
        self.params = params

        # Load the prompt template
        self.prompt_template = self._load_prompt("optimized_prompt")

        logging.info(f"Initializing vLLM with model: {self.params.llm.label_desc.model}")

        # Initialize vLLM Engine with optimal settings for multi-GPU
        self.llm = LLM(
            model=self.params.llm.label_desc.model,
            tensor_parallel_size=self.params.llm.label_desc.tensor_parallel_size,
            gpu_memory_utilization=self.params.llm.label_desc.gpu_memory_utilization,
            trust_remote_code=self.params.llm.label_desc.trust_remote_code,
            # Optimize for throughput
            max_num_batched_tokens=self.params.llm.label_desc.max_num_batched_tokens,
            max_num_seqs=self.params.llm.label_desc.max_num_seqs,
        )

        # Define sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.params.llm.label_desc.temperature,
            top_p=self.params.llm.label_desc.top_p,
            max_tokens=self.params.llm.label_desc.max_gen_len,
            # Ignore EOS to ensure we get full generations
            ignore_eos=False,
        )

        # Configuration
        self.batch_size = self.params.llm.label_desc.batch_size

    def _format_label(self, label: str) -> str:
        """Format label by removing NA suffix if present"""
        splited_label = label.split("->")
        if splited_label[-1] == "NA":
            return splited_label[0]
        return label

    def _format_labels(self, labels: List[str]) -> str:
        """Format multiple labels as semicolon-separated string"""
        return '; '.join([self._format_label(label) for label in labels])

    def _get_samples_candidates(self, samples: List[Dict]) -> Dict[int, List[int]]:
        """Build mapping from label_idx to sample indices"""
        candidates = {}
        for sample in tqdm(samples, desc="Finding candidates"):
            for label_idx in sample["labels_ids"]:
                if label_idx not in candidates:
                    candidates[label_idx] = []
                candidates[label_idx].append(sample["idx"])
        return candidates

    def _get_labels_map(self, samples: List[Dict]) -> Dict[str, int]:
        """Build mapping from label text to label_idx"""
        labels_map = {}
        for sample in tqdm(samples, desc="Getting labels map"):
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels_map[label] = label_idx
        return labels_map

    def _get_text_label_pairs(self, samples: List[Dict], select_ids: List[int]) -> str:
        """Format text-label pairs for few-shot prompting"""
        text_label_pairs = ""
        for sample_idx in select_ids:
            # Truncate text to first 128 words
            text_label_pairs += f"    text: {' '.join(samples[sample_idx]['text'].split()[:128])}\n"
            text_label_pairs += f"    labels: {self._format_labels(samples[sample_idx]['labels'])}\n\n"
        return text_label_pairs

    def _get_label_prompt(self, target_label: str, samples: List[Dict], select_ids: List[int]) -> str:
        """Generate prompt for a specific label"""
        return self.prompt_template.format(
            target_label=target_label,
            text_label_pairs=self._get_text_label_pairs(samples, select_ids)
        )

    def _get_prompt_queue(self, fold_idx: int) -> deque:
        """
        Generate a queue of all prompt tasks for the fold.
        Returns a deque for efficient pop/append operations.
        """
        all_samples = self._load_samples()
        logging.info(f"Generating prompt queue for fold {fold_idx}...")

        prompt_queue = deque()
        num_samples = self.params.llm.label_desc.num_samples

        # Load train/val samples for few-shot examples
        samples = self._load_split_samples(fold_idx, "train") + self._load_split_samples(fold_idx, "val")

        labels_map = self._get_labels_map(samples)
        candidates = self._get_samples_candidates(samples)

        for label, label_idx in tqdm(labels_map.items(), desc="Creating prompt tasks"):
            formated_label = self._format_label(label)

            # Select positive examples
            samples_ids = candidates[label_idx]
            select_ids = samples_ids if len(samples_ids) < num_samples else random.sample(samples_ids, num_samples)

            # Create prompt
            label_prompt = self._get_label_prompt(formated_label, all_samples, select_ids)

            # Add task to queue
            prompt_queue.append(
                PromptTask(
                    label_idx=label_idx,
                    label=formated_label,
                    prompt=label_prompt))

        logging.info(f"Created queue with {len(prompt_queue)} prompt tasks")
        return prompt_queue

    def _checkpoint_descriptions(self, fold_idx: int, batch_results: List[tuple]):
        """
        Incrementally append successful results to HDF5 checkpoint.

        Args:
            fold_idx: Fold index
            batch_results: List of (label_idx, description) tuples
        """
        if not batch_results:
            return

        # Extract data
        label_indices = np.array([r[0] for r in batch_results], dtype=np.int32)
        descriptions = [r[1].encode('utf-8') for r in batch_results]

        # Create variable-length string dtype
        dt = h5py.string_dtype(encoding='utf-8')

        # Append to file
        with h5py.File(f"{self.params.data.dir}fold_{fold_idx}/labels_descriptions.h5", 'a') as f:
            if 'label_idx' not in f:
                # Create datasets
                f.create_dataset('label_idx', data=label_indices, maxshape=(None,),
                                 chunks=True, compression='gzip')
                f.create_dataset('description', data=np.array(descriptions, dtype=dt),
                                 maxshape=(None,), chunks=True, compression='gzip')
            else:
                # Append to existing datasets
                for ds_name, new_data in [('label_idx', label_indices),
                                          ('description', np.array(descriptions, dtype=dt))]:
                    dset = f[ds_name]
                    old_size = dset.shape[0]
                    new_size = old_size + len(new_data)
                    dset.resize((new_size,))
                    dset[old_size:new_size] = new_data

    def _process_prompts_tasks(
            self,
            prompt_tasks: List[PromptTask],
            progress_bar: Optional[tqdm] = None
    ) -> tuple[List[tuple], List[PromptTask]]:
        """
        Process a batch of prompts with vLLM.

        Returns:
            success_results: List of (label_idx, description) tuples
            failed_tasks: List of PromptTask objects that failed
        """

        # Extract prompts
        prompts = [task.prompt for task in prompt_tasks]

        try:
            # Run batch inference
            outputs = self.llm.generate(prompts, self.sampling_params)

            success_tasks = []
            failed_tasks = []

            # Process outputs
            for i, output in enumerate(outputs):
                task = prompt_tasks[i]

                # Check if generation was successful
                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text.strip()
                    # Validate output (non-empty)
                    if generated_text:
                        success_tasks.append((task.label_idx, generated_text))
                    else:
                        # Empty generation - retry
                        failed_tasks.append(task)

                else:
                    # No output generated
                    failed_tasks.append(task)

            if progress_bar:
                progress_bar.update(len(success_tasks))

            return success_tasks, failed_tasks

        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            # Re-queue all tasks for retry
            failed_tasks = []
            for task in prompt_tasks:
                failed_tasks.append(task)

            return [], failed_tasks

    def _process_prompt_queue(self, fold_idx: int, prompt_queue: deque):
        """
        Process the prompt queue with batching and checkpointing.

        Args:
            fold_idx: Fold index
            prompt_queue: Queue of PromptTask objects
        """

        total_tasks = len(prompt_queue)

        logging.info(f"Processing {total_tasks} remaining tasks")

        with tqdm(total=total_tasks, desc=f"Fold {fold_idx} - Generating descriptions") as pbar:
            while prompt_queue:
                # Dequeue batch
                batch_size = min(self.batch_size, len(prompt_queue))
                batch_tasks = [prompt_queue.popleft() for _ in range(batch_size)]

                # Process batch
                success_tasks, failed_tasks = self._process_prompts_tasks(batch_tasks, pbar)

                # Save successful results immediately
                if success_tasks:
                    self._checkpoint_descriptions(fold_idx, success_tasks)

                # Re-queue failed tasks at the end
                prompt_queue.extend(failed_tasks)

                # Log progress
                if len(success_tasks) < len(batch_tasks):
                    logging.info(f"Batch: {len(success_tasks)} success, {len(failed_tasks)} retrying")

    def run(self):
        """Main execution method"""
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Generating descriptions with vLLM for {self.params.data.name} (fold {fold_idx})\n"
                f"Params:\n{OmegaConf.to_yaml(self.params.llm.label_desc)}"
            )

            # Generate prompt queue
            prompt_queue = self._get_prompt_queue(fold_idx)

            # Process queue with batching and checkpointing
            self._process_prompt_queue(fold_idx, prompt_queue)
