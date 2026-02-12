import logging
import pickle
import random
from tqdm import tqdm
from omegaconf import OmegaConf

from vllm import LLM, SamplingParams

from source.helper.Helper import Helper

logging.basicConfig(level=logging.INFO)


class LabelDescriptionHelper(Helper):

    def __init__(self, params):
        super(LabelDescriptionHelper, self).__init__()
        self.params = params

        # Load the prompt template (e.g., optimized_prompt.txt)
        self.prompt_template = self._load_prompt("optimized_prompt")

        logging.info(f"Initializing vLLM with model: {self.params.llm.label_desc.model}")

        # initialize vLLM Engine
        self.llm = LLM(
            model=self.params.llm.label_desc.model,
            tensor_parallel_size=self.params.llm.label_desc.tensor_parallel_size,
            gpu_memory_utilization=self.params.llm.label_desc.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=self.params.llm.label_desc.max_model_len
        )

        # define Sampling Parameters
        self.sampling_params = SamplingParams(
            temperature=self.params.llm.label_desc.temperature,
            top_p=self.params.llm.label_desc.top_p,
            max_tokens=self.params.llm.label_desc.max_gen_len
        )

    def _format_labels(self, labels):
        formatted_labels = []
        for label in labels:
            formatted_labels.append(self._format_label(label))
        return '; '.join(formatted_labels)

    def _format_label(self, label):
        splited_label = label.split("->")
        if splited_label[-1] == "NA":
            return splited_label[0]
        return label

    def _get_candidates(self, samples):
        candidates = {}
        for sample in tqdm(samples, desc="Finding candidates"):
            for label_idx in sample["labels_ids"]:
                if label_idx not in candidates:
                    candidates[label_idx] = []
                candidates[label_idx].append(sample["idx"])
        return candidates

    def _get_labels_map(self, samples):
        labels_map = {}
        for sample in tqdm(samples, desc="Getting labels map"):
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels_map[label] = label_idx
        return labels_map

    def _get_text_label_pairs(self, samples, select_ids):
        text_label_pairs = ""
        for i, sample_idx in enumerate(select_ids):
            text_label_pairs += f"    text: {' '.join(samples[sample_idx]['text'].split()[:128])}\n"
            text_label_pairs += f"    labels: {self._format_labels(samples[sample_idx]['labels'])}\n\n"
        return text_label_pairs

    def _get_label_prompt(self, target_label, samples, select_ids):
        return self.prompt_template.format(
            target_label=target_label,
            text_label_pairs=self._get_text_label_pairs(samples, select_ids)
        )

    def _generate_prompts_list(self, fold_idx):
        """
        Generates the list of prompts for all labels in the specific fold.
        Returns a list of dictionaries containing the prompt text and the associated label_idx.
        """
        all_samples = self._load_samples()
        logging.info(f"Generating prompts for fold {fold_idx}...")

        prompts_data = []

        num_samples = self.params.llm.label_desc.num_samples  # e.g., 5

        # Load train/val splits to extract few-shot examples
        samples = self._load_split_samples(fold_idx, "train") + self._load_split_samples(fold_idx, "val")

        labels_map = self._get_labels_map(samples)
        candidates = self._get_candidates(samples)

        for target_label, label_idx in tqdm(labels_map.items(), desc="Preparing Prompts"):
            target_label_clean = self._format_label(target_label)

            # Select positive examples for few-shot learning
            samples_ids = candidates[label_idx]
            select_ids = samples_ids if len(samples_ids) < num_samples else random.sample(samples_ids, num_samples)

            # Construct the final prompt
            prompt_text = self._get_label_prompt(target_label_clean, all_samples, select_ids)

            prompts_data.append({
                "label_idx": label_idx,
                "prompt": prompt_text,
                "target_label": target_label_clean
            })

        return prompts_data

    def _process_with_vllm(self, prompts_data):
        """
        Sends the complete list of prompts to vLLM for offline inference.
        This maximizes throughput by allowing vLLM to manage batching globally.
        """
        if not prompts_data:
            return {}

        # Extract only the text prompts to pass to the engine
        prompts_texts = [p["prompt"] for p in prompts_data]

        logging.info(f"Running vLLM inference on {len(prompts_texts)} prompts...")

        # Optimized Synchronous Call (Offline Inference)
        # vLLM handles batching and scheduling internally
        outputs = self.llm.generate(prompts_texts, self.sampling_params)

        labels_descriptions = {}

        # Map outputs back to label IDs
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            label_idx = prompts_data[i]["label_idx"]

            labels_descriptions[label_idx] = generated_text

        return labels_descriptions

    def run(self):

        for fold_idx in self.params.data.folds:
            logging.info(
                f"Generating descriptions with vLLM for {self.params.data.name} (fold {fold_idx}) "
                f"using params:\n{OmegaConf.to_yaml(self.params.llm.label_desc)}"
            )

            # 1. Prepare Prompts
            prompts_data = self._generate_prompts_list(fold_idx)

            # 2. Batch Inference (vLLM)
            labels_descriptions = self._process_with_vllm(prompts_data)

            # 3. Save Results
            self._checkpoint_label_descriptions(labels_descriptions, fold_idx)

    def _checkpoint_label_descriptions(self, labels_descriptions, fold_idx):
        output_path = f"{self.params.data.dir}fold_{fold_idx}/labels_descriptions.pkl"
        logging.info(f"Checkpointing {len(labels_descriptions)} labels descriptions to {output_path}")
        with open(output_path, "wb") as labels_desc_file:
            pickle.dump(labels_descriptions, labels_desc_file)