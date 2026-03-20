"""
Data Loading and Processing for Recruiter Persona Fine-tuning
Handles loading, formatting, and collation of interview conversation data
"""

from datasets import Dataset, load_dataset
from typing import Dict, List, Optional
import json


class RecruiterDataLoader:
    """
    Loads and processes recruiter interview conversation data.
    Handles Llama messages format with system prompts, multi-turn conversations.
    """

    def __init__(
        self,
        train_path: str,
        eval_path: str,
        tokenizer,
        max_seq_length: int = 2048,
    ):
        """
        Initialize the data loader.

        Args:
            train_path: Path to training JSONL file
            eval_path: Path to evaluation JSONL file
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length for conversations
        """
        self.train_path = train_path
        self.eval_path = eval_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @staticmethod
    def _is_valid_conversation(example: Dict) -> bool:
        """Return True when a sample has well-formed, non-empty chat messages."""
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            return False

        valid_roles = {"system", "user", "assistant"}
        seen_user = False
        seen_assistant = False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            role = msg.get("role")
            content = msg.get("content")
            if role not in valid_roles:
                return False
            if not isinstance(content, str) or not content.strip():
                return False
            if role == "user":
                seen_user = True
            if role == "assistant":
                seen_assistant = True

        # Keep only interview-like dialogues with both sides present.
        return seen_user and seen_assistant

    def load_datasets(self):
        """
        Load training and evaluation datasets from JSONL files.

        Returns:
            tuple: (train_dataset, eval_dataset)
        """
        print(f"Loading training data from: {self.train_path}")
        train_dataset = load_dataset("json", data_files=self.train_path, split="train")

        print(f"Loading evaluation data from: {self.eval_path}")
        eval_dataset = load_dataset("json", data_files=self.eval_path, split="train")

        print(f"✅ Loaded {len(train_dataset)} training samples")
        print(f"✅ Loaded {len(eval_dataset)} evaluation samples")

        return train_dataset, eval_dataset

    def format_chat_template(self, example: Dict) -> Dict:
        """
        Apply chat template to messages for training.

        Args:
            example: Dataset example with 'messages' field

        Returns:
            Dict with 'text' field containing formatted conversation
        """
        if "messages" not in example:
            raise ValueError("Example missing 'messages' field")

        # Apply the tokenizer's chat template
        formatted_text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,  # We have complete conversations
        )

        return {"text": formatted_text}

    @staticmethod
    def _is_question_like_assistant_turn(messages: List[Dict], idx: int) -> bool:
        """Check whether an assistant message is a suitable single-question target."""
        msg = messages[idx]
        if msg.get("role") != "assistant":
            return False

        # Train only on direct recruiter prompts after user context.
        if idx == 0 or messages[idx - 1].get("role") != "user":
            return False

        content = (msg.get("content") or "").strip()
        if not content:
            return False

        question_marks = content.count("?")
        if question_marks != 1:
            return False

        # Avoid very long responses that tend to include explanation + follow-up.
        if len(content) > 280:
            return False

        return True

    def _build_question_turn_dataset(self, dataset, split_name: str):
        """Convert conversation-level samples into assistant-question supervision samples."""
        records = []
        skipped_too_long = 0

        for example in dataset:
            messages = example.get("messages", [])
            if not isinstance(messages, list) or not messages:
                continue

            for idx, msg in enumerate(messages):
                if not self._is_question_like_assistant_turn(messages, idx):
                    continue

                # Include context up to and including the assistant target turn.
                clipped_messages = messages[: idx + 1]
                text = self.tokenizer.apply_chat_template(
                    clipped_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                token_count = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
                if token_count > self.max_seq_length:
                    skipped_too_long += 1
                    continue

                records.append({"text": text})

        print(
            f"Built {len(records)} {split_name} question-turn samples "
            f"(skipped {skipped_too_long} over max_seq_length={self.max_seq_length})."
        )

        if not records:
            raise ValueError(
                f"No usable {split_name} samples after question-turn extraction. "
                "Relax constraints or review dataset quality."
            )

        return Dataset.from_list(records)

    def prepare_datasets(self):
        """
        Load and format datasets for training.

        Returns:
            tuple: (formatted_train_dataset, formatted_eval_dataset)
        """
        train_dataset, eval_dataset = self.load_datasets()

        # Filter out malformed samples before tokenization/template formatting.
        original_train_size = len(train_dataset)
        original_eval_size = len(eval_dataset)
        train_dataset = train_dataset.filter(
            self._is_valid_conversation,
            desc="Filtering invalid train samples",
        )
        eval_dataset = eval_dataset.filter(
            self._is_valid_conversation,
            desc="Filtering invalid eval samples",
        )
        print(
            f"Kept {len(train_dataset)}/{original_train_size} train samples "
            f"and {len(eval_dataset)}/{original_eval_size} eval samples after validation."
        )

        # Build training examples from assistant turns that match one-question behavior.
        print("\nExtracting question-turn training samples...")
        train_dataset = self._build_question_turn_dataset(train_dataset, "train")

        print("Extracting question-turn evaluation samples...")
        eval_dataset = self._build_question_turn_dataset(eval_dataset, "eval")

        # Print sample
        if len(train_dataset) > 0:
            self._print_sample(train_dataset[0])
        else:
            raise ValueError("No valid training samples remain after filtering.")

        return train_dataset, eval_dataset

    def _print_sample(self, sample: Dict):
        """Print a formatted sample for inspection."""
        print("\n" + "=" * 70)
        print("SAMPLE FORMATTED CONVERSATION:")
        print("=" * 70)
        text = sample.get("text", "")
        # Truncate if too long
        if len(text) > 1000:
            print(text[:1000] + "\n... [truncated] ...")
        else:
            print(text)
        print("=" * 70 + "\n")

    def analyze_dataset(self, dataset):
        """
        Analyze dataset statistics.

        Args:
            dataset: HuggingFace dataset

        Returns:
            Dict with statistics
        """
        stats = {
            "num_samples": len(dataset),
            "conversation_lengths": [],
            "num_turns": [],
            "decision_distribution": {"select": 0, "reject": 0},
        }

        for example in dataset:
            if "messages" in example:
                num_messages = len(example["messages"])
                stats["num_turns"].append(num_messages)

            if "text" in example:
                stats["conversation_lengths"].append(len(example["text"]))

            if "meta" in example and "decision" in example["meta"]:
                decision = example["meta"]["decision"]
                if decision in stats["decision_distribution"]:
                    stats["decision_distribution"][decision] += 1

        # Calculate averages
        if stats["conversation_lengths"]:
            stats["avg_length"] = sum(stats["conversation_lengths"]) / len(stats["conversation_lengths"])
            stats["max_length"] = max(stats["conversation_lengths"])
            stats["min_length"] = min(stats["conversation_lengths"])

        if stats["num_turns"]:
            stats["avg_turns"] = sum(stats["num_turns"]) / len(stats["num_turns"])
            stats["max_turns"] = max(stats["num_turns"])
            stats["min_turns"] = min(stats["num_turns"])

        return stats

    def print_dataset_info(self):
        """Print detailed information about the datasets."""
        print("\n" + "=" * 70)
        print("DATASET ANALYSIS")
        print("=" * 70)

        train_dataset, eval_dataset = self.load_datasets()

        print("\n📊 TRAINING SET:")
        train_stats = self.analyze_dataset(train_dataset)
        self._print_stats(train_stats)

        print("\n📊 EVALUATION SET:")
        eval_stats = self.analyze_dataset(eval_dataset)
        self._print_stats(eval_stats)

        print("\n" + "=" * 70)

    def _print_stats(self, stats: Dict):
        """Print statistics in a formatted way."""
        print(f"  Total Samples: {stats['num_samples']}")

        if "avg_turns" in stats:
            print(f"  Average Turns: {stats['avg_turns']:.1f}")
            print(f"  Turn Range: {stats['min_turns']} - {stats['max_turns']}")

        if "avg_length" in stats:
            print(f"  Average Length: {stats['avg_length']:.0f} chars")
            print(f"  Length Range: {stats['min_length']} - {stats['max_length']} chars")

        if stats["decision_distribution"]["select"] > 0 or stats["decision_distribution"]["reject"] > 0:
            total = stats["decision_distribution"]["select"] + stats["decision_distribution"]["reject"]
            select_pct = 100 * stats["decision_distribution"]["select"] / total
            reject_pct = 100 * stats["decision_distribution"]["reject"] / total
            print(f"  Select/Reject: {stats['decision_distribution']['select']}/{stats['decision_distribution']['reject']} ({select_pct:.1f}%/{reject_pct:.1f}%)")


def validate_data_format(file_path: str) -> bool:
    """
    Validate that the JSONL file has the correct format.

    Args:
        file_path: Path to JSONL file

    Returns:
        bool: True if valid, raises exception otherwise
    """
    required_fields = {"messages"}
    message_roles = {"system", "user", "assistant"}

    print(f"\nValidating data format: {file_path}")

    with open(file_path, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Check required fields
                if not required_fields.issubset(data.keys()):
                    raise ValueError(f"Line {i}: Missing required fields. Expected: {required_fields}")

                # Check messages format
                if not isinstance(data["messages"], list):
                    raise ValueError(f"Line {i}: 'messages' must be a list")

                if len(data["messages"]) == 0:
                    raise ValueError(f"Line {i}: 'messages' list is empty")

                # Check each message
                for j, msg in enumerate(data["messages"]):
                    if "role" not in msg or "content" not in msg:
                        raise ValueError(f"Line {i}, Message {j}: Missing 'role' or 'content'")

                    if msg["role"] not in message_roles:
                        raise ValueError(f"Line {i}, Message {j}: Invalid role '{msg['role']}'. Expected one of: {message_roles}")

                    if not isinstance(msg["content"], str) or len(msg["content"]) == 0:
                        raise ValueError(f"Line {i}, Message {j}: 'content' must be a non-empty string")

            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i}: Invalid JSON - {e}")

            # Only validate first 10 lines for speed
            if i >= 10:
                break

    print("✅ Data format validation passed!")
    return True


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Configuration
    TRAIN_DATA = "../data/final/recruiter_llama_train_messages.jsonl"
    EVAL_DATA = "../data/final/recruiter_llama_eval_messages.jsonl"
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    print("=" * 70)
    print("Recruiter Data Loader - Testing & Analysis")
    print("=" * 70)

    # Validate data format
    print("\n1. Validating data format...")
    try:
        validate_data_format(TRAIN_DATA)
        validate_data_format(EVAL_DATA)
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
        exit(1)

    # Load tokenizer
    print("\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize data loader
    print("\n3. Initializing data loader...")
    data_loader = RecruiterDataLoader(
        train_path=TRAIN_DATA,
        eval_path=EVAL_DATA,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )

    # Analyze datasets
    print("\n4. Analyzing datasets...")
    data_loader.print_dataset_info()

    # Load and format datasets
    print("\n5. Loading and formatting datasets...")
    train_dataset, eval_dataset = data_loader.prepare_datasets()

    print("\n✅ Data loader testing complete!")
