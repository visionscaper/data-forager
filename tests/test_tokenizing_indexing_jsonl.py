from typing import Dict, List, Optional, Union

import json
import os
import shutil
import tempfile
import unittest

from tqdm import tqdm

import numpy as np

from data_forager.datasets.tokens import TokensDataset
from data_forager.indexers.tokenization_indexer import (
    create_tokenize_and_index_jsonl_text_func,
    TokenizerFunc
)
from data_forager.utils import find_files_recursive, natural_sort


def load_jsonl_data(base_path: str) -> List[Dict]:
    input_file_paths = find_files_recursive(
        base_path,
        extension_patterns=['*.jsonl', '*.JSONL']
    )

    input_file_paths = natural_sort(input_file_paths)

    data = []
    for path in input_file_paths:
        with open(path, 'r') as f:
            while text_line := f.readline():
                data.append(json.loads(text_line))

    return data


def generate_tokens_dataset(
        jsonl_data: List[Dict],
        tokenize_text_func: TokenizerFunc,
        eos_idx: int,
        sample_size: Optional[int] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Reference implementation for generating tokenized samples.

    For fixed sample_size: packs documents with EOS separators into fixed-length samples.
    For variable size (sample_size=None): returns list of tokenized documents.
    """
    tokens = []
    rest_tokens = None

    for data in tqdm(jsonl_data, desc='Generating tokens'):
        jsonl_text_tokens = tokenize_text_func(data['text'])

        if sample_size is not None:
            # Always append EOS after each document
            jsonl_text_tokens = jsonl_text_tokens + [eos_idx]

            # Prepend any leftover tokens from previous document
            if rest_tokens is not None:
                jsonl_text_tokens = rest_tokens + jsonl_text_tokens
                rest_tokens = None

            num_tokens = len(jsonl_text_tokens)
            num_samples = num_tokens // sample_size
            num_rest_tokens = num_tokens % sample_size

            if num_rest_tokens > 0:
                # Store remainder tokens (includes EOS from this document)
                rest_tokens = jsonl_text_tokens[-num_rest_tokens:]
                jsonl_text_tokens = jsonl_text_tokens[:num_samples * sample_size]

            tokens += jsonl_text_tokens
        else:
            jsonl_text_tokens = np.array(jsonl_text_tokens)
            tokens += [jsonl_text_tokens]

    if sample_size is not None:
        if rest_tokens is not None:
            print(f'generate_tokens_dataset: Num. rest tokens: {len(rest_tokens)}')

        return np.array(tokens).reshape(-1, sample_size)
    else:
        return tokens


def find_eos_positions(tokens: np.ndarray, eos_idx: int) -> List[int]:
    """Find all positions where EOS token appears."""
    return list(np.where(tokens == eos_idx)[0])


class SimpleTokenizer:
    """Simple character-based tokenizer for testing. Each character = 1 token."""

    def __init__(self):
        # Reserve 0 for EOS
        self.eos_token = 0

    def encode(self, text: str) -> List[int]:
        # Map each character to its ordinal + 1 (to avoid 0 which is EOS)
        return [ord(c) + 1 for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text, converting EOS to empty string."""
        return ''.join(chr(t - 1) if t != self.eos_token else '' for t in tokens)

    def decode_with_eos_marker(self, tokens: List[int]) -> str:
        """Decode tokens back to text, showing EOS as visible marker."""
        return ''.join(chr(t - 1) if t != self.eos_token else '<EOS>' for t in tokens)


def extract_documents_from_samples(
    samples: List[np.ndarray],
    eos_idx: int,
    decode_func,
    rest_tokens: Optional[List[int]] = None,
) -> List[str]:
    """
    Extract original documents from packed samples by splitting on EOS tokens.

    :param samples: List of tokenized samples
    :param eos_idx: EOS token index
    :param decode_func: Function to decode tokens to text
    :param rest_tokens: Optional remainder tokens that weren't included in samples
    :return: List of decoded documents
    """
    # Concatenate all samples and rest tokens
    all_tokens = []
    for sample in samples:
        all_tokens.extend(sample.tolist())
    if rest_tokens:
        all_tokens.extend(rest_tokens)

    # Split on EOS to get individual documents
    documents = []
    current_doc_tokens = []

    for token in all_tokens:
        if token == eos_idx:
            if current_doc_tokens:  # Don't add empty documents
                documents.append(decode_func(current_doc_tokens))
            current_doc_tokens = []
        else:
            current_doc_tokens.append(token)

    # Handle any remaining tokens (incomplete document at the end)
    if current_doc_tokens:
        documents.append(decode_func(current_doc_tokens))

    return documents


class TestTokenizingIndexingJSONL(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.tokenizer = SimpleTokenizer()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_jsonl_file(self, documents: List[str], filename: str = "test.jsonl"):
        """Helper to create a JSONL file from a list of document texts."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            for doc in documents:
                f.write(json.dumps({"text": doc}) + '\n')
        return filepath

    def _cleanup_index(self):
        """Remove index and tokenized-samples directories."""
        shutil.rmtree(os.path.join(self.temp_dir, 'index'), ignore_errors=True)
        shutil.rmtree(os.path.join(self.temp_dir, 'tokenized-samples'), ignore_errors=True)

    def test_eos_after_each_document(self):
        """Test that EOS is placed after each document, regardless of length."""
        sample_size = 10

        # Create documents with unique characters for precise verification
        documents = [
            "ABCD",      # 4 tokens + EOS = 5
            "EFGHIJ",    # 6 tokens + EOS = 7 -> total 12, creates 1 sample, 2 rest
            "KLMN",      # 4 tokens + EOS = 5 -> with 2 rest = 7, no full sample
        ]

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # Should have exactly 1 sample (12 tokens total, sample_size=10)
        self.assertEqual(len(dataset), 1)

        sample = dataset[0]
        eos_positions = find_eos_positions(sample, self.tokenizer.eos_token)

        # EOS should appear at position 4 (after "ABCD")
        self.assertIn(4, eos_positions, "EOS should appear after first document")

        # Verify decode: sample should be "ABCD" + EOS + "EFGHI" (first 5 of second doc)
        decoded = self.tokenizer.decode(sample.tolist())
        self.assertEqual(decoded, "ABCDEFGHI", "Sample should contain first doc + start of second")

    def test_eos_when_document_fits_exactly(self):
        """Test that EOS is added even when document length is exact multiple of sample_size."""
        sample_size = 5

        # Each document has exactly 4 tokens, + EOS = 5 = sample_size
        # This tests the edge case where document fits exactly
        documents = [
            "ABCD",      # 4 tokens + EOS = 5 (exactly sample_size)
            "EFGH",      # 4 tokens + EOS = 5 (exactly sample_size)
        ]

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # Should have 2 samples (each document + EOS fills exactly one sample)
        self.assertEqual(len(dataset), 2)

        # Each sample should end with EOS and decode to original document
        for i, expected_doc in enumerate(documents):
            sample = dataset[i]
            self.assertEqual(
                sample[-1], self.tokenizer.eos_token,
                f"Sample {i} should end with EOS"
            )
            # Decode without EOS should match original document
            decoded = self.tokenizer.decode(sample.tolist())
            self.assertEqual(decoded, expected_doc, f"Sample {i} should decode to original document")

    def test_eos_between_packed_documents(self):
        """Test that EOS separates documents when they're packed together."""
        sample_size = 20

        # Multiple short documents with unique characters
        documents = [
            "ABC",    # 3 + EOS = 4
            "DEF",    # 3 + EOS = 4 -> total 8
            "GHI",    # 3 + EOS = 4 -> total 12
            "JKL",    # 3 + EOS = 4 -> total 16
        ]

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # All 4 documents (16 tokens total) fit in one sample of size 20
        # But we only get samples when we have >= sample_size tokens
        # 16 < 20, so no complete sample
        self.assertEqual(len(dataset), 0)

        # Let's add one more document to get a complete sample
        self._cleanup_index()

        documents.append("MNOPQ")  # 5 + EOS = 6 -> total 22, creates 1 sample with 2 rest

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        self.assertEqual(len(dataset), 1)

        sample = dataset[0]
        eos_positions = find_eos_positions(sample, self.tokenizer.eos_token)

        # Should have EOS at positions 3, 7, 11, 15 (after each of first 4 documents)
        expected_eos_positions = [3, 7, 11, 15]
        self.assertEqual(eos_positions, expected_eos_positions)

        # Verify decode: sample should contain all 4 complete docs + start of 5th
        decoded = self.tokenizer.decode(sample.tolist())
        self.assertEqual(decoded, "ABCDEFGHIJKLMNOP", "Sample should contain 4 docs + 4 chars of 5th")

    def test_matches_reference_implementation(self):
        """Test that indexed samples match the reference implementation."""
        sample_size = 8

        documents = [
            "Hello",
            "World",
            "Test",
            "Documents",
            "For",
            "Validation",
        ]

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        # Load indexed data
        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # Generate reference data
        jsonl_data = load_jsonl_data(self.temp_dir)
        reference_data = generate_tokens_dataset(
            jsonl_data,
            tokenize_text_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )

        # Compare
        self.assertEqual(len(dataset), reference_data.shape[0])

        for idx in range(len(dataset)):
            np.testing.assert_array_equal(
                dataset[idx], reference_data[idx],
                f"Sample {idx} mismatch"
            )

    def test_round_trip_extract_all_documents(self):
        """Test that we can extract all original documents from tokenized samples."""
        sample_size = 10

        # Design documents to create 6+ samples with various boundary conditions
        # Each line shows: text (length) + EOS = total, running total, samples created
        documents = [
            "Alpha",       # 5 + EOS = 6, total 6, 0 samples
            "Beta",        # 4 + EOS = 5, total 11, 1 sample + 1 rest
            "Gamma",       # 5 + EOS = 6, total 7, 0 samples
            "Delta",       # 5 + EOS = 6, total 13, 1 sample + 3 rest
            "Epsilon",     # 7 + EOS = 8, total 11, 1 sample + 1 rest
            "Zeta",        # 4 + EOS = 5, total 6, 0 samples
            "Eta",         # 3 + EOS = 4, total 10, 1 sample + 0 rest (exact fit!)
            "Theta",       # 5 + EOS = 6, total 6, 0 samples
            "Iota",        # 4 + EOS = 5, total 11, 1 sample + 1 rest
            "Kappa",       # 5 + EOS = 6, total 7, 0 samples
            "Lambda",      # 6 + EOS = 7, total 14, 1 sample + 4 rest
        ]
        # Total: 6 complete samples + 4 rest tokens

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # Should have 6 samples
        self.assertEqual(len(dataset), 6, "Should have 6 complete samples")

        # Collect all samples
        samples = [dataset[i] for i in range(len(dataset))]

        # Extract documents from samples
        extracted = extract_documents_from_samples(
            samples,
            eos_idx=self.tokenizer.eos_token,
            decode_func=self.tokenizer.decode,
        )

        # Documents fully contained in samples should be recoverable
        # "Lambda" is split: "Lambd" in sample 6, "a" + EOS in remainder (discarded)
        expected_complete_docs = documents[:-1]  # All except last (Lambda is cut off)

        self.assertEqual(len(extracted), len(expected_complete_docs) + 1,
                         "Should extract all complete docs + partial last doc")

        # Verify complete documents match exactly
        for i, doc in enumerate(expected_complete_docs):
            self.assertEqual(extracted[i], doc, f"Document {i} '{doc}' should match original")

        # Last extracted should be partial "Lambda" -> "Lam"
        # Sample 6 contains: [EOS][K][a][p][p][a][EOS][L][a][m] (10 tokens)
        # Rest (discarded): [b][d][a][EOS] (4 tokens)
        self.assertEqual(extracted[-1], "Lam", "Last doc should be truncated 'Lambda' -> 'Lam'")

    def test_no_sample_size_variable_length(self):
        """Test variable-length mode (sample_size=None)."""
        documents = [
            "Short",
            "A longer document",
            "X",
        ]

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            # sample_size=None (default)
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # Should have 3 samples, one per document
        self.assertEqual(len(dataset), 3)

        # Each sample should decode back to original document
        for i, doc in enumerate(documents):
            sample = dataset[i]
            self.assertEqual(len(sample), len(doc), f"Sample {i} should have correct length")
            decoded = self.tokenizer.decode(sample.tolist())
            self.assertEqual(decoded, doc, f"Sample {i} should decode to original document")

    def test_large_documents_spanning_multiple_samples(self):
        """Test multiple documents that each span multiple samples."""
        sample_size = 5

        # Three large documents, each spanning multiple samples
        # Doc 1: 12 chars + EOS = 13 tokens
        # Doc 2: 14 chars + EOS = 15 tokens
        # Doc 3: 11 chars + EOS = 12 tokens
        # Total: 40 tokens = 8 samples + 0 rest
        documents = [
            "ABCDEFGHIJKL",      # 12 chars + EOS = 13 tokens
            "MNOPQRSTUVWXYZ",    # 14 chars + EOS = 15 tokens
            "0123456789a",       # 11 chars + EOS = 12 tokens
        ]
        # Packed stream: ABCDEFGHIJKL<EOS>MNOPQRSTUVWXYZ<EOS>0123456789a<EOS>
        # Sample 0: ABCDE
        # Sample 1: FGHIJ
        # Sample 2: KL<EOS>MN
        # Sample 3: OPQRS
        # Sample 4: TUVWX
        # Sample 5: YZ<EOS>01
        # Sample 6: 23456
        # Sample 7: 789a<EOS>

        self._create_jsonl_file(documents)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.temp_dir,
            tokenizer_func=self.tokenizer.encode,
            eos_idx=self.tokenizer.eos_token,
            sample_size=sample_size
        )
        tokenize_and_index()

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.temp_dir,
            token_dtype=np.uint16
        )

        # 40 tokens / 5 = 8 complete samples
        self.assertEqual(len(dataset), 8)

        # Verify each sample's content
        expected_samples = [
            ("ABCDE", []),           # Sample 0: no EOS
            ("FGHIJ", []),           # Sample 1: no EOS
            ("KLMN", [2]),           # Sample 2: EOS at position 2 (after "KL")
            ("OPQRS", []),           # Sample 3: no EOS
            ("TUVWX", []),           # Sample 4: no EOS
            ("YZ01", [2]),           # Sample 5: EOS at position 2 (after "YZ")
            ("23456", []),           # Sample 6: no EOS
            ("789a", [4]),           # Sample 7: EOS at position 4 (after "789a")
        ]

        for i, (expected_text, expected_eos_positions) in enumerate(expected_samples):
            sample = dataset[i]
            decoded = self.tokenizer.decode(sample.tolist())
            eos_positions = find_eos_positions(sample, self.tokenizer.eos_token)

            self.assertEqual(decoded, expected_text,
                             f"Sample {i} should decode to '{expected_text}', got '{decoded}'")
            self.assertEqual(eos_positions, expected_eos_positions,
                             f"Sample {i} EOS positions should be {expected_eos_positions}, got {eos_positions}")

        # Verify round-trip: extract all documents from samples
        samples = [dataset[i] for i in range(len(dataset))]
        extracted = extract_documents_from_samples(
            samples,
            eos_idx=self.tokenizer.eos_token,
            decode_func=self.tokenizer.decode,
        )

        self.assertEqual(len(extracted), len(documents), "Should extract all 3 documents")
        for i, doc in enumerate(documents):
            self.assertEqual(extracted[i], doc, f"Document {i} should match original")


class TestTokenizingWithExternalData(unittest.TestCase):
    """
    Tests that use external data files and tiktoken.
    These tests are skipped if the data or tiktoken is not available.
    """

    @classmethod
    def setUpClass(cls):
        cls.input_base_path = "../data/neural-ram/long-fineweb-docs/sample-100BT/test/unzipped"
        cls.data_available = os.path.exists(cls.input_base_path)
        cls.tiktoken_available = False
        cls.encoder = None

        if cls.data_available:
            try:
                import tiktoken
                cls.encoder = tiktoken.get_encoding("gpt2")
                cls.tiktoken_available = True
            except ImportError:
                pass

    def setUp(self):
        if not self.data_available:
            self.skipTest(f"External data not available at {self.input_base_path}")
        if not self.tiktoken_available:
            self.skipTest("tiktoken not installed")

        shutil.rmtree(os.path.join(self.input_base_path, 'index'), ignore_errors=True)
        shutil.rmtree(os.path.join(self.input_base_path, 'tokenized-samples'), ignore_errors=True)

    def test_tokenizing_indexing_with_tiktoken(self):
        """Test with real data and tiktoken tokenizer."""
        def tokenize_text(text: str) -> List[int]:
            return self.encoder.encode_ordinary(text)

        tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
            input_base_path=self.input_base_path,
            tokenizer_func=tokenize_text,
            eos_idx=self.encoder.eot_token,
            sample_size=1024
        )
        tokenize_and_index()

        jsonl_data = load_jsonl_data(self.input_base_path)
        reference_data = generate_tokens_dataset(
            jsonl_data,
            tokenize_text_func=tokenize_text,
            eos_idx=self.encoder.eot_token,
            sample_size=1024
        )

        dataset = TokensDataset.create_from_index_on_filesystem(
            self.input_base_path,
            token_dtype=np.uint16
        )

        self.assertEqual(reference_data.shape[0], len(dataset))

        # Test a random subset
        import random
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for idx in tqdm(indices[:100], desc="Testing samples"):
            np.testing.assert_array_equal(dataset[idx], reference_data[idx])


if __name__ == '__main__':
    unittest.main()
