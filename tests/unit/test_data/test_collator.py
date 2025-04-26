# Copyright 2025 The LLaVA-Pool Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import shutil
import tempfile
import unittest
from argparse import Namespace

import pytest
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from llavapool.data.collator import SFTDataCollatorWith4DAttentionMask
from llavapool.data.converter import load_converter
from llavapool.utils.constants import IGNORE_INDEX


class TestSFTDataCollatorWith4DAttentionMask(unittest.TestCase):
    """Tests for SFTDataCollatorWith4DAttentionMask class."""

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        # Create a demo image for testing
        self.demo_image_path = os.path.join(self.tmpdirname, "demo_image.jpg")
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(self.demo_image_path)

        # Load the actual tokenizer, processor and model
        self.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Load the model with low precision to save memory
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Create a minimal model config for testing if loading fails
            self.model = AutoModelForVision2Seq.config_class.from_pretrained(self.model_name)
            self.model._attn_implementation = "eager"

        # Create data arguments
        self.data_args = Namespace(dataset_dir=self.tmpdirname, ignore_pad_token_for_loss=False)

        # Create model arguments
        self.model_args = Namespace(block_diag_attn=False, compute_dtype="float16")

        # Create training arguments
        self.training_args = Namespace(do_train=True)

        # Load the converter
        self.converter = load_converter(processor=self.processor, tokenizer=self.tokenizer, data_args=self.data_args)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_collator_init(self):
        """Test initialization of the collator."""
        # Initialize with required parameters
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Check that the collator has the expected attributes
        self.assertEqual(collator.converter, self.converter)
        self.assertEqual(collator.pad_to_multiple_of, 8)
        self.assertEqual(collator.label_pad_token_id, self.tokenizer.pad_token_id)
        self.assertEqual(collator.block_diag_attn, False)
        self.assertEqual(collator.attn_implementation, "eager")
        self.assertEqual(collator.compute_dtype, "float16")
        self.assertEqual(collator.model, self.model)
        self.assertEqual(collator.tokenizer, self.tokenizer)

        # Check that the collator class has the expected methods
        self.assertTrue(hasattr(collator, "__call__"))

        # Check the signature of the __call__ method
        signature = inspect.signature(collator.__call__)
        self.assertIn("features", signature.parameters)

    def test_collator_call_text_only(self):
        """Test calling the collator with text-only data using real tokenizer."""
        # Initialize collator with required parameters
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts in English
        texts = ["Hello, how is the weather today?", "The weather is sunny and clear."]

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features
        features = []
        for i in range(len(texts)):
            # Create 4D attention mask (batch_size, num_heads, seq_len, seq_len)
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs have the expected keys
        self.assertIn("input_ids", outputs)
        self.assertIn("labels", outputs)
        self.assertIn("attention_mask", outputs)

        # Check shapes
        max_length = max(len(encodings["input_ids"][0]), len(encodings["input_ids"][1]))
        # Pad to multiple of 8
        padded_max_length = ((max_length + 8 - 1) // 8) * 8
        self.assertEqual(outputs["input_ids"].shape, (2, padded_max_length))
        self.assertEqual(outputs["labels"].shape, (2, padded_max_length))

        # Check that attention_mask is 4D
        self.assertEqual(len(outputs["attention_mask"].shape), 4)
        self.assertEqual(outputs["attention_mask"].shape[0], 2)  # batch_size
        self.assertEqual(outputs["attention_mask"].shape[2], padded_max_length)  # seq_len
        self.assertEqual(outputs["attention_mask"].shape[3], padded_max_length)  # seq_len

        # Verify padding
        shorter_idx = 0 if len(encodings["input_ids"][0]) < len(encodings["input_ids"][1]) else 1
        shorter_len = len(encodings["input_ids"][shorter_idx])

        # Check that padding is done with the correct pad_token_id
        self.assertTrue(all(outputs["input_ids"][shorter_idx, shorter_len:] == self.tokenizer.pad_token_id))

        # Check that labels are padded with label_pad_token_id
        self.assertTrue(all(outputs["labels"][shorter_idx, shorter_len:] == self.tokenizer.pad_token_id))

    def test_collator_with_pad_to_multiple_of(self):
        """Test the collator with different pad_to_multiple_of values."""
        # Initialize with pad_to_multiple_of=16
        pad_to_multiple_of = 16
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts in English
        texts = ["This is the first example sentence.", "This is the second example sentence. It's a bit longer."]

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features
        features = []
        for i in range(len(texts)):
            # Create 4D attention mask
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check shapes
        max_length = max(len(encodings["input_ids"][0]), len(encodings["input_ids"][1]))
        # Pad to multiple of pad_to_multiple_of
        padded_max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        self.assertEqual(outputs["input_ids"].shape, (2, padded_max_length))
        self.assertEqual(outputs["labels"].shape, (2, padded_max_length))
        self.assertEqual(outputs["attention_mask"].shape[2], padded_max_length)
        self.assertEqual(outputs["attention_mask"].shape[3], padded_max_length)

        # Verify that the length is a multiple of pad_to_multiple_of
        self.assertEqual(padded_max_length % pad_to_multiple_of, 0)

    def test_collator_with_block_diag_attn(self):
        """Test the collator with block_diag_attn=True."""
        # Update model_args
        self.model_args.block_diag_attn = True

        # Initialize with block_diag_attn=True
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,  # Enable block diagonal attention
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts in English
        texts = ["Testing block diagonal attention mask.", "Another example for testing."]

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features
        features = []
        for i in range(len(texts)):
            # Create 4D attention mask
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs have the expected keys
        self.assertIn("input_ids", outputs)
        self.assertIn("labels", outputs)
        self.assertIn("attention_mask", outputs)

        # Check that attention_mask is 4D
        self.assertEqual(len(outputs["attention_mask"].shape), 4)

        # Reset model_args for other tests
        self.model_args.block_diag_attn = False

    def test_collator_with_additional_fields(self):
        """Test the collator with additional fields."""
        # Initialize collator
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts in English
        texts = ["This example has additional fields.", "Second example with additional fields."]

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features with additional fields
        features = []
        for i in range(len(texts)):
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            # Add position_ids and token_type_ids as additional fields
            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
                "position_ids": torch.arange(seq_len),
                "token_type_ids": torch.zeros(seq_len, dtype=torch.long),
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs have the expected additional keys
        self.assertIn("position_ids", outputs)
        self.assertIn("token_type_ids", outputs)

        # Check shapes of additional fields
        max_length = max(len(encodings["input_ids"][0]), len(encodings["input_ids"][1]))
        padded_max_length = ((max_length + 8 - 1) // 8) * 8
        self.assertEqual(outputs["position_ids"].shape, (2, padded_max_length))
        self.assertEqual(outputs["token_type_ids"].shape, (2, padded_max_length))

        # Verify padding for additional fields
        shorter_idx = 0 if len(encodings["input_ids"][0]) < len(encodings["input_ids"][1]) else 1
        shorter_len = len(encodings["input_ids"][shorter_idx])

        # Check that position_ids and token_type_ids are padded with 0
        self.assertTrue(all(outputs["position_ids"][shorter_idx, shorter_len:] == 0))
        self.assertTrue(all(outputs["token_type_ids"][shorter_idx, shorter_len:] == 0))

    def test_collator_with_mixed_tensor_types(self):
        """Test the collator with mixed tensor types (int, float)."""
        # Initialize collator
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts in English
        texts = ["This is a mixed tensor type test.", "Second example for the test."]

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features with mixed tensor types
        features = []
        for i in range(len(texts)):
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            # Add float_tensor as an additional field
            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
                "float_tensor": torch.rand(seq_len, dtype=torch.float),  # Random float tensor
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs have the float_tensor key
        self.assertIn("float_tensor", outputs)

        # Check shape of float_tensor
        max_length = max(len(encodings["input_ids"][0]), len(encodings["input_ids"][1]))
        padded_max_length = ((max_length + 8 - 1) // 8) * 8
        self.assertEqual(outputs["float_tensor"].shape, (2, padded_max_length))

        # Verify dtype of float_tensor
        self.assertEqual(outputs["float_tensor"].dtype, torch.float)

        # Verify padding for float_tensor
        shorter_idx = 0 if len(encodings["input_ids"][0]) < len(encodings["input_ids"][1]) else 1
        shorter_len = len(encodings["input_ids"][shorter_idx])

        # Check that float_tensor is padded with 0.0
        self.assertTrue(all(outputs["float_tensor"][shorter_idx, shorter_len:] == 0.0))

    def test_collator_with_multimodal_data(self):
        """Test the collator with multimodal data (text + image)."""
        # Initialize collator
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts with image placeholders in English
        texts = ["<image>\nWhat is in this image?", "<image>\nPlease describe this image in detail."]

        # Process the demo image
        with open(self.demo_image_path, "rb") as f:
            image_bytes = f.read()

        # Process images using processor
        processed_images = []
        for _ in range(len(texts)):
            # Process image using processor's image processor
            processed_image = self.processor.image_processor(Image.open(self.demo_image_path), return_tensors="pt")
            processed_images.append(processed_image["pixel_values"][0])

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features with both text and image data
        features = []
        for i in range(len(texts)):
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
                "pixel_values": processed_images[i],
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs have the expected keys
        self.assertIn("input_ids", outputs)
        self.assertIn("labels", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertIn("pixel_values", outputs)

        # Check shapes
        max_length = max(len(encodings["input_ids"][0]), len(encodings["input_ids"][1]))
        padded_max_length = ((max_length + 8 - 1) // 8) * 8
        self.assertEqual(outputs["input_ids"].shape, (2, padded_max_length))
        self.assertEqual(outputs["labels"].shape, (2, padded_max_length))

        # Check that attention_mask is 4D
        self.assertEqual(len(outputs["attention_mask"].shape), 4)

        # Check that pixel_values are stacked correctly (not padded)
        self.assertEqual(outputs["pixel_values"].shape[0], 2)  # batch_size

    def test_collator_with_real_conversation(self):
        """Test the collator with a real conversation example."""
        # Initialize collator
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample conversation texts in English
        conversations = [
            [
                {"role": "user", "content": "Hello, how is the weather today?"},
                {"role": "assistant", "content": "Hello! The weather is sunny and clear today. How can I help you?"},
                {"role": "user", "content": "What's the chance of rain tomorrow?"},
            ],
            [
                {"role": "user", "content": "Can you show me how to print 'Hello, World!' in Python?"},
                {
                    "role": "assistant",
                    "content": "Sure! Here's how to print 'Hello, World!' in Python:\n\n```python\nprint('Hello, World!')\n```\n\nWhen you run this code, it will output 'Hello, World!' to the console.",
                },
            ],
        ]

        # Format conversations using tokenizer's chat template if available
        formatted_texts = []
        for conversation in conversations:
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_text = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback if chat template is not available
                formatted_text = ""
                for message in conversation:
                    formatted_text += f"{message['role']}: {message['content']}\n"
            formatted_texts.append(formatted_text)

        # Tokenize the formatted texts
        encodings = self.tokenizer(formatted_texts, padding=False, truncation=False, return_tensors=None)

        # Create features
        features = []
        for i in range(len(formatted_texts)):
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len))

            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]),
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs have the expected keys
        self.assertIn("input_ids", outputs)
        self.assertIn("labels", outputs)
        self.assertIn("attention_mask", outputs)

        # Check shapes
        max_length = max(len(encodings["input_ids"][0]), len(encodings["input_ids"][1]))
        padded_max_length = ((max_length + 8 - 1) // 8) * 8
        self.assertEqual(outputs["input_ids"].shape, (2, padded_max_length))
        self.assertEqual(outputs["labels"].shape, (2, padded_max_length))

        # Check that attention_mask is 4D
        self.assertEqual(len(outputs["attention_mask"].shape), 4)
        self.assertEqual(outputs["attention_mask"].shape[0], 2)  # batch_size
        self.assertEqual(outputs["attention_mask"].shape[2], padded_max_length)  # seq_len
        self.assertEqual(outputs["attention_mask"].shape[3], padded_max_length)  # seq_len

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_collator_with_cuda_tensors(self):
        """Test the collator with CUDA tensors."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Initialize collator
        collator = SFTDataCollatorWith4DAttentionMask(
            converter=self.converter,
            pad_to_multiple_of=8 if self.training_args.do_train else None,
            label_pad_token_id=IGNORE_INDEX
            if self.data_args.ignore_pad_token_for_loss
            else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Create sample texts in English
        texts = ["This is a CUDA tensor test.", "Second example for the test."]

        # Tokenize the texts
        encodings = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)

        # Create features with CUDA tensors
        features = []
        for i in range(len(texts)):
            seq_len = len(encodings["input_ids"][i])
            attention_mask_4d = torch.ones((1, 1, seq_len, seq_len)).cuda()

            feature = {
                "input_ids": torch.tensor(encodings["input_ids"][i]).cuda(),
                "attention_mask": attention_mask_4d,
                "labels": torch.tensor(encodings["input_ids"][i]).cuda(),
            }
            features.append(feature)

        # Call collator
        outputs = collator(features)

        # Check that the outputs are on CUDA
        self.assertTrue(outputs["input_ids"].is_cuda)
        self.assertTrue(outputs["labels"].is_cuda)
        self.assertTrue(outputs["attention_mask"].is_cuda)


if __name__ == "__main__":
    unittest.main()
