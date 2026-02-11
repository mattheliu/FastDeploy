"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

Test V100FlashAttentionBackend functionality.
"""

import unittest

import paddle

from fastdeploy.platforms import current_platform


class TestV100FlashAttentionBackend(unittest.TestCase):
    """Test suite for V100FlashAttentionBackend"""

    @classmethod
    def setUpClass(cls):
        """Check if we're on V100 or can test the backend"""
        cls.sm_version = 0
        if current_platform.is_cuda():
            cls.sm_version = current_platform.get_sm_version()
        print(f"Testing on SM{cls.sm_version}")

    def test_backend_import(self):
        """Test that V100FlashAttentionBackend can be imported"""
        from fastdeploy.model_executor.layers.attention import V100FlashAttentionBackend

        self.assertIsNotNone(V100FlashAttentionBackend)

    def test_backend_selection_on_v100(self):
        """Test that V100 correctly selects V100FlashAttentionBackend"""
        if self.sm_version == 0:
            self.skipTest("No CUDA device available")

        from fastdeploy.platforms.base import _Backend

        backend_cls = current_platform.get_attention_backend_cls(_Backend.APPEND_ATTN)

        if self.sm_version < 80:
            # V100 should use V100FlashAttentionBackend
            self.assertIn("V100FlashAttentionBackend", backend_cls)
            print(f"V100 (SM{self.sm_version}) correctly uses V100FlashAttentionBackend")
        else:
            # SM80+ should use AppendAttentionBackend
            self.assertIn("AppendAttentionBackend", backend_cls)
            print(f"SM{self.sm_version} correctly uses AppendAttentionBackend")

    def test_qkv_split(self):
        """Test QKV tensor splitting"""
        from fastdeploy.model_executor.layers.attention import V100FlashAttentionBackend

        # Create a mock backend instance for testing helper methods
        class MockFDConfig:
            class model_config:
                max_model_len = 2048
                causal = True
                head_dim = 128
                num_hidden_layers = 32
                rope_3d = False
                use_3d_rope = False
                start_layer_index = 0

            class cache_config:
                block_size = 64

            class speculative_config:
                method = None
                num_speculative_tokens = 0
                model_type = "main"

            class parallel_config:
                pd_disaggregation_mode = None
                expert_parallel_rank = None
                tensor_parallel_rank = 0
                pipeline_parallel_rank = 0

        # Initialize backend
        backend = V100FlashAttentionBackend(
            fd_config=MockFDConfig(),
            kv_num_heads=8,
            num_heads=32,
            head_dim=128,
        )

        # Create test QKV tensor
        # Shape: [num_tokens, (num_heads + 2 * kv_num_heads) * head_dim]
        # = [4, (32 + 2 * 8) * 128] = [4, 6144]
        num_tokens = 4
        qkv = paddle.randn([num_tokens, (32 + 2 * 8) * 128], dtype="float16")

        # Split QKV
        q, k, v = backend._split_qkv(qkv)

        # Verify shapes
        self.assertEqual(q.shape, [num_tokens, 32, 128])
        self.assertEqual(k.shape, [num_tokens, 8, 128])
        self.assertEqual(v.shape, [num_tokens, 8, 128])
        print("QKV split test passed!")

    def test_rotary_embedding_shapes(self):
        """Test rotary embedding application with various input shapes"""
        from fastdeploy.model_executor.layers.attention import V100FlashAttentionBackend

        class MockFDConfig:
            class model_config:
                max_model_len = 2048
                causal = True
                head_dim = 128
                num_hidden_layers = 32
                rope_3d = False
                use_3d_rope = False
                start_layer_index = 0

            class cache_config:
                block_size = 64

            class speculative_config:
                method = None
                num_speculative_tokens = 0
                model_type = "main"

            class parallel_config:
                pd_disaggregation_mode = None
                expert_parallel_rank = None
                tensor_parallel_rank = 0
                pipeline_parallel_rank = 0

        backend = V100FlashAttentionBackend(
            fd_config=MockFDConfig(),
            kv_num_heads=8,
            num_heads=32,
            head_dim=128,
        )

        # Test with None rotary_embs (should return unchanged)
        q = paddle.randn([4, 32, 128], dtype="float16")
        k = paddle.randn([4, 8, 128], dtype="float16")

        q_out, k_out = backend._apply_rotary_emb(q, k, None)
        self.assertTrue(paddle.allclose(q, q_out))
        self.assertTrue(paddle.allclose(k, k_out))
        print("Rotary embedding (None input) test passed!")

    def test_flash_attn_unpadded_available(self):
        """Test that flash_attn_unpadded is available"""
        try:
            from paddle.nn.functional.flash_attention import flash_attn_unpadded

            self.assertIsNotNone(flash_attn_unpadded)
            print("flash_attn_unpadded is available!")
        except ImportError:
            self.skipTest("flash_attn_unpadded not available in this Paddle version")


class TestV100BackendEnvVariable(unittest.TestCase):
    """Test V100 backend selection via environment variable"""

    def test_v100_flash_attn_enum(self):
        """Test V100_FLASH_ATTN is in _Backend enum"""
        from fastdeploy.platforms.base import _Backend

        self.assertTrue(hasattr(_Backend, "V100_FLASH_ATTN"))
        print("V100_FLASH_ATTN enum value exists!")


if __name__ == "__main__":
    print("=" * 60)
    print("V100FlashAttentionBackend Unit Tests")
    print("=" * 60)

    # Print environment info
    print(f"\nPaddle version: {paddle.__version__}")
    print(f"CUDA available: {paddle.is_compiled_with_cuda()}")

    if paddle.is_compiled_with_cuda():
        try:
            prop = paddle.device.cuda.get_device_properties()
            sm_version = prop.major * 10 + prop.minor
            print(f"GPU: {prop.name}")
            print(f"SM Version: {sm_version}")
            print(f"Is V100 (SM70): {sm_version == 70}")
        except Exception as e:
            print(f"Could not get GPU properties: {e}")

    print("\n" + "=" * 60)
    unittest.main(verbosity=2)
