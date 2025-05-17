#!/usr/bin/env python3
"""Direct test of the transformer MLP dtype fix"""

import torch
import torch.nn as nn
from rfdetr.models.transformer import MLP, gen_sineembed_for_position


def test_ref_point_head_chain():
    """Test the exact scenario from the error - ref_point_head with sine embeddings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the ref_point_head MLP (matching the actual architecture)
    # From transformer.py line 386: self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
    d_model = 256
    ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
    ref_point_head.to(device)
    
    # Simulate evaluation with fp16
    print("Testing ref_point_head with fp16 evaluation...")
    ref_point_head.half()
    
    # Create position tensor (obj_center from the error stack)
    batch_size = 2
    num_queries = 100
    obj_center = torch.rand(batch_size, num_queries, 4).half().to(device)
    
    # Generate sine embeddings
    # From transformer.py line 430-431: 
    # query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)
    query_sine_embed = gen_sineembed_for_position(obj_center, d_model / 2)
    
    print(f"obj_center dtype: {obj_center.dtype}")
    print(f"query_sine_embed dtype: {query_sine_embed.dtype}")
    print(f"query_sine_embed shape: {query_sine_embed.shape}")
    
    # This is exactly where the error occurred:
    # From transformer.py line 441: query_pos = self.ref_point_head(query_sine_embed)
    query_pos = ref_point_head(query_sine_embed)
    
    print(f"✓ Success! query_pos dtype: {query_pos.dtype}, shape: {query_pos.shape}")
    print("\nThe dtype fix works correctly - no more Float/Half mismatch!")
    
    # Also test with regular float32 to ensure backward compatibility
    print("\nTesting with float32 for backward compatibility...")
    ref_point_head_f32 = MLP(2 * d_model, d_model, d_model, 2).to(device)
    obj_center_f32 = torch.rand(batch_size, num_queries, 4).to(device)
    query_sine_embed_f32 = gen_sineembed_for_position(obj_center_f32, d_model / 2)
    query_pos_f32 = ref_point_head_f32(query_sine_embed_f32)
    print(f"✓ Float32 also works: query_pos dtype: {query_pos_f32.dtype}")


if __name__ == "__main__":
    test_ref_point_head_chain()