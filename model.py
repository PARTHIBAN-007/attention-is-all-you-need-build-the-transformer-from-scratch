"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab (not yet solved)
# TODO: implement

# Step 2 - build_id_to_token_vocab (not yet solved)
# TODO: implement

# Step 3 - encode_sentence_to_ids (not yet solved)
# TODO: implement

# Step 4 - decode_ids_to_tokens (not yet solved)
# TODO: implement

# Step 5 - pad_id_sequence (not yet solved)
# TODO: implement

# Step 6 - stack_padded_sequences_to_batch (not yet solved)
# TODO: implement

# Step 7 - scale_embeddings_by_sqrt_d_model (not yet solved)
# TODO: implement

# Step 8 - compute_positional_div_term (not yet solved)
# TODO: implement

# Step 9 - build_position_index_column (not yet solved)
# TODO: implement

# Step 10 - fill_even_indices_with_sin (not yet solved)
# TODO: implement

# Step 11 - fill_odd_indices_with_cos (not yet solved)
# TODO: implement

# Step 12 - build_sinusoidal_positional_encoding (not yet solved)
# TODO: implement

# Step 13 - add_positional_encoding_to_embeddings (not yet solved)
# TODO: implement

# Step 14 - build_padding_mask (not yet solved)
# TODO: implement

# Step 15 - build_causal_mask (not yet solved)
# TODO: implement

# Step 16 - combine_padding_and_causal_masks (not yet solved)
# TODO: implement

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    attn_scores = query @ key.transpose(-1,-2)
    return attn_scores

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    mask_attn = scores.masked_fill(mask==False,-torch.inf)
    return mask_attn

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    mask = masked_scores.softmax(axis = -1)
    return torch.nan_to_num(mask, nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = key.shape[-1]
    attention_score = query @ key.transpose(-1,-2) / (d_k)**0.5
    if mask is not None:
        attention_score = attention_score.masked_fill(~mask , value = float('-inf'))
    attention_weights = attention_score.softmax(axis = -1)
    if mask is not None:
        attention_weights = attention_weights.nan_to_num(nan=0.0)
    context_vector = attention_weights @ value
    return (context_vector,attention_weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    batch_size, seq_len , d_model = tensor.shape
    return tensor.view(batch_size,seq_len,num_heads,d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    return split_tensor.transpose(-2,-3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    batch_size, num_heads, seq_len, d_k = multi_head_tensor.shape
    multi_head_tensor = multi_head_tensor.transpose(-2,-3)
    return multi_head_tensor.reshape(batch_size,seq_len,num_heads * d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    if bias is not None:
        return x @ weight.T + bias
    else:
        return x @ weight.T

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    if b_q is not None:
        query = x @ w_q.T + b_q
    else:
        query = x @ w_q.T
    
    if b_k is not None:
        key = x @ w_k.T + b_k
    else:
        key = x @ w_k.T

    if b_v is not None:
        value = x @ w_v.T + b_v
    else:
        value = x @ w_v.T
    return query,key,value

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    batch_size, seq_len, d_model = q.shape
    q = q.view(batch_size,seq_len,num_heads,d_model//num_heads).transpose(-2,-3)
    k = k.view(batch_size,seq_len,num_heads,d_model//num_heads).transpose(-2,-3)
    v = v.view(batch_size,seq_len,num_heads,d_model//num_heads).transpose(-2,-3)
    return (q,k,v)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    batch_size,num_heads,seq_len,head_dim = k_h.shape
    attention_score = q_h @ k_h.transpose(-2, -1) / (head_dim)**0.5
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==False,-torch.inf)
    attention_weights = attention_score.softmax(dim=-1)
    context_vector = (attention_weights @ v_h)
    return (context_vector,attention_weights)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    batch_size, num_heads, seq_len, head_dim = context.shape
    context = context.transpose(-2,-3)
    context = context.reshape(batch_size,seq_len,num_heads*head_dim)
    if b_o is not None:
        return context @ w_o.T + b_o
    else:
        return context @ w_o.T

# Step 31 - assemble_multi_head_attention_forward (not yet solved)
# TODO: implement

# Step 32 - apply_ffn_first_linear_and_relu (not yet solved)
# TODO: implement

# Step 33 - apply_ffn_second_linear (not yet solved)
# TODO: implement

# Step 34 - position_wise_feed_forward_network (not yet solved)
# TODO: implement

# Step 35 - compute_layer_norm_mean_and_variance (not yet solved)
# TODO: implement

# Step 36 - normalize_and_scale_with_gamma_beta (not yet solved)
# TODO: implement

# Step 37 - apply_residual_add_and_norm (not yet solved)
# TODO: implement

# Step 38 - apply_dropout_with_keep_mask (not yet solved)
# TODO: implement

# Step 39 - encoder_layer_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 40 - encoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 41 - assemble_encoder_layer (not yet solved)
# TODO: implement

# Step 42 - stack_encoder_layers (not yet solved)
# TODO: implement

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

