import torch
from analysis import monosemantic_analysis_for_token

################################################################################
# 7. Single code path for text generation
################################################################################

def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution (1D tensor).

    Returns:
        torch.Tensor: Entropy value (scalar).
    """
    eps = 1e-8  # avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs + eps))
    return entropy


def dynamic_p_from_entropy(entropy: torch.Tensor, min_p=0.85, max_p=0.98) -> float:
    """
    Map entropy to a dynamic p value.

    Args:
        entropy (torch.Tensor): Computed entropy.
        min_p (float): Minimum p value (for low entropy, confident).
        max_p (float): Maximum p value (for high entropy, uncertain).

    Returns:
        float: Adjusted p value for nucleus sampling.
    """
    # Assume typical entropy range [low, high]
    # We rescale it between min_p and max_p

    # These are *rough* typical entropy bounds — you can tune them
    low_entropy = 2.0    # very confident
    high_entropy = 6.0   # very uncertain

    # Clamp entropy to reasonable bounds
    entropy = torch.clamp(entropy, low_entropy, high_entropy)

    # Linearly map entropy to [min_p, max_p]
    normalized = (entropy - low_entropy) / (high_entropy - low_entropy)
    p = min_p + normalized * (max_p - min_p)
    return p.item()

def dynamic_nucleus_sampling(logits: torch.Tensor, base_p=0.95, temperature=1.0) -> torch.Tensor:
    """
    Dynamic nucleus (top-p) sampling based on model uncertainty (entropy).

    Args:
        logits (torch.Tensor): The unnormalized model outputs (1D tensor).
        base_p (float): Default p to fall back on (if needed).
        temperature (float): Temperature scaling.

    Returns:
        torch.Tensor: Index of the sampled token.
    """
    assert logits.dim() == 1, "Logits should be 1D tensor."

    # Apply temperature scaling before softmax
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    # If p=1.0, just sample from full distribution
    if base_p >= 1.0:
        return torch.multinomial(probs.float(), num_samples=1)

    # Compute entropy of the probs
    entropy = compute_entropy(probs)

    # Dynamically adjust p based on entropy
    dynamic_p = dynamic_p_from_entropy(entropy)

    # Sort the probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find where cumulative probability exceeds dynamic_p
    selected = (cumulative_probs >= dynamic_p).nonzero(as_tuple=True)
    transition_index = selected[0][0]

    # Correct for excess probability
    excess = cumulative_probs[transition_index] - dynamic_p
    sorted_probs[transition_index] -= excess

    selected_probs = sorted_probs[:transition_index + 1]
    selected_probs /= selected_probs.sum()

    # Sample from the truncated normalized distribution
    sampled_idx = torch.multinomial(selected_probs.float(), num_samples=1)

    # Map back to original token id
    return sorted_indices[sampled_idx]


def nucleus_sampling(logits, p=0.95, temperature=1.0):
    """
    Nucleus (top-p) sampling with optional temperature scaling.
    
    Args:
        logits (torch.Tensor): Model output logits (1D tensor).
        p (float): Top-p cumulative probability threshold.
        temperature (float): Temperature for scaling logits.
        
    Returns:
        torch.Tensor: Index of the sampled token.
    """
    assert logits.dim() == 1, "Logits should be a 1D tensor."

    # Apply temperature scaling **before** softmax
    if temperature != 1.0:
        logits = logits / temperature

    # Softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)

    if p == 1.0:
        # Sample from full distribution if p=1.0
        return torch.multinomial(probs.float(), num_samples=1)

    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum of sorted probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index where cumulative probability exceeds p
    selected = (cumulative_probs >= p).nonzero(as_tuple=True)
    transition_index = selected[0][0]  # First index where cumsum exceeds p

    # Correct the last selected probability for overshoot
    excess = cumulative_probs[transition_index] - p
    sorted_probs[transition_index] -= excess

    # Select probabilities up to and including the cutoff
    selected_probs = sorted_probs[:transition_index + 1]

    # Normalize selected probabilities to sum to 1
    selected_probs /= selected_probs.sum()

    # Sample from truncated, normalized distribution
    sampled_idx = torch.multinomial(selected_probs.float(), num_samples=1)

    # Map back to original token indices
    return sorted_indices[sampled_idx]




def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  temperature=1,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  dynamic_p=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                if dynamic_p:
                    # Dynamic nucleus sampling based on entropy
                    chosen_token = dynamic_nucleus_sampling(next_logits, base_p=top_p, temperature=temperature)
                else:
                    # Static nucleus sampling
                    chosen_token = nucleus_sampling(next_logits, p=top_p, temperature=temperature)


            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text




