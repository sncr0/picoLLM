import torch
from analysis import monosemantic_analysis_for_token

################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    probs = torch.softmax(logits, dim=-1)
    if p == 1.0:
        return torch.multinomial(probs.float(), num_samples=1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    threshold = p
    #print(cumulative_probs)
    selected = (cumulative_probs >= threshold).nonzero(as_tuple=True)
    transition_index = selected[0][0] #find the point that the cum_probs first exceed 0.95
    #print(selected[0].shape, transition_index)
    excess = cumulative_probs[transition_index] -p
    sorted_probs[transition_index] -= excess
    probs = sorted_probs[:transition_index+1]/p
    i = torch.multinomial(probs.float(), num_samples=1)
    return sorted_indices[i]




def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  debug_top_n_probs=0):
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
            
            # --- Debugging: Print top N probabilities ---
            if debug_top_n_probs > 0:
                probs = torch.softmax(next_logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, k=min(debug_top_n_probs, probs.size(-1))) # Ensure k is not > vocab_size
                print(f"Current Context: {enc.decode(context_tokens)}")
                print(f"Top {top_k_probs.size(0)} predictions for the next token:")
                for i in range(top_k_probs.size(0)):
                    token_id = top_k_indices[i].item()
                    token_prob = top_k_probs[i].item()
                    # Assuming enc.decode can handle a list with a single token ID
                    decoded_token = enc.decode([token_id])
                    # If your tokens are simple numbers and enc.decode([5]) -> "5", this is fine.
                    # If enc.decode gives more complex output, you might adjust the print.
                    print(f"  - Token: \"{decoded_token}\" (ID: {token_id}) - Probability: {token_prob:.4f}")
            # --- End Debugging ---

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

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




