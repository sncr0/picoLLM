import random

def generate_arithmetic_sequence(start_val, diff, length):
    """
    Generates an arithmetic sequence with non-negative integer terms.
    Args:
        start_val (int): The first term of the sequence (non-negative).
        diff (int): The common difference (non-negative).
        length (int): The number of terms in the sequence.
    Returns:
        str: A comma-separated string of sequence terms.
    """
    seq = []
    current_val = start_val
    for _ in range(length):
        seq.append(current_val)
        # Check for potential overflow before adding
        # Since diff is non-negative, only need to check upper bound
        if current_val > (10**12 - diff): # Using a large number as a practical limit
            break # Stop if next term might overflow
        current_val += diff
    return ",".join(map(str, seq))

def generate_geometric_sequence(start_val, ratio, length):
    """
    Generates a geometric sequence with non-negative integer terms.
    Args:
        start_val (int): The first term of the sequence (non-negative).
        ratio (int): The common ratio (non-negative integer).
        length (int): The number of terms in the sequence.
    Returns:
        str: A comma-separated string of sequence terms.
    """
    seq = []
    current_val = start_val
    for i in range(length):
        seq.append(current_val)
        if i < length - 1: # Only calculate next term if not the last one
            if ratio == 0:
                next_val = 0
            elif current_val == 0: # If current is 0, next will be 0 (since ratio is non-negative)
                next_val = 0
            # Check for potential overflow before multiplying (ratio is non-negative)
            elif ratio > 0 and current_val > (10**12 / ratio): # Practical limit
                break # Stop if next term might overflow
            else:
                next_val = current_val * ratio
            
            current_val = next_val
            # If terms become too large, stop to keep sequences manageable
            if current_val > 10**12 and length > 5 : # Allow short sequences to have large numbers
                 if len(seq) > 1 : break
                 else: seq = [start_val] 

    return ",".join(map(str, seq))

def generate_fibonacci_sequence(val1, val2, length):
    """
    Generates a Fibonacci-like sequence with non-negative integer terms.
    Args:
        val1 (int): The first term (non-negative).
        val2 (int): The second term (non-negative).
        length (int): The number of terms in the sequence.
    Returns:
        str: A comma-separated string of sequence terms.
    """
    if length == 0:
        return ""
    if length == 1:
        return str(val1)
    
    seq = [val1, val2] # Both val1 and val2 are non-negative
    for _ in range(2, length):
        # Check for potential overflow before adding (terms are non-negative)
        if seq[-1] > (10**12 - seq[-2]): # Both are positive, check sum
            break
        next_val = seq[-1] + seq[-2]
        # If terms become too large, stop
        if next_val > 10**12 and length > 5:
            break
        seq.append(next_val)
    return ",".join(map(str, seq))

# --- Configuration ---
num_sequences_per_type = 1000  # Number of sequences to generate for each type
min_len = 4                    # Minimum length of a sequence
max_len = 15                   # Maximum length of a sequence
output_filename = "files/many_seqs_positive_integers.txt"

# --- Generation ---
with open(output_filename, "w") as f:
    f.write("# Arithmetic Sequences (Non-Negative Integers Only)\n")
    for i in range(num_sequences_per_type):
        start = random.randint(0, 100) # Start is non-negative
        diff = random.randint(0, 25)   # Difference is non-negative
        # Ensure some sequences have d=0
        if i % 50 == 0: # Every 50th sequence
            diff = 0
        length = random.randint(min_len, max_len)
        f.write(generate_arithmetic_sequence(start, diff, length) + "\n")

    f.write("\n# Geometric Sequences (Non-Negative Integers Only)\n")
    for i in range(num_sequences_per_type):
        start = random.randint(0, 50) # Start is non-negative

        possible_ratios = [0, 1, 2, 3] # Ratios are non-negative
        if i % 100 == 0 and start != 0 : # Occasionally allow larger ratios for short sequences
            possible_ratios.extend([4, 5])

        ratio = random.choice(possible_ratios)
        
        current_max_len = max_len
        if ratio > 1 and start > 1 : # For larger ratios/starts, keep length smaller
            current_max_len = max(min_len, 7) 
        elif ratio > 2 and start == 1:
             current_max_len = max(min_len, 10)

        length = random.randint(min_len, current_max_len)

        f.write(generate_geometric_sequence(start, ratio, length) + "\n")

    f.write("\n# Fibonacci-like Sequences (Non-Negative Integers Only)\n")
    for _ in range(num_sequences_per_type):
        val1 = random.randint(0, 50) # First term is non-negative
        val2 = random.randint(0, 50) # Second term is non-negative
        length = random.randint(min_len, max_len)
        f.write(generate_fibonacci_sequence(val1, val2, length) + "\n")

print(f"Dataset {output_filename} generated with non-negative integer sequences.")
