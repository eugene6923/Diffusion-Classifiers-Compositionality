import numpy as np

def evaluate_scores_v1(scores):
    if isinstance(scores, tuple):
        scores_i2t = scores[1]  # (N, 2, 2)
        scores_t2i = scores[0]  # (N, 2, 2)
    else:
        scores_t2i = scores
        scores_i2t = scores

    # Compute correctness for both queries
    correct_i2t = (scores_i2t[:, 0, 0] > scores_i2t[:, 0, 1]) & (scores_i2t[:, 1, 1] > scores_i2t[:, 1, 0])
    correct_t2i = (scores_t2i[:, 0, 0] > scores_t2i[:, 1, 0]) & (scores_t2i[:, 1, 1] > scores_t2i[:, 0, 1])

    # Compute accuracy where both queries must be correct
    return {"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}

def evaluate_scores_v2(scores):
    if isinstance(scores, tuple):
        scores_i2t = scores[0]
        scores_t2i = scores[1].T  # Transposed
    else:
        scores_t2i = scores
        scores_i2t = scores

    ground_truth_i2t = np.array([0, 1] * (len(scores_i2t))).reshape(len(scores_i2t), 2)
    ground_truth_t2i = np.array([0, 1] * (len(scores_t2i))).reshape(len(scores_t2i), 2)

    predicted_i2t = scores_i2t.argmax(axis=-1)
    predicted_t2i = scores_t2i.argmax(axis=1)

    correct_i2t = np.all(predicted_i2t == ground_truth_i2t, axis=-1)
    correct_t2i = np.all(predicted_t2i == ground_truth_t2i, axis=-1)

    return {"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}

# Generate random test cases
for seed in range(1000):
    np.random.seed(seed)
    N = 100
    random_scores = np.random.rand(N, 2, 2)
    random_tuple_scores = (random_scores, random_scores)

    # Evaluate using both functions
    result_v1 = evaluate_scores_v1(random_scores)
    result_v2 = evaluate_scores_v2(random_scores)

    # Return the results
    if result_v1 != result_v2:
        print(f"Results are different for seed {seed}")
        print(result_v1)
        print(result_v2)
        break
