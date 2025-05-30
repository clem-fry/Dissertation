import models
import evaluation

def find_params():
    param_ranges = {
        'a': (-20, 20),
        'b': (-20, 20),
        'c': (-20, 20),   # c and d positive to keep reaction meaningful
        'd': (-20, 20),
        'e': (-20, 20),
        'w': (-20, 20),
        'Du': (0.1, 1),
        'Dv': (5, 15)
    }

    model = models.linear
    valid_params = evaluation.suggest_parameters(param_ranges, model, n_trials = 5000)

    with open('valid_params.txt', 'w') as f:
        for item in valid_params:
            f.write(str(item) + '\n')

def score_params(valid_params, criteria):
    scored_params = []

    for p, param in enumerate(valid_params):
        a, b, c, d, e, w, Du, Dv, peak, peak_lambda, peak_k, band_width, area = param

        if not peak:
            continue  # skip if no dispersion peak

        # Compute score
        #score = peak_lambda * band_width * area
        score = criteria
        # Append as (score, full_param_tuple)
        scored_params.append((score, p, param))

    # Sort by score descending (best patterns first)
    scored_params.sort(reverse=True, key=lambda x: x[0])

    # Print top 5
    for score, p, param in scored_params[:5]:
        print(f"Score: {score:.4f}, Index: {p} Parameters: {param}")

    return scored_params