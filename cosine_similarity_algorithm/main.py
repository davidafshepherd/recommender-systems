import csv
import numpy as np
from collections import defaultdict


def load_train_data(filename):
    """Loads training data from CSV file.

    Args:
        - filename (str): Path to the training dataset CSV file.

    Returns:
        - user_ratings (dict of dicts): A nested dictionary mapping users to the items they have rated {user: {item: rating}}.
        - item_ratings (dict of dicts): A nested dictionary mapping items to users who have rated them {item: {user: rating}}.
        - baseline_stats (dict): Statistics needed for baseline estimates including global mean, user bias and item bias.
    """

    # Dictionaries
    user_ratings = defaultdict(dict)            # Mapping from user ID and item ID to item rating
    item_ratings = defaultdict(dict)            # Mapping from item ID and user ID to item rating

    user_rating_sums = defaultdict(float)       # Mapping from user ID to sum of ratings
    item_rating_sums = defaultdict(float)       # Mapping from item ID to sum of ratings

    user_rating_counts = defaultdict(int)       # Mapping from user ID to count of ratings
    item_rating_counts = defaultdict(int)       # Mapping from item ID to count of ratings

    total_ratings = 0
    total_sum = 0

    # Load all ratings and compute sums
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            user, item, rating = int(row[0]), int(row[1]), float(row[2])
            user_ratings[user][item] = rating
            item_ratings[item][user] = rating
            user_rating_sums[user] += rating
            item_rating_sums[item] += rating
            user_rating_counts[user] += 1
            item_rating_counts[item] += 1
            total_sum += rating
            total_ratings += 1

    # Compute global mean
    global_mean = total_sum / total_ratings if total_ratings > 0 else 0

    # Compute user and item biases
    user_biases = {}
    item_biases = {}

    # Compute user biases
    for user in user_rating_sums:
        user_mean = user_rating_sums[user] / user_rating_counts[user]
        user_biases[user] = user_mean - global_mean

    # Compute item biases
    for item in item_rating_sums:
        item_mean = item_rating_sums[item] / item_rating_counts[item]
        item_biases[item] = item_mean - global_mean

    # Package baseline statistics
    baseline_stats = {
        'global_mean': global_mean,
        'user_biases': user_biases,
        'item_biases': item_biases
    }

    return user_ratings, item_ratings, baseline_stats


def compute_item_similarity(user_ratings, item_ratings, baseline_stats):
    """Computes the cosine similarity between items using baseline-adjusted ratings. Applies significance weighting to account for number of common users.

    Args:
        - user_ratings (dict of dicts): A nested dictionary mapping users to the items they have rated {user: {item: rating}}.
        - item_ratings (dict of dicts): A nested dictionary mapping items to users who have rated them {item: {user: rating}}.
        - baseline_stats (dict): Statistics needed for baseline estimates including global mean, user bias, and item bias.

    Returns:
        - item_similarities (dict of dicts): A nested dictionary mapping items to their similarity score {item: {item: item}}.
    """

    # Mapping from items to similarity score
    item_similarities = {}

    # Extract baseline components
    global_mean = baseline_stats['global_mean']
    user_biases = baseline_stats['user_biases']
    item_biases = baseline_stats['item_biases']

    # Significance weighting threshold (minimum number of common users)
    min_common_users = 50

    # Compute similarity between items using the adjusted cosine similarity
    for item1 in item_ratings:
        item_similarities[item1] = {}
        for item2 in item_ratings:
            # Skip if self-similarity
            if item1 == item2:
                continue

            # Find common users who rated both items
            common_users = set(item_ratings[item1].keys()) & set(item_ratings[item2].keys())

            # Skip if no common users
            if not common_users:
                continue

            # Compute baseline-adjusted ratings
            ratings1 = []
            ratings2 = []
            for u in common_users:
                # Calculate baseline estimates
                baseline1 = global_mean + user_biases[u] + item_biases[item1]
                baseline2 = global_mean + user_biases[u] + item_biases[item2]
                
                # Get baseline-adjusted ratings
                ratings1.append(user_ratings[u][item1] - baseline1)
                ratings2.append(user_ratings[u][item2] - baseline2)

            # Convert to numpy arrays for efficient computation
            ratings1 = np.array(ratings1)
            ratings2 = np.array(ratings2)

            # Compute similarity using cosine formula
            numerator = np.dot(ratings1, ratings2)
            denominator = np.sqrt(np.dot(ratings1, ratings1)) * np.sqrt(np.dot(ratings2, ratings2))

            # Calculate base similarity
            similarity = numerator / denominator if denominator != 0 else 0

            # Apply significance weighting
            n_common = len(common_users)
            significance_weight = n_common / (n_common + min_common_users)
            similarity = similarity * significance_weight

            item_similarities[item1][item2] = similarity

    return item_similarities


def predict_rating(user, item, user_ratings, item_similarities, baseline_stats, k=20):
    """Predicts rating for a given user-item pair using k most similar items.

    Args:
        - user (int): The user ID.
        - item (int): The item ID to predict a rating for.
        - user_ratings (dict of dicts): A nested dictionary mapping users to the items they have rated {user: {item: rating}}.
        - item_similarities (dict of dicts): a nested dictionary mapping items to their similarity score {item: {item: item}}.
        - baseline_stats (dict): Statistics needed for baseline estimates including global mean, user bias, and item bias.
        - k (int): Number of most similar items to use for prediction (default: 20).

    Returns:
        - predicted_rating (float): The estimated rating for the user-item pair, rounded to the nearest integer and bounded between 1 and 5.
    """

    # Get baseline components
    global_mean = baseline_stats['global_mean']
    user_biases = baseline_stats['user_biases']
    item_biases = baseline_stats['item_biases']

    # If user is not in training data, return global average rating
    if user not in user_biases:
        return global_mean

    # If item is not in similarity matrix, return baseline estimate
    if item not in item_similarities:
        baseline = global_mean + user_biases[user] + item_biases.get(item, 0)
        return max(1, min(5, round(baseline)))

    # Calculate baseline for target item
    baseline_i = global_mean + user_biases[user] + item_biases.get(item, 0)

    # Get similar items
    sim_items = item_similarities[item]

    # Get items that the user has rated
    user_rated_items = user_ratings[user]
    relevant_items = [(rated_item, sim_items[rated_item]) 
                     for rated_item in user_rated_items 
                     if rated_item in sim_items]

    # If no similar items exist, return baseline estimate
    if not relevant_items:
        return max(1, min(5, round(baseline_i)))

    # Sort by absolute similarity and take top k
    relevant_items.sort(key=lambda x: abs(x[1]), reverse=True)
    relevant_items = relevant_items[:k]

    # Initialize numerator and denominator for weighted sum
    numerator = 0
    denominator = 0

    # Compute the weighted sum
    for rated_item, similarity in relevant_items:
        # Calculate baseline for the rated item j
        baseline_j = global_mean + user_biases[user] + item_biases[rated_item]
        # Get deviation from baseline
        deviation = user_ratings[user][rated_item] - baseline_j
        # Add to sums
        numerator += similarity * deviation
        denominator += abs(similarity)

    # Compute predicted rating
    if denominator != 0:
        prediction = baseline_i + (numerator / denominator)
        # Round to nearest integer and bound between 1 and 5
        prediction = max(1, min(5, round(prediction)))
    else:
        # If no valid predictions, return baseline estimate
        prediction = max(1, min(5, round(baseline_i)))

    return prediction


def generate_predictions(train_file, test_file, output_file):
    """Generates predictions for the test set and saves them to a CSV file.

    Args:
        - train_file: Path to the training dataset CSV file.
        - test_file: Path to the test dataset CSV file.
        - output_file: Path to the results CSV file.
    """

    # Load training data
    user_ratings, item_ratings, baseline_stats = load_train_data(train_file)

    # Compute item similarities
    item_similarities = compute_item_similarity(user_ratings, item_ratings, baseline_stats)

    # List for storing predicted ratings
    predictions = []

    # Open test set in read mode
    with open(test_file, 'r') as file:
        reader = csv.reader(file)

        # Predict the rating for each given user-item pair
        for row in reader:
            user, item, timestamp = int(row[0]), int(row[1]), int(row[2])
            pred_rating = predict_rating(user, item, user_ratings, item_similarities, baseline_stats)
            predictions.append([user, item, pred_rating, timestamp])

    # Save predictions to output CSV
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["user", "item", "rating", "timestamp"])
        writer.writerows(predictions)

if __name__ == '__main__':
    # Generate predictions
    generate_predictions("train_100k_withratings.csv", "test_100k_withoutratings.csv", "predictions.csv")