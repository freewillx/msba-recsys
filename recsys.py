import numpy as np
import pandas as pd
import logging

''' Create logger '''
logger = logging.getLogger('recsys')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
log_handler = logging.StreamHandler()
# log_handler = logging.FileHandler('/tmp/recsys.log', 'w')

# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(log_handler)


''' Load data files '''
train_data_file = './data/Tiny_training.csv'
test_data_file = './data/Tiny_test.csv'
# train_data_file = './data/toy_train.csv'
# test_data_file = './data/toy_test.csv'
# train_data_file = './data/restaurant_train.csv'
# test_data_file = './data/restaurant_test.csv'

train_data = pd.read_csv(train_data_file)
train_data.columns = ['user_id', 'item_id', 'rating']

test_data = pd.read_csv(test_data_file)
test_data.columns = ['user_id', 'item_id', 'rating']

logger.info('Loaded %d training data' % (len(train_data)))


''' Create rating matrix from training data '''
training_data_matrix = train_data.pivot_table(values='rating', index='user_id', columns='item_id')
assert training_data_matrix.shape == (len(train_data.user_id.unique()), len(train_data.item_id.unique()))
logger.info('Training data matrix shape: %s users X %s businesses' % (
    training_data_matrix.shape[0], training_data_matrix.shape[1]))


''' Define Functions '''
# Calculate cosine similarity
def cosine_sim(u1_rating_np_array, u2_rating_np_array):
    u1_rating = np.nan_to_num(u1_rating_np_array)
    u2_rating = np.nan_to_num(u2_rating_np_array)
    u1_rating_sqrt_sum = np.sum(u1_rating ** 2)
    u2_rating_sqrt_sum = np.sum(u2_rating ** 2)
    return np.sum(u1_rating * u2_rating) / np.sqrt(u1_rating_sqrt_sum * u2_rating_sqrt_sum)


# Find the N nearest neighbors for a specific user from the training data
# For higher performance, rated_item_filter can be used to exclude neighbors who did not rated the same item
def rated_neighbors_sim_score(train_matrix, user_id, rated_item, n=3):
    user_ratings = train_matrix.loc[user_id].tolist()
    # Convert list to NumPy array
    user_ratings = np.asarray(user_ratings)

    # Filter training matrix to only find neighbors rated the item
    train_matrix = train_matrix.loc[train_matrix[rated_item].notnull()]

    neighbor_list = list(train_matrix.iterrows())

    sim_score_list = []
    for neighbor in neighbor_list:
        neighbor_id = neighbor[0]

        # Skip user it self
        if (neighbor_id != user_id):
            neighbor_rattings = np.asarray(neighbor[1])

            # Calculate Cosine Similarity
            sim_score = cosine_sim(user_ratings, neighbor_rattings)

            # Calculate average ratings for rated products
            avg_rating = np.nanmean(neighbor_rattings)

            # Get the neighbor's rating for the product
            item_rating = neighbor[1][rated_item]

            sim_score_list.append((neighbor_id, sim_score, avg_rating, item_rating))

    # Sort and get the top neighbors filter by n
    sim_score_list = sorted(sim_score_list, key=lambda x: x[1], reverse=True)[:n]

    # Convert results to Dataframe
    sim_score_df = pd.DataFrame.from_records(sim_score_list,
                                             columns=['user_name', 'sim_score', 'avg_rating', 'item_rating'])
    return sim_score_df.set_index('user_name')


# Function for calculate the rating prediction based on calculated neighbor similarity scores
def predict_rating(training_matrix, neighbor_scores, user_id, item_id):
    # Calculate user's average rating
    user_ratings = training_matrix.loc[user_id]
    user_avg_rating = np.nanmean(user_ratings)

    numerator = np.sum(neighbor_scores['sim_score'] * (neighbor_scores['item_rating'] - neighbor_scores['avg_rating']))
    denominator = np.sum(neighbor_scores['sim_score'])

    predicted_rating = user_avg_rating + numerator / denominator

    return predicted_rating


# Compare result using RMSE
def calculate_rmse(actual, prediction):
    return np.sqrt(((prediction - actual) ** 2).mean())


''' Calculate Rating Predictions from the test set '''
# TODO - Assignment 6, filter matrix by context

logger.info("Calculating prediction for %d test data" % len(test_data))

# Create a new column in the test dataframe to store predicted value
test_data['predict_rating'] = np.nan

for row in test_data.itertuples():
    neighbor_scores = rated_neighbors_sim_score(training_data_matrix, row[1], row[2], n=40)
    pred = predict_rating(training_data_matrix, neighbor_scores, row[1], row[2])
    test_data.set_value(row[0], 'predict_rating', pred)

# Calculate RMSE
# TODO - Assignment 6, change to new evaluation
logger.info('Calculating RMSE for user-business rating predictions: %s ' % (len(test_data.index)))
rmse = calculate_rmse(test_data['rating'], test_data['predict_rating'])
logger.info("RMSE on test dataset is: %s" % rmse)
