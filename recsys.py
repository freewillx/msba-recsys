import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

"""Create logger"""
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


"""Define Functions"""
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

    # Filter training matrix to only find test user in training set and neighbors rated the item
    if (rated_item == None or rated_item not in train_matrix.columns or user_id not in train_matrix.index):
        return None
    else:
        user_ratings = train_matrix.loc[user_id].tolist()
        # Convert list to NumPy array
        user_ratings = np.asarray(user_ratings)

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

                if (sim_score >0):
                    # Calculate average ratings for rated products
                    avg_rating = np.nanmean(neighbor_rattings)

                    # Get the neighbor's rating for the product
                    item_rating = neighbor[1][rated_item]
                    sim_score_list.append((neighbor_id, sim_score, avg_rating, item_rating))

        if (len(sim_score_list) >0):
            # Sort and get the top neighbors filter by n
            sim_score_list = sorted(sim_score_list, key=lambda x: x[1], reverse=True)[:n]

            # Convert results to Dataframe
            sim_score_df = pd.DataFrame.from_records(sim_score_list,
                                                     columns=['user_name', 'sim_score', 'avg_rating', 'item_rating'])
            return sim_score_df.set_index('user_name')
        else:
            return None


# Function for calculate the rating prediction based on calculated neighbor similarity scores
def calculate_predicted_rating(training_matrix, neighbor_scores, user_id, item_id):
    # Calculate user's average rating
    user_ratings = training_matrix.loc[user_id]
    user_avg_rating = np.nanmean(user_ratings)

    numerator = np.sum(neighbor_scores['sim_score'] * (neighbor_scores['item_rating'] - neighbor_scores['avg_rating']))
    denominator = np.sum(neighbor_scores['sim_score'])

    predicted_rating = user_avg_rating + numerator / denominator

    return predicted_rating

def predict_new_ratings(new_data, train_matrix):

    ''' Calculate Rating Predictions from the test set '''
    # Create a new dataframe to store predicted value
    prediction = pd.DataFrame(new_data[['user_id', 'rating']])
    prediction['predict_rating'] = None
    for row in new_data.itertuples():
        neighbor_scores = rated_neighbors_sim_score(train_matrix, row[1], row[2], n=3)
        if (neighbor_scores is not None):
            pred = calculate_predicted_rating(train_matrix, neighbor_scores, row[1], row[2])
            prediction.set_value(row[0], 'predict_rating', pred)

    # Remove entries that are unable to produce prediction
    prediction = prediction[prediction['predict_rating'].notnull()]

    if(len(prediction) >0):
        prediction['grade'] = prediction.apply(lambda r: 'Good' if r['rating'] >= 11 else 'Bad', axis=1)
        prediction['predict_grade'] = prediction.apply(lambda r: 'Good' if r['predict_rating'] >= 11 else 'Bad', axis=1)
    else:
        prediction = None

    return prediction

# Compare result using RMSE
def calculate_rmse(actual, prediction):
    return np.sqrt(((prediction - actual) ** 2).mean())


# Compare result using F-measure
def calculate_f_measure(actual, prediction, data_labels):

    """
    By definition a confusion matrix C is such that C{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
    Thus in binary classification, the count of true negatives is C{0,0}, false negatives is C{1,0}, true positives is C{1,1} and false positives is C{0,1}.
    """
    cf_matrix = confusion_matrix(actual, prediction, data_labels)

    tp = cf_matrix[0, 0]
    fp = cf_matrix[1, 0]
    fn = cf_matrix[0, 1]
    tn = cf_matrix[1, 1]

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure


"""Load data files"""
rating_data_file = './data/Ratings.csv'
rating_data = pd.read_csv(rating_data_file)
logger.info('Loaded %d raw data' % (len(rating_data)))

'''
  user_id: User ID
  imdb_id: Movie ID
  rating: Movie Rating (scale: 1 - 13)
  with_whom: Friends - 1, Parents - 2, Girlfriend/Boyfriend - 3, Alone - 4, Siblings - 5, Spouse - 6, Children - 7, Colleagues - 8
  day_of_wk: Dayofweek, Weekend - 1, Weekday - 2, Don't remember - 3
  venue : Movie Venue, Theater - 1, Home - 2
'''
rating_data.columns = ['user_id', 'imdb_id', 'rating', 'with_whom', 'day_of_wk', 'venue']

# Data validations
rating_data = rating_data[(rating_data['rating'] >= 1) & (rating_data['rating'] <= 13)]
rating_data = rating_data[(rating_data['with_whom'] >= 1) & (rating_data['with_whom'] <= 8)]
rating_data = rating_data[(rating_data['day_of_wk'] >= 1) & (rating_data['day_of_wk'] <= 3)]
rating_data = rating_data[(rating_data['venue'] >= 1) & (rating_data['venue'] <= 2)]
logger.info('Loaded %d valid data' % (len(rating_data)))


"""
User Based Collective Filtering with 10 folds cross validations - average f_measure 0.428707
"""
kf = KFold(n_splits=10, shuffle=True)
kfg = kf.split(list(rating_data.index))

f_measure_list = []
for train_data, test_data in kfg:

    train_data = rating_data.loc[rating_data.index[train_data]]
    # Create rating sparse matrix from rating data
    train_rating_matrix = train_data.pivot_table(values='rating', index='user_id', columns='imdb_id')

    test_data = rating_data.loc[rating_data.index[test_data]]

    logger.info('UBCF 10 fold cross validation with %d training data and %d testing data' % (len(train_data), len(test_data)))
    prediction = predict_new_ratings(test_data, train_rating_matrix)

    f_measure_list.append(calculate_f_measure(prediction['grade'], prediction['predict_grade'], ['Good', 'Bad']))

logger.info('UBCF average f_measure %f' % (np.mean(f_measure_list)))

'''
Exact Pre-filtering (EPF) method with conditions (Saturday night, Friends, Movie Theater)
with_whom=1
day_of_wk=1
venue=1
Filter is too restrictive, no predictions can be generated
'''
epf_rating_data = rating_data[rating_data['with_whom'] == 1]
epf_rating_data = epf_rating_data[epf_rating_data['day_of_wk'] == 1]
epf_rating_data = epf_rating_data[epf_rating_data['venue'] == 1]

kfg = kf.split(list(epf_rating_data.index))

f_measure_list = []
for train_data, test_data in kfg:

    train_data = rating_data.loc[rating_data.index[train_data]]
    # Create rating sparse matrix from rating data
    train_rating_matrix = train_data.pivot_table(values='rating', index='user_id', columns='imdb_id')

    test_data = rating_data.loc[rating_data.index[test_data]]

    logger.info('EPF 10 fold cross validation with %d training data and %d testing data' % (len(train_data), len(test_data)))
    prediction = predict_new_ratings(test_data, train_rating_matrix)
    if (prediction is not None):
        f_measure_list.append(calculate_f_measure(prediction['grade'], prediction['predict_grade'], ['Good', 'Bad']))

logger.info('EPF average f_measure list size: %d' % (len(f_measure_list)))

train_rating_matrix = epf_rating_data.pivot_table(values='rating', index='user_id', columns='imdb_id')
logger.info('EPF too restrictive, no item is rated by more than one user, no neighbors can be found:\n %s' % (train_rating_matrix))
