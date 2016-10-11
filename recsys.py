import numpy as np
import pandas as pd
import logging

# create logger
logger = logging.getLogger('recsys')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# log_handler = logging.StreamHandler()
log_handler = logging.FileHandler('/tmp/recsys.log', 'w')
log_handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# add formatter to ch
log_handler.setFormatter(formatter)

# add ch to logger
logger.addHandler(log_handler)

# Load data files
train_data_file = './data/toy_train.csv'
test_data_file = './data/toy_test.csv'
#train_data_file = './data/restaurant_train.csv'
#test_data_file = './data/restaurant_test.csv'

train_data = pd.read_csv(train_data_file)
train_data.columns = ['user_id', 'item_id', 'rating']

test_data = pd.read_csv(test_data_file)
test_data.columns = ['user_id', 'item_id', 'rating']

# Compare training data users and test data users and determine if any test users are not in the training data set
unique_train_user = train_data.user_id.unique()
unique_test_user = test_data.user_id.unique()
unique_compare_mask = np.in1d(unique_test_user, unique_train_user, invert=True)
assert len(unique_test_user[unique_compare_mask]) == 0

logger.info('Loaded %d training data' % (len(train_data)))

'''
From the output we know that all test data set users exists in the training data.
To optimize the computation, we will first find the 40 nearest neighbours for each user using the training data set
and we calculate the prediction with test data
'''

# Create rating matrix from training data
training_data_matrix = train_data.pivot_table(values='rating', index='user_id', columns='item_id')
assert training_data_matrix.shape == (len(train_data.user_id.unique()), len(train_data.item_id.unique()))
logger.info('Training data matrix shape: %s users X %s businesses' % (
    training_data_matrix.shape[0], training_data_matrix.shape[1]))


# Create a recommendation predictor object to predict user ratings
class RecommendationPredictor(object):
    # Define cosine similarity function

    @staticmethod
    def __cosine_sim_tuple(u1_rating, u2_rating):
        u1_rating = np.nan_to_num(u1_rating)
        u2_rating = np.nan_to_num(u2_rating)
        u1_rating_sqrt_sum = np.sum(u1_rating ** 2)
        u2_rating_sqrt_sum = np.sum(u2_rating ** 2)
        return np.sum(u1_rating * u2_rating) / np.sqrt(u1_rating_sqrt_sum * u2_rating_sqrt_sum)

    @staticmethod
    def __update_top_naighbours(c_uid, o_uid, sim_score, sim_score_dict, top_n=40):
        if not sim_score_dict.has_key(c_uid):
            sim_score_dict[c_uid] = {o_uid: sim_score}
        else:
            rattings = sim_score_dict[c_uid]
            if len(rattings) < top_n:
                sim_score_dict[c_uid][o_uid] = sim_score
            else:
                # Already has top_n neighbour, determine if this new entry has a higher score and replace if it does
                min_sim_score = min(rattings.values())
                if min_sim_score < sim_score:
                    rattings = {key: value for key, value in rattings.items() if value is not min_sim_score}
                    rattings[o_uid] = sim_score
                    sim_score_dict[c_uid] = rattings

    def __init__(self, train_matrix):
        self.train_matrix = train_matrix

        # Calculate average rating for all users
        self.user_mean_ratings = pd.DataFrame(data=train_matrix.mean(axis=1), index=train_matrix.index)

        # Calculate similarities between all
        self.similarity_dict = self.__calculate_user_sim_dict(training_data_matrix)

    # Define function to calculate similarities between all users using formula 2.5 from the text book
    def __calculate_user_sim_dict(self, mtrx):

        # sim_score_df = pd.DataFrame(index=mtrx.index, columns=mtrx.index)
        sim_score_dict = {}

        # Faster way to loop
        tuple_list = list(mtrx.itertuples())
        # we don't need to compute the last entry of similarity to itself
        for i in range(0, (len(tuple_list) - 1)):
            # Remove the top user as current user
            current = tuple_list.pop(0)
            c_uid = current[0]
            c_rattings = np.asarray(current[1:len(current)])
            logger.debug('calculating similarity for %dth user - %sc_uid' % (i, c_uid))

            others = tuple_list
            for other in others:
                o_uid = other[0]
                o_rattings = np.asarray(other[1:len(other)])

                sim_score = self.__cosine_sim_tuple(c_rattings, o_rattings)

                self.__update_top_naighbours(c_uid, o_uid, sim_score, sim_score_dict)
                self.__update_top_naighbours(o_uid, c_uid, sim_score, sim_score_dict)

        return sim_score_dict

    # Define prediction function using formula 2.3 from the text book
    def predict_rating_dict(self, user_name, item_name):
        try:

            # Get the user's average ratings
            u_avg_rate = self.user_mean_ratings.ix[user_name]

            # Get the user's 40 nearest neighbors from the similarity_matrix
            neighbors = pd.DataFrame.from_dict(self.similarity_dict[user_name], orient='index')
            neighbors.columns = ['sim_score']

            # Get average ratings of all the neighbors
            neighbors['avg_rating'] = self.user_mean_ratings.ix[neighbors.index]

            # Get ratings of all the neighbors on the specific item for prediction
            neighbors['product_rating'] = self.train_matrix.ix[neighbors.index, item_name]

            # neutralize NAN ratings by setting the default rating to the average rating
            neighbors['product_rating'] = neighbors['product_rating'].fillna(neighbors['avg_rating'])

            # Calculate prediction with formula 2.3
            n_nominator = (neighbors['sim_score'] * (neighbors['product_rating'] - neighbors['avg_rating'])).sum(
                skipna=True)

            pred_rating = u_avg_rate + n_nominator / (neighbors['sim_score']).sum(skipna=True)

            return pred_rating

        except KeyError as e:
            print e.message
            raise

    def get_user_mean_ratings(self):
        return self.user_mean_ratings

    def get_similarity_dict(self):
        return self.similarity_dict


rec_predictor = RecommendationPredictor(training_data_matrix)
cosine_sim_dict = rec_predictor.get_similarity_dict()
logger.info('Calculated similarity for %d users' % (len(cosine_sim_dict)))

# Predict user ratings on test data set

# Create a new column in the test dataframe for predicted value
test_data['predicted'] = np.nan

logger.info("Calculating prediction for %d test data" % len(test_data))
for row in test_data.itertuples():
    pred = rec_predictor.predict_rating_dict(row[1], row[2])
    test_data.set_value(row[0], 'predicted', pred)

logger.info('Calculating RMSE for user-business rating predictions: %s ' % (len(test_data.index)))
# Compare result using RMSE
def calculate_rmse(actual, prediction):
    return np.sqrt(((prediction - actual) ** 2).mean())


rmse = calculate_rmse(test_data['rating'], test_data['predicted'])
logger.info("RMSE on test dataset is: %s" % rmse)
