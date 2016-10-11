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
# train_data_file = './data/restaurant_train.csv'
# test_data_file = './data/restaurant_test.csv'

train_data = pd.read_csv(train_data_file)
train_data.columns = ['user_id', 'item_id', 'rating']

test_data = pd.read_csv(test_data_file)
test_data.columns = ['user_id', 'item_id', 'rating']

# Compare training data users and test data users and determine if any test users are not in the training data set
unique_train_user = train_data.user_id.unique()
unique_test_user = test_data.user_id.unique()
unique_compare_mask = np.in1d(unique_test_user, unique_train_user, invert=True)
assert len(unique_test_user[unique_compare_mask]) == 0

'''
From the output we know that all test data set users exists in the training data.
To optimize the computation, we will first find the 40 nearest neighbours for each user using the training data set
and we calculate the prediction with test data
'''

# Create rating matrix from training data
training_data_matrix = train_data.pivot_table(values='rating', index='user_id', columns='item_id')
assert training_data_matrix.shape == (len(train_data.user_id.unique()), len(train_data.item_id.unique()))
logger.info('Training data matrix shape: %s users X %s businesses' % (training_data_matrix.shape[0], training_data_matrix.shape[1]))


# Create a recommendation predictor object to predict user ratings
class RecommendationPredictor(object):
    # Define cosine similarity function
    @staticmethod
    def __cosine_sim(u1_rating, u2_rating):
        u1_rating_sqrt_sum = np.sum(u1_rating.apply(lambda x: x ** 2))
        u2_rating_sqrt_sum = np.sum(u2_rating.apply(lambda x: x ** 2))
        return np.sum(u1_rating * u2_rating) / np.sqrt(u1_rating_sqrt_sum * u2_rating_sqrt_sum)

    def __init__(self, train_matrix):
        self.train_matrix = train_matrix

        # Calculate average rating for all users
        self.user_mean_ratings = pd.DataFrame(data=train_matrix.mean(axis=1), index=train_matrix.index)

        # Calculate similarities between all users
        self.similarity_matrix = self.__calculate_user_sim_matrix(train_matrix)

    # Define function to calculate similarities between all users using formula 2.5 from the text book
    def __calculate_user_sim_matrix(self, mtrx):
        sim_score_df = pd.DataFrame(index=mtrx.index, columns=mtrx.index)

        # Loop over the rows of each user to get their ratings
        for uid in range(0, len(mtrx)):
            logger.debug('calculating similarity for user %d' % (uid))

            # Get user ratings as dataframe
            u = mtrx.iloc[uid]
            others = mtrx.iloc[(uid + 1):len(mtrx)]

            sim_serie = others.apply(lambda x: self.__cosine_sim(u.squeeze(), x.squeeze()), axis=1)

            # Fill users row
            sim_score_df.ix[uid, :].loc[sim_serie.index] = sim_serie

            # Fill users column
            sim_score_df.ix[:, uid].loc[sim_serie.index] = sim_serie

        return sim_score_df

    # Define prediction function using formula 2.3 from the text book
    def predict_rating(self, user_name, item_name):
        try:
            # Get the user's average ratings
            u_avg_rate = self.user_mean_ratings.ix[user_name]

            # Get the user's 40 nearest neighbors from the similarity_matrix
            # Excluding oneself
            neighbors = self.similarity_matrix.ix[user_name].order(ascending=False)[1:41]
            neighbors = neighbors.to_frame(name='sim_score')

            # Get average ratings of all the neighbors
            neighbors['avg_rating'] = self.user_mean_ratings.ix[neighbors.index]

            # Get ratings of all the neighbors on the specific item for prediction
            neighbors['product_rating'] = self.train_matrix.ix[neighbors.index, item_name]

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

    def get_similarity_matrix(self):
        return self.similarity_matrix


rec_predictor = RecommendationPredictor(training_data_matrix)
cosine_sim_matrix = rec_predictor.get_similarity_matrix()
logger.info(
    'Calculated user similarity matrix shape: %s X %s users' % (cosine_sim_matrix.shape[0], cosine_sim_matrix.shape[1]))

# Predict user ratings on test data set

# Create a new column in the test dataframe for predicted value
test_data['predicted'] = np.nan

logger.info("Calculating prediction for %d test data" % len(test_data))
for row in test_data.itertuples():
    pred = rec_predictor.predict_rating(row[1], row[2])
    test_data.set_value(row[0], 'predicted', pred)

logger.info('Calculating RMSE for user-business rating predictions: %s ' % (len(test_data.index)))
# Compare result using RMSE
def calculate_rmse(actual, prediction):
    return np.sqrt(((prediction - actual) ** 2).mean())


rmse = calculate_rmse(test_data['rating'], test_data['predicted'])
logger.info("RMSE on test dataset is: %s" % rmse)
