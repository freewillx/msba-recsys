import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import NMF
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


"""Define Collaborative Filtering Functions"""
# Calculate cosine similarity
def cosine_sim(u1_rating_np_array, u2_rating_np_array):
    u1_rating = np.nan_to_num(u1_rating_np_array)
    u2_rating = np.nan_to_num(u2_rating_np_array)
    u1_rating_sqrt_sum = np.sum(u1_rating ** 2)
    u2_rating_sqrt_sum = np.sum(u2_rating ** 2)
    return np.sum(u1_rating * u2_rating) / np.sqrt(u1_rating_sqrt_sum * u2_rating_sqrt_sum)


# Find the N nearest neighbors for a specific user from the training data
# For higher performance, rated_item_filter can be used to exclude neighbors who did not rated the same item
def rated_neighbors_sim_score(train_matrix, user_id, rated_item, n=40):

    # Filter training matrix to only find test user in training set and neighbors rated the item
    if (rated_item == None or rated_item not in train_matrix.columns or user_id not in train_matrix.index):
        return None
    else:
        user_ratings = train_matrix.loc[user_id].tolist()
        # Convert list to NumPy array
        user_ratings = np.asarray(user_ratings)

        # Only consider the users who have rated the item
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
def calculate_cf_rating_score(training_matrix, neighbor_scores, user_id, item_id):
    # Calculate user's average rating
    user_ratings = training_matrix.loc[user_id]
    user_avg_rating = np.nanmean(user_ratings)

    numerator = np.sum(neighbor_scores['sim_score'] * (neighbor_scores['item_rating'] - neighbor_scores['avg_rating']))
    denominator = np.sum(neighbor_scores['sim_score'])

    predicted_rating = user_avg_rating + numerator / denominator

    return predicted_rating


# Calculate Rating Predictions from the test set
def predict_new_cf_ratings(new_data, train_matrix):

    # Create a new dataframe to store predicted value
    preds = pd.DataFrame(new_data[['user_id', 'business_id', 'rating']])
    preds['predict_rating'] = None
    for row in new_data.itertuples():
        # Use 40 neighbours for rating predictions
        neighbor_scores = rated_neighbors_sim_score(train_matrix, row[1], row[2], n=40)
        if (neighbor_scores is not None):
            pred = calculate_cf_rating_score(train_matrix, neighbor_scores, row[1], row[2])
            preds.set_value(row[0], 'predict_rating', pred)

    # Remove entries that are unable to produce prediction
    preds = preds[preds['predict_rating'].notnull()]

    if len(preds) == 0:
        preds = None

    return preds


"""Define Matrix Factorization Functions"""
# Calculate Rating Predictions from the test set
def predict_new_mf_ratings(new_data, train_matrix, n_features, reg=0):

    # Create a new dataframe to store predicted value
    pred_df = pd.DataFrame(new_data[['user_id', 'business_id', 'rating']])
    pred_df['predict_rating'] = None

    for row in new_data.itertuples():
        logger.debug('Processing entry %d' % (row[0]))
        uid = row[1]
        bid = row[2]

        # Use 40 neighbours for rating predictions
        neighbor_scores = rated_neighbors_sim_score(train_matrix, uid, bid, n=40)
        if (neighbor_scores is not None):
            try:
                # Find users who have also rated this business
                rate_matrix = train_matrix.loc[neighbor_scores.index]

                # If the predicting user is not in the list, add to decomposing matrix
                if uid not in rate_matrix.index:
                    rate_matrix = rate_matrix.append(train_matrix.loc[uid])

                # Number of users in the decomposing matrix should be smaller than the number of features
                if (n_features > rate_matrix.shape[0]):
                    n_features = rate_matrix.shape[0]

                rate_matrix = rate_matrix.dropna(axis=1, how='all').fillna(0)
                # Matrix factorization with given NMF model
                model = NMF(init='nndsvd', n_components=n_features, alpha=reg, random_state=3, max_iter=500)
                user_feature_mtrx = model.fit_transform(rate_matrix)
                item_feature_mtrx = model.components_

                # Compute prediction matrix
                pred_matrix = np.dot(user_feature_mtrx, item_feature_mtrx)
                pred_matrix = pd.DataFrame(pred_matrix, index=rate_matrix.index, columns=rate_matrix.columns)

                pred = pred_matrix.loc[uid][bid]

            except KeyError:
                pred = np.nan
        else:
            pred = np.nan

        pred_df.set_value(row[0], 'predict_rating', pred)

    return pred_df


"""Define Error Estimation Functions"""
# Compare result using RMSE
def calculate_rmse(actual, prediction):
    return np.sqrt(((prediction - actual) ** 2).mean())


# Compare result using F-measure
def calculate_f_measure(actual, prediction, data_labels):

    try:
        """
        By definition a confusion matrix C is such that C{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
        Thus in binary classification, the count of true negatives is C{0,0}, false negatives is C{1,0}, true positives is C{1,1} and false positives is C{0,1}.
        """
        cf_matrix = confusion_matrix(actual, prediction, data_labels)

        tp = cf_matrix[0, 0]
        fp = cf_matrix[1, 0]
        fn = cf_matrix[0, 1]

        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)

        f_measure = 0
        if ((precision + recall) != 0):
            f_measure = 2 * (precision * recall) / (precision + recall)
        return f_measure

    except Exception as e:
        print(e)


"""Load data files"""
train_data_file = './data/assignment7-train.csv'
train_data = pd.read_csv(train_data_file)
train_data.columns= ['user_id', 'business_id', 'rating']
logger.info('Loaded %d raw data' % (len(train_data)))

test_data_file = './data/assignment7-test.csv'
test_data = pd.read_csv(test_data_file)
test_data.columns= ['user_id', 'business_id', 'rating']
logger.info('Loaded %d raw data' % (len(test_data)))

# Create rating sparse matrix from rating data
train_rating_matrix = train_data.pivot_table(values='rating', index='user_id', columns='business_id')

"""Calculate prediction with CF method - RMSE using CF method: 0.908192"""
cf_predictions = predict_new_cf_ratings(test_data, train_rating_matrix)

logger.info('Calculated %d predictions using CF method' % (len(cf_predictions)))

cf_rmse = calculate_rmse(cf_predictions['rating'], cf_predictions['predict_rating'])
logger.info('Calculated RMSE using CF method: %f ' % (cf_rmse))


"""Calculate prediction with MF method"""
#Grid Search - regularization coefficient 0.001 ~ 0.05, number of features 2~20
#Determined the best number of features 0.001, 10 features with RMSE of 1.747664
rmse_dict = {}
for reg in range(0,5):
    reg_coefficent = float(reg)*0.001
    logger.info('Computing RMSE with Alpha %f' % (reg_coefficent))
    feature_dict = {}
    for n in range(2,21):
        mf_predictions = predict_new_mf_ratings(test_data.head(10), test_data.pivot_table(values='rating', index='user_id', columns='business_id'), n_features = n, reg=reg_coefficent)
        mf_rmse = calculate_rmse(mf_predictions['rating'], mf_predictions['predict_rating'])
        feature_dict[n] = mf_rmse
    rmse_dict[reg_coefficent] = feature_dict

min_alpha = None;
min_n = None
min_rmse = None
for reg in rmse_dict:
    feature_dict = rmse_dict[reg]
    for f in feature_dict:
        if min_rmse == None or min_rmse > feature_dict[f]:
            min_alpha = reg
            min_n = f
            min_rmse = feature_dict[f]
        logger.info('Alpha %f RMSE for %d features is %f' % (reg, f, feature_dict[f]))

logger.info('Best alpha: %f, best number of features: %f with RMSE of%f' % (min_alpha, min_n, min_rmse))

# Select the number of latent features = 10
# Regularization coefficient 0.001
mf_predictions = predict_new_mf_ratings(test_data, test_data.pivot_table(values='rating', index='user_id', columns='business_id'), n_features = 10, reg=0.001)
mf_rmse = calculate_rmse(mf_predictions['rating'], mf_predictions['predict_rating'])
logger.info('Calculated RMSE using MF method: %f ' % (mf_rmse))

'''Apply ensemble method to combine CF ratings and MF ratings with blending ratings'''
# Ensemble with weighted average
cf_predictions=cf_predictions.rename(columns = {'predict_rating':'cf_predict_rating'})
mf_predictions=mf_predictions.rename(columns = {'predict_rating':'mf_predict_rating'})

logger.info('Number of predicted rating with CF method: %f ' % (len(cf_predictions)))
logger.info('Number of predicted rating with MF method: %f ' % (len(mf_predictions)))

ensemble_result = pd.merge(cf_predictions, mf_predictions, how='outer', on=['user_id', 'business_id', 'rating'])
ensemble_result = ensemble_result.drop_duplicates(['user_id', 'business_id', 'rating'])

ensemble_predict_df = pd.DataFrame(ensemble_result[['user_id', 'business_id', 'rating']])

plot_alpha_serie = []
plot_rmse_serie = []
for i in range(0,11):

    alpha = i * 0.1
    ensemble_predict_df['ensemble_rating'] = (ensemble_result['cf_predict_rating'] * alpha) + (ensemble_result['mf_predict_rating'] * (1-alpha))
    ensemble_rmse = calculate_rmse(ensemble_predict_df['rating'], ensemble_predict_df['ensemble_rating'])

    plot_alpha_serie.append(alpha)
    plot_rmse_serie.append(ensemble_rmse)
    logger.info('Ensemble RMSE with Alpha %f is: %f ' % (alpha, ensemble_rmse))

'''Plot Ensemble RMSE graph - Best result of Ensemble RMSE with Alpha 0.800000 is: 0.868461'''

import matplotlib.pyplot as plt

dots = plt.plot(plot_alpha_serie, plot_rmse_serie, 'ro')
line = plt.plot(plot_alpha_serie, plot_rmse_serie, color = 'r', linewidth=2.0)
plt.xlabel('Alpha')
plt.ylabel('Ensemble RMSE value')
plt.show()

