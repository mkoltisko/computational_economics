from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

start_set = pd.read_csv('final_train.csv')
final_set = pd.read_csv('final_test.csv')

# PRE-PROCESSING IMPUTATION
categorical_data = start_set[['cut', 'color', 'clarity']]
float_data = start_set.drop(['cut', 'color', 'clarity'], axis=1)

imputer = KNNImputer(n_neighbors=3)
x = imputer.fit_transform(float_data)
new_start = pd.DataFrame(x, columns=float_data.columns, index=float_data.index)
new_start[['cut', 'color', 'clarity']] = categorical_data

categories = {
    'cut': set(new_start['cut']),
    'color': set(new_start['color']),
    'clarity': set(new_start['clarity'])
}

# remove nan values from each set of ranks
categories['cut'] = {x for x in categories['cut'] if x == x}
categories['color'] = {x for x in categories['color'] if x == x}
categories['clarity'] = {x for x in categories['clarity'] if x == x}
print(categories)


def check_result(observations, predictions, title=''):
    fig_i = plt.figure(figsize=(8, 8))
    ax_i = fig_i.add_subplot(1, 1, 1)
    ax_i.scatter(observations, predictions)
    ax_i.plot([0, 500000], [0, 500000], color='red')
    ax_i.set_title(title)
    ax_i.axhline(y=0, color='k')
    ax_i.axvline(x=0, color='k')
    bounds = max(observations.max(), predictions.max())
    ax_i.set_xlim(0, bounds)
    ax_i.set_ylim(0, bounds)
    ax_i.set_xlabel("observed value")
    ax_i.set_ylabel("predicted value")


# Look at all data which follows a certain rank
def separate(data):
    separated_data = {'cut': {}, 'color': {}, 'clarity': {}}
    for variable, rankings in categories.items():
        for place in rankings:
            isolated = deepcopy(data[data[variable] == place])
            isolated = isolated.drop('cut', axis=1)
            isolated = isolated.drop('color', axis=1)
            isolated = isolated.drop('clarity', axis=1)
            separated_data[variable][place] = isolated
    return separated_data


# R Squared Score evaluator
def evaluate(observations, predictions):
    average = np.mean(observations)
    ssr = sum((observations - predictions) ** 2)
    sst = sum((observations - average) ** 2)
    return 1 - (ssr / sst)


# Root Mean Squared Error evaluator
def rms_error(observations, predictions):
    return sum((observations - predictions) ** 2) / len(observations)


# STRATIFIED SPLIT FOR DISTRIBUTING TRAIN AND TEST DATA
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
new_start['carat'].plot(kind='hist', bins=100,
                        figsize=(8, 8), title='Carat Bins\n'
                                              'Chunks at 0.5, 0.75, 1, 1.5, 2, 2.5\n'
                                              'Falls off after 3')

new_start['carat_category'] = pd.cut(new_start['carat'],
                                     bins=[0, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, np.inf],
                                     labels=[1, 2, 3, 4, 5, 6, 7, 8])
train = []
test = []
for train_index, test_index in sss.split(new_start, new_start['carat_category']):
    train = new_start.iloc[train_index]
    test = new_start.iloc[test_index]

train = train.drop('carat_category', axis=1)  # extraneous column which isn't needed since we have carat
test = test.drop('carat_category', axis=1)

# EXPLORATORY ANALYSIS
"""
From this exploration we can see that each rank has close to the same y-intercept but increasing slopes with rank
Thus, a proper model should weigh in where the diamond places in each category
"""
lin_reg = LinearRegression()  # basic regression to see variation across ranks
exploratory = separate(train)
result_explore = deepcopy(train)
for category, ranks in categories.items():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Linear Regression Outcomes for each rank of ' + category)
    for rank in ranks:
        train_features = exploratory[category][rank].copy()
        train_labels = train_features['price']
        train_features = train_features.drop('price', axis=1)
        lin_reg.fit(train_features, train_labels)
        sorted_features = train_features.sort_values('carat')
        lin_predict = lin_reg.predict(sorted_features)
        ax.plot(sorted_features['carat'], lin_predict, label=rank)
        exploratory[category][rank][category + '_based_price'] = lin_predict
    combined_exp = pd.concat(exploratory[category].values())
    result_explore[category + '_based_price'] = combined_exp[category + '_based_price']
    ax.legend()
    ax.set_xlabel('carat')
    ax.set_ylabel('predicted price')


diffs = pd.DataFrame()
diffs['cut_diffs'] = abs(result_explore['price'] - result_explore['cut_based_price'])
diffs['color_diffs'] = abs(result_explore['price'] - result_explore['color_based_price'])
diffs['clarity_diffs'] = abs(result_explore['price'] - result_explore['clarity_based_price'])
diffs['minimums'] = diffs[['cut_diffs', 'color_diffs', 'clarity_diffs']].min(axis=1)
diffs['cut_best'] = diffs['minimums'] == diffs['cut_diffs']
diffs['color_best'] = diffs['minimums'] == diffs['color_diffs']
diffs['clarity_best'] = diffs['minimums'] == diffs['clarity_diffs']
scores = [sum(diffs['cut_best']), sum(diffs['color_best']), sum(diffs['clarity_best'])]
labels = ['Cut', 'Color', 'Clarity']
fig1, ax1 = plt.subplots()
ax1.pie(scores, labels=labels, autopct='%1.1f%%', )
ax1.set_title('Percentage of Time Each Regression \nCategory is Closest to Actual Price')
ax1.axis('equal')

# BASELINE EXAMPLE TO IMPROVE UPON
poor_predict = train['carat'] * 1000
check_result(train['price'], poor_predict, 'Example Set Prediction')
print('Poor Prediction Result (From Example)')
print('R2  : ' + str(evaluate(train['price'], poor_predict)))
print('RMS : ' + str(rms_error(train['price'], poor_predict)))
print('-----------------')


# FORM CONSENSUS BY WEIGHING IN CUT, COLOR, AND CLARITY

# Predict based on individual rankings and take the mean of each prediction for final predicted price 'consensus'
def predict_consensus(data_set, trained_regressions):
    result = data_set.copy()
    separated_data = separate(data_set)
    for variable, rankings in categories.items():
        for ranking in rankings:
            current_features = separated_data[variable][ranking].drop('price', axis=1)
            separated_data[variable][ranking][variable + '_based_price'] = \
                trained_regressions[variable][ranking].predict(current_features)
        combined_i = pd.concat(separated_data[variable].values())
        result[variable + '_based_price'] = combined_i[variable + '_based_price']
    result['consensus_price'] = result[['cut_based_price', 'color_based_price', 'clarity_based_price']].mean(axis=1)
    return result


# Look at the separated data and fit a model to data of a particular ranking
def build_regressions(data_set, regression_model):
    regressions = {'cut': {}, 'color': {}, 'clarity': {}}
    separated_train = separate(data_set)
    for variable, rankings in categories.items():
        for ranking in rankings:
            current_features = separated_train[variable][ranking].copy()
            current_labels = current_features['price']
            current_features = current_features.drop('price', axis=1)
            regressions[variable][ranking] = deepcopy(regression_model.fit(current_features, current_labels))
    return regressions


starting_regressions = build_regressions(train, RandomForestRegressor(n_estimators=10, random_state=0))

result_train = predict_consensus(train, starting_regressions)
check_result(result_train['price'], result_train['consensus_price'], 'Training Set Prediction')
print('Train Fit Result')
print('R2  : ' + str(evaluate(result_train['price'], result_train['consensus_price'])))
print('RMS : ' + str(rms_error(result_train['price'], result_train['consensus_price'])))

print('-----------------')

result_test = predict_consensus(test, starting_regressions)
check_result(result_test['price'], result_test['consensus_price'], 'Testing Set Prediction')
print('Test Fit Result')
print('R2  : ' + str(evaluate(result_test['price'], result_test['consensus_price'])))
print('RMS : ' + str(rms_error(result_test['price'], result_test['consensus_price'])))

print('-----------------')

# APPLY TO FINAL DATA SET AND WRITE TO FILE

categorical_data = final_set[['cut', 'color', 'clarity']]
float_data = final_set.drop(['cut', 'color', 'clarity'], axis=1)

imputer = KNNImputer(n_neighbors=3)
x = imputer.fit_transform(float_data)
new_final = pd.DataFrame(x, columns=float_data.columns, index=float_data.index)
new_final[['cut', 'color', 'clarity']] = categorical_data
new_final['price'] = 0

result_final = predict_consensus(new_final, starting_regressions)
result_final['price'] = round(result_final['consensus_price'], 2)
result_final['product_id'] = result_final['product_id'].astype(int)
file_output = result_final[['product_id', 'price']]
print('Entries to File : ' + str(int(file_output.size / 2)))
file_output.to_csv('prediction.csv', index=False)

plt.show()
