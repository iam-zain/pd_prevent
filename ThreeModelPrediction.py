
import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr
from keras.optimizers import Adam, Nadam
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.read_csv('Feats45_unCategAge_APPRDX_Gender.csv')
df1 = df.drop(['PATNO','APPRDX'], axis = 1)
df2 = pd.read_csv('NonMotor_Empty.csv')
df3 = df2.drop('Patient_ID', axis = 1)
new_age = int(sys.argv[1])
new_gender = int(sys.argv[2])
df3.at[0, 'Age'] = new_age
df3.at[0, 'Gender'] = new_gender
col_age = df3.pop('Age')
df3.insert(0, "Age", col_age)
dframe1 = pd.read_csv('Feats45_unCategSparse_APPRDX.csv')
dframe = dframe1.drop(['PATNO','APPRDX','Age', 'Gender'], axis = 1)
dframe = dframe.add(1)
tests_scores = []
num_columns = int(sys.argv[3])
def update_values(df3, dframe):
    # Input from the user for number of columns
    num_columns = int(sys.argv[3])
    if num_columns < 5 or num_columns > 45:
        print("Invalid input, please enter a number from 5 to 45")
        sys.exit()

    columns = []
    values = []

    for i in range(num_columns):
        column = sys.argv[i*2 + 4]
        value = float(sys.argv[i*2 + 5])
        columns.append(column)
        values.append(value)
        tests_scores.append([column, value])
    for i in range(num_columns):
        df3.loc[df3[columns[i]] != values[i], columns[i]] = values[i] 
    for col in dframe.columns:
        if col not in columns and col != 'Age': 
            dframe = dframe.drop(col, axis=1)
    
    return df3, dframe

df3, dframe = update_values(df3, dframe)
col_age = df3.pop('Age')
df3.insert(0, "Age", col_age)
col_gender = df3.pop('Gender')
df3.insert(1, "Gender", col_gender)
df4 = df3
df3 = df3.dropna(axis = 1)
dframe = dframe.dropna()
col_age = dframe1.pop('Age')
col_gender = dframe1.pop('Gender')
dframe.insert(0, "Age", col_age)
dframe.insert(1, "Gender", col_gender)
col_Apprdx = dframe1.pop('APPRDX')
dframe.insert(0, 'APPRDX', col_Apprdx)
dframes = dframe.drop(['APPRDX'], axis = 1)
dframe.loc[:, "APPRDX"] = dframe["APPRDX"].apply(lambda x: x - 1)
count_0 = dframe['APPRDX'].value_counts()[0]
count_1 = dframe['APPRDX'].value_counts()[1]
min_count = min(count_0, count_1)
subset_dframe = dframe.groupby('APPRDX').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
X = subset_dframe.iloc[:, 1:].values
y = subset_dframe.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
np.random.seed(1)
tf.random.set_seed(1)
model_ann = load_model('model_ann.h5')
new_input_layer = Input(shape=dframes.shape[1:])
new_hidden_layer1 = Dense(units=12, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.057, l2=0.0001))(new_input_layer)
new_hidden_layer1_dropout = Dropout(0.0)(new_hidden_layer1)
pretrained_hidden_layer2_weights = model_ann.layers[1].get_weights()
pretrained_output_layer_weights = model_ann.layers[2].get_weights()
new_hidden_layer2 = Dense(units=pretrained_hidden_layer2_weights[0].shape[1], activation='relu', weights=pretrained_hidden_layer2_weights)(new_hidden_layer1_dropout)
new_output_layer = Dense(units=pretrained_output_layer_weights[0].shape[1], activation='sigmoid', weights=pretrained_output_layer_weights)(new_hidden_layer2)
new_model = Model(inputs=new_input_layer, outputs=new_output_layer)
optimizer = Nadam(learning_rate=0.001)
new_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
for layer in new_model.layers[3:]:
    layer.trainable = False
results_list = []
for _ in range(5):
    new_model.fit(X_train, y_train, batch_size=10, epochs=50, shuffle=True, verbose=0)
    y_pred = new_model.predict(X_test, verbose = 0)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    prediction = new_model.predict(df3, verbose=0)
    
    if prediction[0][0] > 0.5:
        predicted_class = "Healthy"
        scaled_prediction = prediction[0][0]
    else:
        scaled_prediction = 1 - (prediction[0][0] - 0) * (1 - 0.5) / (0.5 - 0)
        predicted_class = "Patient"
        scaled_prediction = scaled_prediction

    results_list.append({'Category': predicted_class, 'Percentage': scaled_prediction, 'Accuracy': acc_score})
result_df_NeuNet = pd.DataFrame(results_list)
final_output = result_df_NeuNet['Category'].value_counts().idxmax()
mean_scaled_prediction = result_df_NeuNet[result_df_NeuNet['Category'] == final_output]['Percentage'].mean()
final_result_df_NeuNet = pd.DataFrame({'Category': [final_output],
                                       'Percentage': [mean_scaled_prediction]})
#print(result_df_NeuNet)

# 2nd Model
df4.insert (0, 'Patient_ID', df2['Patient_ID'])
merged_df = pd.concat ([df1, df4], axis = 0)
merged_df.reset_index(inplace = True)
merged_df1 = merged_df.drop(['index','Patient_ID'], axis = 1)
scaler = MinMaxScaler (feature_range = (1,5))
df_scaled = scaler.fit_transform(merged_df1)
df_scaled = pd.DataFrame (df_scaled, columns = merged_df1.columns)
df_scaled.insert (0, 'Patient_ID', merged_df['Patient_ID'], True)
df_melted = df_scaled.melt(id_vars='Patient_ID', var_name='Feature', value_name='Value')
feat_rating_count = df_melted.groupby('Patient_ID')['Value'].count()
feat_rating_count = pd.DataFrame(feat_rating_count)
feat_rating_count.columns = ['Value_count']
feat_rating_count['Patient_ID'] = feat_rating_count.index
feat_rating_count = feat_rating_count.reset_index(drop=True)
dframe = df_melted.merge(feat_rating_count, on = 'Patient_ID', how = 'inner')
RatingMat = dframe.pivot_table(index=['Patient_ID'],columns=['Feature'],values=['Value'],fill_value=0)
Original_RatingMat = RatingMat.copy()
RatingMat.columns = RatingMat.columns.droplevel()

RatingMat_centered = RatingMat.sub(RatingMat.mean(axis=1), axis=0)
user_sim_cos = cosine_similarity(RatingMat)
user_sim_cos_df = pd.DataFrame(user_sim_cos,index=RatingMat.index,columns=RatingMat.index)
user_sim_pear = 1 - pairwise_distances(RatingMat_centered, metric='correlation')
user_sim_pear_df = pd.DataFrame(user_sim_pear, index=RatingMat.index, columns=RatingMat.index)
user_sim_euc = 1 - pairwise_distances(RatingMat, metric="euclidean")
user_sim_euc_df = pd.DataFrame(user_sim_euc, index=RatingMat.index, columns=RatingMat.index)
current_user_rating = dframe[(dframe.Patient_ID == 999) & (dframe.Value != 0)]['Feature']
current_user_rating = pd.DataFrame(current_user_rating, columns=['Feature'])
def categorize(x):
    if x < 165:
        return ('Patient')
    else:
        return ('Healthy')

# put similarity of current user i.e. 999 in a dataframe because later we need for weighted average..
curr_user_sim_cos = pd.DataFrame(user_sim_cos_df.loc[999])
curr_user_sim_cos.rename(columns={999:'Similarity_Score'}, inplace=True)
curr_user_sim_cos.sort_values(by='Similarity_Score', ascending=False, inplace=True)
curr_user_sim_cos1 = curr_user_sim_cos.iloc[1:, :]
curr_user_sim_cos1.reset_index(inplace=True)
curr_user_sim_cos1.rename(columns={'index': 'Index_Column'}, inplace=True)
curr_user_sim_cos1['Patient_Type'] = curr_user_sim_cos1['Patient_ID'].apply(categorize)

curr_user_sim_pear = pd.DataFrame(user_sim_pear_df.loc[999])
curr_user_sim_pear.rename(columns={999:'Similarity_Score'}, inplace=True)
curr_user_sim_pear.sort_values(by='Similarity_Score', ascending=False, inplace=True)
curr_user_sim_pear1 = curr_user_sim_pear.iloc[1:, :]
curr_user_sim_pear1.reset_index(inplace=True)
curr_user_sim_pear1.rename(columns={'index': 'Index_Column'}, inplace=True)
curr_user_sim_pear1['Patient_Type'] = curr_user_sim_pear1['Patient_ID'].apply(categorize)

curr_user_sim_euc = pd.DataFrame(user_sim_euc_df.loc[999])
curr_user_sim_euc.rename(columns={999:'Similarity_Score'}, inplace=True)
curr_user_sim_euc.sort_values(by='Similarity_Score', ascending=False, inplace=True)
curr_user_sim_euc1 = curr_user_sim_euc.iloc[1:, :]
curr_user_sim_euc1.reset_index(inplace=True)
curr_user_sim_euc1.rename(columns={'index': 'Index_Column'}, inplace=True)
curr_user_sim_euc1['Patient_Type'] = curr_user_sim_euc1['Patient_ID'].apply(categorize)

similar_user_cos = curr_user_sim_cos1.iloc[:5, :]
similar_user_pear = curr_user_sim_pear1.iloc[:5, :]
similar_user_euc = curr_user_sim_euc1.iloc[:5, :]

counts_cos = similar_user_cos['Patient_Type'].value_counts()
counts_pear = similar_user_pear['Patient_Type'].value_counts()
counts_euc = similar_user_euc['Patient_Type'].value_counts()

counts_all = pd.concat([counts_cos, counts_pear, counts_euc], axis=1).fillna(0)
total_counts = counts_all.sum(axis=1)
total_counts = pd.DataFrame(total_counts)
most_occur_value = total_counts.idxmax()
similar_users = pd.concat([similar_user_cos, similar_user_pear, similar_user_euc])
rec_sys_counts = similar_users['Patient_Type'].value_counts()
if 'Healthy' in rec_sys_counts:
    healthy_pct = rec_sys_counts['Healthy'] / len(similar_users)
else:
    healthy_pct = 0

if 'Patient' in rec_sys_counts:
    patient_pct = rec_sys_counts['Patient'] / len(similar_users)
else:
    patient_pct = 0


if healthy_pct > patient_pct:
    category = 'Healthy'
    percentage = healthy_pct
else:
    category = 'Patient'
    percentage = patient_pct

result_df_RecSys = pd.DataFrame({'Category': [category], 'Percentage': [percentage]})

most_occurring = similar_users['Patient_Type'].value_counts().index[0]
select_simil_user = similar_users.loc[similar_users['Patient_Type'] == most_occurring]
df_simil = df1[df1['Patient_ID'].isin(select_simil_user['Patient_ID'])]
select_simil_user = select_simil_user.drop(['Patient_Type'], axis=1)
df_score = pd.merge(similar_users, df_simil, on='Patient_ID', how='inner')
df_score = df_score.drop_duplicates(subset = ['Patient_ID'], keep='first')
colX = df_simil
colY = df4.drop(['Patient_ID', 'Age'], axis=1)
col_weights = df_score.iloc[:, 1].reset_index(drop=True)
values = colX.iloc[:, 2:]
weighted_avg_cols = np.average(values, axis=0, weights=col_weights)
fill_values = {col: val for col, val in zip(colY.columns, weighted_avg_cols)}
colY = colY.fillna(fill_values)
col_Age = df4.pop('Age')
colY.insert(0, 'Age', col_Age)


# 3rd Model
predictions = []
for i in range(num_columns):
    np.random.seed(1)
    tf.random.set_seed(1)
    column = tests_scores[i][0]
    user_data = tests_scores[i][1]
    user_data = np.array(user_data).reshape(-1, 1)
    LASSO_model = joblib.load(column + '_Lasso_model.joblib')
    Lasso_prediction = LASSO_model.predict(user_data)
    svmL_model = joblib.load(column + '_svmL_model.joblib')
    SVM_prediction = svmL_model.predict(user_data)
    enet_model = joblib.load(column + '_enet_model.joblib')
    enet_prediction = enet_model.predict(user_data)
    predictions.append([Lasso_prediction[0], SVM_prediction[0],enet_prediction[0]])

feat_pred_df = pd.DataFrame(predictions, columns=["Lasso", "SVM Linear", "Elastic Net"])
counts = feat_pred_df.apply(pd.Series.value_counts, axis=1).fillna(0).astype(int)
result = counts.idxmax(axis=1)
max_inAll = pd.DataFrame()
max_inAll["Maximum_Occurrence"] = pd.DataFrame(result)
max_inAll = max_inAll.iloc[:, ]
counts = max_inAll['Maximum_Occurrence'].value_counts()
most_occurring_value = counts.index[0]
#print("The user might fall under category of", most_occurring_value)
patient_count = ((feat_pred_df == "Patient").sum(axis=1) > 0).sum()
healthy_count = ((feat_pred_df == "Healthy").sum(axis=1) > 0).sum()
total_count = len(feat_pred_df)
patient_percent = (patient_count / total_count)
healthy_percent = (healthy_count / total_count)
if patient_percent > healthy_percent:
    category = "Patient"
    percentage = patient_percent
else:
    category = "Healthy"
    percentage = healthy_percent
result_df_IndiMod = pd.DataFrame({"Category": [category], "Percentage": [percentage]})
#print(result_df_IndiMod)

pred_all = pd.concat([result_df_IndiMod, result_df_RecSys, final_result_df_NeuNet])
#print(pred_all)

most_frequent = pred_all['Category'].mode()[0]
low_cat = pred_all['Category'].value_counts().idxmin()
high_cat = pred_all['Category'].value_counts().idxmax()

low_cat_sum = pred_all.loc[pred_all['Category'] == low_cat, 'Percentage'].sum()
high_cat_sum = pred_all.loc[pred_all['Category'] == high_cat, 'Percentage'].sum()
if high_cat_sum == low_cat_sum:
    sum_difference = (high_cat_sum)/3 * 100
else: 
    sum_difference = (high_cat_sum - low_cat_sum)/3 * 100
round_differ = "{:.0f}".format(sum_difference)

print( most_frequent + "#" + (round_differ))
