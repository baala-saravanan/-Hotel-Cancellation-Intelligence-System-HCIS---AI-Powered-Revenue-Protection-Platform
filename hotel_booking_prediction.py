# #!/usr/bin/env python
# # coding: utf-8

# # Inital Exploratory Data Analysis (EDA)

# # In[2]:


# import pandas as pd

# df = pd.read_csv("D:\hotel_booking_prediction\hotel_booking.csv")
# df.head(10) # show first 10 rows


# # In[3]:


# df.tail(10)


# # In[4]:


# df.shape


# # In[5]:


# df.info()


# # In[6]:


# data = []

# for col in df.columns:
#     data.append({
#         "Columns": col,
#         "Column_Uniques": df[col].unique(),
#         "No_of_Unique": df[col].nunique(),
#         "Missing_Values": df[col].isnull().sum()
#     })

# a = pd.DataFrame(data)
# a


# # In[7]:


# df.describe()


# # In[8]:


# df.describe(include = "object")


# # In[9]:


# categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


# # In[10]:


# categorical_columns


# # In[11]:


# numerical_columns


# # In[12]:


# df[categorical_columns].head(10)


# # In[13]:


# df[categorical_columns].tail(10)


# # In[14]:


# df[numerical_columns].head(10)


# # In[15]:


# df[numerical_columns].tail(10)


# # Pre-processing

# # In[16]:


# df.isna().sum()


# # In[17]:


# null_percentage = (df.isna().sum() / df.shape[0]) * 100
# null_percentage


# # In[18]:


# high_null_features = null_percentage[null_percentage > 50]
# high_null_features


# # In[19]:


# for high_null_col in high_null_features.index:
#   print(f'Data type of {high_null_col}: {df[high_null_col].dtype},\n \n unique values of {high_null_col}: {df[high_null_col].unique()},\n \n total no. of unique values of {high_null_col}: {df[high_null_col].nunique()}')


# # In[20]:


# df = df.drop(['agent', 'company'], axis=1)


# # Feature Engineering [Converting Categorical to Numerical Features] - LabelEncoder

# # In[21]:


# from sklearn.preprocessing import LabelEncoder

# label_encoders = {le_col: LabelEncoder() for le_col in categorical_columns}

# for le_col in categorical_columns:
#     non_nan_values = df[le_col].dropna()
#     encoded_values = label_encoders[le_col].fit_transform(non_nan_values)
#     df.loc[non_nan_values.index, le_col] = encoded_values
# df[categorical_columns].head(10)


# # In[22]:


# df[categorical_columns].tail(10)


# # In[23]:


# df[categorical_columns].isna().sum()


# # In[24]:


# # Explicitly remove 'agent' and 'company' from the numerical_columns list
# if 'agent' in numerical_columns:
#     numerical_columns.remove('agent')
# if 'company' in numerical_columns:
#     numerical_columns.remove('company')


# # In[25]:


# df[numerical_columns].isna().sum()


# # In[26]:


# df.info()


# # In[27]:


# df.isna().sum()


# # In[28]:


# df


# # Handling Missing Values [NaN] - KNNImputer

# # In[29]:


# # Add the encoded 'country' column back to the DataFrame as a numerical column
# df['country_encoded'] = df['country']
# numerical_columns.append('country_encoded')
# categorical_columns.remove('country')


# # In[30]:


# from sklearn.impute import KNNImputer

# # Initialize KNNImputer
# imputer = KNNImputer(n_neighbors=5, weights='distance')

# imputed_array = imputer.fit_transform(df)

# df = pd.DataFrame(imputed_array, columns=df.columns)
# df.head(10)


# # In[31]:


# df.isna().sum()


# # In[32]:


# df.shape


# # In[33]:


# df.info()


# # In[34]:


# # Drop the original 'country' column
# df = df.drop('country', axis=1)

# # Rename 'country_encoded' to 'country'
# df = df.rename(columns={'country_encoded': 'country'})


# # In[35]:


# df.info()


# # In[36]:


# df


# # Top Features Selection - XGBoost

# # In[37]:


# # Install XGBoost if not available
# get_ipython().run_line_magic('pip', 'install xgboost')

# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split

# # Assuming your dataset is already label-encoded
# # Define features (X) and target (y)
# X = df.drop('is_canceled', axis=1)
# y = df['is_canceled']

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize XGBoost Classifier
# xgb_model = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1,
#     eval_metric='logloss'
# )

# # Fit the model
# xgb_model.fit(X_train, y_train)

# # Extract feature importance
# importance_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': xgb_model.feature_importances_
# }).sort_values(by='Importance', ascending=False)


# # In[38]:


# # Display top 15 important features
# top_features = importance_df.head(15)
# print("🔥 Top 15 Most Important Features (Based on XGBoost):")
# print(top_features)


# # In[39]:


# # Optional: select only top features for final model
# selected_features = top_features['Feature'].tolist()
# X_selected = X[selected_features]
# X_selected


# # In[65]:


# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
# plt.gca().invert_yaxis()
# plt.title("Top 15 Important Features (XGBoost)")
# plt.xlabel("Feature Importance Score")
# plt.show()


# # Feature Scaling

# # In[41]:


# for col in X_selected.columns:
#   print(f"Unique values for column '{col}':")
#   print(X_selected[col].unique())
#   print(X_selected[col].nunique())
#   print("-" * 30)


# # In[42]:


# from sklearn.preprocessing import StandardScaler

# # Define the columns to scale
# columns_to_scale = ['previous_bookings_not_canceled', 'booking_changes', 'country', 'lead_time', 'assigned_room_type', 'previous_cancellations']

# # Initialize StandardScaler
# scaler = StandardScaler()

# # Scale the selected columns
# df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])


# # In[43]:


# # Display the first few rows of the DataFrame with scaled columns
# display(df.head(20))


# # 
# # Model Building - Without Hyperparameter Tuning

# # In[44]:


# get_ipython().run_line_magic('pip', 'install catboost')


# # In[45]:


# get_ipython().run_line_magic('pip', 'install lightgbm')


# # In[ ]:


# # from sklearn.linear_model import LogisticRegression
# # from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
# # from sklearn.svm import SVC
# # from sklearn.naive_bayes import GaussianNB
# # from catboost import CatBoostClassifier
# # # from xgboost import XGBClassifier
# # from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# classifiers = {
#     # 'Logistic Regression': LogisticRegression(),
#     # 'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(),
#     # 'Random Forest': RandomForestClassifier(),
#     # 'Support Vector Classifier': SVC(),
#     # 'Naive Bayes': GaussianNB(),
#     # 'XGBoost': XGBClassifier(),
#     # 'AdaBoost': AdaBoostClassifier(),
#     # 'Extra Trees': ExtraTreesClassifier(),
#     # 'HistGradientBoosting': HistGradientBoostingClassifier(),
#     # 'CatBoost': CatBoostClassifier(verbose=0),
#     # 'LightGBM': LGBMClassifier()
# }

# results = {}

# for name, clf in classifiers.items():

#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     results[name] = {'Accuracy': accuracy, 'F1 Score': f1}

# results_df = pd.DataFrame(results).T

# results_df_sorted = results_df.sort_values(by='F1 Score', ascending=False)
# results_df_sorted


# # In[ ]:


# # RF_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# # RF_model.fit(X_train, y_train)

# # y_pred_RF = RF_model.predict(X_test)

# # accuracy_RF = accuracy_score(y_test, y_pred_RF)
# # f1_score_RF = f1_score(y_test, y_pred_RF)

# # print(f'Random Forest Accuracy: {accuracy_RF}\n')
# # print(f'Random Forest F1 Score: {f1_score_RF}')


# # In[48]:


# DT_model = DecisionTreeClassifier(random_state=42)

# DT_model.fit(X_train, y_train)

# y_pred_DT = DT_model.predict(X_test)

# accuracy_DT = accuracy_score(y_test, y_pred_DT)
# f1_score_DT = f1_score(y_test, y_pred_DT)

# print(f'Decision Tree Accuracy: {accuracy_DT}\n')
# print(f'Decision Tree F1 Score: {f1_score_DT}')


# # In[ ]:


# # xgb_model = XGBClassifier(
# #     n_estimators=300,
# #     learning_rate=0.05,
# #     max_depth=6,
# #     subsample=0.8,
# #     colsample_bytree=0.8,
# #     random_state=42,
# #     n_jobs=-1,
# #     eval_metric='logloss'
# # )

# # xgb_model.fit(X_train, y_train)

# # y_pred_xgb = xgb_model.predict(X_test)

# # accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# # f1_score_xgb = f1_score(y_test, y_pred_xgb)

# # print(f'XGBoost Accuracy: {accuracy_xgb}\n')
# # print(f'XGBoost F1 Score: {f1_score_xgb}')


# # In[ ]:


# # adaboost_model = AdaBoostClassifier(random_state=42)

# # adaboost_model.fit(X_train, y_train)

# # y_pred_adaboost = adaboost_model.predict(X_test)

# # accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
# # f1_score_adaboost = f1_score(y_test, y_pred_adaboost)

# # print(f'AdaBoost Accuracy: {accuracy_adaboost}\n')
# # print(f'AdaBoost F1 Score: {f1_score_adaboost}')


# # In[ ]:


# # catboost_model = CatBoostClassifier(iterations=100, depth=7, learning_rate=0.1, verbose=0, random_state=42)

# # catboost_model.fit(X_train, y_train)

# # y_pred_catboost = catboost_model.predict(X_test)

# # accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
# # f1_score_catboost = f1_score(y_test, y_pred_catboost)

# # print(f'CatBoost Accuracy: {accuracy_catboost}\n')
# # print(f'CatBoost F1 Score: {f1_score_catboost}')


# # In[ ]:


# # lgbm_model = LGBMClassifier()

# # lgbm_model.fit(X_train, y_train)

# # y_pred_lgbm = lgbm_model.predict(X_test)

# # accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
# # f1_score_lgbm = f1_score(y_test, y_pred_lgbm)

# # print(f'LightGBM Accuracy: {accuracy_lgbm}\n')
# # print(f'LightGBM F1 Score: {f1_score_lgbm}')


# # In[ ]:


# # hgb_model = HistGradientBoostingClassifier(random_state=42)

# # hgb_model.fit(X_train, y_train)

# # y_pred_hgb = hgb_model.predict(X_test)

# # accuracy_hgb = accuracy_score(y_test, y_pred_hgb)
# # f1_score_hgb = f1_score(y_test, y_pred_hgb)

# # print(f'HistGradientBoosting Accuracy: {accuracy_hgb}\n')
# # print(f'HistGradientBoosting F1 Score: {f1_score_hgb}')


# # In[ ]:


# # et_model = ExtraTreesClassifier(random_state=42, n_jobs=-1)

# # et_model.fit(X_train, y_train)

# # y_pred_et = et_model.predict(X_test)

# # accuracy_et = accuracy_score(y_test, y_pred_et)
# # f1_score_et = f1_score(y_test, y_pred_et)

# # print(f'Extra Trees Accuracy: {accuracy_et}\n')
# # print(f'Extra Trees F1 Score: {f1_score_et}')


# # Confusion Matrix

# # In[55]:


# get_ipython().run_line_magic('pip', 'install seaborn')

# import seaborn as sns

# models = {
#     'Random Forest': RF_model,
#     'Decision Tree': DT_model,
#     'XGBoost': xgb_model,
#     'AdaBoost': adaboost_model,
#     'CatBoost': catboost_model,
#     'LightGBM': lgbm_model,
#     'HistGradientBoosting': hgb_model,
#     'Extra Trees': et_model
# }

# for name, model in models.items():
#     y_pred = model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)

#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix for {name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()


# # Classification Report

# # In[56]:


# from sklearn.metrics import classification_report

# models = {
#     'Random Forest': RF_model,
#     'Decision Tree': DT_model,
#     'XGBoost': xgb_model,
#     'AdaBoost': adaboost_model,
#     'CatBoost': catboost_model,
#     'LightGBM': lgbm_model,
#     'HistGradientBoosting': hgb_model,
#     'Extra Trees': et_model
# }

# for name, model in models.items():
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred)
#     print(f'Classification Report for {name}:\n{report}\n')


# # K-Fold Cross-Validation

# # In[ ]:


# # from sklearn.model_selection import KFold, cross_val_score

# # k = 3 #10
# # kf = KFold(n_splits=k, shuffle=True, random_state=4)

# # kf_results = {}

# # for name, clf in classifiers.items():

# #     accuracy_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
# #     f1_scores = cross_val_score(clf, X, y, cv=kf, scoring='f1_weighted')

# #     kf_results[name] = {
# #         'Mean Accuracy': accuracy_scores.mean(),
# #         'Mean F1 Score': f1_scores.mean()
# #     }

# # kf_results_df = pd.DataFrame(kf_results).T

# # kf_results_df_sorted = kf_results_df.sort_values(by='Mean F1 Score', ascending=False)

# # kf_results_df_sorted


# # Hyperparameter Tuning Using Grid Search

# # In[ ]:


# # from sklearn.model_selection import GridSearchCV

# # hp_classifiers = {
# #     # 'Logistic Regression': LogisticRegression(),
# #     # 'K-Nearest Neighbors': KNeighborsClassifier(),
# #     'Decision Tree': DecisionTreeClassifier(),
# #     'Random Forest': RandomForestClassifier(),
# #     # 'Support Vector Classifier': SVC(),
# #     # 'Naive Bayes': GaussianNB(),
# #     'XGBoost': XGBClassifier(),
# #     'AdaBoost': AdaBoostClassifier(),
# #     'Extra Trees': ExtraTreesClassifier(),
# #     'HistGradientBoosting': HistGradientBoostingClassifier(),
# #     'CatBoost': CatBoostClassifier(verbose=0),
# #     'LightGBM': LGBMClassifier()
# # }

# # hp_grids = {
# #     # 'Logistic Regression': {
# #     #     'C': [0.1, 1, 10],
# #     #     'solver': ['liblinear', 'lbfgs']
# #     # },
# #     # 'K-Nearest Neighbors': {
# #     #     'n_neighbors': [3, 5, 7, 9],
# #     #     'weights': ['uniform', 'distance'],
# #     #     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
# #     # },
# #     'Decision Tree': {
# #         'max_depth': [3, 5, 7, 10],
# #         'min_samples_split': [2, 5, 10],
# #         'min_samples_leaf': [1, 2, 4]
# #     },
# #     'Random Forest': {
# #         'n_estimators': [50, 100, 200],
# #         'max_depth': [3, 5, 7],
# #         'min_samples_split': [2, 5, 10],
# #         'min_samples_leaf': [1, 2, 4]
# #     },
# #     # 'Support Vector Classifier': {
# #     #     'C': [0.1, 1, 10],
# #     #     'kernel': ['linear', 'rbf'],
# #     #     'gamma': ['scale', 'auto']
# #     # },
# #     # 'Naive Bayes': {},
# #     'XGBoost': {
# #         'n_estimators': [50, 100, 200],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'max_depth': [3, 5, 7],
# #         'subsample': [0.8, 1.0]
# #     },
# #     'AdaBoost': {
# #         'n_estimators': [50, 100, 200],
# #         'learning_rate': [0.01, 0.1, 1.0]
# #     },
# #     'Extra Trees': {
# #         'n_estimators': [50, 100, 200],
# #         'max_depth': [3, 5, 7],
# #         'min_samples_split': [2, 5, 10],
# #         'min_samples_leaf': [1, 2, 4]
# #     },
# #     'HistGradientBoosting': {
# #         'max_iter': [100, 200, 300],
# #         'max_depth': [3, 5, 7],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'min_samples_leaf': [1, 2, 4]
# #     },
# #     'CatBoost': {
# #         'iterations': [50, 100, 200],
# #         'depth': [3, 5, 7],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'l2_leaf_reg': [1, 3, 5]
# #     },
# #     'LightGBM': {
# #         'n_estimators': [50, 100, 200],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'max_depth': [3, 5, 7],
# #         'num_leaves': [31, 50, 70]
# #     }
# # }

# # hp_tuned_results = {}

# # for name, clf in hp_classifiers.items():
# #     print(f"Performing grid search for {name}...")

# #     hp_grid_search = GridSearchCV(clf, hp_grids.get(name, {}), cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
# #     # hp_grid_search = GridSearchCV(clf, hp_grids[name], cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)
# #     hp_grid_search.fit(X_train, y_train)

# #     hp_best_model = hp_grid_search.best_estimator_
# #     y_pred = hp_best_model.predict(X_test)

# #     accuracy = accuracy_score(y_test, y_pred)
# #     f1 = f1_score(y_test, y_pred)

# #     hp_tuned_results[name] = {
# #         'Best Params': hp_grid_search.best_params_,
# #         'Accuracy': accuracy,
# #         'F1 Score': f1
# #     }

# # hp_tuned_results_df = pd.DataFrame(hp_tuned_results).T

# # hp_tuned_results_df_sorted = hp_tuned_results_df.sort_values(by='Accuracy', ascending=False)
# # # hp_tuned_results_df_sorted = hp_tuned_results_df.sort_values(by='F1 Score', ascending=False)

# # hp_tuned_results_df_sorted


# # Conclusion:
# # All eight models — Decision Tree, Random Forest, Extra Trees, XGBoost, AdaBoost, HistGradientBoosting, CatBoost, and LightGBM — achieved perfect Accuracy and F1-score (1.0) after hyperparameter tuning, indicating a highly separable dataset.
# # Among them, the Decision Tree stands out as the best choice because it achieved the same perfect performance even without tuning, making it the simplest, most interpretable, and least complex model while maintaining excellent predictive power.

# # In[76]:


# # ----------------------------
# # Target meaning
# # ----------------------------
# target_labels = {0: "Not Cancelled", 1: "Cancelled"}

# # ----------------------------
# # 1. reservation_status
# # ----------------------------
# mapping_reservation_status = {1: 'Check-Out', 0: 'Canceled', 2: 'No-Show'}
# df['reservation_status_label'] = df['reservation_status'].map(mapping_reservation_status)

# plt.figure(figsize=(8, 4))
# sns.countplot(data=df, x='reservation_status_label', hue='is_canceled', palette='Set2')
# plt.title('Reservation Status vs is_canceled', fontsize=14)
# plt.xlabel('Reservation Status')
# plt.ylabel('Count')
# plt.legend(title='Booking Status', labels=target_labels.values())
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # 2. deposit_type
# # ----------------------------
# mapping_deposit_type = {0: 'No Deposit', 1: 'Non Refund', 2: 'Refundable'}
# df['deposit_type_label'] = df['deposit_type'].map(mapping_deposit_type)

# plt.figure(figsize=(8, 4))
# sns.countplot(data=df, x='deposit_type_label', hue='is_canceled', palette='Set2')
# plt.title('Deposit Type vs is_canceled', fontsize=14)
# plt.xlabel('Deposit Type')
# plt.ylabel('Count')
# plt.legend(title='Booking Status', labels=target_labels.values())
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # 3. market_segment
# # ----------------------------
# mapping_market_segment = {
#     6: 'Online TA', 5: 'Offline TA/TO', 4: 'Groups', 3: 'Direct',
#     2: 'Corporate', 1: 'Complementary', 0: 'Aviation', 7: 'Undefined'
# }
# df['market_segment_label'] = df['market_segment'].map(mapping_market_segment)

# plt.figure(figsize=(10, 5))
# sns.countplot(data=df, x='market_segment_label', hue='is_canceled', palette='Set2')
# plt.title('Market Segment vs is_canceled', fontsize=14)
# plt.xlabel('Market Segment')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.legend(title='Booking Status', labels=target_labels.values())
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # 4. total_of_special_requests
# # ----------------------------
# # plt.figure(figsize=(8, 4))
# # sns.countplot(data=df, x='total_of_special_requests', hue='is_canceled', palette='Set2')
# # plt.title('Special Requests vs is_canceled', fontsize=14)
# # plt.xlabel('Total of Special Requests')
# # plt.ylabel('Count')
# # plt.legend(title='Booking Status', labels=target_labels.values())
# # plt.tight_layout()
# # plt.show()

# # ----------------------------
# # 5. required_car_parking_spaces
# # ----------------------------
# # plt.figure(figsize=(8, 4))
# # sns.barplot(
# #     data=df, x='required_car_parking_spaces',
# #     y='is_canceled', palette='Set2', ci=None
# # )
# # plt.title('Car Parking Spaces vs Cancellation Rate', fontsize=14)
# # plt.xlabel('Required Car Parking Spaces')
# # plt.ylabel('Average Cancellation Rate')
# # plt.tight_layout()
# # plt.show()

# # ----------------------------
# # 6. country (top 10)
# # ----------------------------
# # top_countries = df['country'].value_counts().head(10).index
# # df_top_countries = df[df['country'].isin(top_countries)]

# # plt.figure(figsize=(10, 5))
# # sns.barplot(
# #     data=df_top_countries,
# #     x='country', y='is_canceled',
# #     palette='Set2', ci=None
# # )
# # plt.title('Top 10 Countries vs Cancellation Rate', fontsize=14)
# # plt.xlabel('Country')
# # plt.ylabel('Average Cancellation Rate')
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.show()

# # ----------------------------
# # 7. lead_time
# # ----------------------------
# # plt.figure(figsize=(8, 4))
# # sns.boxplot(data=df, x='is_canceled', y='lead_time', palette='Set2')
# # plt.title('Lead Time Distribution by Cancellation Status', fontsize=14)
# # plt.xlabel('is_canceled (0=Not, 1=Yes)')
# # plt.ylabel('Lead Time (days)')
# # plt.tight_layout()
# # plt.show()

# # ----------------------------
# # 8. previous_cancellations
# # ----------------------------
# # plt.figure(figsize=(8, 4))
# # sns.countplot(data=df, x='previous_cancellations', hue='is_canceled', palette='Set2')
# # plt.title('Previous Cancellations vs is_canceled', fontsize=14)
# # plt.xlabel('Previous Cancellations')
# # plt.ylabel('Count')
# # plt.legend(title='Booking Status', labels=target_labels.values())
# # plt.tight_layout()
# # plt.show()

# # ----------------------------
# # 9. customer_type
# # ----------------------------
# mapping_customer_type = {
#     2: 'Transient', 3: 'Transient-Party', 0: 'Contract', 1: 'Group'
# }
# df['customer_type_label'] = df['customer_type'].map(mapping_customer_type)

# plt.figure(figsize=(8, 4))
# sns.countplot(data=df, x='customer_type_label', hue='is_canceled', palette='Set2')
# plt.title('Customer Type vs is_canceled', fontsize=14)
# plt.xlabel('Customer Type')
# plt.ylabel('Count')
# plt.legend(title='Booking Status', labels=target_labels.values())
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # 10. arrival_date_day_of_month
# # ----------------------------
# # plt.figure(figsize=(8, 4))
# # sns.boxplot(data=df, x='is_canceled', y='arrival_date_day_of_month', palette='Set2')
# # plt.title('Arrival Day of Month vs is_canceled', fontsize=14)
# # plt.xlabel('is_canceled (0=Not, 1=Yes)')
# # plt.ylabel('Arrival Day of Month')
# # plt.tight_layout()
# # plt.show()


# # Streamlit web app - LIVE predictions

# # In[84]:


# import joblib # for loading your trained model

# # Example: After training
# # clf.fit(X_train, y_train)

# # Save the trained model
# joblib.dump(DT_model, "decision_tree_model.pkl")

# print("Model saved successfully!")


# # In[83]:


# # Installation command for streamlit
# get_ipython().run_line_magic('pip', 'install streamlit')

# # app.py
# import streamlit as st

# # Load your trained model
# model = joblib.load("decision_tree_model.pkl")

# # Encoding maps
# mapping_reservation_status = {1: 'Check-Out', 0: 'Canceled', 2: 'No-Show'}
# mapping_deposit_type = {0: 'No Deposit', 1: 'Non Refund', 2: 'Refundable'}
# mapping_market_segment = {
#     6: 'Online TA', 5: 'Offline TA/TO', 4: 'Groups', 3: 'Direct',
#     2: 'Corporate', 1: 'Complementary', 0: 'Aviation', 7: 'Undefined'
# }
# mapping_customer_type = {
#     2: 'Transient', 3: 'Transient-Party', 0: 'Contract', 1: 'Group'
# }

# # Streamlit App Title
# st.title("🏨 Hotel Booking Cancellation Prediction")

# # Get user inputs
# reservation_status = st.selectbox("Reservation Status", list(mapping_reservation_status.values()))
# deposit_type = st.selectbox("Deposit Type", list(mapping_deposit_type.values()))
# market_segment = st.selectbox("Market Segment", list(mapping_market_segment.values()))
# total_of_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=5, value=0)
# required_car_parking_spaces = st.number_input("Car Parking Spaces Required", min_value=0, max_value=8, value=0)
# customer_type = st.selectbox("Customer Type", list(mapping_customer_type.values()))

# # Convert selected strings back to encoded numbers
# rev_status_key = [k for k, v in mapping_reservation_status.items() if v == reservation_status][0]
# dep_type_key = [k for k, v in mapping_deposit_type.items() if v == deposit_type][0]
# market_seg_key = [k for k, v in mapping_market_segment.items() if v == market_segment][0]
# cust_type_key = [k for k, v in mapping_customer_type.items() if v == customer_type][0]

# # Predict button
# if st.button("🔍 Predict Cancellation"):
#     # Prepare data for model
#     input_data = np.array([[rev_status_key, dep_type_key, market_seg_key,
#                             total_of_special_requests, required_car_parking_spaces, cust_type_key]])
    
#     prediction = model.predict(input_data)
    
#     if prediction[0] == 1:
#         st.error("⚠️ The booking is likely to be **CANCELED**.")
#     else:
#         st.success("✅ The booking is **NOT likely** to be canceled.")

# # ***********************************

# Installation command for streamlit
# %pip install streamlit

# 🏨 app.py — Hotel Booking Cancellation Prediction

import streamlit as st
import joblib

# Load your trained model (ensure this file exists)
model = joblib.load("random_forest_model.pkl")  # or decision_tree_model.pkl

# ---------------------------
# 🔢 Encoding maps
# ---------------------------
mapping_reservation_status = {1: 'Check-Out', 0: 'Canceled', 2: 'No-Show'}
mapping_deposit_type = {0: 'No Deposit', 1: 'Non Refund', 2: 'Refundable'}
mapping_market_segment = {
    6: 'Online TA',
    5: 'Offline TA/TO',
    4: 'Groups',
    3: 'Direct',
    2: 'Corporate',
    1: 'Complementary',
    0: 'Aviation',
    7: 'Undefined'
}

# ---------------------------
# 🎯 Streamlit App UI
# ---------------------------
st.title("🏨 Hotel Booking Cancellation Prediction App")
st.markdown("This app predicts whether a booking is likely to be **canceled** or **not canceled** based on key booking details.")

# ---------------------------
# 📋 User Inputs
# ---------------------------
reservation_status = st.selectbox("Reservation Status", list(mapping_reservation_status.values()))
deposit_type = st.selectbox("Deposit Type", list(mapping_deposit_type.values()))
market_segment = st.selectbox("Market Segment", list(mapping_market_segment.values()))
total_of_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=9, step=1)
required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=9, step=1)

# ---------------------------
# 🔁 Convert Selections Back to Numeric Keys
# ---------------------------
rev_status_key = [k for k, v in mapping_reservation_status.items() if v == reservation_status][0]
dep_type_key = [k for k, v in mapping_deposit_type.items() if v == deposit_type][0]
market_seg_key = [k for k, v in mapping_market_segment.items() if v == market_segment][0]

# ---------------------------
# 🔮 Prediction Section
# ---------------------------
if st.button("🔍 Predict Cancellation"):
    # Prepare data for model
    input_data = np.array([[rev_status_key, dep_type_key, market_seg_key,
                            total_of_special_requests, required_car_parking_spaces]])

    # Prediction
    prediction = model.predict(input_data)

    # Output
    if prediction[0] == 1:
        st.error("⚠️ The booking is **likely to be CANCELED.**")
    else:
        st.success("✅ The booking is **NOT likely** to be canceled.")

# ---------------------------
# 🧾 Note
# ---------------------------
st.caption("Model used: Random Forest Classifier (trained on hotel booking data)")