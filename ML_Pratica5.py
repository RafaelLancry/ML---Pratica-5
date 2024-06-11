import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from scipy.stats import uniform

# Carregar os dados
train_data = pd.read_csv('flight_delays_train.csv')
test_data = pd.read_csv('flight_delays_test.csv')

# Preencher valores ausentes
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)

# Nome da coluna de destino
target_col = 'dep_delayed_15min'

# Converter a coluna de destino em binária
train_data[target_col] = train_data[target_col].apply(lambda x: 1 if x == 'Y' else 0)

# Identificar colunas categóricas
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Aplicar One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(train_data[categorical_cols])
encoded_test = encoder.transform(test_data[categorical_cols])

# Converter para DataFrames
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

# Adicionar dados codificados aos DataFrames originais
train_data = train_data.drop(categorical_cols, axis=1).join(encoded_train_df)
test_data = test_data.drop(categorical_cols, axis=1).join(encoded_test_df)

# Selecionar recursos e alvo
X = train_data.drop(target_col, axis=1)
y = train_data[target_col]

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(test_data)

# Dividir em conjunto de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento e avaliação de modelos
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    print(f"\n{model_name}")
    print("Accuracy:", accuracy_score(y_val, predictions))
    print("ROC AUC Score:", roc_auc_score(y_val, predictions))
    print(classification_report(y_val, predictions))

# Hiperparâmetros para GridSearchCV e RandomizedSearchCV
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1)
svm_grid_search.fit(X_train, y_train)

best_svm_model = svm_grid_search.best_estimator_

print("\nBest SVM Model")
svm_best_predictions = best_svm_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, svm_best_predictions))
print("ROC AUC Score:", roc_auc_score(y_val, svm_best_predictions))
print(classification_report(y_val, svm_best_predictions))

# Salvar previsões do melhor modelo no conjunto de teste
test_predictions = best_svm_model.predict(X_test)
output = pd.DataFrame({'Id': test_data.index, 'Prediction': test_predictions})
output.to_csv('flight_delays_predictions.csv', index=False)

print("Predictions on Test Data Saved to 'flight_delays_predictions.csv'")
