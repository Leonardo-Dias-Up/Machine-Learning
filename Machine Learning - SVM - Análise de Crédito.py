from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import OneHotEncoder
# =============================================================================
# Base de dados - Crédito
# =============================================================================
df = pd.read_csv(r"D:\01. Python\Diversos\Machine Learning\machine-learning-master\dados\exemplo3.csv")
df.head()

df.describe()

# Correlação 
corr_df = df.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.title('Backcasting')
plt.show()

# Mapa de Calor a Procura de Valor Nulos
sns.heatmap(df.isnull(), cmap='plasma')
plt.show()

# Plot
plt.figure(figsize=(15, 8))
plt.scatter(df[df.risco == 'ruim'].idade, df[df.risco == 'ruim'].conta_corrente)
plt.scatter(df[df.risco == 'bom'].idade, df[df.risco == 'bom'].conta_corrente)
plt.xlabel('idade')
plt.ylabel('conta corrente')
plt.legend(['ruim', 'bom'])

# Separando inputs e outputs
X = df.drop('risco', axis=1)
y = df.risco

# =============================================================================
# OneHotEnconder - Binarização das Variáveis Categóricas
# =============================================================================
X_cat = X.select_dtypes(include='object')
onehot = OneHotEncoder(sparse=False, drop="first")
X_bin = onehot.fit_transform(X_cat)
X_bin

# =============================================================================
# Pré processando inputs - Normallizador - Variaveis Númericas
# =============================================================================
X_num = X.select_dtypes(exclude='object')
normalizador = MinMaxScaler()
X_norm = normalizador.fit_transform(X_num)
X_norm
    
X_all = np.append(X_num, X_bin, axis=1)
X_all

# Modelo normalizado por MinMax
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=1/3, random_state=42)

# =============================================================================
# Treinamento do classificador SVM - All Models
# =============================================================================

# Treinamento do classificador SVM - Linear
svcLinear = SVC(kernel='linear')
svcLinear.fit(X_train, y_train)

# Treinamento do classificador SVM - Polinomial
svcPolinomial = SVC(kernel='poly', degree=5)
svcPolinomial.fit(X_train, y_train)

# Treinamento do classificador SVM - Rbf
svcRbf = SVC(kernel='rbf')
svcRbf.fit(X_train, y_train)

# Treinamento do classificador SVM - Sigmoid
svcSigmoid = SVC(kernel='sigmoid')
svcSigmoid.fit(X_train, y_train)
    

# =============================================================================
# Previsões - Melhor Modelo
# =============================================================================
def accuracy_scoreSVM(): 
    # Cálculo da precisão - Linear
    a = accuracy_score(y_test, svcLinear.predict(X_test))
    # Cálculo da precisão - Polinomial
    b = accuracy_score(y_test, svcPolinomial.predict(X_test))
    # Cálculo da precisão - Rbf
    c = accuracy_score(y_test, svcRbf.predict(X_test))
    # Cálculo da precisão - Sigmoid
    d = accuracy_score(y_test, svcSigmoid.predict(X_test))
    
    # Max Value
    numbers = pd.DataFrame({'model':['Linear', 'Polinomial', 'Rbf', 'Sigmoid'],
                            'accuracy': [a,b,c,d]}).set_index('model')
    max_value = max(numbers['accuracy'])
    
    def return_column_value(df,column_value,value):
        df_column = df.iloc[:,column_value]
        index = np.where(df_column == value)
        value = index[0]
        result = df.iloc[value]
        return result
    z = return_column_value(numbers, 0, max_value)        
    return print('Melhor',z,'\n\n', numbers)
    
accuracy_scoreSVM()


# =============================================================================
# Predict
# =============================================================================
# Base
df_new = pd.DataFrame({'idade': [20, 25, 50, 35, 75], 
                      'conta_corrente': [800, 4000, 2200, 3200, 1000], 
                      'sexo': ['masculino', 'feminino', 'masculino', 'feminino', 'feminino']})
df_new.head()

# Transformação
X_new_bin = onehot.transform(df_new.select_dtypes(include=['object']))
X_new_num = normalizador.transform(df_new.select_dtypes(exclude=['object']))
X_new = np.append(X_new_num, X_new_bin, axis=1)

# Comparando aos Modelos
svcLinear.predict(X_new)
svcPolinomial.predict(X_new)
svcRbf.predict(X_new)
svcSigmoid.predict(X_new)

# Retornando aos valores Original
df_previsao = df_new.copy()
df_previsao['previsao'] = svcLinear.predict(X_new)
df_previsao











