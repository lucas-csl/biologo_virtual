
# coding: utf-8

# # CRIANDO UM ROBÔ BOTANICO:
#  ## Desenvolvendo um algoritmo de ML Supervisionado para classificar plantas.

# ### Para começar, iremos importar as bibliotecas:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Carregando o CSV em um dataframe (df):

# In[2]:


df = pd.read_csv ("iris.csv")


# ### Vizualizando quais as colunas existentes em nosso df:

# In[3]:


df.columns


# ### Uma tecnica importante para Data Visualization é utilizar a função describe():

# In[4]:


df.describe()


# ### Plotando a dispersão dos dados utilizando o seaborn:

# In[5]:


sb.pairplot (df, hue = "species")


# ### Selecionando nossas caracteristicas ou "Features" para classificação em um array NUMPY:

# In[6]:


X = np.array (df.drop ('species', 1))
X


# ### Selecionando as classes ou "Targets" para classificação em um array NUMPY:

# In[9]:


y = np.array (df.species)
y


# # Classificando com KNN:
# 
#  O algoritmo KNN (K - vizinhos mais próximos em português) é um método utilizado para a classificação e regressão. Em ambos os casos, a entrada consiste em calcular a distância entre os k exemplos de treinamento mais próximos.
#  
#  Na classifcação com o K-NN, a saída é uma associação com uma classe. Um objeto é classifcado pela votação majoritária da classse dos vizinhos, sendo o objeto atribuído à classe mais comum dentro seus k vizinhos mais próximos.
#  
#  K é um inteiro positivo, tipicamente pequeno e impar.
#  Se k=1, então o objeto é simplesmente atribuído à classe desse único vizinho mais próximo.

# ### Importando KNN do scikit-learn:

# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# ### Criando nosso classificador:

# In[11]:


knn = KNeighborsClassifier (n_neighbors = 3)


# ### Treinando nosso classificador:

# In[12]:


knn.fit (X, y)


# # Predizendo flores novas com nosso modelo de ML:

# In[13]:


knn.predict ([[6.5, 6.5, 4.7, 1.3]])


# In[14]:


knn.predict ([[4.2, 3.1, 2.1, 0.9]])

