# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:55:51 2019

@author: Estudiante
"""

# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

datos = pd.read_csv(r"C:\Users\Estudiante\Desktop\europa_demografia.csv")
print("Cantidad de filas y columnas")
print(datos.shape)
print("primeras 3 filas de datos")
print(datos.head(3))
print("promedio del area pib y desempleo")
print(datos[["pib", "area", "desempleo"]].mean())
print("valor maximo de expectativa de vida, inflacion y crecimiento poblacional")
print(datos[["expectativa.vida", "inflacion", "Crecimiento.pob"]].max())
print("valor minimo de todas las variables")
print(datos.min())
print("Grafica corelacion entre inflacion, pib y área")
sns.pairplot(datos[["pib", "inflacion", "area"]])
print("Grafica de corelacion entre expectativa vida, pib y desempleo")
sns.pairplot(datos[["expectativa.vida", "pib", "desempleo"]])
plt.show()
#Analisis de Cluster 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

mod_Kmedias = KMeans(n_clusters = 3, random_state = 1)
variables = datos._get_numeric_data().dropna(axis = 1)
mod_Kmedias.fit(variables)
etiquetas = mod_Kmedias.labels_
acp = PCA(2)
columnas = acp.fit_transform(variables)
plt.scatter(x = columnas[:,0], y = columnas[:,1], c = etiquetas)
plt.show()
jc = [(datos['pais'][id], etiquetas[id]) for id in range(0, len(etiquetas))
        if etiquetas[id] == 2]
##-----------------------
etiqueta0=[(datos['pais'][id], etiquetas[id]) for id in range(0, len(etiquetas))
        if etiquetas[id] == 0]
print("Cantidad de paises en la etiqueta 0:",len(etiqueta0))
etiqueta1=[(datos['pais'][id], etiquetas[id]) for id in range(0, len(etiquetas))
        if etiquetas[id] == 1]
print("Cantidad de paises en la etiqueta 1:",len(etiqueta1))
etiqueta2=[(datos['pais'][id], etiquetas[id]) for id in range(0, len(etiquetas))
        if etiquetas[id] == 2]
print("Cantidad de paises en la etiqueta 2:",len(etiqueta2))
