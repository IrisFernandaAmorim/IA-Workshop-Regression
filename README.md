# 🎬 Regresión Lineal con TMDB 5000 Movie Dataset

Ejercicio de regresión lineal supervisada usando el dataset público de películas de TMDB, con el objetivo de predecir la **recaudación** (`revenue`) de una película a partir de sus características.

> Este ejercicio fue guiado por el repositorio de Factoría F5:
> **https://github.com/Factoria-F5-madrid/IA-Workshop-Regression/tree/ejercicio-regresion**

---

## Descripción del ejercicio

El notebook recorre de forma progresiva todo el flujo de un proyecto de machine learning de regresión, desde la carga de datos hasta la comparación de tres modelos con distinto nivel de complejidad.

---

##  Pasos realizados

### 1. Descarga y carga de datos
Descarga automática del dataset con `kagglehub` y carga de los dos archivos CSV: `tmdb_5000_movies.csv` (4803 × 20) y `tmdb_5000_credits.csv` (4803 × 4).

### 2. Merge de datasets
Join entre los dos DataFrames usando `id` / `movie_id` como clave, con limpieza de columnas duplicadas generadas por el join.

### 3. Parseo de columnas JSON embebido
Las columnas `genres`, `cast` y `crew` contienen JSON como texto plano. Se implementaron funciones para parsearlas y extraer: lista de géneros, género principal, director y año de estreno.

### 4. Limpieza de datos
Filtrado de películas con valores implícitamente nulos (budget = 0, revenue = 0) y aplicación de criterios de calidad: budget > 100K, revenue > 100K, status = "Released", runtime > 0, director conocido. Dataset limpio: **3154 películas**.

### 5. Análisis exploratorio
- Histogramas de `revenue` y `budget` (distribuciones muy sesgadas a la derecha)
- Revenue mediano por género (top 10)
- Scatter de budget vs revenue

### 6. Transformación logarítmica
Aplicación de `np.log1p()` a `revenue`, `budget`, `popularity` y `vote_count` para normalizar las distribuciones y mejorar la linealidad. Análisis de correlación con heatmap.

### 7. Regresión lineal simple
Modelo con una sola feature (`log_budget` → `log_revenue`):
- R² = **0.406**
- RMSE = **1.304**
- Coeficiente β₁ = 0.806 (por cada 1% de aumento en budget, el revenue sube ~0.8%)

### 8. Diagnóstico de residuos
Tres gráficos para validar los supuestos del modelo: residuos vs predichos, histograma de residuos y real vs predicho.

### 9. Regresión múltiple
Incorporación de 5 features (`log_budget`, `log_popularity`, `log_vote_count`, `vote_average`, `runtime`) con estandarización via `StandardScaler`:
- R² = **0.624**
- RMSE = **1.038**

### 10. Ingeniería de características
Enriquecimiento del modelo con:
- **Dummies de género** para los 8 géneros más frecuentes
- **`director_avg_revenue`**: promedio histórico de recaudación logarítmica por director

Resultado con 14 features en total:
- R² = **0.770**
- RMSE = **0.812**

#### Comparativa de modelos

| Modelo | R² | RMSE |
|---|---|---|
| Simple (1 feature) | 0.406 | 1.304 |
| Múltiple (5 features) | 0.624 | 1.038 |
| Enriquecido (14 features) | 0.770 | 0.812 |

---

### 11. Preguntas de análisis y reflexión
Análisis de los resultados.

**1. ¿Qué variable numérica tiene mayor correlación con log_revenue y por qué tiene sentido?**
    La variable con mayor correlación es log_vote_count (0.72), seguida de cerca por log_popularity (0.67) y log_budget (0.64). Tiene sentido porque el número de votos refleja cuánta gente vio la película y se tomó el tiempo de valorarla — es casi un proxy directo de la audiencia. Una película muy taquillera naturalmente acumula más votos. log_budget también correlaciona fuerte, pero log_vote_count le gana ligeramente porque captura el resultado real (exposición al público) más que la inversión inicial.

**2. ¿Cuánto mejoró el R² del modelo simple al enriquecido?**
    El R² pasó de 0.406 (modelo simple) a 0.770 (modelo enriquecido), una mejora de +0.364 puntos. Eso es bastante significativo — casi duplicamos la varianza explicada. El salto más grande se dio al pasar del modelo simple al múltiple (+0.218), y añadir la ingeniería de características sumó otros +0.146 adicionales. Esto confirma que variables como director_avg_revenue y el género aportan información real que log_budget sola no capturaba.

**3. ¿Qué género tiene el coeficiente positivo más alto? ¿Coincide con lo que esperarías?**
    Mirando los coeficientes estandarizados del modelo enriquecido, el género con mayor coeficiente positivo es Animation, lo cual sí tiene bastante sentido: las películas animadas (especialmente las de grandes estudios como Disney, Pixar o DreamWorks) tienen un alcance global enorme porque no tienen barreras de idioma tan fuertes y atraen tanto a niños como adultos. Eso se traduce en recaudaciones desproporcionadamente altas respecto a su presupuesto.

---

## Stack

- Python 3
- pandas, numpy
- scikit-learn (LinearRegression, StandardScaler, train_test_split, cross_val_score)
- matplotlib, seaborn
- kagglehub

---

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — Kaggle

---
_Ejercicio realizado en el bootcamp de IA (P6) de Factoria F5_
