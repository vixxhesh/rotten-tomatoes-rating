﻿# Movie Audience Rating Prediction

## Overview

This project involves building a machine learning model to predict audience ratings of movies based on various metadata, such as genre, runtime, and critics' reviews. Using the Random Forest Regressor, the model achieved an R² score of 0.51, demonstrating its ability to explain 51% of the variance in audience ratings. The project incorporates preprocessing pipelines, feature engineering, and model evaluation techniques.

---

## Features

- **Regression Model**: Built using Random Forest Regressor to predict audience ratings.
- **Data Preprocessing**: Included handling missing values, encoding categorical data, and scaling numerical features.
- **Evaluation**: Conducted cross-validation, residual analysis, and calculated performance metrics.
- **Technologies Used**: Python, scikit-learn, pandas, matplotlib, seaborn.

---

## Dataset Description

The dataset contains metadata and reviews for movies, including:

| Column Name          | Description                                    |
| -------------------- | ---------------------------------------------- |
| `movie_title`        | Title of the movie                             |
| `movie_info`         | Short description of the movie                 |
| `rating`             | Target variable, audience rating (0-100 scale) |
| `genre`              | Genre of the movie, encoded as integers        |
| `directors`          | List of directors                              |
| `cast`               | List of main cast members                      |
| `runtime_in_minutes` | Movie runtime in minutes (scaled)              |
| `tomatometer_status` | Rotten Tomatoes critic review status (encoded) |
| `tomatometer_rating` | Rotten Tomatoes rating (scaled)                |
| `tomatometer_count`  | Number of critic reviews (scaled)              |

---

## Data Analysis and Visualization

### Key Insights

1. **Distribution of Ratings**:

   - Audience ratings are concentrated in the mid-range with fewer extreme values.

2. **Runtime Impact**:

   - Longer movies tend to have slightly higher ratings.

3. **Tomatometer Influence**:
   - Higher critic ratings and counts are strongly correlated with better audience ratings.

### Visualizations

#### Rating Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(data['rating'], bins=20, kde=True)
plt.title('Distribution of Audience Ratings')
plt.xlabel('Audience Rating')
plt.ylabel('Frequency')
plt.show()
```

#### Correlation Heatmap

```python
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

#### Runtime vs. Rating

```python
sns.scatterplot(x=data['runtime_in_minutes'], y=data['rating'])
plt.title('Runtime vs. Audience Rating')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Audience Rating')
plt.show()
```

---

## Process

### 1. Data Preprocessing

- **Missing Values**: Handled missing or null values using mean/mode imputation.
- **Feature Scaling**: Scaled numerical features (e.g., `runtime_in_minutes`, `tomatometer_rating`, `tomatometer_count`).
- **Encoding**: Transformed categorical variables like `genre` and `tomatometer_status` using label encoding.

### 2. Feature Engineering

- Extracted and transformed relevant features for training.
- Encoded genres and review statuses to numerical formats.

### 3. Model Development

- Split the dataset into training and testing sets (80%-20%).
- Used Random Forest Regressor as the predictive model.
- Conducted hyperparameter tuning for optimal performance.

### 4. Model Evaluation

- **Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score.
- **Cross-Validation**: Performed k-fold cross-validation to validate model robustness.

---

## Results

- **Model Performance**:
  - MAE: 11.60
  - MSE: 214.40
  - R² Score: 0.51
- **Residual Analysis**: Errors are mostly normally distributed, with slight deviations at the extremes.

---

## How to Run the Project

### Prerequisites

- Python 3.8+
- Libraries: pandas, scikit-learn, matplotlib, seaborn

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/vixxhesh/rotten-tomatoes-rating.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script:
   ```bash
   python main.py
   ```
4. View results and visualizations in the output folder.

---

## Future Improvements

- Add more features like user demographics and social media mentions.
- Experiment with other machine learning models like Gradient Boosting or Neural Networks.
- Deploy the model using a web application framework such as Flask or Django.

---

## Acknowledgments

- Data Source: Hypothetical dataset inspired by rotten tomatoes movie metadata.
- Tools: scikit-learn, pandas, matplotlib, seaborn.
