
# 🏠 Boston Housing Price Prediction App

This Streamlit web application predicts the **median price of a house in Boston** using a **Linear Regression** model. Users can interactively adjust neighborhood and housing features, and the model will estimate the property value in real-time.

---

## 🔍 Overview

- **Model**: Linear Regression
- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn
- **Visualization**: Matplotlib
- **Data**: Boston Housing Dataset (`BostonHousing.csv`)

---

## 🚀 How It Works

1. **Data Loading**  
   The app loads the Boston Housing dataset containing information such as:
   - Crime rate
   - Residential land zoning
   - Number of rooms per dwelling
   - Distance to employment centers
   - Property tax rate  
   And more...

2. **User Input (Sidebar)**  
   Users set values for 13 housing and environmental variables using sliders and dropdowns. These simulate real-world property characteristics.

3. **Model Training**  
   A **Linear Regression** model is trained on the dataset. The model learns a mathematical relationship between the input variables and house prices (`medv`).

4. **Price Prediction**  
   Based on user input, the trained model estimates the predicted median house value in dollars.

5. **Visualization**  
   The app also displays a scatter plot comparing **actual vs. predicted** prices on a test dataset, helping users understand model performance.

---

## 📈 Why Linear Regression?

**Linear Regression** is a fundamental and interpretable machine learning algorithm used for regression tasks. It fits a straight line (or hyperplane) that best describes the relationship between input features and the target value.

### 📊 Key Benefits:
- **Simple & Fast** to train
- **Interpretability**: Coefficients show how features influence price
- **Baseline Model**: Ideal starting point before trying more complex models

It assumes that changes in features like crime rate or number of rooms cause linear changes in house price, which often holds true in real-world scenarios.

---

## 🛠️ Technologies Used

| Tool            | Purpose                             |
|----------------|-------------------------------------|
| `Python`        | Programming language                |
| `Streamlit`     | Frontend for user interface         |
| `pandas`        | Data manipulation                   |
| `scikit-learn`  | Machine learning modeling           |
| `matplotlib`    | Visualization of results            |

---

## 📁 Folder Structure

```

linear\_regression\_boston\_housing/
│
├── data/
│   └── BostonHousing.csv         # Dataset
│
├── app.py                        # Main Streamlit app
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies

````

---

## ▶️ Run the App Locally

### 1. Clone the repository:
```bash
git clone https://github.com/nanier216/linear-regression-boston-housing-price-prediction.git
cd linear-regression-boston-housing-price-prediction
````

### 2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📊 Sample Features Used

| Feature Name | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| `crim`       | Per capita crime rate by town                                       |
| `zn`         | Proportion of residential land zoned for lots over 25,000 sq.ft     |
| `indus`      | Proportion of non-retail business acres                             |
| `chas`       | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| `nox`        | Nitric oxides concentration (parts per 10 million)                  |
| `rm`         | Average number of rooms per dwelling                                |
| `age`        | Proportion of owner-occupied units built before 1940                |
| `dis`        | Weighted distances to employment centers                            |
| `rad`        | Index of accessibility to radial highways                           |
| `tax`        | Property tax rate per \$10,000                                      |
| `ptratio`    | Pupil-teacher ratio by town                                         |
| `b`          | 1000(Bk - 0.63)^2, where Bk is the proportion of Black population   |
| `lstat`      | % lower status of the population                                    |
| `medv`       | Median value of owner-occupied homes in \$1000s (target)            |

---

## 📬 Contact

Created by **Nanier**
[GitHub Profile](https://github.com/nanier216)

---

## ✅ Future Improvements

* Add cross-validation
* Incorporate more advanced models (e.g., Ridge, Lasso, XGBoost)
* Add feature importance chart
* Save/load trained models

---

> *"Price is what you pay. Value is what you get." – Warren Buffett*
