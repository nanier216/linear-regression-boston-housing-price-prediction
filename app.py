import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv("data/BostonHousing.csv")
    X = df.drop("medv", axis=1)
    y = df["medv"]
    return X, y

X, y = load_data()

st.title("🏠 Boston Housing Price Prediction")
st.write("Use the sliders to set features and predict the house price.")

# Sidebar inputs
st.sidebar.header("Set Features")

def user_input_features():
    crim = st.sidebar.slider('Crime Rate (per capita)', float(X.crim.min()), float(X.crim.max()), float(X.crim.mean()))
    zn = st.sidebar.slider('Residential Land Zoned (%)', float(X.zn.min()), float(X.zn.max()), float(X.zn.mean()))
    indus = st.sidebar.slider('Non-retail Business Acres (%)', float(X.indus.min()), float(X.indus.max()), float(X.indus.mean()))
    chas = st.sidebar.selectbox('Charles River Boundary (0 = No, 1 = Yes)', [0, 1])
    nox = st.sidebar.slider('Nitric Oxides Concentration (ppm)', float(X.nox.min()), float(X.nox.max()), float(X.nox.mean()))
    rm = st.sidebar.slider('Avg. Rooms per Dwelling', float(X.rm.min()), float(X.rm.max()), float(X.rm.mean()))
    age = st.sidebar.slider('Proportion of Older Homes (%)', float(X.age.min()), float(X.age.max()), float(X.age.mean()))
    dis = st.sidebar.slider('Distance to Employment Centers', float(X.dis.min()), float(X.dis.max()), float(X.dis.mean()))
    rad = st.sidebar.slider('Accessibility to Highways (Index)', int(X.rad.min()), int(X.rad.max()), int(X.rad.median()))
    tax = st.sidebar.slider('Property Tax Rate ($ per $10,000)', float(X.tax.min()), float(X.tax.max()), float(X.tax.mean()))
    ptratio = st.sidebar.slider('Pupil-Teacher Ratio', float(X.ptratio.min()), float(X.ptratio.max()), float(X.ptratio.mean()))
    b = st.sidebar.slider('Black Population Measure', float(X.b.min()), float(X.b.max()), float(X.b.mean()))
    lstat = st.sidebar.slider('Lower Status Population (%)', float(X.lstat.min()), float(X.lstat.max()), float(X.lstat.mean()))
    
    data = {
        'crim': crim,
        'zn': zn,
        'indus': indus,
        'chas': chas,
        'nox': nox,
        'rm': rm,
        'age': age,
        'dis': dis,
        'rad': rad,
        'tax': tax,
        'ptratio': ptratio,
        'b': b,
        'lstat': lstat
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Cache model training
@st.cache_resource
def train_model():
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# Predict house price
prediction = model.predict(input_df)

st.markdown(
    "<h2 style='color:#1E90FF;'>Predicted House Price</h2>",
    unsafe_allow_html=True
)

predicted_price = prediction[0] * 1000
st.markdown(
    f"""<p style='font-size:48px; font-weight:bold;'>
        <span style='font-size:72px; vertical-align:middle; color:#228B22;'>$</span>
        {predicted_price:,.2f}
    </p>""",
    unsafe_allow_html=True
)




# Show raw data toggle
if st.checkbox('Show raw data'):
    st.subheader('Boston Housing Dataset Sample')
    st.write(X.head())

# Plot actual vs predicted prices on test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
y_pred_test = model.predict(X_test)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(y_test, y_pred_test, alpha=0.7)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Price ($1000s)')
ax.set_ylabel('Predicted Price ($1000s)')
ax.set_title('Actual vs Predicted Prices on Test Set')
st.pyplot(fig)
