# Disease-Prediction
Disease prediction systems are computer-based tools that use data and statistical models to predict the likelihood of an individual developing a particular disease or health condition.

# Tech Stack
* Front-End: HTML, CSS, JavaScript, Bootstrap
* Back-End: Flask, TinyDB, Jinja, Pickle
* IDE: Jupyter Notebook, Visual Studio Code

# How to run this app
* First create a virtual environment by using this command:
* conda create -n myenv python=3.9
* Activate the environment using the below command:
* conda activate myenv
* Then install all the packages by using the following command
* pip install -r requirements.txt
* Now for the final step. Run the app
* python app.py OR python -m flask run

# Data Collection: 
[Disease Prediction Dataset](https://www.kaggle.com/) from Kaggle

# Data Preprocessing: 
* Missing Values Handled by Random Sample Imputation to maintain the variance
* Categorical Values are handled by using One Hot Encoding
* Outliers are handled using BoxPlot
* Imbalanced Dataset was handled using SMOTE

# Model Creation:
* Different types of models were trained like Random Forest, Support Vector Classifier, Gaussian Naive Bayes Classifier & Bernoulli Naive Bayes Classifier
* An Ensemble model was created by combining all 4
* The conclusion were arrived at, based on classification metrics, accuracy score, f1-score