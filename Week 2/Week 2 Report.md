# **Week 2 Report : Predictive Model Building & Evaluation**

---

This document details the complete analysis performed during Week 2 of the project, corresponding to the "Model Building, Training, Evaluation & Serialization" milestone. 

All code and analysis described here can be found in the Model\_Building.ipynb Jupyter Notebook. 

## **1\. Objective** 

* Following the data cleaning and feature engineering in Week 1, the objective for Week 2 was to build, train, and evaluate a machine learning model to accurately forecast the key target variable of the project :   
  * price\_actual 

## **2\. Methodology & Code Analysis** 

* The notebook follows a structured approach to model development, ensuring that the data is handled correctly for a time-series problem and that the model is robustly evaluated. 

### **Step 2.1 : Data Loading & Preparation** 

* **Load Data :** The analysis begins by loading the processed\_dataset.csv file, which was the final output from Week 1\. This dataset includes all engineered temporal features, generation data, and external forecasts.   
* Define Features (X) and Target (y) :   
  * **Target (y) :** The target variable was defined as price\_actual.   
  * **Features (X) :** The feature set (X) was composed of all other relevant columns, including generation sources (generation\_biomass, generation\_natural\_gas, etc.), forecasts (solar\_forecast, wind\_onshore\_forecast, load\_forecast), and temporal features (hour, day\_of\_week, etc.). 

### **Step 2.2 : Feature Scaling** 

* A StandardScaler was employed to normalize the features.   
* The scaler was fit only on the training data (X\_train) and then used to transform both X\_train and X\_test. This critical step prevents "data leakage," where information from the test set could improperly influence the model's training.   
* The fitted scaler was saved as scaler.joblib for later use in production. 

### **Step 2.3 : Train-Test Split & Visualization** 

#### **A. Time-Series Split :** 

* The data was split into training (80%) and testing (20%) sets.   
* Crucially, shuffle=False was used during the split. This is essential for time-series data, as it ensures that the model is trained on past data (the first 80% of the dataset) and tested on the most recent, "future" data (the final 20%), simulating a real-world forecasting scenario. 

#### **B. Split Visualization :** 

* The graph below clearly shows this split, with the blue line representing the training data used to build the model and the orange line representing the unseen test data used for evaluation. 

![](https://github.com/Sanjana-Dhabale/ai-electricity-price-forecasting/blob/dea0557a14d6797d028803fb361fa16c6f59ac4f/Week%202/train_test_split.png)
Train/Test Split Visualization

### **Step 2.4 : Model Selection & Training** 

* **Algorithm :** An XGBRegressor (Extreme Gradient Boosting) was chosen as the modeling algorithm. This is a powerful, tree-based ensemble model renowned for its state-of-the-art performance on tabular and time-series data, and it is highly effective at capturing complex, non-linear relationships.   
* **Model Training :** A single XGBRegressor model was trained on the scaled training dataset (X\_train, y\_train). 

### **Step 2.5 : Model Performance Evaluation** 

* The model's performance was evaluated by comparing its predictions against the unseen test data. 

#### **A. Visual Evaluation :** 

* A primary qualitative evaluation was conducted by plotting the model's predictions against the actual values for a one-week period from the test set.   
* As shown in the graph below, the Predicted Price (orange dashed line) tracks the Actual Price (blue line) with extremely high fidelity. The model successfully captures the major price spikes, dips, and daily cyclical patterns. 

![](https://github.com/Sanjana-Dhabale/ai-electricity-price-forecasting/blob/dea0557a14d6797d028803fb361fa16c6f59ac4f/Week%202/price_forecast_1_week.png)
1-Week Price Forecast vs. Actual Values

#### **B. Conclusion :** 

* The close visual correspondence between the predicted and actual values, along with strong quantitative metrics (such as R-squared, MAE, and MSE, as calculated in the notebook), indicates that the XGBRegressor model is highly effective and has successfully learned the complex patterns driving electricity pricing. 

### **Step 2.6 : Model Serialization** 

* As the final step, the fully trained model and its associated scaler were saved (serialized) to disk using joblib :   
  * model\_price\_forecast.joblib   
  * scaler.joblib   
* These files, along with the saved scaler.joblib, allow the model to be easily reloaded for future predictions in a production environment without needing to be retrained. 
