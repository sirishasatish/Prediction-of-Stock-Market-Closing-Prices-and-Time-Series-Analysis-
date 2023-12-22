# Databricks notebook source
# DBTITLE 1,CS6350 Big Data Management & Analysis 
# Title - Prediction of Stock Market Closing Prices and Time Series Analysis using Recurrent Neural Networks with Long-Short Term Memory 
# 1. Sirisha Satish (sxs210095)
# 2. Anish Joshi (axj200101)
# 3. Sandra Jayakumar (sxj210016)
# 4. Sherin Jayakumar (sxj210018)

# COMMAND ----------

pip install pandas

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# COMMAND ----------

github_csv_url1 = "https://raw.githubusercontent.com/sirishasatish/BDA-dataset/main/AAPL.csv"
databricks_target_path1 = "/mnt/my-data/AAPL.csv"
dbutils.fs.cp(github_csv_url1, databricks_target_path1)

github_csv_url2 = "https://raw.githubusercontent.com/sirishasatish/BDA-dataset/main/ABM.csv"
databricks_target_path2 = "/mnt/my-data/ABM.csv"
dbutils.fs.cp(github_csv_url2, databricks_target_path2)

github_csv_url3 = "https://raw.githubusercontent.com/sirishasatish/BDA-dataset/main/ADBE.csv"
databricks_target_path3 = "/mnt/my-data/ADBE.csv"
dbutils.fs.cp(github_csv_url3, databricks_target_path3)

# COMMAND ----------

df_apple = spark.read.csv(databricks_target_path1, header=True, inferSchema=True)
df_abm = spark.read.csv(databricks_target_path2, header=True, inferSchema=True)
df_adobe = spark.read.csv(databricks_target_path3, header=True, inferSchema=True)

# COMMAND ----------

print((df_apple.count(), len(df_apple.columns)))
print((df_abm.count(), len(df_abm.columns)))
print((df_adobe.count(), len(df_adobe.columns)))

# COMMAND ----------

df_apple.show()

# COMMAND ----------

df_abm.show()

# COMMAND ----------

df_adobe.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

# COMMAND ----------

def check_null_values(*dfs):
    """
    Check for null values in one or more DataFrames.
    
    Parameters:
    *dfs: Variable number of PySpark DataFrames
    
    Returns:
    None (prints information about null values)
    """
    for df in dfs:
        
        nullCounts = {col: df.where(df[col].isNull()).count() for col in df.columns}
        
        for col, cnt in nullCounts.items():
            if cnt > 0:
                print(f"Column '{col}' has {cnt} null values.")
        
        if any(nullCounts.values()):
            print("Null values Detected.\n")
        else:
            print("Null values not detected.\n")


# COMMAND ----------

check_null_values(df_apple, df_abm, df_adobe)

# COMMAND ----------

df_apple.describe().select("summary", "Low", "Open", "Volume", "High", "Close", "Adjusted Close").show()


# COMMAND ----------

df_abm.describe().select("summary", "Low", "Open", "Volume", "High", "Close", "Adjusted Close").show()

# COMMAND ----------


df_adobe.describe().select("summary", "Low", "Open", "Volume", "High", "Close", "Adjusted Close").show()

# COMMAND ----------

def plot_close_prices(df, title="Title of the Plot", limit=None):
    """
    Plot the 'Open' column over time for a given PySpark DataFrame.

    Parameters:
    - df: PySpark DataFrame containing 'Date' and 'Close' columns.
    - title: Title for the plot (default is "Title of the Plot").
    - limit: Number of rows to limit for plotting (default is None, i.e., no limit).

    Returns:
    None (displays the plot).
    """
    sample_data = df.select("Date", "Close").toPandas()
    plt.plot(sample_data['Date'], sample_data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title(title)
    plt.show()

# COMMAND ----------

plot_close_prices(df_apple, title="Apple Stock Prices")
plot_close_prices(df_abm, title="ABM Stock Prices")
plot_close_prices(df_adobe, title="Adobe Stock Prices")

# COMMAND ----------

from pyspark.sql import functions as F
def plot_open_prices_quarterly_subplots(df, title="Title of the Plot"):
    """
    Plot the quarterly trends of 'Open' prices for a given PySpark DataFrame.

    Parameters:
    - df: PySpark DataFrame containing 'Date' and 'Open' columns.
    - title: Title for the plot (default is "Title of the Plot").

    Returns:
    None (displays the subplots).
    """
    df = df.withColumn("Quarter", F.quarter("Date"))
    df = df.withColumn("Year", F.year("Date"))
    quarterly_data = df.groupBy("Year", "Quarter").agg(F.avg("Close").alias("AvgClose"))
    quarterly_data = quarterly_data.orderBy("Year", "Quarter")
    sample_data = quarterly_data.toPandas()
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharey=True)
    for quarter in range(1, 5):  
        data_quarter = sample_data[sample_data['Quarter'] == quarter]
        axs[quarter - 1].plot(data_quarter['Year'], data_quarter['AvgClose'])
        axs[quarter - 1].set_title(f'Quarter {quarter}')
        axs[quarter - 1].set_xlabel('Year')
    plt.suptitle(title)
    plt.show()


# COMMAND ----------

plot_open_prices_quarterly_subplots(df_apple, title="Apple Stock Prices Quarterly Trends")
plot_open_prices_quarterly_subplots(df_abm, title="ABM Stock Prices Quarterly Trends")
plot_open_prices_quarterly_subplots(df_adobe, title="Adobe Stock Prices Quarterly Trends")

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
def generate_pairplot_for_datasets(*datasets, target_column='Close', numerical_columns=None, sample_size=1000):
    """
    Generate pairplots for the specified target column and numerical columns in multiple PySpark DataFrames.

    Parameters:
    - datasets: Variable number of PySpark DataFrames.
    - target_column: Name of the target column (default is 'Close').
    - numerical_columns: List of numerical columns to include in the pairplot.
                         If None, all numerical columns (excluding the target) will be considered.
    - sample_size: Number of samples to use for pairplot (default is 1000).

    Returns:
    None (displays the pairplots).
    """
    if numerical_columns is None:
        numerical_columns = [col for col, dtype in datasets[0].dtypes if dtype in ('double', 'float') and col != target_column]
    for i, df in enumerate(datasets, 1):
        sampled_data = df.select([target_column] + numerical_columns).sample(False, sample_size / df.count())
        pandas_df = sampled_data.toPandas()
        sns.pairplot(pandas_df, markers='.', diag_kind='kde', height=3)
        plt.suptitle(f'Pairplot for {target_column} and Other Variables - Dataset {i}')
        plt.show()
generate_pairplot_for_datasets(df_apple, df_abm, df_adobe, target_column='Close', numerical_columns=['Open', 'High', 'Low', 'Volume', 'Adjusted Close'])


# COMMAND ----------

#Correlation Matrix for the apple dataset
selected_features = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
panda_df = df_apple.toPandas()
selected_df=panda_df[selected_features]
matrixOfCorr= selected_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(matrixOfCorr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matrix depicting correlation between the features and target variable for the Apple Dataset')
plt.show()

# COMMAND ----------

#Plotting the difference between Open and Close price
sampled_data_apple = df_apple.select("Date", "Open", "Close").sample(False, 0.1)
sampled_data_abm = df_abm.select("Date", "Open", "Close").sample(False, 0.1)
sampled_data_adobe = df_adobe.select("Date", "Open", "Close").sample(False, 0.1)
pandas_df_apple = sampled_data_apple.toPandas()
pandas_df_abm = sampled_data_abm.toPandas()
pandas_df_adobe = sampled_data_adobe.toPandas()
pandas_df_apple['Open_Close_Difference'] = pandas_df_apple['Close'] - pandas_df_apple['Open']
pandas_df_abm['Open_Close_Difference'] = pandas_df_abm['Close'] - pandas_df_abm['Open']
pandas_df_adobe['Open_Close_Difference'] = pandas_df_adobe['Close'] - pandas_df_adobe['Open']
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
sns.lineplot(x='Date', y='Open_Close_Difference', data=pandas_df_apple)
plt.title('Apple')
plt.subplot(1, 3, 2)
sns.lineplot(x='Date', y='Open_Close_Difference', data=pandas_df_abm)
plt.title('ABM')
plt.subplot(1, 3, 3)
sns.lineplot(x='Date', y='Open_Close_Difference', data=pandas_df_adobe)
plt.title('Adobe')
plt.tight_layout()
plt.show()

# COMMAND ----------

from pyspark.sql.functions import year, avg
def plot_price_difference(df, stock_name):
    df = df.withColumn("Year", year(df["Date"]))
    avg_price_diff = df.groupBy("Year").agg(avg(df["Close"] - df["Open"]).alias("AvgPriceDifference"))
    pandas_df = avg_price_diff.toPandas()
    plt.bar(pandas_df["Year"], pandas_df["AvgPriceDifference"])
    plt.xlabel('Year')
    plt.ylabel('Average Price Difference (Close - Open)')
    plt.title(f'Average Price Difference Over the Years - {stock_name}')
    plt.show()

# COMMAND ----------

plot_price_difference(df_apple, "Apple")
plot_price_difference(df_abm, "ABM")
plot_price_difference(df_adobe, "Adobe")

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
pandas_df_apple = df_apple.toPandas()
pandas_df_abm = df_abm.toPandas()
pandas_df_adobe = df_adobe.toPandas()
columns_to_scale = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
scaler = MinMaxScaler()
for col in columns_to_scale:
    pandas_df_apple[col + '_scaled'] = scaler.fit_transform(pandas_df_apple[[col]])
    pandas_df_abm[col + '_scaled'] = scaler.fit_transform(pandas_df_abm[[col]])
    pandas_df_adobe[col + '_scaled'] = scaler.fit_transform(pandas_df_adobe[[col]])
X_apple = pandas_df_apple[[col + '_scaled' for col in columns_to_scale if col != 'Close']].to_numpy()
X_abm = pandas_df_abm[[col + '_scaled' for col in columns_to_scale if col != 'Close']].to_numpy()
X_adobe = pandas_df_adobe[[col + '_scaled' for col in columns_to_scale if col != 'Close']].to_numpy()
Y_apple = pandas_df_apple['Close_scaled'].to_numpy()
Y_abm = pandas_df_abm['Close_scaled'].to_numpy()
Y_adobe = pandas_df_adobe['Close_scaled'].to_numpy()
features = ['Low_scaled', 'Open_scaled', 'volume_scaled', 'High_scaled', 'Adjusted Close_scaled']

# COMMAND ----------

#Plotting correlation matrix for the ABM dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
selected_features = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
selected_df = pandas_df_abm[selected_features]
matrixOfCorr = selected_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(matrixOfCorr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matrix depicting correlation between the features and target variable for the ABM Dataset')
plt.show()

# COMMAND ----------

#Plotting the correlation matrix for Adobe Dataset
selected_features = ['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close']
selected_df = pandas_df_adobe[selected_features]
matrixOfCorr = selected_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(matrixOfCorr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matrix depicting correlation between the features and target variable for the Adobe Dataset')
plt.show()

# COMMAND ----------


xAppleTrain, xAppleTest, yAppleTrain, yAppleTest = train_test_split(X_apple, Y_apple, test_size=0.2, random_state=42)
xAbmTrain, xAbmTest, yAbmTrain, yAbmTest = train_test_split(X_abm, Y_abm, test_size=0.2, random_state=42)
xAdobeTrain, xAdobeTest, yAdobeTrain, yAdobeTest = train_test_split(X_adobe, Y_adobe, test_size=0.2, random_state=42)

print("Apple Train Dataset Shape:", xAppleTrain.shape, yAppleTrain.shape)
print("ABM Train Dataset Shape:", xAbmTrain.shape, yAbmTrain.shape)
print("Adobe Train Dataset Shape:", xAdobeTrain.shape, yAdobeTrain.shape)

print("Apple Test Dataset Shape:", xAppleTest.shape, yAppleTest.shape)
print("ABM Test Dataset Shape:", xAbmTest.shape, yAbmTest.shape)
print("Adobe Test Dataset Shape:", xAdobeTest.shape, yAdobeTest.shape)

# COMMAND ----------

# We will be using the Xavier initialization for weights
import numpy as np
class LSTM:
    def __init__(self, inpDim, numOfNeurons, outDim, lr):
        self.inpDim = inpDim
        self.numOfNeurons = numOfNeurons
        self.outDim = outDim
        self.lr = lr

        # Xavier/Glorot initialization for weights
        self.W_forget = np.random.randn(numOfNeurons, inpDim + numOfNeurons) / np.sqrt(inpDim + numOfNeurons)
        self.b_forget = np.zeros(numOfNeurons)

        self.W_input = np.random.randn(numOfNeurons, inpDim + numOfNeurons) / np.sqrt(inpDim + numOfNeurons)
        self.b_input = np.zeros(numOfNeurons)

        self.W_output = np.random.randn(numOfNeurons, numOfNeurons) / np.sqrt(numOfNeurons)
        self.b_output = np.zeros(outDim)

        # Initialize cell state and hidden state
        self.c_state = np.zeros(numOfNeurons)
        self.h_state = np.zeros(numOfNeurons)

        # Intermediate variables for backpropagation
        self.f_gate = None
        self.i_gate = None
        self.c_tilde = None
        self.o_gate = None

        # Gradients
        self.d_W_forget = None
        self.d_W_input = None
        self.d_W_output = None

    def sigmoidActivationFunction(self, x):
        return 1 / (1 + np.exp(-x))

    def tanhActivationFunction(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def forward(self, x):
        # Forget gate
        self.f_gate = self.sigmoidActivationFunction(np.dot(self.W_forget, np.concatenate([x, self.h_state])) + self.b_forget)

        # Input gate
        self.i_gate = self.sigmoidActivationFunction(np.dot(self.W_input, np.concatenate([x, self.h_state])) + self.b_input)

        # Cell state update
        self.c_tilde = self.tanhActivationFunction(np.dot(self.W_input, np.concatenate([x, self.h_state])) + self.b_input)
        self.c_state = self.f_gate * self.c_state + self.i_gate * self.c_tilde

        # Output gate
        self.o_gate = self.sigmoidActivationFunction(np.dot(self.W_output, self.c_state) + self.b_output)
        self.h_state = self.o_gate * self.tanhActivationFunction(self.c_state)

        return self.h_state

    def backward(self, x, y):
        """
        Performs backward propagation through the LSTM network.

        Args:
            x: The input data.
            y: The target output.

        Returns:
            gradients 
        """
        # Calculate binary cross-entropy loss gradient
        loss_grad = self.h_state - y.flatten()

        # Output gate gradient
        dh_state = loss_grad * self.o_gate * (1 - np.square(np.tanh(self.c_state)))

        # Cell state gradient
        dc_state = dh_state * self.o_gate * (1 - np.square(np.tanh(self.c_state))) + loss_grad * self.f_gate * np.tanh(self.c_state)

        # Input, Forget, and Cell State Update Gradients
        di_gate = dc_state * np.tanh(self.c_tilde)
        df_gate = dc_state * self.c_state
        dc_tilde = dc_state * self.i_gate

        # Backpropagation through Gates
        df_gate = df_gate * self.f_gate * (1 - self.f_gate)
        di_gate = di_gate * self.i_gate * (1 - self.i_gate)
        dc_tilde = dc_tilde * (1 - np.square(np.tanh(self.c_tilde)))

        self.d_W_forget = np.outer(df_gate, np.concatenate([x, self.h_state]))
        self.d_W_input = np.outer(di_gate, np.concatenate([x, self.h_state]))
        self.d_W_output = np.outer(dh_state, self.c_state).reshape((self.c_state.shape[0], self.numOfNeurons))

        return self.d_W_forget, self.d_W_input, self.d_W_output

    def updateWgts(self):
        self.W_forget -= self.lr * self.d_W_forget
        self.W_input -= self.lr * self.d_W_input
        self.W_output -= self.lr * self.d_W_output.T

    def meanSqError(yTrue, yPred):
        return np.mean(np.square(yTrue - yPred))
    
    def fit_lstm(model, X_train, y_train, num_epochs, lr):
        """

        Args:
            model
            X_train: array
            y_train: array
            num_epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            List of mean squared errors for each epoch.
        """
        mse_history = []

        for ep in range(num_epochs):
            epochLoss = 0.0

            for i in range(len(X_train)):
                x = X_train[i]
                y = y_train[i]

                # Forward pass
                output = model.forward(x)

                # Backward pass
                model.backward(x, y)

                # Update weights
                model.updateWgts()

                # Calculate mean squared error 
                epochLoss += LSTM.meanSqError(y, output)

            #  Calculating RMSE
            mse = epochLoss / len(X_train)
            rmse=np.sqrt(mse)
            mse_history.append(rmse)

            
            print(f"Epoch {ep + 1}/{num_epochs}, Training RMSE: {rmse}")

        return mse_history
    
    def predict(self, X):
        """
        Predicts the output for input data X.

        Args:
            X: array
        Returns:
            Predicted output.
        """
        predictions = []

        for x in X:
            output = self.forward(x)
            predictions.append(output)

        return np.array(predictions)

# COMMAND ----------

def plot_epochs_rmse(mse_vals):
   epochsList = list(range(1, len(mse_vals) + 1))
   plt.figure(figsize=(16,8))
   plt.plot(epochsList, mse_vals, marker='o', linestyle='-')
   plt.title('RMSE Values vs. Number of epochs')
   plt.xlabel('Epochs')
   plt.ylabel('RMSE')
   plt.grid(True)
   plt.xticks(range(2, len(epochsList) + 2, 2))
   plt.show()

# COMMAND ----------

def calculate_rmse(true_values, predictions):
    """
    Calculate Root Mean Squared Error (RMSE) for each output separately.

    Args:
        true_values: Actual values.
        predictions: Predicted values.

    Returns:
        List of RMSE values
    """
    rmseVals = []
    for i in range(predictions.shape[1]):
        mse = np.mean((true_values - predictions[:, i])**2)
        rmse = np.sqrt(mse)
        rmseVals.append(rmse)
    return rmseVals

def calculate_r2(true_values, predictions):
    """
    Calculate R-squared (R2) score for each output separately.

    Args:
        true_values: Actual values.
        predictions: Predicted values.

    Returns:
        List of R-squared scores.
    """
    r2Vals = []
    for i in range(predictions.shape[1]):
        mean_true = np.mean(true_values)
        totalSumofSq = np.sum((true_values - mean_true)**2)
        residualSumofSq = np.sum((true_values - predictions[:, i])**2)
        r2 = 1 - (residualSumofSq / totalSumofSq)
        r2Vals.append(r2)
    return r2Vals

# COMMAND ----------

#For apple dataset
inpDim = len(features)
outDim = 1 
numOfNeurons = 10
no_epochs=50
lstm_model_apple = LSTM(inpDim=inpDim, numOfNeurons=numOfNeurons, outDim=outDim, lr=0.03)
mse_history_apple = LSTM.fit_lstm(lstm_model_apple, xAppleTrain, yAppleTrain, num_epochs=no_epochs, lr=0.03)

# COMMAND ----------

#testing on Apple dataset
predictions_apple = lstm_model_apple.predict(xAppleTest)

# COMMAND ----------

#Calculating RMSE and R2 score for Apple
rmse_apple = calculate_rmse(yAppleTest, predictions_apple)
print(f"RMSE value: {rmse_apple}")
r2_apple = calculate_r2(yAppleTest, predictions_apple)
print(f"R-2 Score: {r2_apple}")
avg_rmse_apple = np.mean(rmse_apple)
avg_r2_apple = np.mean(r2_apple)
print("--"*100)
print(f"Root Mean Squared Error (RMSE) for Test set: {avg_rmse_apple}")
print(f" R-2 Score for Test set: {avg_r2_apple}")

# COMMAND ----------

#For ABM dataset
inpDim = len(features)  
outDim = 1  
numOfNeurons = 10
no_epochs=50
lstm_model_abm = LSTM(inpDim=inpDim, numOfNeurons=numOfNeurons, outDim=outDim, lr=0.05)
mse_history_abm = LSTM.fit_lstm(lstm_model_abm, xAbmTrain, yAbmTrain, num_epochs=no_epochs, lr=0.05)

# COMMAND ----------

predictions_abm = lstm_model_abm.predict(xAbmTest)
type(predictions_abm)

# COMMAND ----------

#Calculating RMSE and R2 score for ABM dataset
rmse_abm = calculate_rmse(yAbmTest, predictions_abm)
print(f"RMSE value): {rmse_abm}")
r2_abm = calculate_r2(yAbmTest, predictions_abm)
print(f"R-2 Score: {r2_abm}")
avg_rmse_abm = np.mean(rmse_abm)
avg_r2_abm = np.mean(r2_abm)
print("--"*100)
print(f"Root Mean Squared Error (RMSE) for Test set: {avg_rmse_abm}")
print(f"R-2 Score for Test set: {avg_r2_abm}")

# COMMAND ----------

# For Adobe Dataset
inpDim = len(features)  
outDim = 1  
your_num_neurons = 10
no_epochs=50
lstm_model_adobe = LSTM(inpDim=inpDim, numOfNeurons=numOfNeurons, outDim=outDim, lr=0.05)
mse_history_adobe = LSTM.fit_lstm(lstm_model_adobe, xAdobeTrain, yAdobeTrain, num_epochs=no_epochs, lr=0.05)

# COMMAND ----------

predictions_adobe = lstm_model_adobe.predict(xAdobeTest)
type(predictions_adobe)

# COMMAND ----------

rmse_adobe = calculate_rmse(yAdobeTest, predictions_adobe)
print(f"RMSE value: {rmse_adobe}")
r2_adobe = calculate_r2(yAdobeTest, predictions_adobe)
print(f"R-2 Score: {r2_adobe}")
avg_rmse_adobe = np.mean(rmse_adobe)
avg_r2_adobe = np.mean(r2_adobe)
print("--"*100)
print(f"Root Mean Squared Error (RMSE) for Test set: {avg_rmse_adobe}")
print(f" R-2 Score for Test set: {avg_r2_adobe}")

# COMMAND ----------

print("Plotting RMSE for each epoch -- Apple Dataset")
plot_epochs_rmse(mse_history_apple)
print("--"*100)
print("Plotting RMSE for each epoch -- ABM Dataset")
plot_epochs_rmse(mse_history_abm)
print("--"*100)
print("Plotting RMSE for each epoch -- Adobe Dataset")
plot_epochs_rmse(mse_history_adobe)

# COMMAND ----------


