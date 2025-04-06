## 📈 Stock Price Prediction using LSTM
This project predicts future stock prices using an LSTM (Long Short-Term Memory) deep learning model. It's built using Python, Keras, and Streamlit, and visualized through a Jupyter Notebook.

![LSTM Stock Prediction](https://img.shields.io/badge/LSTM-Stock--Prediction-blue)  


## 🧠 Model Summary
The model uses an LSTM neural network to learn from historical stock price data and make predictions. It's trained and saved as Stock_Price_LSTM_model.h5.

## 🔧 Tech Stack
Python 🐍

Jupyter Notebook 📓

TensorFlow / Keras 🤖

NumPy, Pandas, Matplotlib 📊

Streamlit 🌐

## 📂 Project Structure

├── app.py                        # Streamlit frontend to serve predictions  
├── requirements.txt             # Python dependencies  
├── Stock_Price_LSTM_Model.ipynb # Jupyter notebook (EDA + training)  
├── Stock_Price_LSTM_model.h5    # Saved trained LSTM model  




## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Nakkshh/StockPrediction.git  
cd StockPrediction
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv  
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## 📊 Jupyter Notebook

To explore the data preprocessing, LSTM training, and prediction logic:

```bash
jupyter notebook Stock_Price_LSTM_Model.ipynb
```

## ⚙️ Future Improvements

- Add frontend to input custom stock symbols  
- Integrate live stock market data API  
- Deploy the app using Streamlit Cloud  

## 📬 Contact

Made with ❤️ by [Nakkshh](https://github.com/Nakkshh)
