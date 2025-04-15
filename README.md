
# 🚗 Used Cars Data Analysis & Price Prediction

This project performs exploratory data analysis (EDA) and linear regression modeling on a dataset of used cars. The goal is to uncover insights about car sales patterns and predict the selling price of cars based on key features.

## 📁 Dataset

The dataset (`used_cars.csv`) includes 1000 records of used car listings with features like:

- `manufacturer`: Car brand (e.g., Toyota, Ford)
- `model`: Car model name
- `year`: Year of manufacture
- `mileage`: Distance driven (in miles)
- `engine`: Engine specification
- `transmission`: Type of transmission (Manual/Automatic)
- `fuel_type`: Fuel category (Petrol, Diesel, etc.)
- `mpg`: Mileage per gallon
- `price`: Selling price

## 📊 Exploratory Data Analysis

The code performs various visual analyses using `matplotlib` and `seaborn`:

- **Brand popularity**: Count of vehicles listed by manufacturer
- **Average selling price**: Bar plot of average car prices by brand
- **Year-wise distribution**: Histogram of cars based on their year of manufacture
- **Mileage trends**: Line plot comparing mileage across brands based on transmission type
- **Engine vs. Mileage**: Scatter plot to examine the relationship between engine size and fuel efficiency

## 🤖 Machine Learning Model

A simple linear regression model is used to predict the selling price of a used car based on:

- `mileage`
- `year`
- `mpg`

### Model Steps:
1. Split data into training and testing sets.
2. Fit a `LinearRegression` model from `sklearn`.
3. Predict car prices using test data.
4. Calculate **Mean Squared Error (MSE)** for model evaluation.

## 🧪 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 How to Run

1. Update the CSV path if needed:
   ```python
   used_cars = pd.read_csv(r"C:\Users\Anshika\Downloads\used_cars.csv")
   ```

2. Run the script in a Python environment or Jupyter Notebook.

3. Visualizations will display automatically and model performance (MSE) will be printed.

## 📈 Output Example

```
Mean Squared Error: 1305224.77
```
