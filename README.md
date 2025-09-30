# CORD-19 Data Explorer - Streamlit Application

## Assignment Overview
This project explores a subset of the CORD-19 research dataset and demonstrates basic data analysis and visualization techniques. A simple Streamlit web application is included to interactively display insights from COVID-19 research papers.

The assignment focuses on fundamental data analysis skills suitable for beginners, including data loading, cleaning, visualization, and creating an interactive web app.

---

## Learning Objectives
By completing this project, you will:

- Practice loading and exploring a real-world dataset
- Learn basic data cleaning techniques
- Create meaningful visualizations
- Build a simple interactive web application
- Present data insights effectively

---

## Dataset Information
We work with the `metadata.csv` file from the CORD-19 dataset, which contains information about COVID-19 research papers, including:

- Paper titles and abstracts  
- Publication dates  
- Authors and journals  
- Source information  

You can download the dataset from Kaggle:  
[CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

> Note: The full dataset is very large. For this assignment, we use either the full metadata file or a smaller sample.

---

## Required Tools
- Python 3.7+  
- pandas (data manipulation)  
- matplotlib / seaborn (visualization)  
- Streamlit (web application)  
- Jupyter Notebook (optional, for exploration)  

Install required packages using:

```bash
pip install pandas matplotlib seaborn streamlit
````

---

## Project Steps

### Part 1: Data Loading and Exploration

* Load the `metadata.csv` file into a pandas DataFrame
* Inspect the first few rows and data types
* Check for missing or empty values
* Generate summary statistics for numeric columns

### Part 2: Data Cleaning and Preparation

* Handle missing values (fill or remove as appropriate)
* Convert date columns to datetime
* Extract the publication year for time-based analysis
* Create new features such as `abstract_word_count`

### Part 3: Data Analysis and Visualization

* Count papers by publication year
* Identify top journals publishing COVID-19 research
* Create a word cloud of most frequent words in titles
* Plot distribution of papers by source

### Part 4: Streamlit Application

* Build an interactive web app with Streamlit
* Include a basic layout with title and description
* Display visualizations and a sample of the dataset
* Example interactive element: year range slider

### Part 5: Documentation and Reflection

* Comment your code for clarity
* Write a brief summary of findings
* Reflect on challenges and what you learned

---

## Evaluation Criteria

Your project is evaluated based on:

* **Implementation (40%)**: Completion of all tasks
* **Code Quality (30%)**: Readable and well-commented code
* **Visualizations (20%)**: Clear, appropriate charts
* **Streamlit App (10%)**: Functional and interactive

---

## Example Code Snippets

```python
# Load the data
import pandas as pd
df = pd.read_csv('metadata.csv')

# Basic info
print(df.shape)
print(df.info())

# Check missing values
print(df.isnull().sum())

# Simple visualization
import matplotlib.pyplot as plt
df['year'] = pd.to_datetime(df['publish_time']).dt.year
year_counts = df['year'].value_counts().sort_index()
plt.bar(year_counts.index, year_counts.values)
plt.title('Publications by Year')
plt.show()
```

---

## Expected Outcomes

After completing this project, you will have:

* A Python script or Jupyter notebook with your analysis
* Visualizations showing patterns in COVID-19 research publications
* A functional Streamlit web app presenting your findings
* Practical experience with the data science workflow

---

## Repository Structure

```
Frameworks_Assignment/
│
├─ main.py                 # Streamlit application
├─ metadata_sample.csv     # Sample dataset
├─ README.md               # Project documentation
├─ requirements.txt        # Optional: list of Python dependencies
└─ notebooks/              # Optional: exploratory Jupyter notebooks
```

---

## How to Run the Streamlit App

```bash
streamlit run main.py
```

The app will open in your default browser, allowing you to explore the data interactively.

