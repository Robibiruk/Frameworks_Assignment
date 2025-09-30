import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="CORD-19 Data Explorer", layout="wide")
st.title("CORD-19 Data Explorer and Analysis")

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
st.header("Load and Preview Data")
df = pd.read_csv("metadata.csv", low_memory=False)
st.write(f"Dataset shape: {df.shape}")
st.dataframe(df.head(10))

st.subheader("Code Used")
st.code("""import pandas as pd
df = pd.read_csv("metadata.csv")
df.head(10)""", language='python')

# -----------------------------
# 2️⃣ Data Inspection
# -----------------------------
st.header("Data Inspection")
st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Empty Strings per Column")
st.write((df == "").sum())

st.subheader("Numerical Summary")
st.write(df.describe())

# -----------------------------
# 3️⃣ Feature Engineering & Cleaning
# -----------------------------
st.header("Feature Engineering & Cleaning")

# Fill missing numeric/categorical values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Date & numeric features
if 'publish_time' in df.columns:
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['publish_year'] = pd.to_numeric(df['publish_time'].dt.year, errors='coerce')

# Feature engineering
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
df['title_word_count'] = df['title'].fillna("").apply(lambda x: len(x.split()))
df['num_authors'] = df['authors'].fillna("").apply(lambda x: len(x.split(';')))

st.write("Feature engineering complete. Sample:")
st.dataframe(df.head(5))

st.subheader("Code Used")
st.code("""# Fill missing numeric/categorical
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature creation
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))
df['title_word_count'] = df['title'].fillna("").apply(lambda x: len(x.split()))
df['num_authors'] = df['authors'].fillna("").apply(lambda x: len(x.split(';')))""", language='python')

# -----------------------------
# 4️⃣ Visualizations
# -----------------------------
st.header("Visualizations")

# Publications by Year
st.subheader("Number of Publications by Year (1990+)")
df_year = df.dropna(subset=['publish_year'])
df_year = df_year[df_year['publish_year'] >= 1990]
papers_per_year = df_year['publish_year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=papers_per_year.index.astype(int), y=papers_per_year.values, color='steelblue', alpha=0.8, ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
ax.set_title("Number of Publications by Year (1990+)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Top 10 Journals
st.subheader("Top 10 Journals")
top_journals = df['journal'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=top_journals.values, y=top_journals.index, palette='viridis', ax=ax)
ax.set_xlabel("Number of Papers")
ax.set_ylabel("Journal")
ax.set_title("Top 10 Journals Publishing COVID-19 Papers")
st.pyplot(fig)

# Word Cloud
st.subheader("Word Cloud of Paper Titles")
all_titles = " ".join(df['title'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_titles)
fig, ax = plt.subplots(figsize=(15,7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title("Most Frequent Words in Paper Titles")
st.pyplot(fig)

# Distribution by Source (Top 10 + Other)
st.subheader("Distribution of Papers by Source")
source_counts = df['source_x'].value_counts()
top_sources = source_counts.nlargest(10)
df['source_reduced'] = df['source_x'].apply(lambda x: x if x in top_sources.index else 'Other')
source_counts_final = df['source_reduced'].value_counts()
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=source_counts_final.index, y=source_counts_final.values, palette='Set2', ax=ax)
ax.set_xlabel("Source")
ax.set_ylabel("Number of Papers")
ax.set_title("Distribution of Papers by Source (Top 10 + Other)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# 5️⃣ Random Forest Classification
# -----------------------------
st.header("Abstract Length Classification (Short vs Long)")

# Create 'journal_reduced' for Random Forest
top_journals_rf = df['journal'].value_counts().nlargest(20).index
df['journal_reduced'] = df['journal'].apply(lambda x: x if x in top_journals_rf else 'Other')

# One-hot encode
df_encoded = pd.get_dummies(df, columns=['journal_reduced'], drop_first=True)
df_encoded = df_encoded.dropna(subset=['abstract_word_count', 'publish_year'])

# Target: short vs long abstracts
median_wc = df_encoded['abstract_word_count'].median()
df_encoded['long_abstract'] = (df_encoded['abstract_word_count'] > median_wc).astype(int)

# Features & target
feature_cols = ['publish_year', 'title_word_count', 'num_authors'] + \
               [c for c in df_encoded.columns if c.startswith('journal_reduced_')]
X = df_encoded[feature_cols]
y = df_encoded['long_abstract']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Cross-validation accuracy
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
st.write(f"Cross-validated Accuracy: {cv_scores.mean():.2%}")

# Classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Short','Long']).plot(cmap='Blues', ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("Top 20 Feature Importances")
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)[:20]
fig, ax = plt.subplots(figsize=(10,6))
importances.plot(kind='barh', color='teal', ax=ax)
ax.set_xlabel("Importance")
ax.invert_yaxis()
st.pyplot(fig)

# Predicted probabilities vs Year
st.subheader("Predicted Probability of Long Abstract by Year")
probs = clf.predict_proba(X_test)[:,1]
fig, ax = plt.subplots(figsize=(10,6))
scatter = ax.scatter(X_test['publish_year'], probs, c=y_test, cmap='bwr', alpha=0.6)
fig.colorbar(scatter, ax=ax, label='Actual Class (0=Short,1=Long)')
ax.set_xlabel("Year")
ax.set_ylabel("Predicted Probability of Long Abstract")
ax.set_title("Predicted Probability of Long Abstract by Year")
st.pyplot(fig)

# Histogram of predicted probabilities
st.subheader("Histogram of Predicted Probabilities")
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(probs[y_test==0], bins=20, alpha=0.6, label='Short Abstracts', color='blue')
ax.hist(probs[y_test==1], bins=20, alpha=0.6, label='Long Abstracts', color='red')
ax.set_xlabel("Predicted Probability of Long Abstract")
ax.set_ylabel("Number of Abstracts")
ax.set_title("Predicted Probabilities: Short vs Long Abstracts")
ax.legend()
st.pyplot(fig)