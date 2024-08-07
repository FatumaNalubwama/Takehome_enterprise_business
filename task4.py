import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder


file_path = 'Books_data.csv'
books_df = pd.read_csv(file_path)


books_df = books_df.dropna(subset=['Book Title', 'Author'])

books_df['Copies Left'] = pd.to_numeric(books_df['Copies Left'].replace('50+', '50'), errors='coerce')


books_df.columns = books_df.columns.str.strip()

# Fill missing values for 'Reviews', 'Ratings', 'Copies Left', and 'Category'
books_df['Reviews'] = books_df['Reviews'].fillna(books_df['Reviews'].median())
books_df['Ratings'] = books_df['Ratings'].fillna(books_df['Ratings'].median())
books_df['Copies Left'] = books_df['Copies Left'].fillna(books_df['Copies Left'].median())
books_df['Category'] = books_df['Category'].fillna(books_df['Category'].mode()[0])


books_df = books_df.drop_duplicates()

if 'Publication' in books_df.columns:
    books_df['Publication'] = pd.to_datetime(books_df['Publication'], errors='coerce')


books_df['Book Title'] = books_df['Book Title'].str.title()
books_df['Author'] = books_df['Author'].str.title()


label_encoder = LabelEncoder()
books_df['Category'] = label_encoder.fit_transform(books_df['Category'])


print(books_df.info())
print(books_df.describe())


sns.set(style="whitegrid")


plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=books_df, palette='viridis', hue='Category', dodge=False, legend=False)
plt.title('Distribution of Book Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(books_df['Ratings'], bins=20, kde=True)
plt.title('Distribution of Book Ratings')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(books_df['Reviews'], bins=20, kde=True)
plt.title('Distribution of Book Reviews')
plt.xlabel('Reviews')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Ratings', y='Reviews', data=books_df, hue='Category', palette='viridis')
plt.title('Ratings vs. Reviews')
plt.xlabel('Ratings')
plt.ylabel('Reviews')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Book Length (Pages)', y='Copies Left', data=books_df, hue='Category', palette='viridis')
plt.title('Book Length vs. Copies Left')
plt.xlabel('Book Length (Pages)')
plt.ylabel('Copies Left')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=0)
books_df['Cluster'] = kmeans.fit_predict(books_df[['Ratings', 'Reviews', 'Book Length (Pages)', 'Copies Left']])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Ratings', y='Reviews', hue='Cluster', data=books_df, palette='viridis')
plt.title('KMeans Clustering (3 Clusters)')
plt.xlabel('Ratings')
plt.ylabel('Reviews')
plt.show()


X = books_df[['Ratings', 'Book Length (Pages)', 'Copies Left']]
y = books_df['Reviews']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print('Mean Squared Error:', mean_squared_error(y_test, y_pred))


X = books_df[['Ratings', 'Book Length (Pages)', 'Copies Left']]
y = books_df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = LinearRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred = [round(value) for value in y_pred]


unique_labels = sorted(set(y_test.unique()) | set(y_pred))

print(classification_report(y_test, y_pred, labels=unique_labels, target_names=label_encoder.inverse_transform(unique_labels)))
