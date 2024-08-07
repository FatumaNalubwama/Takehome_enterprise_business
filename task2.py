import pandas as pd

file_path = 'Books_data.csv'
books_df = pd.read_csv(file_path)


print("Column Names in the Dataset:")
print(books_df.columns)

print("\nBefore Cleaning:")
print(books_df.head())


missing_values = books_df.isnull().sum()
print("\nMissing Values Before Cleaning:")
print(missing_values)

books_df = books_df.dropna(subset=['Book Title', 'Author'])

books_df['Copies Left'] = pd.to_numeric(books_df['Copies Left'].replace('50+', 50), errors='coerce')


books_df['Reviews'] = books_df['Reviews'].fillna(books_df['Reviews'].median())
books_df['Ratings '] = books_df['Ratings '].fillna(books_df['Ratings '].median())
books_df['Copies Left'] = books_df['Copies Left'].fillna(books_df['Copies Left'].median())


books_df['Category'] = books_df['Category'].fillna(books_df['Category'].mode()[0])


books_df = books_df.drop_duplicates()


if 'Publication' in books_df.columns:
    books_df['Publication'] = pd.to_datetime(books_df['Publication'], errors='coerce')


print("\nData Types After Converting Publication Date:")
print(books_df.dtypes)

books_df['Book Title'] = books_df['Book Title'].str.title()
books_df['Author'] = books_df['Author'].str.title()

print("\nAfter Cleaning:")
print(books_df.head())


cleaned_file_path = 'cleaned_books_dataset.xlsx'
books_df.to_excel(cleaned_file_path, index=False)
print("\nCleaned dataset saved to:", cleaned_file_path)
