import pandas as pd


additional_data = {
    'Book Title': ['The Catcher In The Rye', 'To Kill A Mockingbird', 'The Great Gatsby', '1984', 'Pride And Prejudice'],
    'ISBN': ['978-0-316-76948-0', '978-0-06-112008-4', '978-0-7432-7356-5', '978-0-452-28423-4', '978-0-19-953556-9'],
    'Publisher': ['Little, Brown and Company', 'J.B. Lippincott & Co.', 'Scribner', 'Plume', 'Oxford University Press']
}


additional_df = pd.DataFrame(additional_data)


print("Additional Dataset:")
print(additional_df)


file_path = 'Books_data.csv'
books_df = pd.read_csv(file_path)


books_df = books_df.dropna(subset=['Book Title', 'Author'])

books_df['Copies Left'] = pd.to_numeric(books_df['Copies Left'].replace('50+', '50'), errors='coerce')


books_df['Reviews'] = books_df['Reviews'].fillna(books_df['Reviews'].median())
books_df['Ratings '] = books_df['Ratings '].fillna(books_df['Ratings '].median())
books_df['Copies Left'] = books_df['Copies Left'].fillna(books_df['Copies Left'].median())

books_df['Category'] = books_df['Category'].fillna(books_df['Category'].mode()[0])


books_df = books_df.drop_duplicates()


if 'Publication' in books_df.columns:
    books_df['Publication'] = pd.to_datetime(books_df['Publication'], errors='coerce')


books_df['Book Title'] = books_df['Book Title'].str.title()
books_df['Author'] = books_df['Author'].str.title()


merged_df = pd.merge(books_df, additional_df, on='Book Title', how='left')

print("\nMerged Dataset:")
print(merged_df.head())

merged_file_path = 'merged_books_dataset.xlsx'
merged_df.to_excel(merged_file_path, index=False)

merged_file_path
