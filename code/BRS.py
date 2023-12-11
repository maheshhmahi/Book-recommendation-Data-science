import pandas as pd
import numpy as np

cleaned_data = pd.read_csv(
    "D:/M@hii/WSU/Sem 1/Data Science/Project - Book recommendation System/Code/Book-Recommendation-System/cleaned data/dataset.csv",
    sep=",",
    dtype="unicode",
)

## We remove the ratings with 0 and keep only other ratings
cleaned_data["book_rating"] = cleaned_data["book_rating"].astype(int)
cleaned_data = cleaned_data[cleaned_data["book_rating"] != 0]
cleaned_data = cleaned_data.reset_index(drop=True)


from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

bookName = input("Enter a book name: ")
number = int(input("Enter number of books to recommend: "))

# bookName = "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
# number = 5

popularity_threshold = 50

data = (
    cleaned_data.groupby(by=["Book-Title"])["book_rating"]
    .sum()
    .reset_index()
    .rename(columns={"book_rating": "Total-Rating"})[["Book-Title", "Total-Rating"]]
)

result = pd.merge(data, cleaned_data, left_on="Book-Title", right_on="Book-Title")
result = result[result["Total-Rating"] >= popularity_threshold]
result = result.reset_index(drop=True)

matrix = result.pivot_table(
    index="Book-Title", columns="User-ID", values="book_rating"
).fillna(0)
up_matrix = csr_matrix(matrix)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(up_matrix)

distances, indices = model.kneighbors(
    matrix.loc[bookName].values.reshape(1, -1), n_neighbors=number + 1
)
print("\nRecommended books:\n")
for i in range(0, len(distances.flatten())):
    if i > 0:
        print(matrix.index[indices.flatten()[i]])
