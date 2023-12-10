import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

text_box_color = "#fff"  # Set your preferred color

# Apply custom styles to the text box




data=pd.read_excel("task3.xlsx")
data.head(1)

data.describe()
data.nunique()
data['Unnamed: 12'].unique()
data.isnull().sum()
df=data.dropna(axis=1)


def eda():
    st.title("Exploratory Data Analysis (EDA)")
    
    plt.figure(figsize=(10,8))
    books = df['title'].value_counts()[:20]
    rating = df.average_rating[:20]
    sns.barplot(x = books, y = books.index, palette='deep')
    plt.title("Most Occurring Books")
    plt.xlabel("Number of occurances")
    plt.ylabel("Books")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.button('Most Occurring Books')
    st.pyplot()

    plt.figure(figsize=(10,8))
    ax = df.groupby('language_code')['title'].count().plot.bar()
    plt.title('Language Code')
    st.button('Mostly Used Language')
    st.pyplot()

    st.write(df['authors'].value_counts().head(10))

    df['publication_date'].astype(str)
    df['publication_date']=pd.to_datetime(df['publication_date'],errors='coerce')
    df['year'] = df['publication_date'].dt.year
    df['year'].astype('Int64')
    most_rated = df.sort_values('ratings_count', ascending=False).head(10).set_index('title')
    plt.figure(figsize=(10, 8))
    sns.barplot(x=most_rated['ratings_count'],y= most_rated.index, palette='rocket')
    st.button('Top Rated Books')
    st.pyplot()

    author_counts = df['authors'].value_counts().head(10).reset_index()
    author_counts.columns = ['authors', 'title']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='title', y='authors', data=author_counts, palette='viridis')
    plt.xlabel('Number of Books')
    plt.ylabel('Author')
    plt.title('Top 10 Authors with Most Books')
    plt.show()
    st.button('Top 10 Authors with Most Books')
    st.pyplot()

    Authors=['Stephen King','P.G. Wodehouse','Rumiko Takahashi','Orson Scott Card','Agatha Christie','Piers Anthony','Sandra Brown','Mercedes Lackey','Dick Francis','Terry Pratchett']

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True, sharey=True)
    fig.suptitle('Ratings Count for Top 10 Authors', fontsize=16)

    for i, (author_name, ax) in enumerate(zip(Authors, axes.flatten())):
        author_df = df[df['authors'] == author_name]
        if not author_df.empty:
            sns.barplot(x='year', y='average_rating', data=author_df, ax=ax, palette='deep')
            ax.set_title(f'{author_name}')
            ax.set_ylabel('Average Rating')
            ax.set_xlabel('Year')
            ax.set_ylim(0, 5)  # Set y-axis limit for better comparison
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Show only integer ticks on x-axis
            plt.xticks(rotation=30)


        else:
         fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent title overlap
    plt.show()
    st.button('Average Rating by Year for Top 10 Authors')
    st.pyplot()

    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['average_rating'].dropna(), bins=20, kde=False)
    plt.xlabel('Average Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Ratings')
    plt.show()
    st.button('Distribution of Average Ratings')
    st.pyplot()

    top_books = df.sort_values(by='text_reviews_count', ascending=False).head(10)

    

    plt.figure(figsize=(10,8))
    
    sns.set_context('paper')
    ax =sns.jointplot(x="average_rating",y='text_reviews_count', kind='scatter',  data= df[['text_reviews_count', 'average_rating']])
    ax.set_axis_labels("Average Rating", "Text Review Count")
    plt.show()
    st.button('Average Rating vs Text Review Count')
    st.pyplot()

    



df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
rating_bins = [0, 1, 2, 3, 4, 5]
rating_labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
df['Ratings_Class'] = pd.cut(df['average_rating'], bins=rating_bins, labels=rating_labels, include_lowest=True)
df.head(2)

books_features = pd.concat([df['Ratings_Class'].str.get_dummies(sep=","),df['bookID'], df['average_rating'], df['ratings_count']], axis=1)

books_features=books_features.drop('nan',axis=1)
books_features.sample(10)

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
features = books_features.drop(['bookID', 'average_rating', 'ratings_count'], axis=1)
min_max_scaler = MinMaxScaler()
features_scaled = min_max_scaler.fit_transform(features)
books_features_scaled = pd.concat([pd.DataFrame(features_scaled, columns=features.columns), df[['bookID', 'average_rating', 'ratings_count']]], axis=1)
model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
model.fit(features_scaled)

# Find nearest neighbors
distance, indices = model.kneighbors(features_scaled)


def get_index_from_name(name):
    return df[df["title"]==name].index.tolist()[0]

all_books_names = list(df.title.values)


def get_id_from_partial_name(partial):
    matches = [name for name in all_books_names if isinstance(name, str) and partial in name]

    if not matches:
        st.info(f"No matches found for '{partial}'.")
        return
    st.subheader("Matching Books:")
    for idx, name in enumerate(matches):
        st.write(f"{name}: {idx}")
        
def print_similar_books(query=None,id=None):
    if id:
        st.subheader("Similar Books:")
        for id in indices[id][1:]:
            st.write(df.iloc[id]["title"])
    if query:
        found_id = get_index_from_name(query)
        st.subheader("Similar Books:")
        for id in indices[found_id][1:]:
            st.write(df.iloc[id]["title"])



def main():

    st.title("Book Recommendation and EDA App")

    # Sidebar for user input
    st.sidebar.header("User Input")
    partial_name = st.sidebar.text_input("Enter partial book name:", "")

    # Main content
    if partial_name:
        get_id_from_partial_name(partial_name)

    # Recommendation and EDA section
    st.sidebar.header("Book Recommendations / EDA")
    option = st.sidebar.radio("Choose an option:", ("By Title", "By Index", "Perform EDA"))

    if option == "By Title":
        selected_book = st.sidebar.text_input("Enter the book title:", "")
        if selected_book:
            print_similar_books(query=selected_book)
    elif option == "By Index":
        selected_index = st.sidebar.number_input("Enter the book index:", min_value=0, max_value=len(df)-1)
        if selected_index is not None:
            print_similar_books(id=int(selected_index))
    elif option == "Perform EDA":
        eda()

if __name__ == "__main__":
    main()



