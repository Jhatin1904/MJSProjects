import pandas as pd
import re
import requests

# Function to clean the HTML content
def clean_article_content(article_content):
    clean_content = re.sub('<.*?>', '', article_content)  # Remove HTML tags
    clean_content = re.sub(r'\s+', ' ', clean_content).strip()  # Remove multiple spaces, newlines, etc.
    clean_content = clean_content.lower()  # Convert to lowercase
    return clean_content

# Function to get article content from a URL
def get_article_content(url):
    try:
        # Ensure the URL is valid and fetch the content
        response = requests.get(url)
        if response.status_code == 200:  # Check if request was successful
            return response.text
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch any request exceptions like connection errors
        print(f"Error fetching {url}: {e}")
        return None

# Example: Creating a DataFrame with sample URLs
data = {
    'link': [
        'https://medium.com/',  # Example URL
        'https://dev.to/'   # Example URL
    ]
}

articles_df = pd.DataFrame(data)

# Apply the content fetching and cleaning function to each link
articles_df['clean_content'] = articles_df['link'].apply(
    lambda url: clean_article_content(get_article_content(url)) if get_article_content(url) else None
)

# Drop articles with missing data
articles_df.dropna(subset=['clean_content'], inplace=True)

# Save the cleaned data
articles_df.to_csv('cleaned_articles.csv', index=False)
