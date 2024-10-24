import pandas as pd
import re
import requests

def clean_article_content(article_content):
    clean_content = re.sub('<.*?>', '', article_content) 
    clean_content = re.sub(r'\s+', ' ', clean_content).strip() 
    clean_content = clean_content.lower() 
    return clean_content

def get_article_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200: 
            return response.text
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

data = {
    'link': [
        'https://medium.com/',  
        'https://dev.to/'  
    ]
}

articles_df = pd.DataFrame(data)

articles_df['clean_content'] = articles_df['link'].apply(
    lambda url: clean_article_content(get_article_content(url)) if get_article_content(url) else None
)

articles_df.dropna(subset=['clean_content'], inplace=True)

articles_df.to_csv('cleaned_articles.csv', index=False)
