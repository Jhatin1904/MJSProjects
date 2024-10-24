import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Set headers and user-agent to prevent getting blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

def scrape_articles(url):
    articles = []
    while url:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all articles in the page
        for article in soup.find_all('div', class_='postArticle'):
            title = article.find('h3').get_text() if article.find('h3') else 'No Title'
            author = article.find('span', class_='ds-link').get_text() if article.find('span', class_='ds-link') else 'Unknown'
            date = article.find('time')['datetime'] if article.find('time') else 'No Date'
            link = article.find('a')['href'] if article.find('a') else 'No Link'
            articles.append({'title': title, 'author': author, 'date': date, 'link': link})

        # Find the pagination button/link to the next page
        next_page = soup.find('a', class_='next')
        url = next_page['href'] if next_page else None

        # Sleep to avoid rate limiting/blocking
        time.sleep(random.randint(1, 5))

    return articles

# Example: Start scraping from the first page of articles by the author
author_url = "https://medium.com/specific-author"
articles_data = scrape_articles(author_url)

# Store the scraped data into a DataFrame
articles_df = pd.DataFrame(articles_data)
articles_df.to_csv('scraped_articles.csv', index=False)
