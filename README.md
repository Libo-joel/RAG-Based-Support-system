# RAG-Based-Support-system
Web Content Q&A with Cohere and Pinecone
This project allows you to scrape web content, create embeddings using Cohere, and store them in a Pinecone vector database for efficient retrieval. You can then query the content and get answers using an integrated language model.

Features
Scrape Web Content: Extracts text content from a given URL and saves it into a file.
Embeddings Creation: Uses Cohere to create embeddings for the scraped content.
Vector Store with Pinecone: Stores embeddings in a Pinecone index for efficient similarity search.
Question Answering: Uses Cohere's language model to answer questions based on the scraped content.
Setup
Prerequisites
Python 3.7 or higher
Cohere API Key
Pinecone API Key
Installation
cd web-content-qa
Installation
Install the required libraries:

pip install cohere pinecone-client numpy
Set up API keys
Replace "feqm6L1rQdncmas9yquTodYg04XgLhJ5iVTFeAxU" with your Cohere API key.

Replace "69ed45f3-116d-4e7a-a399-566d0dc3bcbe" with your Pinecone API key.

Usage
Scrape Web Content
Use your preferred method to scrape the webpage content and save it to a file named scraped_data.txt. For example, you can use BeautifulSoup and Selenium for web scraping:

Beautiful Soup: A Python library for pulling data out of HTML and XML files.
Selenium: A web browser automation tool to interact with web elements.
Here's a basic example of using BeautifulSoup and Selenium to scrape web content and save it:

Install BeautifulSoup and Selenium:
pip install beautifulsoup4 selenium
