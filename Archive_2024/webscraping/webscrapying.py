import requests
from bs4 import BeautifulSoup
import os

BASE_URL = 'https://www.awm.gov.au'
COLLECTION_URL = 'https://www.awm.gov.au/collection/C963607'
SAVE_PATH = r'C:\Users\10840\Desktop\8715\leaflets2'#don't forget change the save_path when you run it!!!
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}


# Get all collection links on a page
def get_collection_links(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    return [link['href'] for link in links if '/collection/C' in link['href']]


# Download PDF
def download_pdf_from_collection(url, file_number):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_link = soup.find('a', class_='iconic media__iconic jsCollectionItemDownloadLink')

    if pdf_link and pdf_link.has_attr('href'):
        pdf_url = pdf_link['href']
        pdf_response = requests.get(pdf_url, headers=HEADERS)

        if pdf_response.status_code == 200:
            file_name = os.path.join(SAVE_PATH, f"{file_number}.pdf")
            with open(file_name, 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)
            print(f'PDF {file_number} successfully downloaded!')
        else:
            print(f'Failed to retrieve the PDF for {url}')


if __name__ == "__main__":
    collection_links = get_collection_links(COLLECTION_URL)

    for idx, link in enumerate(collection_links, 1):
        collection_page_url = BASE_URL + link
        download_pdf_from_collection(collection_page_url, idx)
