import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Read codes from the file
with open("codes.txt", "r") as file:
    codes = [line.strip() for line in file.readlines()]

# Base URL
base_url = "https://www.awm.gov.au/collection/"

# Initialize the webdriver
driver = webdriver.Chrome()  # You need to have Chrome driver installed

for code in codes:
    # Construct the full URL
    full_url = base_url + code

    # Use requests to fetch the HTML content
    response = requests.get(full_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the download link using BeautifulSoup
    download_link = soup.find("a", class_="iconic media__iconic jsCollectionItemDownloadLink")
    if download_link:
        download_url = download_link["href"]
        print(f"Downloading {download_url}")
        
        # Download the file using requests
        file_response = requests.get(download_url)
        
        # Save the file
        with open(f"{code}.pdf", "wb") as file:
            file.write(file_response.content)
    else:
        print(f"No download link found for {code}")

# Close the browser window
driver.quit()
