import logging
import time
from typing import Dict, List

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from src.settings import CHROME_WEBDRIVER_PATH, FILMWEB_PASSWORD, FILMWEB_USERNAME

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)


class FilmwebScraper:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(CHROME_WEBDRIVER_PATH)
        self.base_url = "https://www.filmweb.pl"
        self.login_url = "https://www.filmweb.pl/login"

    def login(self):
        self.driver.get(self.login_url)
        self.driver.find_element_by_class_name('fwBtn--gold').click()
        self.driver.find_element_by_class_name('authButton--filmweb').click()
        self.driver.find_element_by_name('j_username').send_keys(FILMWEB_USERNAME)
        self.driver.find_element_by_name('j_password').send_keys(FILMWEB_PASSWORD)
        self.driver.find_element_by_class_name('materialForm__submit').click()

    def open_page(self, url: str):
        self.driver.get(url)
        time.sleep(5)
        return self.driver.page_source

    def get_user_ratings(self, user_name: str) -> List[Dict[str, str]]:
        log.info('Scrapping from: {}'.format(self.base_url + f'/user/{user_name}/films'))
        page = self.open_page(self.base_url + f'/user/{user_name}/films')
        rated_movies = self._scrap_rated_movies(page)

        soup = BeautifulSoup(page, "html.parser")
        next_paginated = soup.find('a', {'class': 'pagination__link', 'title': 'następna'},
                                   href=True)
        while next_paginated is not None:
            log.info('Scrapping from: {}'.format(next_paginated['href']))

            page = self.open_page(self.base_url + next_paginated['href'])
            rated_movies.extend(self._scrap_rated_movies(page))
            soup = BeautifulSoup(page, "html.parser")
            next_paginated = soup.find('a', {'class': 'pagination__link', 'title': 'następna'},
                                       href=True)
        return rated_movies

    @staticmethod
    def _scrap_rated_movies(page):
        rated_movies = []
        soup = BeautifulSoup(page, "html.parser")
        boxes = soup.find_all('div', {'class': 'myVoteBox'})
        for box in boxes:
            title = box.find('h2', {'class': 'filmPreview__title'})
            origin_title = box.find('div', {'class': 'filmPreview__originalTitle'})
            year = box.find('div', {'class': 'filmPreview__year'})
            rate = box.find('span', {'class': 'userRate__rate'})

            rated_movies.append({
                'origin_title': origin_title.text if origin_title else title.text,
                'year': year.text,
                'rate': rate.text,
            })
        return rated_movies
