import time
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import requests
import queue



table = queue.Queue()
failed_url = queue.Queue()
PROXY_POOL_URL = 'http://localhost:5555/random'
headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
        }


def fetch(url):
    try:
        response = requests.get(PROXY_POOL_URL, timeout=100)
        assert response.status_code == 200
        proxy = response.text

        proxies = {'http': 'http://'+proxy, 'https': 'https://'+proxy}
        response = requests.get(url, headers=headers, proxies=proxies, timeout=50)
        assert response.status_code == 200
        response.encoding = 'utf-8'
        return response.text
    except:
        return None

    
def parser(html, movie):
    if not html:
        failed_url.put(movie)
        return None
    try:
        soup = BeautifulSoup(html, 'lxml')
        items = list(soup.find_all("div", class_="comment-item"))
        if len(items) == 0:
            failed_url.put(movie)
            return None
        for item in items:
            comment = item.find('p').get_text().strip()
            content = np.array([*movie, comment])
            table.put(content)
    except:
        print('parser error')
        pass


def download(movie):
    html = fetch(movie[4])
    parser(html, movie)

if __name__ == '__main__':
    #movies = pd.read_csv('reviews_url.csv')
    movies = pd.read_csv('failed136.csv')
    print('*' * 50)
    t1 = time.time()

    tasks = [movie for movie in movies.values]
    pool = ThreadPool(800)
    
    pool.map_async(download, tasks)
    pool.close()
    failed_url.join()
    table.join()

    pool.join()

    time.sleep(10)
    df = pd.DataFrame(list(table.queue), columns=['name', 'rating', 'subject_href', 'comment_nums', 'comment_href', 'label', 'content'])
    df_failed = pd.DataFrame(list(failed_url.queue), columns=['name', 'rating', 'subject_href', 'comment_nums', 'comment_href', 'label'])
    df_failed.to_csv('failed/failed{}.csv'.format(len(df_failed)), index=False)
    df.to_csv('result/{}.csv'.format(len(df)), index=False)
    t2 = time.time()
    print('time consumption：%s' % (t2 - t1))
    print('*' * 50)
