#!/usr/bin/env python3

import requests
import time
import sqlite3
from bs4 import BeautifulSoup as soup
import pickle

def download(cl, limit):
    url = 'http://www.kinopoisk.ru/reviews/type/comment/status/{0}/period/month/page/{1}/#list'
    texts = []
    p = 1

    while True:
        user_agent = {'User-agent': 'Mozilla/5.0'}
        r = requests.get(url.format(cl, p), headers = user_agent)
        s = soup(r.text)
        for div in s.find_all('div', {'class': 'userReview'}):
            div_text = div.find('div', {'class': 'brand_words'})
            #div_text =  unicode(div_text.text.encode('cp1251', 'ignore'))
            div_text = div_text.text
            texts.append((div_text,))
        print('Processed page {0}, {1} texts'.format(p, len(texts)))
        if len(texts) >= limit:
            break
        p += 1
        time.sleep(1)

    return texts[:limit]



con = sqlite3.connect('test.db')
cur = con.cursor()

cur.execute('drop table docs')
cur.execute('''
	create table docs(
		id integer primary key autoincrement,
		text text,
		class text
	)
	''')

limit = 300

texts_pos = download('good', limit)
texts_neg = download('bad', limit)

cur.executemany('insert into docs (class, text) values ("pos", ?)', texts_pos)
cur.executemany('insert into docs (class, text) values ("neg", ?)', texts_neg)
pickle.dump( texts_pos, open('texts_pos.p', 'wb'))
pickle.dump( texts_neg, open('texts_neg.p', 'wb'))
texts = cur.execute('SELECT * FROM docs').fetchall()
pickle.dump(texts, open('texts_300.p', 'wb'))

con.commit()
con.close()