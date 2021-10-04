import pandas as pd

import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait

from time import sleep
# conda install -c conda-forge pathos
from pathos.multiprocessing import ProcessingPool as Pool

'''
참고 자료
https://greeksharifa.github.io/references/2020/10/30/python-selenium-usage/
https://www.hanumoka.net/2020/07/05/python-20200705-python-selenium-install-start/

https://conservative-vector.tistory.com/entry/%EC%85%80%EB%A0%88%EB%8B%88%EC%9B%80%EC%9D%84-%EC%93%B0%EB%8A%94%EB%8D%B0-%EB%A9%80%ED%8B%B0%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1%EC%9D%B4-%EC%95%88-%EB%8F%BC%EC%9A%94
https://beomi.github.io/gb-crawling/posts/2017-07-05-HowToMakeWebCrawler-with-Multiprocess.html
chrome webdriver 다운 필수!
'''

class Parser(object):
    def __init__(self, df, url, processes = 8):
        self.url = url
        self.df = df
        self.processes = processes
        self.pool = Pool(processes=self.processes)
    
    def open_browser(self, num):
        options = webdriver.ChromeOptions()
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        driver = webdriver.Chrome(chrome_options=options, executable_path='chromedriver')
        driver.get(url=self.url)
        driver.implicitly_wait(time_to_wait=10)
        sleep(4//self.processes*num)

        input_box = driver.find_element_by_xpath('//*[@id="txtSource"]')
        result_box = driver.find_element_by_xpath('//*[@id="txtTarget"]')
        change_language_btn = driver.find_element_by_xpath('//*[@id="ddTargetLanguage2"]/div[1]/button[2]')
        english_btn = driver.find_element_by_xpath('//*[@id="ddTargetLanguage2"]/div[2]/ul/li[2]/a')
        japanese_btn = driver.find_element_by_xpath('//*[@id="ddTargetLanguage2"]/div[2]/ul/li[3]/a')
        back_translation = driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/section/div/div[1]/div[1]/div/div[2]/button')

        change_language_btn.click()
        japanese_btn.click()
        input_box.click()
        df = self.df

        translated = []

        for i in range(1250):
            try:
                tmp = df.iloc[i*self.processes + num]
            except:
                break
            # print('-'*30)
            # print(tmp.sentence, '\n')
            input_box.send_keys(tmp.sentence)
            sleep(6.5)
            back_translation.click()
            sleep(3)

            result = result_box.text
            # print(result)
            translated.append([i*self.processes + num, result, tmp.sub_entity, tmp.obj_entity, tmp.label])
            
            back_translation.click()
            sleep(0.5)
            input_box.clear()
        
        return translated

    def multi_processing(self):
        num = range(self.processes)
        return self.pool.map(self.open_browser, num)



df = pd.read_csv('preprocessed.csv', index_col=0)

URL = 'https://papago.naver.com/'

if __name__ == '__main__':
    parser = Parser(df = df, url=URL, processes=10)
    translated_list = parser.multi_processing()
    sleep(3)
    translated = []
    for results in translated_list:
        for result in results:
            translated.append(result)
    pd.DataFrame(translated, columns=["id", "sentence", "sub_entity", "obj_entity", "label"]).sort_values(by='id', ascending=True).to_csv('ppg_jp_backtrans1.csv', index=False)

# 로딩 대기 (10초)
# driver.implicitly_wait(time_to_wait=10)

# driver.close()