#  키워드 추출 기반 경제 트렌드 뉴스레터 서비스 "비요뜨"
"비요뜨" 서비스는 매일매일의 뉴스를 키워드 형태로 정보를 압축하여 전달해주는 서비스입니다.
키워드 형태로 정보를 압축하여 정보를 받은 고객은 경제 트랜드를 파악할 수 있게 됩니다.

# 목차
1. 뉴스 수집 및 전처리

2. 기사별 Top-k개 키워드 선정
   - 빈도수 기반(TF-IDF)
   - 코사인 유사도 거리 기반

3. 대주제 탐색(DBSCAN)

4. 주제별 기사 요약

5. 결과물

< NLP 프로세스 >
![image](https://github.com/Yoon-Hee-Jae/bitamin_project/assets/140389762/9fd21c27-1e6a-43dd-b534-134114a5b321)


# 1. 뉴스 수집 및 전처리

## 뉴스 수집
뉴스는 네이버 뉴스에서 크롤링을 진행하였습니다.
크롤링에는 셀레니움을 사용하였고 헤드라인 기사 대신 실시간 최신 뉴스에서 수집을 하였습니다.
이번 프로젝트에서는 1573개의 기사를 크롤링해왔습니다.

< 링크 선택 >
'''python
def extract_url(page=1, sid=101):
    ### 뉴스 분야(sid)와 페이지(page)를 입력하면 그에 대한 링크들을 리스트로 추출하는 함수 ###

    ## 1. url의 html 구조를 가져온다
    url = f"https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1={sid}#&date=%2000:00:00&page={page}"
    driver = webdriver.Chrome(options=options)  # for Mac
    driver.get(url)
    req = driver.page_source
    soup = BeautifulSoup(req, "html.parser")

    # 크롤링 대상 섹션 중에서 a태그가 있는 것을 고른다
    body_text = soup.find(name="div", attrs={"class":"section_body"}).find_all(name="a")

    ## 2. a태그의 href 속성을 리스트로 추출하여, 크롤링할 페이지 리스트를 생성 (모든 기사 링크 리스트)
    page_urls = []
    for index in body_text:
        if "href" in index.attrs:  # href가 있는 것만 고름
            if ( f"sid={sid}" in index["href"] # 경제
                and "article" in index["href"] # 기사인 것 (광고 등 제외)
               ):
                page_urls.append(index["href"])

    driver.close()
    return page_urls


## 뉴스 전처리
뉴스를 크롤링해오는 과정에서 본문에 다양한 특수기호와 경제 뉴스의 특성상 ㎡ ㎥ ㎞ ㎏ ㎖와 같은 기호가 특수문자로 입력이 되어있었습니다. 또한 한문이 다수 포함되어 있었으며 기자들의 이메일이 포함되어 있어 삭제하는 전처리 과정을 추가하였습니다. 
또한, △▲ 등은 소제목(또는 나열된 항목), ↑↓ 는 상승, 하락을 의미할 것으로 보이기 때문에 
-> 상승(증가), 하락(감소)으로 대체하였습니다. 

<이메일, url 삭제>




<특수기호 처리>






