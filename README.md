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
```python

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
```

## 뉴스 전처리
뉴스를 크롤링해오는 과정에서 본문에 다양한 특수기호와 경제 뉴스의 특성상 ㎡ ㎥ ㎞ ㎏ ㎖와 같은 기호가 특수문자로 입력이 되어있었습니다. 또한 한문이 다수 포함되어 있었으며 기자들의 이메일이 포함되어 있어 삭제하는 전처리 과정을 추가하였습니다. 
또한, △▲ 등은 소제목(또는 나열된 항목), ↑↓ 는 상승, 하락을 의미할 것으로 보이기 때문에 
-> 상승(증가), 하락(감소)으로 대체하였습니다. 

<이메일, url, 특수문자 처리>
```python
def preprocess(text):

    '''기사 형식'''
    # 바이라인: 이메일, url 제거
    import re
    pattern_mail = re.compile(r'[\w.+]+\s?@\s?[\w]+\.[\w.]+') # 공백 포함: mbcjebo @ mbc.co.kr
#     pattern_mail = re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-.])+')
#     pattern_mail = re.compile(r'[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', flags=re.IGNORECASE) # .com .co.kr 등
    text = re.sub(pattern_mail, '', text)
    pattern_url = re.compile(r'(?:https?:\/\/)+?[-_0-9a-z]+(?:[\.\/][-_0-9a-z]+)*', flags=re.IGNORECASE)
    text = re.sub(pattern_url, '', text)
    pattern_call = re.compile(r'\d{2,3}-\d{3,4}-\d{4}')
    text = re.sub(pattern_call, '', text)

#     text = text.replace('기자 단독', '')
#     text = text.replace('[단독]', '')

    '''특수문자'''
    # 한자를 한글로 치환 (兆, 조)
    text = text.replace('比', '대비')
    text = hanja.translate(text, 'substitution')
    text = text.replace('年', '년')
    text = text.replace('李', '이') # 이창용 한은 총재

    # 단위 치환
    text = text.replace('㎞', 'km')
    text = text.replace('㎏', 'kg')
    text = text.replace('㎖', 'ml')
    text = text.replace('㎎', 'mg')
    text = text.replace('ℓ', 'l')
    text = text.replace('㎝', 'cm')
    text = text.replace('㏗', 'pH')
    text = text.replace('$', '달러')
    text = text.replace('ｍ', 'm')
    text = text.replace('ｇ', 'g')
    text = text.replace('㎾', 'kW')
    text = text.replace('㎐', 'Hz') # 헤르츠도 있음 - 동의어 사전 필요함
    text = text.replace('℃', '섭씨')
    text = text.replace('㎉', '칼로리')
    text = text.replace('％', '%')
    text = text.replace('%', '퍼센트')
    text = text.replace('...', '…') # 추후 띄어쓰기 위함

    # 특수문자 해석
    # 주가 상승, 인구 이동 증가 등등 문맥에 맞게 변환해야 함. 해결방법???
#     text = text.replace('↑', ' 상승')
#     text = text.replace('↓', ' 하락')

    # 특수문자 제거
    # 예외: .(1.2%), &(B&S 홀딩스, S&P), ~(30~40)
    pattern_special = r'[^가-힣0-9a-zA-Z\.,\%\&\~\s]' # 한글, 숫자, 영어, 공백, 기타 문자만 유지
    text = re.sub(pattern_special, ' ', text)

    # [], (), {}, <> 괄호와 괄호 안 문자 제거하기
#( ) 괄호 안에 1문자 이상 있으면 제거
#     pattern = r'\([^)]*\)'  # () # ('\(.+?\)', '')
#     s = re.sub(pattern=pattern, repl='', string=s)

    # 예외: 숫자 (4.5조, 4.5%, 3.75%, 90% ...)
#     TBD
#     text = re.sub('[-=+,#\/\?:^\.@*\"※~ㆍ!』|\(\)\[\]`\'…》\”\“\‘\’·▷▶▲ⓒ◆■]', ' ', text) # 이 중 하나와 매치
#     text = re.sub('[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]', ' ', text)

    # 다중 공백 제거
    text = ' '.join(text.split()) # 공백 기준으로 분리 후 단어 사이에 공백 삽입
    return text
```


<>






