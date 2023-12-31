#  비요뜨
   키워드 추출 기반 뉴스 구독 서비스 "비요뜨" 서비스는 매일매일의 뉴스를 키워드 형태로 정보를 압축하여 전달해주는 서비스입니다. 키워드 형태로 정보를 압축하여 정보를 받은 고객은 경제 트랜드를 빠르게 파악할 수 있게 됩니다.

# 목차
1. 뉴스 수집 및 전처리

2. 기사별 Top-k개 키워드 선정
   - N- gram과 빈도수 기반(TF-IDF) 2가지 방법을 사용해 키워드 추출
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

# 2. 키워드 추출
키워드 추출의 경우 정확성을 높이기 위해서 N-gram과 TF-IDF 2가지 방식으로 키워드를 추출하였습니다.
이렇게 두가지 방식으로 추출된 키워드를 합친 다음 제목과 본문과의 코사인 유사도를 계산하여
최종적으로 기사마다 코사인 유사도가 높은 순으로 10개의 키워드를 선정하였습니다. 

## 2-1 N-gram
"연어"란 두 단어가 연속적으로 쓰여 뜻을 가지는 단어를 의미합니다.
하나의 단어로만 키워드를 선정하는 것보다 연어로 추출하는 경우가 결과가 더 좋은 것을 확인하였습니다. 연어로 키워드를 추출하기 위해서 음절의 단위를 형태소 분석 결과로 나누어서 정확도를 높인 연어가 뽑힐 수 있도록 설정하였습니다.
```python
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramAssocMeasures
from nltk.metrics.association import QuadgramAssocMeasures
from nltk.collocations import BigramCollocationFinder

def collocations_single(num):

    ngram = [(BigramAssocMeasures(),BigramCollocationFinder),
             (TrigramAssocMeasures(),TrigramCollocationFinder),
              (QuadgramAssocMeasures(),QuadgramCollocationFinder)]

    # 텍스트 지정, 출력
    doc = selectsample(num, content=True)
    print('제목/', selectsample(num, content=False))
    print(' ')
    print('본문/', doc)
    print(' ')

    # 형태소 분석기 - 명사만 추출
    okt = Okt()
    nouns = okt.nouns(doc) # get nouns
    tagged_words = okt.pos(doc) # 품사 부착
    words = [w for w, t in tagged_words] # 단어+품사 중 단어만 선택
    ignored_words = [u'되다', '되었다', '됐다', '하다', '하였다', '했다', '이다', '이었다', '였다', '있다', '없다'] # 불용어

    # 각 알고리즘별로 bigram, trigram, quadgram 10개씩 뽑아 리스트 만들기
    founds_from_4measure = []
    for measure, finder in ngram:
        '''명사 vs 전체 단어 택 1'''
    #     finder = finder.from_words(nouns)
        finder = finder.from_words(words) # 숫자, 알파벳(LG, NH), 영어 포함

        finder.apply_word_filter(lambda w: len(w) < 2 or w in ignored_words) # 제외조건: 길이가 1이거나 ignored_words에 포함되면 제외
        finder.apply_freq_filter(3) # only bigrams that appear 3+ times
        founds = finder.nbest(measure.pmi, 10)       # pmi - 상위 10개 추출
        founds += finder.nbest(measure.chi_sq, 10)   # chi_sq - 상위 10개 추출
        founds += finder.nbest(measure.mi_like, 10)  # mi_like - 상위 10개 추출
        founds += finder.nbest(measure.jaccard, 10)  # jaccard - 상위 10개 추출

        founds_from_4measure += founds

    # 나뉜 연어를 한 단어로 묶고, 3/4 voting된 최종 연어 출력
    collocations = [' '.join(collocation) for collocation in founds_from_4measure]
    collocations = [(w,f) for w,f in Counter(collocations).most_common() if f > 2]
    pprint(collocations)
```

## 2-2 TF-IDF를 통해 키워드 추출
TF-IDF란 문서내에서 특정 단어의 중요도를 계산하는 방법입니다.
TF = 하나의 문서내에서 특정 단어가 나타나는 빈도
IDF = 여러 문서에서 특정 단어가 얼마나 공통적으로 등장하는지를 수치화한 값
TF-IDF = 한 문서내에서 많이 등장하면서 다른 문서에서 적게 등장할수록 중요한 단어로 인식합니다.

```python
# TF-IDF를 사영하여 문서 벡터화 진행
tfidf_matrix = tfidfv.fit_transform(df1['content_p'])

for i in range(len(df1)):
    doc_tfidf = tfidf_matrix[i].toarray().flatten() # 벡터화시킨 것을 다시 문자로
    sorted_indices = doc_tfidf.argsort()[::-1] # tf-idf값이 높을수록 중요한 키워드인데 높은 순으로 정렬
    top_keywords = [tfidfv.get_feature_names_out()[idx] for idx in sorted_indices[:15]]  # 상위 10개 단어만 추출 # 숫자나 영어 단어만 나올 경우 제거하고 다음 키워드를 올려서 쓸거기 때문에 넉넉히 뽑음
    # 키워드를 저장할 빈 리스트 생성
    keywords = []
    keywords.append(top_keywords)
    # 각 행에 알맞은 키워드 부여
    df1['TF-keywords'][i] = keywords
print(df1['TF-keywords'][:10])

```

## 2-3 제목과 본문과의 코사인유사도를 통해 최종 키워드 선정
앞에서 2가지 방법으로 구한 키워드를 합쳐서 최종 키워드 선별 과정을 추가하였습니다.
추출된 키워드들과 제목과 본문과의 코사인유사도를 구한 다음 가장 유사도가 높은 키워드들로 선정을 하였습니다. 
코사인 유사도를 계산하기 위해서 제목과 본문을 벡터화 하는 과정을 진행하였고 두 벡터값을 가중합(제목=0.7, 본문=0.3)하여 의미 벡터를 생성하였습니다. 이를 통하여 의미 벡터와 키워드 벡터간의 코사인 유사도 계산을 가능하게 하였습니다.

```python
for i in range(len(df)):
  #  제목, 본문, 키워드 후보 데이터
  title = df['title_p'][i]
  body = df['content_p'][i]
  keyword_candidates = df['merged_col'][i]

  # TF-IDF 벡터화 객체 생성
  tfidf_vectorizer = TfidfVectorizer()

  # 제목과 본문을 따로 TF-IDF 벡터화
  title_vector = tfidf_vectorizer.fit_transform([title])
  body_vector = tfidf_vectorizer.transform([body])

  # 키워드 후보들을 TF-IDF 벡터화
  keyword_candidate_vectors = tfidf_vectorizer.transform(keyword_candidates)

  # 제목과 본문의 TF-IDF 벡터를 가중합(Weighted Sum) 및 정규화(Normalization)하여 의미 벡터 생성
  alpha = 0.7  # 제목 벡터 가중치 (0과 1 사이의 값을 선택)
  beta = 0.3   # 본문 벡터 가중치 (0과 1 사이의 값을 선택)

  title_vector = title_vector.toarray()
  body_vector = body_vector.toarray()
  title_body_vector = alpha * title_vector + beta * body_vector

  # 정규화(Normalization)하여 의미 벡터를 단위 벡터로 변환
  title_body_vector = normalize(title_body_vector)[0]

  # 키워드 후보 벡터와의 의미 유사도를 측정하여 Top-K개의 키워드 선정
  k = 3  # Top-K 개수를 선택해주세요

def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  return dot_product / (norm_vec1 * norm_vec2)

similarity_scores = [cosine_similarity(title_body_vector, candidate.toarray()[0]) for candidate in keyword_candidate_vectors]
top_k_indices = np.argsort(similarity_scores)[-k:]
selected_keywords = [keyword_candidates[i] for i in top_k_indices]

print("선택된 키워드:")
for keyword in selected_keywords:
    print(keyword)
```

# 3-1. 대주제 탐색(DBSCAN)
당일 수 많은 뉴스들 사이에서 대표적인 주제들을 선정하기 위해서 군집화 기술 중 하나인 DBSCAN을 사용하였습니다. 군집화에서 K-MEANS 대신 DBSCAN을 사용한 이유는 K-MEANS의 경우 사전에 군집의 개수를 설정해줘야만 한다는 문제점이 있었기 때문입니다. 매일매일 대주제의 개수가 정해져있는 것이 아니기 때문에 사전에 군집의 개수를 설정하지 않고 자동으로 군집의 개수를 정하는 DBSCAN을 선택하였습니다. 
DBSCAN의 경우 각각 제목과 키워드 2가지 방식으로 군집화를 실시하였고 그 중 결과가 더 나은 키워드로 군집화한 것을 선택하였습니다. 
* DBSCAN의 하이퍼파라미터의 경우(eps,min_samples) 여러 값을 시도해보며 군잡화된 결과물을 직접 눈으로 보고 팀원들과의 상의 후 결정하였습니다.

<tf-idf 임베딩(+normalize)>

```python
from sklearn.feature_extraction.text import TfidfVectorizer

#text = [" ".join(noun) for noun in df['final_key']]
text = df['title_p']
tfidf_vectorizer = TfidfVectorizer(min_df = 3, ngram_range=(1,5))
tfidf_vectorizer.fit(text)
vector = tfidf_vectorizer.transform(text).toarray()

vector = np.array(vector)
```

<DBSCAN>

```python

2 DBSCAN Clustering

from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.6,min_samples=12, metric = "cosine")
     거리 계산 식으로는 Cosine distance를 이용
     eps이 낮을수록, min_samples 값이 높을수록 군집으로 판단하는 기준이 까다로움.
result = model.fit_predict(vector)
df['result'] = result

print('군집개수 :', result.max())
df

```
<DBSCAN 결과 시각화>

![image](https://github.com/Yoon-Hee-Jae/bitamin_project/assets/140389762/512a0674-c2aa-48d4-9fcc-12baa7305d01)


<군집화된 기사의 제목>

![image](https://github.com/Yoon-Hee-Jae/bitamin_project/assets/140389762/633a0ba3-2447-4450-ba7a-5c124ff15b9c)

비슷한 주제의 기사들이 군집화된 것을 볼 수 있음







# 3-2 대주제 탐색 - keybert
군집화 후 각 군집속에 포함된 기사의 제목들을 모아 keybert를 사용하여 대주제 키워드를 선정하였습니다. 

```python

## title로 한단어~두단어 키워드 출력
for i in range(1,10):
    dff = df[df['cluster_num']==i]['title']
    dff = ' '.join(dff)
    morphemes = okt.pos(dff)
    filtered_words = [word for word, pos in morphemes if pos in ['Noun', 'Number', 'Alpha', 'Suffix']]
    filtered_words = ' '.join(filtered_words)
    kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
    keywords = kw_model.extract_keywords(filtered_words,
                                    keyphrase_ngram_range=(1,2),
                                    use_maxsum=True,
                                    nr_candidates = 20,
                                    top_n = 3,
                                    use_mmr = True,
                                    diversity=0.7)
    print(keywords)
    result = []
    result.append(keywords)

```

# 4. 기사 요약
군집화된 개수만큼 총 8개의 기사 요약문을 생성하기로 하였습니다.
요약문의 경우 각 군집에 속한 기사를 각각 요약한 후 기사 개수만큼 만들어진 요약문을 합쳐서 다시 요약을 진행하였습니다.

```python

for i in range(len(news_lst)):
  # 뉴스요약

    tokenizer =  PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
    model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")

    inputs_ids = tokenizer.encode(data['content'][i], return_tensors="pt")

# kobert가 정해둔 데이터 크기의 한계가 있어서 텍스트가 긴 뉴스같은 경우는 잘라서 진행
# 모델 입력 최대값 구하는 코드
    max_input_length = model.config.max_position_embeddings

    inputs_ids = inputs_ids[:, :max_input_length]

    summary_text_ids = model.generate(
        input_ids = inputs_ids,
        bos_token_id = model.config.bos_token_id,
        eos_token_id = model.config.eos_token_id,
        length_penalty = 2.0,
        max_length = 128,
        min_length = 32,
        num_beams = 4)
  # 요약한 값을 summary라는 새 열에 저장
    #data['summary'][i] = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)

```




