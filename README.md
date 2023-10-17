# Fandom2Vec: Uncovering the types of K-pop fandoms with network embedding

## 연구 요약
1) Background
- 팬덤은 매우 충성적인 고객 집단으로(Tanner & Brown, 2001), 높은 구매 능력과 지속적인 소비 및 시장 내 소비 트렌드의 주도로 인해 그 상업적 가치가 매우 큼(Liang & Shen, 2016; J. Zhang, 2022)
- 최근, 스포츠나 엔터테인먼트 산업과 같은 전통적인 fan-driven 비즈니스 외에 전자기기나 패션 등 다양한 산업들도 팬덤을 새롭게 구축하고 효율적으로 관리하기 위해 노력하고 있음(Lee & Jung, 2018)
- 팬들은 일반적인 충성 고객들과는 달라 전통적인 충성 고객 연구로 설명될 수 없는 부분이 존재(Roşca, 2013; Xu, 2022; Yin, 2020)
2) Motivation
- 팬덤 연구는 팬덤을 하나의 문화적 현상으로서 관찰하는 정성적 문화 연구에 초점이 맞춰져 있음(Bacon-Smith, 1992; Jenkins, 2012b; Yun et al., 2021)
- 팬덤은 그 대상에 따라 다른 특성을 보이므로(Exsha et al., 2022; Wang et al., 2021), 다양한 유형의 팬덤을 이해하는 것이 중요
- 팬을 유형화하는 연구들(Dimmock & Grove, 2005; Hunt et al., 1999; Jaeger, 2021)은 있지만, 팬덤을 유형화하는 연구는 거의 없음
- 팬들의 사회적 상호작용에 초점을 맞춰, 각 팬덤의 고유한 특성을 소셜 네트워크 관점에서 정의하는 사례가 늘어남(Loria et al., 2021; Tuguinay et al., 2022)
3) Purpose
- 팬덤의 소셜 네트워크 정보를 사용하여 fandom taxonomy를 구축할 수 있는 접근 방식을 제안
- K-pop 팬덤을 사례로 팬덤 유형화 분석을 수행하고, 해석하여 K-pop 팬덤 산업에 managerial implication을 제공

## 연구 프레임워크
![image](https://github.com/jinjinzara/twitter-fandom-analysis/assets/82082271/a33d9233-d450-482d-8e1f-5a05af47825e)

## Fandom2Vec
### Data collection
- 데이터 출처: Twitter
- 수집 방법
1. 트위터 팔로워 수 기준 K-pop 아티스트 상위 100팀 선정(2022년 11월 9일 기준)
  - 출처: https://kpopping.com/database/group/twitter-followers
    
2. Twitter API v2로 한 팀당 공식 계정 팔로워(팬) 5000명의 계정 정보 수집

  (1) 2022년 5월 1일 이전에 생성된 계정
  
  (2) 2022년 5월 1일에서 10월 31일 사이 트윗이 존재
  
  (3) 비인증 계정
  
  - 위의 세 가지 조건을 만족하는 계정만 필터링
    
3. 모든 팬덤에 대해 팬을 각 100명씩 random sampling

### Network construction
- 팬덤 내 팬들의 멘션 관계 수집
  - 멘션(Mention): 트윗(게시글)을 올릴 때 다른 사용자의 unique id를 언급해서 기재하는 트위터 기능
- 수집 방법
1. 팬들이 올린 트윗 중에서 멘션 기능을 사용한 것만 필터링
2. 멘션된 유저가 멘션한 유저와 같은 팬덤에 소속되어 있는지 확인
  - K-pop 아티스트의 공식 계정 팔로워 = 팬
3. 같은 팬덤 내에서 발생한 멘션 상호작용일 경우, 팬덤 네트워크에 edge 추가

### Network embedding
- graph2vec: https://github.com/benedekrozemberczki/graph2vec
- 하이퍼 파라미터
```Python
dimensions: 16. 임베딩 벡터의 차원 수.
workers: 5. 모델 훈련 시 스레드 수. 높을수록 훈련 속도 증가.
epochs: 20. 전체 데이터 학습 횟수.
min_count: 5. 최소 빈도 수. 해당 수보다 적게 나온 토큰은 학습에서 제외.
wl_iterations: 5. subgraph에서 합칠 이웃 노드의 수. 네트워크가 클수록 큰 값으로 설정.
learning rate: 0.015. 학습률.
```
