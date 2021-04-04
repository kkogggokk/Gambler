# Gambler
## Pockemon datacheck
[포켓몬도감](https://www.kaggle.com/terminus7/pokemon-challenge/discussion/40057)
[포켓몬위키](https://pokemon.fandom.com/ko/wiki/비조푸_(포켓몬))
```
포켓몬 데이터를 통해서 두 포켓몬의 배틀 결과를 예측하자
```

### Data check

- 각 타입별 분포
  - HP
  - Attack
  - Defence
  - Sp. Atk	
  - Sp. Def	
  - Speed	

- Type 1, Type 2 별
  - HP
  - Attack
  - Defence
  - Sp. Atk	
  - Sp. Def	
  - Speed	

- Data type 별
  - categorical
  - numeric



## EDA

- Null check
  - Null 비율
- y와의 관계
  - 승패 / 승률 
    - generation 별 승률
    - Tpye 1, 2 별 승률
    - Stat 별 승률
      - general
        - HP
        - Attack
        - Speed
      - Special
        - Sp. Atk
        - Sp. Def



## Feature Engineering

- Null
  - 'Name' 처리
  - 'Type 2' 결측치 처리  
- Scaling
  - pipeline 사용
- Encoding
  - Label Encoding
    - 추후 적용 모델
      - Decisiontree
      - Randomforest
      - XGBoost
      - Lightgbm
      - Catboost 
  - One Hot Encoding
    - SVM
    - KNN
  - ID -> 카테고리화: int -> object
    - First_pokemon
    - Second_pokemon
    - Winner -> y 값 처리
  - Label Encoding

- 정규화
  - 컬럼별 데이터분포를 그려보고 skewed되어있다면, log정규화 취하기 


- interaction-feature 생성 (idea)
  - 'First attack' : 먼저 공격 여부 true(1), false(0)
  - 'total stat' : 



## Modeling

- pipeline
  - scaler
    - standard
    - min-max
  - model
    ```
      1. combat 데이터프레임만으로 예측하기
      2. merge한 데이터프레임으로 예측하기
    ```
    - SVM / KNN (진명)
    - Randomforest / Decision tree (현아)
    - XGBoost(희진)
    - Lightgbm(종해)
  - cv
    - 그리드
    - 랜덤
    - 베이지안 
- Ensemble
- Stacking



## Evaluation

- Precision / Recall
  - ROC AUC
- Feature importance
  - feature engineering

