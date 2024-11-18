# 포케몬스터 승률 예측
# 요약

| 기간 | 2021.03.29 ~ 2021.04.09 (2주)  |
| --- | --- |
| 팀원 | 문종해, 송현아, 안희진, 현진명 |
| 주제 | 포켓몬스터의 승률 예측 모델 설계  |
| 내용 | 포켓몬 대결 데이터를 기반으로 승률이 높은 특성을 분석하여 ***95% 이상의 예측 정확도***를 목표로 모델을 설계하였습니다. 모델은 KNN, SVM, Decision Tree, Random Forest, XGBoost 을 사용하였습니다. 기법으로 Feature Engineering과 Ensemble 을 통해 650회 이상 실험하여 최적 조합 도출했습니다. 
최종적으로 0.9768 정확도를 달성했습니다. |
| 결과 | 최고 정확도: 0.9768(XGBoost) |
| 기술스택 | - 데이터 처리 및 분석: Pandas, Numpy, Seaborn, Matplotlib
- 엔지니어링: Label Encoding, Feature Engineering 
- 모델링: : KNN, SVM, Decision Tree, Random Forest, XGBoost |
| 코드URL | https://github.com/kkogggokk/Gambler |

# Data Check

데이터 출처: [캐글 Pokemon](https://www.kaggle.com/datasets/terminus7/pokemon-challenge/data)

- 포켓몬 정보 :  pokemon.csv
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image.png)
    
    | 필드명 | 내용 | 형태 |
    | --- | --- | --- |
    | # | 포켓몬 번호 | int |
    | Name | 포켓몬 이름 | String |
    | Type1 | 속성1 | Categorical |
    | Type2 | 속성2 | Categorical |
    | HP | 체력 | int |
    | Attack | 공격력 | int |
    | Defense | 방어력 | int |
    | Sp.Atk | 공격 스피드 | int |
    | Sp.Def | 방어 서피드 | int |
    | Speed | 스피드 | int |
    | Generation | 포켓몬 세대 | int |
    | Legendary  | 전설의 포켓몬 여부 | Boolean |
- 포켓몬 대결정보 : combat.cvs
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%201.png)
    
    첫번째 포켓몬(`First_pokemon`)과 두번째 포켓몬(`Second_pokemon`) 대결한 결과, 이긴 포켓몬(`Winner`) 대해 알 수 있다. 
    
    예를 들어 266포켓몬 vs 298 포켓몬 대결하여 결과는? 298 승리 
    

- 정리: Pandas Profiling Report

# Data Preprocessing

## null check

파일: pokemon.csv

![Screenshot 2024-11-18 at 3.19.10 PM.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/Screenshot_2024-11-18_at_3.19.10_PM.png)

### Name : 799 / 800

isnull로 null값 확인 시, 63번 포켓몬 이름이 없다. 1세대 포켓몬 중 격투 포켓몬에서 해당 능력치를 갖고 있는 포켓몬을 찾아보자. 62번 포켓몬이 `망키`(Mankey)이므로 망키의 진화 포켓몬인 `성원숭`(Primeape)에 대해 찾아보자. 

![Picture1.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/Picture1.png)

Null값에 해당하는 능력치값들과 성원숭의 능력치가 동일하다. 이를 통해 null 값에 해당하는 포켓몬은 [성원숭](https://pokemon.fandom.com/ko/wiki/%EC%84%B1%EC%9B%90%EC%88%AD_(%ED%8F%AC%EC%BC%93%EB%AA%AC))이다. 

![Screenshot 2024-11-18 at 1.29.20 PM.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/Screenshot_2024-11-18_at_1.29.20_PM.png)

### Type2 : 414 / 800

Type1만 있는 것은 순수 포켓몬을 의미한다. 따라서 Type2 컬럼에 Null값은 순수 포켓몬을 의미하므로 null값이 있다는 것에 큰 문제가 되지 않는다. 

## 승률 영향 미치는 컬럼 조사

승률에 영향을 미치는 컬럼들에 대해 조사를 해보자. 

### win_percentage column

각각의 포켓몬에 대해 승률를 구해보자. 전체 배틀(`Total Fight`)에서 이긴횟수(`numberOfWins`)를 나눠 승률(`WinPercentage`)컬럼을 생성한다. 

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%202.png)

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%203.png)

### Seaborn pairplot

승률과 각 컬럼들의 관계를 확인할 수 있는 Seaborn 확인. 속도(`Speed`)컬럼에서 비례관계

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%204.png)

### boxplot

일반포켓몬과 전설의 포켓몬을 비교하면 전설의 포켓몬 속성이 승률 예측력을 높이는데 도움이 될 것으로 예상 

결론

- `속도`에 따른 승률
- `전설의 포켓몬` 여부에 따른 승률
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%205.png)
    

## Label Encoding

Categorical 데이터나 boolean 데이터를 ***수치 데이터로 변환***하여 머신러닝 모델에서 활용할 수 있도록 하기 위해서 라벨인코딩을 진행

- Type1, Type2 필드 타입 : Categorical
- Legendary 필드 타입 : Boolean

결과파일: `Pokemon_encoded.csv` 

## Merge dataframe

포켓몬의 속성을 통해 승률을 예측하기 위해서 대결 정보(`combat.csv`)파일과 포켓몬정보(`pokemon.csv`)파일을 합치는 작업을 진행

결과파일: Pokemon_df.csv

# Modeling

![Model.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/Model.png)

- **STEP1. Modeling(모델별 성능 확인)**
- **STEP2. Feature Combination (특징 조합)**
- **STEP3. Ensemble(앙상블)**

```jsx
- pipielince
    - Scaler
        - standard
        - min-max
    - Model
        - SVM / KNN
        - Randomforest / Decision Tree
        - XGBoost
    - cv
        - 그리드
        - 랜덤
        - 베이지안
- Ensemble 앙상블
- Stacking
```

## STEP1. **Modeling(모델별 성능 확인)**

### KNN

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%206.png)

### SVM

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%207.png)

### Decision Tree

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%208.png)

### Random Forest

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%209.png)

### XGBoost

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2010.png)

## STEP2. **Feature Combination (특징 조합)**

| 결과 파일 | [model test sheet](https://docs.google.com/spreadsheets/d/1TMqu_tlFWc4r4BlzY15iZgir4Izm0YE_IvIjJaCEfoM/edit?gid=305708277#gid=305708277) |
| --- | --- |
| 유효한 필드 | `Speed`, `Attack`, `Defense`, `Generation`, `Type`, `legendary`  |

총 650회 실험

![Screenshot 2024-11-18 at 2.58.05 PM.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/Screenshot_2024-11-18_at_2.58.05_PM.png)

## STEP3. **Ensemble(앙상블)**

```jsx
- Precision / Recall
    - ROC-AUC
- Feature importance
    - Feature engineering
```

### XGBoost Model Best Score : 0.9768

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2011.png)

### Ensemble model best score : 0.9567

![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2012.png)

# 추가 활동

- Q. 지우는 피카츄를 대결에 잘 내보내지 않는가? 약해서 일까?
    
    A. 약하지 않다. (피카츄는 단지 평화주의자) 
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2013.png)
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2014.png)
    
- Q. 어느 타입이 제일 쎈가?
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2015.png)
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2016.png)
    
- Q. 타입별  승률 비교
    
    조건1. 순수한 포켓몬(Type1) 대결 매치
    
    ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2017.png)
    
    - 물 포켓몬 vs 아이스 포켓몬
        
        물 포켓몬 
        
        ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2018.png)
        
    - 유령 포켓몬 vs 격투 포켓몬
        
        유령 포켓몬 
        
        ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2019.png)
        
    - 물 포켓몬 vs 불 포켓몬
        
        불 포켓몬 
        
        ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2020.png)
        
    - 일반 포켓몬 vs 전설의 포켓몬
        
        전설의 포켓몬 
        
        그러나 승률 TOP10에는 전설의 포켓몬이 없다. 희망을 갖자
        
        ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2021.png)
        
        ![image.png](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/image%2022.png)
        

# 발표영상

[pockemon_presentation.mp4](%E1%84%91%E1%85%A9%E1%84%8F%E1%85%A6%E1%84%86%E1%85%A9%E1%86%AB%E1%84%89%E1%85%B3%E1%84%90%E1%85%A5%20%E1%84%89%E1%85%B3%E1%86%BC%E1%84%85%E1%85%B2%E1%86%AF%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%201083ab0d217080f2af98d4ffdba92b6a/pockemon_presentation.mp4)
