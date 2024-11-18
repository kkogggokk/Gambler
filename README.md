# 포케몬스터 승률 예측
# 🔹 프로젝트 요약
| 항목 | 내용 |
| --- | --- |
| 기간 | 2021.03.29 ~ 2021.04.09 (2주)  |
| 팀원 | 문종해, 송현아, 안희진, 현진명 |
| 주제 | 포켓몬스터의 승률 예측 모델 설계  |
| 내용 | 포켓몬 대결 데이터를 기반으로 승률이 높은 특성을 분석하여 95% 이상의 예측 정확도를 목표로 모델을 설계하였습니다. 모델은 KNN, SVM, Decision Tree, Random Forest, XGBoost 을 사용하였습니다. 기법으로 Feature Engineering과 Ensemble 을 통해 650회 이상 실험하여 최적 조합 도출했습니다. 최종적으로 0.9768 정확도를 달성했습니다. |
| 결과 | 최고 정확도: 0.9768(XGBoost) |
| 기술스택 | - 데이터 처리 및 분석: Pandas, Numpy, Seaborn, Matplotlib     - 엔지니어링: Label Encoding, Feature Engineering    - 모델링: : KNN, SVM, Decision Tree, Random Forest, XGBoost |
| 코드URL | [포켓몬스터 승률 예측 프로젝트](https://github.com/kkogggokk/Gambler)|

# 🔹 1.Data Check

데이터 출처: [캐글 Pokemon](https://www.kaggle.com/datasets/terminus7/pokemon-challenge/data)

- 포켓몬 정보 :  pokemon.csv    
    <img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/pokemon-csv.png" alt="포켓몬 정보" width="600">
- 포켓몬 대결정보 : combat.cvs    
  첫번째 포켓몬(`First_pokemon`)과 두번째 포켓몬(`Second_pokemon`) 대결한 결과, 이긴 포켓몬(`Winner`) 대해 알 수 있다.    
  예를 들어 266포켓몬 vs 298 포켓몬 대결하여 결과는? 298 승리     
    <img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/combat-csv.png?raw=true" alt="포켓몬 대결 정보" width="600">    

    

- 정리: [Pandas Profiling Report]()

# 🔹2.Data Preprocessing
## 2.1 null check
- 파일: pokemon.csv    
    Name, Type 필드의 값이 800개가 안되는 상황
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/nullCheck.png" alt="Null Check" width="600">

- ***Name 필드 (799개)***  
isnull로 null값 확인 시, 63번 포켓몬 이름이 없다. 1세대 포켓몬 중 격투 포켓몬에서 해당 능력치를 갖고 있는 포켓몬을 찾아보자.    
62번 포켓몬이 `망키`(Mankey)이므로 망키의 진화 포켓몬인 `성원숭`(Primeape)에 대해 찾아보자.     
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/nullCheck_name(1).png" alt="성원숭 찾기" width="600">
Null값에 해당하는 능력치값들과 성원숭의 능력치가 동일하다.    
이를 통해 null 값에 해당하는 포켓몬은 `성원숭`이다.     
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/nullCheck_name(2).png" alt="성원숭 확인" width="600">

- ***Type2 필드 (414개)***    
Type1만 있는 것은 순수 포켓몬을 의미한다. 따라서 Type2 컬럼에 Null값은 순수 포켓몬을 의미하므로 null값이 있다는 것에 큰 문제가 되지 않는다.     

## 2.2 승률 영향 미치는 컬럼 조사
승률에 영향을 미치는 컬럼들에 대해 조사를 해보자.     

- ***win_percentage column***    
각각의 포켓몬에 대해 승률를 구해보자. 전체 배틀(`Total Fight`)에서 이긴횟수(`numberOfWins`)를 나눠 승률(`WinPercentage`)컬럼을 생성한다.     
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/Win_persentage(1).png" alt="Win Percentage 계산" width="600">

- ***Seaborn pairplot***    
승률과 각 컬럼들의 관계를 확인할 수 있는 Seaborn 확인. 속도(`Speed`)컬럼에서 비례관계    
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/Seaborn.png" alt="Seaborn 관계" width="600">    

- ***boxplot***    
일반포켓몬과 전설의 포켓몬을 비교하면 전설의 포켓몬 속성이 승률 예측력을 높이는데 도움이 될 것으로 예상     

***결론***
- `속도`에 따른 승률
- `전설의 포켓몬` 여부에 따른 승률    
    <img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/result(1).png" alt="결론" width="600">

## 2.3 Label Encoding
Categorical 데이터나 boolean 데이터를 ***수치 데이터로 변환***하여 머신러닝 모델에서 활용할 수 있도록 하기 위해서 라벨인코딩을 진행    

- Type1, Type2 필드 타입 : Categorical    
- Legendary 필드 타입 : Boolean    
- 파일: `Pokemon_encoded.csv`     

## 2.4 Merge dataframe

포켓몬의 속성을 통해 승률을 예측하기 위해서 대결 정보(`combat.csv`)파일과 포켓몬정보(`pokemon.csv`)파일을 합치는 작업을 진행    
- 파일: Pokemon_df.csv

# 🔹 3. Modeling
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/Model.png" alt="모델링 과정" width="600">

- STEP1. Modeling(모델별 성능 확인)
- STEP2. Feature Combination (특징 조합)
- STEP3. Ensemble(앙상블)

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
<img src="https://raw.githubusercontent.com/kkogggokk/Gambler/refs/heads/main/__backup/model-knn.png" alt="KNN 성능" width="600">
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/model-svm.png?raw=true" alt="SVM 성능" width="600">
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/model-decisionTree.png?raw=true" alt="Decision Tree 성능" width="600">
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/model-RF.png?raw=true" alt="Random Forest 성능" width="600">
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/model-XGBoost.png?raw=true" alt="XGBoost 성능" width="600">

## STEP2. **Feature Combination (특징 조합)**
| 항목 | 내용 |
| --- | --- |
| 유효한 필드 | `Speed`, `Attack`, `Defense`, `Generation`, `Type`, `legendary`  |

<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/Feature%20combination.png?raw=true" alt="Feature Combination" width="600">

[총 650회 실험](https://docs.google.com/spreadsheets/d/1TMqu_tlFWc4r4BlzY15iZgir4Izm0YE_IvIjJaCEfoM/edit?gid=305708277#gid=305708277)

## STEP3. **Ensemble(앙상블)**

```jsx
- Precision / Recall
    - ROC-AUC
- Feature importance
    - Feature engineering
```

### XGBoost Model Best Score : 0.9768

<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/model_result-XGBoost_9768.png?raw=true" alt="XGBoost 성능" width="600">

### Ensemble model best score : 0.9567

<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/model_result_Ensemble_.png?raw=true" alt="Ensemble 성능" width="600">

# 🔹 4. 추가 활동

- Q. 지우는 피카츄를 대결에 잘 내보내지 않는가? 약해서 일까?    
A. 약하지 않다. (피카츄는 단지 평화주의자)     
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/Q1_pikachu.png?raw=true" alt="피카츄 성능" width="300">
    
타입별  승률 비교 (단, 순수한 포켓몬(Type1) 대결 매치)    
- 물 포켓몬 vs 아이스 포켓몬 ? 물 포켓몬       
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/Q2_water_ice.png?raw=true" alt="물 vs 아이스" width="300">
        
- 유령 포켓몬 vs 격투 포켓몬 ? 유령 포켓몬    
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/Q3.Ghost_fight.png?raw=true" alt="유령 vs 격투" width="300">
        
- 물 포켓몬 vs 불 포켓몬 ? 불 포켓몬    
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/Q4.water_fire.png?raw=true" alt="물 vs 불" width="300">
        
- 일반 포켓몬 vs 전설의 포켓몬 ? 전설의 포켓몬        
그러나 승률 TOP10에는 일반포켓몬이다. 희망을 갖자    
<img src="https://github.com/kkogggokk/Gambler/blob/main/__backup/Q5.legendary(1).png?raw=true" alt="전설의 포켓몬" width="300">

# 🔹 5. 발표영상
<iframe width="560" height="315" src="https://www.youtube.com/embed/0TElfPy0alY?si=ymvwBsHDlv5y2f8P" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
