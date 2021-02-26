# Spark-ML-Study
https://spark.apache.org/docs/latest/ml-guide.html

## Spark-3.0.2 Install
https://spark.apache.org/downloads.html

~~~
$ tar -xf spark-3.0.2-bin-hadoop2.7.tgz
~~~
~~~
$ cd spark-3.0.2-bin-hadoop2.7
$ bin/pyspark
~~~
Move the directory into /usr/local/
~~~
$ sudo mv spark-3.0.2-bin-hadoop2.7/ /usr/local/spark
~~~
You can access WebUI with following address   
0.0.0.0:4040

~~~
$ pip install pyspark
~~~

## Pipelines

### 구성요소
#### Dataframe   
Spark SQL의 DataFrame을 다양한 데이터 유형을 보유 할 수있는 ML 데이터 세트로 사용합니다.   
#### Transformer   
Transformer는 기능 변환기 및 학습 된 모델을 포함하는 추상화입니다.   
원시 데이터를 다양한 방식으로 변환하는 메서드입니다.   
데이터프레임에 새로운 칼럼을 추가하고 데이터를 변형합니다.   
##### spark.ml.feature에 제공되는 트랜스포머 종류   
  - Binarizer: 이 함수는 주어진 임계치를 기준으로 연속적인 변수를 이진 변수로 변환한다.   
  - Bucketizer: Binarizer와 비슷하다. 이 함수는 연속적인 변수를 주어진 임꼐치의 리스트를 기반으로 쪼개어 몇 개의 범위로 변환한다.   
  - ChiSqSelector: 이 함수는 모든 카테고리 변수들 중, 파라미터로 주어진 numTopFeatures개의 카테고리 변수들을 선택한다.   
     여기서 선택된 변수들은 타깃의 분산을 잘 나타내는 변수들이다. 이는 차이-스퀘어 데스트를 통해 가능하다.   
  - CountVectorizer: 이 함수는 [['Leaning','Pyspark'],['us']] 와 같은 분리된 텍스트에 유용하다.   
    우선 fit()을 수행해 데이터셋의 패턴을 학습하고 그 결과를 CountVectorizerModel로 변형한다.
  - DCT(Discrete Cosine Transform): 이 함수는 실수로 이뤄진 벡터를 입력으로 받고, 다른 빈도로 진동하는 같은 길이의 벡터를 리턴한다.   
    데이터셋에서의 기본 빈도를 추출하거나 데이터를 압축할 때 유용하다.   
  - ElementwiseProduct: 이 함수는 이 함수에 전달된 벡터와 scalingVec 파라미터를 곱한 것을 리턴하는 함수다.   
    예를 들어 [10.0,3.0,15.0]을 벡터로 전달하고 scalingVec파라미터가 [0.99,3.30,0.66]라면 [9.9,9.9,9.9]를 리턴한다.
  - HashingTF: 분리된 텍스트를 리스트로 입력받아서 카운트 벡터를 리턴하는 Hashing Trick Transformer이다.   
  - ID(Inverse Document Frequency): 주어진 도큐먼트 리스트에 대한 IDF값을 구한다.   
    도큐먼트는 HashingTF나 CountVectorizer을 이용해 미리 벡터로 표현돼 있어야 한다.   
  - IndexToString: 이 함수는 StringIndexer 함수에 대한 보완이다.   
  - MaxAbsScaler: 데이터를 [-1.0,1.0]범위 사이로 재조정한다.
  - MinMaxScaler: 이는 [0.0,1.0]범위 사이로 재조정한다.
  - NGram: 분리된 텍스트를 입력으로 받아서 n-gram을 리턴한다. (2,3,n개의 단어를 합쳐줌.)   
  - Nomalizer: p-norm 값을 이용해 데이터를 단위 크기로 조정한다.(기본값으로 p를 2로 설정해 L2를 사용함.)   
  - OneHotEncoder: 카테고리 칼럼을 이진 벡터 칼럼으로 인코딩한다.   
  - PCA(Principal Component Analysis): 데이터 축소를 수행한다.   
  - PolynomialExpansion: 한 벡터에 대해 다항 확장을 수행한다.   
  - QuantileDiscretizer: Bucketizer 함수와 비슷하나 splits파라미터를 전달하는 대신에 numBuckets라는 파라미터를 전달한다.   
    대략적인 데이터의 양을 계산해 어느 정도로 나눌 것인지를 정한다.   
  - RegexTokenizer: 정규 표현식을 이용한 스트링 분리기이다.
  - StandardScaler: 칼럼이 평균0, 표준편차1인 표준정규분포를 갖도록 한다.   
  - StopWordsRemover: 분리된 텍스트로부터 'the'나 'a'같은 단어들을 제거한다.   
  - StringIndexer: 한 칼럼에 주어진 모든 워드 리스트에 대해 이 함수는 인덱스 벡터를 생성한다.   
  - Tokenizer: 스트링을 소문자로 변환하고 스페이스를 기준으로 분리하는 함수이다.   
  - VectorAssembler: 여러 개의 숫자 칼럼을 벡터 형태의 한 칼럼으로 변환해주는 **아주 유용한** 트랜스포머이다.   
  - VectorIndexer: 카테고리 칼럼을 벡터 인덱스로 변환하는 데 쓰인다.   
  - VectorSlicer: dense든 sparse든 관계없이 피처 벡터에 대해 동작한다. 주어진 인덱스 리스트에 대해 피처 벡터의 값을 추출한다.   
  - Word2Vec: 스트링 문자를 입력으로 취해 {스트링,벡터} 형태로 변형한다.   
   
#### Estimator   
훈련용 데이터 세트에 머신 러닝 모델을 훈련시키거나 적합하도록 조정합니다.   
Estimator는 DataFrame을 받아들이고 Transformer 인 Model을 생성하는 fit() 메서드를 구현합니다.   
##### 제공하는 일곱 개의 분류 모델
  - LogisticRegression: 로지스틱 회귀는 데이터가 특정 클래스로 속하는 확률을 구하기 위해 로지스틱 함수를 사용한다.(현재 이진분류를 지원함.)   
  - DecisionTreeClassifier: 관찰된 데이터에 대한 클래스를 예측하는 결정트리 모델을 생성한다.   
  - GBTClassifier: Gradient Boosted Tree모델이다. 여러 개의 약한 모델들을 뭉쳐서 강한 모델을 만들어내는 앙상블 모델 그룹에 속한다.   
  - RandomForestClassifier: 여러 개의 결정 트리를 만들고, 그 결정 트리들의 결과들을 예측 값으로 사용한다.   
  - NaiveBayes: 베이즈 이론에 기반하여 데이터를 분류하기 위해 조건부 확률을 사용한다.    
  - MultilayerPerceptronClassifier: 입력층과 은닉층에 있는 신경은 시그모이드 활성 함수를 이용, 출력층에서는 소프트맥스 함수를 이용한다.      
  - OneVsRest: 여러 클래스에 대한 분류를 이진 분류로 축소한다.   
##### 제공하는 일곱 개의 회귀 모델
  - AFTSurvivalRegression: Accelerated Failure Time 모델을 학습니다.   
  - DecisionTreeRegressor: 결정 트리 분류 모델과 비슷하나 레이블이 이진 데이터가 아니라 연속적인 데이터이거나 여러개의 레이블을 가지는 데이터다.   
  - GBTRegressor: DecisionTreeRegressor에서처럼 레이블 타입만 다르다.   
  - GeneralizedLinearRegression: 다른 커널 함수를 사용하는 선형 모델이다.(여러 연결 함수로 가우시안, 이진, 감마, 푸아송 분포에 대한 오차 분포를 지원함.)   
  - IsotonicRegression: 감소하지 않는 자유로운 형태의 데이터에 대한 회귀모델이다.(정렬되고 증가하는 형태의 데이터셋에 유용함.)   
  - LinearRegression: 가장 간단한 회귀 모델이다. 피처들 사이의 관계가 선형이고 레이블이 연속적인 값이며 오차가 정규 분포를 띈다는 것을 가정한 모델이다.   
  - RandomForestRegerssor: 분리된 값이 아닌 연속적인 값들에 대해 학습한다.(DecisionTreeRegressor나 GBTRegressor와 비슷함.)   

##### 제공하는 대표적인 군집화 모델
  - BisectingKMeans: 계층적 군집화와 K-평균 군집화 알고리즘의 조합이다. 모든 데이터를 하나의 군집으로 놓고 시작하며 반복적으로 K개의 군집으로 나눈다.   
  - KMeans: 각각의 데이터와 군집 사이 거리 제곱의 합을 최소화하는 중심을 반복적으로 계산하면서 가장 최적화된 K개의 중심을 찾아가는 모델이다.   
  - GaussianMixture: 이 함수는 데이터셋을 분석하기 위해 K 가우시안 분포를 알려지지 않은 파라미터와 같이 사용한다.   
  - LDA: 자연어 처리에서 토픽 모델링을 하는 데에 사용한다.   
   
#### Pipeline   
파이프 라인은 여러 Transformer와 Estimator을 함께 연결하여 ML workflow를 지정합니다.   
ML workflow는 데이터 프로세싱, 특성 추출, 모델 훈련까지 구성하는 것입니다.   
머신 러닝에서는 데이터를 처리하고 학습하기 위해 일련의 알고리즘을 실행하는 것이 일반적입니다.   
MLlib는 특정 순서로 실행될 일련의 PipelineStage (Transformer 및 Estimator)로 구성된 Pipeline과 같은 Workflow를 제공합니다.   
예를 들어 간단한 텍스트 문서 처리 워크플로에는 다음의 단계가 포함될 수 있습니다.   
  - 각 문서의 텍스트를 단어로 분할합니다.   
  - 각 문서의 단어를 숫자 특징 벡터로 변환합니다.   
  - 특징 벡터 및 레이블을 사용하여 예측 모델을 학습합니다.   
#### Parameter
  
### 어떻게 동작하는가?
파이프 라인은 일련의 단계로 지정되며 각 단계는 Transformer 또는 Estimator입니다.   
이러한 단계는 순서대로 실행되며 입력 DataFrame은 각 단계를 통과 할 때 변환됩니다.   
Transformer 스테이지의 경우 transform() 메서드가 DataFrame에서 호출됩니다.   
Estimator 단계의 경우 fit() 메서드가 호출되어 Transformer (PipelineModel 또는 피팅 된 Pipeline의 일부가 됨)를 생성하고   
해당 Transformer의 transform() 메서드가 DataFrame에서 호출됩니다.   

다음은 간단한 텍스트 문서에 대한 workflow입니다.   
![Pipeline_workflow1](https://spark.apache.org/docs/latest/img/ml-Pipeline.png)
위의 행은 세 단계가 있는 pipeline을 나타냅니다.   
처음 두 개 (Tokenizer 및 HashingTF)는 **Transformer**(파란색)이고    
세 번째 (LogisticRegression)는 **Estimator**(빨간색)입니다.   
아래 행은 파이프 라인을 통해 흐르는 데이터를 나타내며 **실린더는 DataFrame**을 나타냅니다.   
   
Pipeline.fit() 메서드는 원시 텍스트 문서와 레이블이있는 원본 DataFrame에서 호출됩니다.   
Tokenizer.transform() 메서드는 원시 텍스트 문서를 단어로 분할하여 단어가있는 새 열을 DataFrame에 추가합니다.     
HashingTF.transform() 메서드는 단어 열을 특징 벡터로 변환하여 해당 벡터가있는 새 열을 DataFrame에 추가합니다.   
이제 **LogisticRegression은 Estimator**이므로 pipeline은 먼저 LogisticRegression.fit()을 호출하여 LogisticRegressionModel을 생성합니다.   
    
만약 pipeline에 Estimator가 더 많으면,
DataFrame을 다음 단계로 전달하기 전에 DataFrame에서 LogisticRegressionModel의 transform() 메서드를 호출합니다.   
   
파이프 라인은 Estimator입니다. 따라서 **Pipeline의 fit() 메소드가 실행 된 후, Transformer인 PipelineModel을 생성**합니다. 이 **PipelineModel은 테스트에 사용**됩니다. 아래 그림은 이 사용법을 보여줍니다.   
![Pipeline_workflow2](https://spark.apache.org/docs/latest/img/ml-PipelineModel.png)
위의 그림에서 PipelineModel은 원래 Pipeline과 동일한 수의 단계를 갖지만, **원래 파이프 라인의 모든 Estimator은 Transformer가 되었습니다.**   
Pipelines과 PipelineModels은 학습 및 테스트 데이터가 동일한 기능 처리 단계를 거치도록합니다.   

### Parameters   
MLlib Estimator 및 Transformer는 매개 변수를 지정하기 위해 균일한 API를 사용합니다.   
Param은 자체 포함된 문서가 있는 명명된 매개 변수입니다. ParamMap은 (매개변수, 값)쌍의 집합입니다.   
알고리즘에 매개 변수를 전달하는 두 가지 주요 방법이 있습니다.   
- **인스턴스에 대한 매개 변수를 설정**합니다.   
  예를 들어, lr이 LogisticRegression의 인스턴스 인 경우,
  lr.setMaxIter (10)를 호출하여 lr.fit ()이 최대 10 개의 반복을 사용하도록 할 수 있습니다. 
  이 API는 spark.mllib 패키지에서 사용되는 API와 유사합니다.
- **ParamMap을 fit() 또는 transform()에 전달**합니다.   
