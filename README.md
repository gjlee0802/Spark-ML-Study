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
- Dataframe   
Spark SQL의 DataFrame을 다양한 데이터 유형을 보유 할 수있는 ML 데이터 세트로 사용합니다.   
- Transformer   
Transformer는 기능 변환기 및 학습 된 모델을 포함하는 추상화입니다.   
원시 데이터를 다양한 방식으로 변환하는 메소드입니다.   
기술적으로 Transformer는 일반적으로 하나 이상의 열을 추가하여 하나의 DataFrame을 다른 DataFrame으로 변환하는 transform () 메서드를 구현합니다.   

- Estimator   
훈련용 데이터 세트에 머신 러닝 모델을 훈련시키거나 적합하도록 조정합니다.   
Estimator는 DataFrame을 받아들이고 Transformer 인 Model을 생성하는 fit () 메서드를 구현합니다.

- Pipeline   
파이프 라인은 여러 Transformer와 Estimator을 함께 연결하여 ML workflow를 지정합니다.   
머신 러닝에서는 데이터를 처리하고 학습하기 위해 일련의 알고리즘을 실행하는 것이 일반적입니다.   
MLlib는 특정 순서로 실행될 일련의 PipelineStage (Transformer 및 Estimator)로 구성된 Pipeline과 같은 Workflow를 제공합니다.   
예를 들어 간단한 텍스트 문서 처리 워크플로에는 다음의 단계가 포함될 수 있습니다.   
  - 각 문서의 텍스트를 단어로 분할합니다.   
  - 각 문서의 단어를 숫자 특징 벡터로 변환합니다.   
  - 특징 벡터 및 레이블을 사용하여 예측 모델을 학습합니다.   
  
### 어떻게 동작하는가?
파이프 라인은 일련의 단계로 지정되며 각 단계는 Transformer 또는 Estimator입니다.   
이러한 단계는 순서대로 실행되며 입력 DataFrame은 각 단계를 통과 할 때 변환됩니다.   
Transformer 스테이지의 경우 transform () 메서드가 DataFrame에서 호출됩니다.   
Estimator 단계의 경우 fit () 메서드가 호출되어 Transformer (PipelineModel 또는 피팅 된 Pipeline의 일부가 됨)를 생성하고   
해당 Transformer의 transform () 메서드가 DataFrame에서 호출됩니다.   

다음은 간단한 텍스트 문서에 대한 workflow입니다.   
![Pipeline_workflow1](https://spark.apache.org/docs/latest/img/ml-Pipeline.png)
위의 행은 세 단계가 있는 pipeline을 나타냅니다.   
처음 두 개 (Tokenizer 및 HashingTF)는 **Transformer**(파란색)이고    
세 번째 (LogisticRegression)는 **Estimator**(빨간색)입니다.   
아래 행은 파이프 라인을 통해 흐르는 데이터를 나타내며 **실린더는 DataFrame**을 나타냅니다.   
