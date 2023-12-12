# KULLM enhancement project

pKLUE 방법론을 이용해서 KULLM의 벤치마크상 성능을 올려 보는 프로젝트입니다.

# 한국어 모델 평가 방법
```test_do_evalharness.sh``` 코드를 이용하는데, 먼저 다음과 같이 필요한 lm_eval을 설치합니다.  

### 한국어 태스크가 추가된 lm_eval 설치
(superheavytail이 태스크 추가한 버전으로 lm_eval 설치)
```bash
git clone https://github.com/superheavytail/lm-evaluation-harness.git
cd lm-evaluation-harness
git switch git-refactor
pip install -e .
```
### evaluation
이후 ```test_do_evalharness.sh```를 수정하고 실행하면 됩니다.  

즉, ```--model_args pretrained=...``` 부분만 원하는 모델의 체크포인트 경로 또는 허깅페이스로 다운로드 가능한 이름으로 바꾸면 됩니다.  

주석 처리되어 있는 예제들을 참고하세요!
