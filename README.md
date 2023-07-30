# Korean LLM (Large Language Models) hands-on labs

## Overview

최근 Gnerative AI의 두뇌 역할을 하는 다양한 파운데이션 모델이 공개되었고 많은 기업에서 파운데이션(Foundation) 모델을 활용하는 애플리케이션을 검토하거나 개발하고 있습니다. 
하지만, 대규모 모델은 단일 GPU로 추론이 쉽지 않고 이를 프로덕션용으로 서빙하거나 파인튜닝하는 것은 쉽지 않습니다.

이 핸즈온은 Generative AI를 빠르게 검토하고 프로덕션에 적용하고자 하는 분들을 위해 작성되었으며, 한국어 대규모 모델을 AWS 인프라 상에서 효율적으로 서빙하고 파인튜닝하는 법에 대해 스텝-바이-스텝으로 안내합니다.

## QLoRA fine-tuning 
- [KULLM-Polyglot-12.8B](fine-tuning)

## Model Serving

- [KULLM-Polyglot-5.8B-v2](kullm-polyglot-5.8b)
- [KULLM-Polyglot-12.8B-v2](kullm-polyglot-12.8b)
- [KoAlpaca-Polyglot-12.8B](koalpaca-polyglot-12.8b)
- [KoAlpaca-KoRWKV-6B](koalpaca-korwkv-6b)

### Filenames
- `1_local-inference.ipynb`: 모델을 허깅페이스 허브에서 로드해서 간단한 추론을 수행합니다. 필수는 아니지만 모델을 체험해 보고 싶다면 이 과정부터 시작하는 것을 권장합니다.
- `2_local-inference-deepspeed.py` & `2_run.sh`: DeepSpeed 분산 추론을 실험합니다. 여러 장의 GPU가 탑재된 인스턴스나 서버를 권장합니다. (예: `ml.g5.12xlarge`)
- `3_sm-serving-djl-deepspeed-from-hub.ipynb`: SageMaker DJL (Deep Java Library) 서빙 컨테이너 (DeepSpeed 분산 추론) 를 사용해 SageMaker 모델 서빙을 수행합니다. 호스팅 서버는 허깅페이스 허브에서 모델을 직접 다운로드합니다.
- `3_sm-serving-djl-deepspeed-from-hub.ipynb`: SageMaker DJL (Deep Java Library) 서빙 컨테이너 (DeepSpeed 분산 추론) 를 사용해 SageMaker 모델 서빙을 수행합니다. 호스팅 서버는 S3에서 모델을 다운로드합니다. 내부적으로 s5cmd로 병렬적으로 파일을 다운로드하기 때문에 다운로드 속도가 매우 빠릅니다.
- `3_sm-serving-tgi-from-hub.ipynb`: SageMaker TGI (Text Generation Inferface) 서빙 컨테이너를 사용해 SageMaker 모델 서빙을 수행합니다. TGI는 허깅페이스에서 개발한 분산 추론 서버로 매우 빠른 추론 속도를 보여줍니다.
- `3_sm-serving-djl-fastertransformer-nocode.ipynb`: SageMaker DJL (Deep Java Library) 서빙 컨테이너 (NVIDIA FasterTransformer 분산 추론) 를 사용해 SageMaker 모델 서빙을 수행합니다. 지원되는 모델에 한해서 DeepSpeed보다 더욱 빠른 속도를 보여줍니다.

## Requirements

이 핸즈온을 수행하기 위해 아래 사양의 인스턴스를 준비하는 것을 권장합니다.

### Notebook instance
- `ml.t3.medium` (최소 사양)
- `ml.m5.xlarge` (권장 사양)

### Training instance
- `ml.g5.2xlarge` (최소 사양)
- `ml.g5.12xlarge` (권장 사양)

### Hosting instance
- `ml.g5.2xlarge`: 7B 파라미터 이하 모델
- `ml.g5.12xlarge` (권장 사양)

## License Summary

이 샘플 코드는 MIT-0 라이선스에 따라 제공됩니다. 라이선스 파일을 참조하세요.