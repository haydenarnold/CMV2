# CMV2

## `predict_moralbert.py` 실행 방법

### 1. 서버 접속

SSH를 이용해 서버에 접속합니다. (juheechoi 자리에는 본인의 서버 id)

```bash
ssh -p 16022 juheechoi@143.248.2.3
```

### 2. 프로젝트 폴더 구조 설정

프로젝트 폴더를 아래와 같이 구성합니다.

```text
moralbert_project/
├── predict_moralbert.py
├── requirements.txt
├── data/
│   ├── dataset_2013.feather
│   ├── dataset_2014.feather
│   └── ...
└── outputs/
```

각 파일과 폴더의 역할은 다음과 같습니다.

- `predict_moralbert.py`: MoralBERT prediction 실행 코드
- `requirements.txt`: 필요한 Python 패키지 목록
- `data/`: 연도별 입력 Feather 파일 저장 폴더
- `outputs/`: prediction 결과와 checkpoint 저장 폴더

필요한 폴더는 다음 명령어로 생성할 수 있습니다.

```bash
mkdir -p data outputs
```

### 3. Python 환경 및 패키지 설정

가상환경을 활성화합니다.

```bash
source moralbert_env/bin/activate
```

필요한 패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

### 4. Hugging Face 로그인

다음 명령어를 실행합니다.

```bash
hf auth login
```

토큰 입력 화면이 나타나면 Hugging Face에서 발급받은 토큰을 입력합니다.

모델 다운로드만 수행하는 경우에는 일반적으로 `read` 권한의 토큰이면 충분합니다.

로그인 상태는 다음 명령어로 확인할 수 있습니다.

```bash
hf auth whoami
```

### 5. 사용 가능한 GPU 확인

다음 명령어로 각 GPU의 메모리 사용량과 연산 사용률을 확인합니다.

```bash
nvidia-smi
```

`Memory-Usage`와 `GPU-Util`이 낮고 실행 중인 프로세스가 없는 GPU를 선택합니다.

예를 들어 GPU 2가 비어 있다면 실행 명령어 앞에 다음 설정을 붙입니다.

```bash
CUDA_VISIBLE_DEVICES=2
```

> 주의: 공용 서버에서는 비어 있는 GPU라도 다른 사용자가 예약해 둔 장비일 수 있으므로 서버 사용 규칙을 먼저 확인합니다.

### 6. MoralBERT prediction 실행

다음 명령어를 사용해 연도별 데이터에 MoralBERT prediction을 수행합니다.

```bash
CUDA_VISIBLE_DEVICES=2 python predict_moralbert.py \
  --year 2023 \
  --data-dir ./data \
  --output-dir ./outputs \
  --text-batch-size 128 \
  --chunk-batch-size 512 \
  --use-amp
```

`--year` 값은 처리할 데이터 연도에 맞게 직접 변경합니다.

예를 들어 2013년 데이터를 처리하려면 다음과 같이 실행합니다.

```bash
CUDA_VISIBLE_DEVICES=2 python predict_moralbert.py \
  --year 2013 \
  --data-dir ./data \
  --output-dir ./outputs \
  --text-batch-size 128 \
  --chunk-batch-size 512 \
  --use-amp
```

입력 파일은 다음 경로에 있어야 합니다.

```text
data/dataset_2013.feather
```

완료된 결과는 다음 경로에 저장됩니다.

```text
outputs/dataset_2013_scored.feather
```

### 7. 주요 실행 옵션

| 옵션 | 설명 |
|---|---|
| `--year` | 처리할 데이터의 연도 |
| `--data-dir` | 입력 Feather 파일이 있는 폴더 |
| `--output-dir` | 결과 파일과 checkpoint를 저장할 폴더 |
| `--text-batch-size` | 한 번에 전처리하는 원문 수 |
| `--chunk-batch-size` | 모델에 한 번에 전달하는 텍스트 청크 수 |
| `--use-amp` | GPU mixed-precision inference 사용 |

배치 방식으로 여러 텍스트 청크를 한꺼번에 추론하므로, 각 행을 개별적으로 처리하는 방식보다 실행 시간을 크게 줄일 수 있습니다.

### 8. 테스트 실행

전체 데이터를 처리하기 전에 일부 데이터와 하나의 Moral Foundation만 사용해 정상 작동 여부를 확인하는 것을 권장합니다.

```bash
CUDA_VISIBLE_DEVICES=2 python predict_moralbert.py \
  --year 2023 \
  --data-dir ./data \
  --output-dir ./outputs \
  --sample-size 1000 \
  --only-mft care \
  --text-batch-size 32 \
  --chunk-batch-size 64 \
  --use-amp
```

테스트가 정상적으로 완료된 후 전체 데이터를 실행합니다.

### 9. GPU 메모리 부족 오류

다음과 같은 오류가 발생하면:

```text
CUDA out of memory
```

`--chunk-batch-size`를 낮춰 다시 실행합니다.

```bash
--chunk-batch-size 256
```

그래도 메모리가 부족하면 다음과 같이 더 낮춥니다.

```bash
--chunk-batch-size 128
```

`--chunk-batch-size`는 GPU 메모리 사용량에 직접적인 영향을 줍니다.
