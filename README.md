# CMV2
## predict_moralbert.py 돌리는 방법
1. 서버 접속
2. 아래와 같이 폴더 구조를 설정
moralbert_project/
├── predict_moralbert.py
├── requirements.txt
├── data/
│   ├── dataset_2013.feather
│   ├── dataset_2014.feather
│   └── ...
└── outputs/
3. Hugging Face login
   -  `hf auth login` 명령어 실행
   - 직접 토큰 입력 보기를 설정한 후, huggingface에서 발급받은 토큰(read 권한으로 충분)을 입력
4. nvidia-smi를 통해 비어있는 GPU 확인 (필자인 주희는 GPU2가 남아있던 상태)
5. 아래 명령어를 통해 output 만들기 (batch로 묶어서 진행하니 시간이 훨씬 단축됨!)
   ```bash
   CUDA_VISIBLE_DEVICES=2 python predict_moralbert.py \
  --year 2023 \ # year는 직접 설정
  --data-dir ./data \
  --output-dir ./outputs \
  --text-batch-size 128 \
  --chunk-batch-size 512 \
  --use-amp
   ```
