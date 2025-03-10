import pickle
import torch

# Wrapper들이 있는 파일 경로를 기준으로 모듈을 import합니다.
# 예를 들어, /home/sk/FT-w2v2-ser/pretrain/dataloader.py 파일에 정의되어 있다면,
# 프로젝트의 루트가 /home/sk/FT-w2v2-ser 라고 가정하면 아래와 같이 import할 수 있습니다.
from modules.FeatureFuser import Wav2vecWrapper, Wav2vec2Wrapper, Wav2vec2PretrainWrapper

# 모델 경로: 로컬에 저장된 wav2vec2-large-960h 모델 디렉토리 경로
model_path = "/home/sk/MERTools/MER2023/tools/transformers/wav2vec2-large-960h/pytorch_model.bin"

# Wav2vecWrapper 테스트
try:
    wrapper1 = Wav2vecWrapper(model_path)
    pickle.dumps(wrapper1)
    print("Wav2vecWrapper 피클링 성공")
except Exception as e:
    print("Wav2vecWrapper 피클링 에러:", e)

# Wav2vec2Wrapper 테스트 (pretrain=True)
try:
    wrapper2 = Wav2vec2Wrapper(pretrain=True)
    pickle.dumps(wrapper2)
    print("Wav2vec2Wrapper 피클링 성공")
except Exception as e:
    print("Wav2vec2Wrapper 피클링 에러:", e)

# Wav2vec2PretrainWrapper 테스트
try:
    wrapper3 = Wav2vec2PretrainWrapper()
    pickle.dumps(wrapper3)
    print("Wav2vec2PretrainWrapper 피클링 성공")
except Exception as e:
    print("Wav2vec2PretrainWrapper 피클링 에러:", e)
