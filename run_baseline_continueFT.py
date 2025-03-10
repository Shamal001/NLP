import pytorch_lightning

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pretrain.trainer import ContinueFinetuningBaseline
import os
import torch

# ✅ 하드코딩된 설정값
hparams = {
    "saving_path": "pretrain/checkpoints_baseline",
    "lr": 1e-4,
    "batch_size": 64,
    "max_epochs": 50, 
    "warmup_epochs": 5,
    "maxseqlen": 10.0,
    "resume_checkpoint": None,  # ✅ 체크포인트 경로 설정 가능
    "precision": 16,  # 16-bit precision 사용
    "distributed": False,  # ✅ 분산 학습 사용 여부
    "accelerator": "gpu",  # ✅ 'ddp' 대신 'gpu' 사용
    "train_bucket_size": 50,
    "val_bucket_size": 20,
    "use_additional_obj": False,
    "use_bucket_sampler": False,
    "unsupdatadir": None,
    "num_clusters": None,
    "save_top_k": 2,
    "datadir": "/home/sk/FT-w2v2-ser/Dataset/IEMOCAP/Audio_16k/",
    "labelpath": "/home/sk/FT-w2v2-ser/Dataset/IEMOCAP/labels_sess/label_1.json",
}

# ✅ 기존 argparse 코드 (주석 처리)
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--saving_path', type=str, default='pretrain/checkpoints_baseline')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--training_step', type=int, default=120000)
parser.add_argument('--warmup_step', type=int, default=4000)
parser.add_argument('--maxseqlen', type=float, default=10.0)
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--accelerator', type=str, default='ddp')
parser.add_argument('--train_bucket_size', type=int, default=50)
parser.add_argument('--val_bucket_size', type=int, default=20)
parser.add_argument('--use_additional_obj', action='store_true')
parser.add_argument('--use_bucket_sampler', action='store_true')
parser.add_argument('--unsupdatadir', type=str, default=None)
parser.add_argument('--num_clusters', type=str, default=None)
parser.add_argument('--save_top_k', type=int, default=2)
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labelpath', type=str, default=None)
args = parser.parse_args()
"""

# ✅ num_clusters 변환
nclusters = None
if hparams["num_clusters"]:
    nclusters = [int(x) for x in hparams["num_clusters"].split(',')]

# ✅ 체크포인트 저장 설정
if not os.path.exists(hparams["saving_path"]):
    os.makedirs(hparams["saving_path"])

checkpoint_callback = ModelCheckpoint(
    dirpath=hparams["saving_path"],
    filename="w2v2-{epoch:02d}-{valid_loss:.2f}",
    verbose=True,
    save_last=True,
)

# ✅ Trainer 설정
wrapper = Trainer(
    precision=16 if hparams["precision"] == 16 else 32,
    callbacks=[checkpoint_callback],
    devices=2, accelerator="gpu",  # GPU 2개 사용
    strategy="ddp",  # 'ddp' 방식으로 분산 학습
    num_sanity_val_steps=2,
)




# ✅ 모델 초기화
model = ContinueFinetuningBaseline(
    max_epochs=hparams["max_epochs"],
    batch_size=hparams["batch_size"],
    lr=hparams["lr"],
    warmup_epochs=hparams["warmup_epochs"],
    maxseqlen=int(16000 * hparams["maxseqlen"]),
    nclusters=nclusters,
    datadir=hparams["datadir"],
    labelpath=hparams["labelpath"],
    distributed=hparams["distributed"],
    use_bucket_sampler=hparams["use_bucket_sampler"],
    train_bucket_size=hparams["train_bucket_size"],
    val_bucket_size=hparams["val_bucket_size"],
    use_additional_obj=hparams["use_additional_obj"],
)

# ✅ 학습 진행
print("🚀 Training started...")
wrapper.fit(model, ckpt_path=hparams["resume_checkpoint"])
print("✅ Training completed!")
