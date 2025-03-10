import os
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.outputlib import WriteConfusionSeaborn
import torch
from tqdm import tqdm  # ✅ 진행률 표시 추가

# ✅ 하드코딩된 설정값
hparams = {
    "batch_size": 64,
    "lr": 1e-4,
    "max_epochs": 15,
    "maxseqlen": 10,
    "nworkers": 4,
    "precision": 16,  # 16-bit precision 사용
    "saving_path": "downstream/checkpoints/custom",
    "datadir": "/home/sk/FT-w2v2-ser/Dataset/IEMOCAP/Audio_16k",
    "labeldir": "/home/sk/FT-w2v2-ser/Dataset/IEMOCAP/labels_sess",
    "pretrained_path": None,
    "model_type": "wav2vec2",
    "save_top_k": 1,
    "num_exps": 1,
    "outputfile": "OUTPUT_FILE",
}

# ✅ 기존 argparse 코드 (주석 처리)
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--maxseqlen', type=float, default=10)
parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labeldir', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2'], default='wav2vec2')
parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--num_exps', type=int, default=1)
parser.add_argument('--outputfile', type=str, default=None)
args = parser.parse_args()
hparams = args
"""

from downstream.Custom.trainer import DownstreamGeneral

if not os.path.exists(hparams["saving_path"]):
    os.makedirs(hparams["saving_path"])

nfolds = len(os.listdir(hparams["labeldir"]))
for foldlabel in os.listdir(hparams["labeldir"]):
    assert foldlabel[-5:] == ".json"

metrics, confusion = np.zeros((4, hparams["num_exps"], nfolds)), 0.

# ✅ tqdm을 사용하여 진행률 표시
for exp in tqdm(range(hparams["num_exps"]), desc="🔄 Running Experiments"):
    for ifold, foldlabel in enumerate(tqdm(os.listdir(hparams["labeldir"]), desc=f"🗂️ Processing Folds", leave=False)):
        print(f"\n🚀 Running experiment {exp+1}/{hparams['num_exps']}, fold {ifold+1}/{nfolds}...")
        
        hparams["labelpath"] = os.path.join(hparams["labeldir"], foldlabel)
        model = DownstreamGeneral(hparams)

        checkpoint_callback = ModelCheckpoint(
            dirpath=hparams["saving_path"],
            filename="{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}" if hasattr(model, "valid_met") else None,
            save_top_k=hparams["save_top_k"] if hasattr(model, "valid_met") else 0,
            verbose=True,
            save_weights_only=True,
            monitor="valid_UAR" if hasattr(model, "valid_met") else None,
            mode="max",
        )

        trainer = Trainer(
            precision=hparams["precision"],
            callbacks=[checkpoint_callback] if hasattr(model, "valid_met") else None,
            check_val_every_n_epoch=1,
            max_epochs=hparams["max_epochs"],
            num_sanity_val_steps=2 if hasattr(model, "valid_met") else 0,
            logger=False,
        )
        
        print("🚀 Training started...")
        trainer.fit(model, ckpt_path=hparams["resume_checkpoint"] if hparams["resume_checkpoint"] else None)
        print("✅ Training completed!")

        print("🔍 Running test evaluation...")
        if hasattr(model, "valid_met"):
            trainer.test()
        else:
            trainer.test(model)
        print("✅ Test evaluation completed!")

        met = model.test_met
        metrics[:, exp, ifold] = np.array(
            [met.uar * 100, met.war * 100, met.macroF1 * 100, met.microF1 * 100]
        )
        confusion += met.m

# ✅ 결과 출력
print("\n📊 Generating summary...")
outputstr = "+++ SUMMARY +++\n"
for nm, metric in zip(("UAR [%]", "WAR [%]", "macroF1 [%]", "microF1 [%]"), metrics):
    outputstr += f"Mean {nm}: {np.mean(metric):.2f}\n"
    outputstr += f"Fold Std. {nm}: {np.mean(np.std(metric, 1)):.2f}\n"
    outputstr += f"Fold Median {nm}: {np.mean(np.median(metric, 1)):.2f}\n"
    outputstr += f"Run Std. {nm}: {np.std(np.mean(metric, 1)):.2f}\n"
    outputstr += f"Run Median {nm}: {np.median(np.mean(metric, 1)):.2f}\n"

if hparams["outputfile"]:
    with open(hparams["outputfile"], "w") as f:
        f.write(outputstr)
else:
    print(outputstr)

# ✅ Confusion Matrix 저장
print("📊 Saving Confusion Matrix...")
WriteConfusionSeaborn(
    confusion,
    model.dataset.emoset,
    os.path.join(hparams["saving_path"], "confmat.png"),
)
print("✅ Confusion Matrix saved successfully!")
