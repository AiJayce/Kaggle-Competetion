"""Generate the implementation-grounded Korean segmentation report."""

from datetime import date
import json
from pathlib import Path
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.textpath import TextToPath
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = Path("/home/Data_Drive_8TB_2/kykim/polyp/benchmark/seed42_448")
DEST = RESULTS / "Polyp_Segmentation_Model_Report_KO.pdf"
FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fontManager.addfont(FONT)
plt.rcParams.update({"font.family": FontProperties(fname=FONT).get_name(),
                     "pdf.fonttype": 42, "axes.unicode_minus": False})
metrics = pd.read_csv(RESULTS / "comparison.csv").set_index("model")
public = pd.read_csv(RESULTS / "validation_public_barplot_values.csv").set_index("model")
config = json.loads((RESULTS / "config.json").read_text())["config"]
MODELS = ["unet", "pranet", "hardnet_mseg", "dinov2", "segformer", "polyp_pvt"]


@lru_cache(maxsize=None)
def character_width(char):
    return TextToPath().get_text_width_height_descent(
        char, FontProperties(fname=FONT, size=10.4), ismath=False)[0]


def wrap(text, width=480):
    lines, line, length = [], "", 0
    for char in text:
        count = character_width(char)
        if length + count > width and line:
            lines.append(line)
            line, length = "", 0
        line += char
        length += count
    if line:
        lines.append(line)
    return lines


page_number = 0


def base(title, subtitle=""):
    global page_number
    page_number += 1
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.08, 0.946, title, size=19, weight="bold", color="#17365b")
    if subtitle:
        fig.text(0.08, 0.911, subtitle, size=10, color="#555555")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.895, 0.895], color="#4477AA", lw=1.5))
    fig.text(0.08, 0.042, "Polyp Segmentation | 구현 기반 실험 보고서 | 2026-11-05",
             size=8, color="#666666")
    fig.text(0.91, 0.042, str(page_number), size=9, ha="right")
    return fig


def text_page(pdf, title, sections, subtitle=""):
    fig = base(title, subtitle)
    y = 0.865
    for heading, content in sections:
        if heading:
            fig.text(0.08, y, heading, size=12, weight="bold", color="#17365b", va="top")
            y -= 0.031
        for paragraph in content.split("\n"):
            for line in wrap(paragraph):
                fig.text(0.085, y, line, size=10.4, va="top")
                y -= 0.021
            y -= 0.009
        y -= 0.012
    if y < 0.08:
        raise RuntimeError(f"Report page overflow: {title}, y={y}")
    pdf.savefig(fig)
    plt.close(fig)


def score_text(model):
    row = metrics.loc[model]
    return (
        f"Validation mean Dice {row.val_dice:.5f} / IoU {row.val_iou:.5f} / "
        f"Boundary F1 {row.val_boundary_f1:.5f}\n"
        f"Kaggle Public Dice {public.loc[model, 'public_dice']:.5f} / "
        f"학습 {int(row.epochs_run)} epoch / 선택 checkpoint {int(row.best_epoch)} epoch"
    )


def image_page(pdf, title, filename, caption):
    fig = base(title)
    image = plt.imread(RESULTS / filename)
    ax = fig.add_axes([0.055, 0.23, 0.89, 0.63])
    ax.imshow(image)
    ax.axis("off")
    y = 0.20
    for line in wrap(caption):
        fig.text(0.08, y, line, size=10, va="top")
        y -= 0.023
    pdf.savefig(fig)
    plt.close(fig)


def main():
    if set(metrics.index) != set(MODELS) or not public.status.eq("SubmissionStatus.COMPLETE").all():
        raise ValueError("Six completed, scored model records are required.")
    if config["size"] != 448 or config["threshold"] != 0.5:
        raise ValueError("Report assumptions differ from the actual experiment.")
    with PdfPages(DEST, metadata={
        "Title": "용종 분할 6개 모델: 학습 및 추론 방식 비교",
        "Author": "Polyp Segmentation experiment",
        "Subject": "지도학습 segmentation과 모델별 encoder/decoder 추론 과정",
    }) as pdf:
        text_page(pdf, "용종 분할 6개 모델 비교 보고서", [
            ("보고서의 질문", "각 모델은 이미지를 어떤 특징으로 바꾸고, 어떤 경로로 용종 영역을 예측했는가? "
             "사전학습의 지도·자기지도 여부와 이번 segmentation 학습의 지도 여부를 구분하여 설명한다."),
            ("가장 중요한 결론", "이번 용종 segmentation 학습은 6개 모두 지도학습이다. 640장의 이미지와 "
             "정답 마스크를 함께 사용해 용종의 위치·모양·경계를 학습했다. 정답 없이 segmentation을 "
             "학습한 실험은 아니다.\nDINOv2는 자기지도학습으로 만들어진 초기 encoder를 사용하지만, "
             "이번에는 decoder를 붙여 정답 용종 마스크로 encoder와 decoder를 함께 미세조정했다."),
            ("대상 모델", "CNN: U-Net / PraNet / HarDNet-MSEG\n"
             "Transformer: DINOv2 + decoder / SegFormer / Polyp-PVT"),
            ("사실과 해석의 구분", "설명은 실제 실행한 코드와 저장된 결과에 근거한다. 논문의 일반적인 구조와 "
             "이번 구현이 다른 부분은 별도로 표시했다. 구조가 어떤 성능 차이를 일으켰다는 설명은 "
             "가능한 가설이지, 이 실험에서 인과관계가 증명된 것은 아니다."),
            ("실험 결과 요약", "이 고정 분할 실험의 최고 성능은 SegFormer이다. Validation mean Dice "
             "0.92406, Kaggle Public Dice 0.92330이다. 6개 모두 제출 및 채점 COMPLETE를 확인했다. "
             "최고 epoch는 validation으로 선택했으며 Kaggle 점수로 다시 선택하지 않았다."),
        ], "실험: seed 42 · 원본 해상도 평가 · scrna · GPU 1 / 작성일: " + date.today().isoformat())

        fig = base("1. 지도·자기지도 구분", "사전학습 방식과 현재 과제의 학습 방식은 별개이다.")
        ax = fig.add_axes([0.07, 0.43, 0.86, 0.39])
        ax.axis("off")
        data = [
            ["U-Net", "없음", "해당 없음", "지도 / 처음부터"],
            ["PraNet", "ImageNet\nRes2Net-50", "지도 분류", "지도 / 미세조정"],
            ["HarDNet-MSEG", "ImageNet\nHarDNet-68", "지도 분류", "지도 / 미세조정"],
            ["DINOv2 + decoder", "DINOv2\nViT-S/14", "자기지도", "지도 / 미세조정"],
            ["SegFormer", "ImageNet\nMiT-B2", "지도 분류", "지도 / 미세조정"],
            ["Polyp-PVT", "ImageNet\nPVTv2-B2", "지도 분류", "지도 / 미세조정"],
        ]
        table = ax.table(cellText=data, colLabels=["모델", "초기 encoder", "기존 사전학습", "이번 분할 학습"],
                         colWidths=[0.25, 0.24, 0.21, 0.30], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 3.2)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("#dce8f4" if r == 0 else ("#f3f6fa" if r % 2 else "white"))
        y = 0.36
        for paragraph in [
            "Training은 손실을 계산하고 가중치를 업데이트하는 과정이다. 지도학습·자기지도학습 모두 "
            "training을 수행한다. 사전학습 가중치는 외부에서 이미 학습된 것을 가져왔고, 이번에 사전학습을 다시 수행하지 않았다.",
            "Annotation은 정답 마스크를 만드는 작업이다. 별도의 '용종 이름 붙이기 분류 단계'를 "
            "학습한 것이 아니라, 마스크의 각 픽셀을 배경 0 / 용종 1로 삼아 segmentation 자체를 지도학습했다.",
            "테스트 때 정답 마스크 없이 예측하는 것은 추론(inference)이다. 이것을 비지도학습이라고 부르지 않는다.",
        ]:
            for line in wrap(paragraph):
                fig.text(0.08, y, line, size=10.4, va="top")
                y -= 0.022
            y -= 0.023
        pdf.savefig(fig)
        plt.close(fig)

        text_page(pdf, "2. 공통 학습 및 용종 추론 절차", [
            ("학습: 정답으로 가중치를 수정", "800장의 제공 학습 이미지를 정렬한 뒤 seed 42 순열로 "
             "640장 train / 160장 validation으로 고정했다. 환자 ID가 없어 환자 단위 분할은 아니다.\n"
             "RGB 448×448 입력을 ImageNet mean/std로 정규화한다. 마스크는 nearest resize 후 "
             "값 >127을 1로 이진화한다. train에만 회전·반전·밝기/대비 증강을 적용한다."),
            ("손실과 업데이트", "픽셀별 BCE는 배경/용종 판단을, soft Dice loss는 예측 영역과 정답 영역의 "
             "겹침을 학습시킨다. 총 손실은 BCE + 1 - soft Dice이다. 여러 출력이 있으면 출력별 손실을 평균한다.\n"
             "AdamW(lr=0.0001, weight decay=0.0001), cosine scheduler, batch 4 × accumulation 2, "
             "BF16, gradient clipping 1.0을 사용했다. encoder 동결 없이 전체 모델을 최적화한다."),
            ("검증과 checkpoint 선택", "최대 50 epoch, patience 10, min_delta 0.0001로 조기 종료한다. "
             "최고 validation mean Dice checkpoint를 보존한다. min_delta는 대기 횟수 계산에만 쓰고 "
             "best checkpoint는 실제 최고값을 저장한다. 검증 데이터는 역전파에 쓰이지 않는다."),
            ("추론: 정답 없이 용종 확률 생성", "새 이미지 → 같은 전처리 → 학습된 모델 → 단일 logit map "
             "→ sigmoid 확률 → 원본 높이·너비로 bilinear 복원 → 확률 ≥0.5를 용종으로 판정한다.\n"
             "PraNet은 마지막 reverse-attention 출력, Polyp-PVT는 마지막 SAM 출력을 사용한다. "
             "여러 출력의 평균이나 앙상블을 사용하지 않는다. TTA·연결요소 제거 등 추가 후처리도 없다."),
            ("제출", "원본 크기의 이진 마스크를 행 우선(C order), 1-based start/length RLE로 바꾼다. "
             "id 0–199는 000.png–199.png에 대응한다. 모든 CSV의 200행과 RLE 복호화 일치를 확인했다."),
        ])

        descriptions = [
            ("unet", "3. U-Net", [
                ("이미지 segmentation 경로", "448×448 RGB → 5단계 convolution encoder "
                 "(32→64→128→256→512 채널) → 4단계 업샘플링 decoder → 1×1 convolution → logit map.\n"
                 "Max pooling으로 공간 크기를 줄이며 더 넓은 문맥을 학습하고, decoder는 "
                 "transposed convolution과 보간으로 크기를 복원한다. 같은 단계의 encoder 특징을 "
                 "skip connection으로 이어 붙여 위치·윤곽 정보를 함께 사용한다."),
                ("용종 영역을 어떻게 추론했는가", "학습 중 정답 마스크와 비교하며 용종에 관련된 "
                 "색·질감·윤곽·문맥의 조합을 convolution 필터가 학습한다. 추론에서는 이 필터가 "
                 "새 이미지의 특징을 계산하고 decoder가 픽셀별 용종 logit을 만든다. "
                 "확률 ≥0.5인 픽셀을 용종으로 선택한다. 사람이 경계를 클릭하거나 위치를 알려주지 않는다."),
                ("이번 구현의 조건", "사전학습 없이 무작위 초기화부터 지도학습했다. "
                 "원 논문의 64채널·valid convolution U-Net을 그대로 복제한 것은 아니며, "
                 "base 32 / padded convolution / BatchNorm을 사용한 U-Net 변형이다."),
                ("해석 범위", "Skip connection은 세부 위치 복원에 유용할 수 있지만 이 실험에서 "
                 "구조만의 효과를 분리하지 않았다. 다른 모델과 달리 사전학습이 없으므로 낮은 점수를 "
                 "CNN 전체의 한계로 해석할 수 없다."),
            ]),
            ("pranet", "4. PraNet", [
                ("이미지 segmentation 경로", "ImageNet Res2Net-50 encoder → 깊은 3개 단계의 특징 "
                 "→ RFB(여러 receptive field) → partial decoder의 초기 용종 map "
                 "→ 3단계 reverse-attention 보정 → 최종 logit.\n"
                 "Res2Net은 한 block 안에서 여러 스케일의 특징을 표현한다. RFB는 다양한 범위의 "
                 "문맥을 결합하고 aggregation은 초기 위치 추정을 만든다."),
                ("용종 영역을 어떻게 추론했는가", "각 보정 단계에서 이전 예측의 sigmoid를 p라 하면 "
                 "1-p를 encoder 특징에 곱한다. 이미 강하게 예측된 부분 이외의 특징을 강조하여 "
                 "놓친 영역이나 불확실한 경계를 보완하도록 학습한다. 보정 logit을 이전 map에 더한다.\n"
                 "이것은 정답 없는 영역 탐색이 아니라 정답 마스크 손실로 학습된 attention 연산이다."),
                ("이번 구현의 조건", "ImageNet encoder를 불러오고 decoder는 새로 학습했다. "
                 "초기 map과 3개 보정 map, 총 4개 출력에 각각 BCE+Dice를 적용해 평균한다. "
                 "추론에는 마지막 lateral_map_2만 사용한다."),
                ("해석 범위", "Reverse attention은 실제 예측 보정 장치이지 의학적 설명 또는 원인 증거가 아니다. "
                 "어떤 이미지에서 경계를 개선했는지는 저장된 검증 마스크로 확인해야 한다."),
            ]),
            ("hardnet_mseg", "5. HarDNet-MSEG", [
                ("이미지 segmentation 경로", "ImageNet HarDNet-68 encoder → 서로 다른 해상도의 특징 "
                 "→ 깊은 3개 특징(320/640/1024채널)에 RFB 적용 → 각 32채널로 변환 "
                 "→ aggregation → 8배 bilinear 확대 → 단일 용종 logit map.\n"
                 "HarDNet은 선택적인 dense 연결로 앞선 특징을 재사용하면서 특징 전달 비용을 줄이도록 "
                 "설계된 CNN encoder이다."),
                ("용종 영역을 어떻게 추론했는가", "깊은 특징의 넓은 문맥과 상대적으로 높은 해상도의 "
                 "위치 정보를 결합하여 픽셀별 용종 여부를 예측한다. RFB와 aggregation이 "
                 "여러 크기의 용종 단서를 통합하도록 정답 마스크로 학습된다."),
                ("이번 구현에서 특히 주의할 점", "사용한 공식 HarDMSEG.forward는 RFB와 aggregation으로 "
                 "만든 lateral_map_5 하나만 반환한다. 클래스에 reverse-attention용 layer가 "
                 "선언되어 있어도 실제 forward에서 호출하지 않는다. 따라서 이 실험을 PraNet과 "
                 "동일한 reverse-attention 추론이라고 설명하면 틀리다.\n"
                 "일부 선언된 미사용 파라미터는 gradient를 받지 않으며, 보고된 총 파라미터 수에는 포함된다."),
                ("학습과 해석", "활성 encoder와 decoder를 함께 지도 fine-tuning하고 단일 출력에 "
                 "BCE+Dice를 적용했다. 효율적인 설계 목적이 이 환경에서의 속도 우위를 자동으로 "
                 "보증하는 것은 아니므로 실제 시간 지표와 함께 해석해야 한다."),
            ]),
            ("dinov2", "6. DINOv2 + 학습 가능한 decoder", [
                ("이미지 segmentation 경로", "448×448 RGB → 14×14 patch 단위 토큰화 "
                 "(32×32 위치 격자) → ViT-S/14 Transformer → 3·6·9·12번째 block 특징 추출.\n"
                 "각 384채널 특징을 1×1 convolution으로 96채널로 줄인 뒤 연결한다. "
                 "Convolution fusion 128채널 → 4배 확대 → refinement 64채널 → "
                 "1채널 head → 입력 크기로 확대한다."),
                ("용종 영역을 어떻게 추론했는가", "Self-attention으로 이미지의 여러 patch 사이 문맥을 "
                 "계산하고, 직접 추가한 decoder가 그 특징을 픽셀별 용종 logit으로 변환한다. "
                 "최종 확률을 원본 크기로 복원한 뒤 공통 0.5 기준으로 마스크를 얻는다.\n"
                 "여러 block 특징은 깊이가 다르지만 같은 32×32 격자이다. 이것을 SegFormer와 "
                 "동일한 공간적 multi-resolution pyramid라고 부르지는 않는다."),
                ("자기지도와 이번 지도학습의 관계", "초기 encoder는 일반 DINOv2 자기지도 사전학습 "
                 "가중치(timm vit_small_patch14_dinov2.lvd142m)이다. 이번에는 정답 용종 마스크를 "
                 "사용해 ViT encoder와 새 decoder를 모두 업데이트했다. Encoder를 고정한 linear probe나 "
                 "정답 없는 용종 clustering 실험이 아니다."),
                ("해석 범위", "점수는 DINOv2 표현학습 자체의 절대 성능이 아니라 이 decoder·분할·예산을 "
                 "포함한 구성의 결과다. Patch 격자나 decoder가 경계 성능에 영향을 줬을 가능성은 있지만 "
                 "추가 대조 실험 없이 원인으로 확정할 수 없다."),
            ]),
            ("segformer", "7. SegFormer", [
                ("이미지 segmentation 경로", "ImageNet MiT-B2 encoder → 4단계 계층형 Transformer 특징 "
                 "(입력 대비 약 1/4, 1/8, 1/16, 1/32 크기) → SegFormer decode head.\n"
                 "각 단계 특징을 MLP로 동일한 768채널에 투영하고, 1/4 해상도로 보간하여 연결한다. "
                 "1×1 fusion, BatchNorm, 활성화, dropout, classifier로 단일 logit map을 만든다."),
                ("용종 영역을 어떻게 추론했는가", "상대적으로 세밀한 특징은 위치·경계 정보를, 깊은 특징은 "
                 "넓은 문맥을 제공하도록 학습한다. Decode head는 이들을 결합하여 각 위치가 "
                 "용종일 가능성을 예측한다. 1/4 크기 logit을 448×448로 확대한 뒤 sigmoid를 적용하고 "
                 "원본 크기 확률로 복원하여 threshold한다."),
                ("이번 구현의 조건", "Hugging Face nvidia/mit-b2의 ImageNet encoder만 불러왔다. "
                 "ADE20K 또는 용종 데이터의 segmentation head를 가져온 것이 아니다. "
                 "새 head는 num_labels=1이고 encoder와 함께 정답 마스크로 지도 fine-tuning했다."),
                ("관측 결과와 해석", "이 실험에서는 validation Dice와 Public Dice 모두 가장 높았다. "
                 "그러나 계층형 특징 융합이 단독 원인이라고 증명한 실험은 아니다. "
                 "사전학습과 모델 용량·최적화 조건이 함께 작용한 결과로 해석한다."),
            ]),
            ("polyp_pvt", "8. Polyp-PVT", [
                ("이미지 segmentation 경로", "ImageNet PVTv2-B2 encoder → 4단계 계층형 특징 "
                 "(64/128/320/512채널) → CFM·CIM·SAM → 2개 용종 logit map.\n"
                 "CFM(Cascaded Fusion Module)은 깊은 3개 단계 특징을 융합한다. "
                 "CIM(Camouflage Identification Module)은 낮은 단계의 특징에 채널·공간 attention을 "
                 "적용하여 용종 관련 단서를 포착하도록 학습한다."),
                ("용종 영역을 어떻게 추론했는가", "SAM(Similarity Aggregation Module)은 융합 특징과 "
                 "낮은 단계 단서를 이용해 anchor 유사도와 graph 연산으로 특징 간 정보를 결합하고 "
                 "보정한다. CFM 출력과 SAM 출력을 각각 단일 logit으로 투영하여 확대한다.\n"
                 "이 SAM은 Polyp-PVT 논문의 Similarity Aggregation Module이며, "
                 "Meta의 Segment Anything Model이 아니다. 클릭·박스 prompt도 사용하지 않았다."),
                ("이번 구현의 조건", "두 출력에 BCE+Dice를 적용하고 손실을 평균했다. 최종 추론에는 "
                 "SAM 뒤 prediction2_8만 사용한다. 전체 용종 사전학습 가중치가 아니라 "
                 "ImageNet encoder 가중치만 사용하고 전체 활성 네트워크를 지도 fine-tuning했다."),
                ("경고와 재현성", "구형 timm import와 F.upsample 경고는 API 노후화 안내였다. "
                 "Registry 경고는 별도 프로세스에서 공식 클래스를 직접 쓰는 구성에서 발생했다. "
                 "CUDA adaptive pooling backward는 완전 결정적이지 않아 같은 seed라도 재학습 "
                 "수치가 달라질 수 있다. 경고가 결과의 완전 재현성을 보증하지는 않는다."),
            ]),
        ]
        for model, title, sections in descriptions:
            text_page(pdf, title, sections + [("실제 결과", score_text(model))],
                      "공통 후처리: 원본 크기 확률 복원 → 0.5 threshold → RLE")

        image_page(pdf, "9. Validation 지표 비교", "validation_metrics_barplots.png",
                   "모든 모델에 같은 160장 validation을 적용했다. Dice/IoU/precision/recall/"
                   "boundary F1은 이미지별 점수 평균이다. Global Dice는 전체 TP/FP/FN을 합쳐 계산한다. "
                   "Boundary F1은 대각선의 0.5%(최소 1px)를 경계 허용 거리로 사용한다.")
        image_page(pdf, "10. Validation과 Kaggle Public", "validation_vs_public_barplot.png",
                   "Kaggle COMPLETE 제출을 메시지와 CSV SHA-256으로 해당 모델에 연결했다. "
                   "Public은 숨겨진 테스트의 공개 평가 부분이고 로컬 validation과 다른 표본이다. "
                   "두 점수 차이를 바로 과적합의 크기로 단정하지 않는다. SegFormer가 두 지표 모두 최고였다.")
        text_page(pdf, "11. 해석 한계와 구현 근거", [
            ("이 실험으로 말할 수 있는 것", "동일한 용종 학습·검증 분할과 공통 예산에서, 사용한 "
             "encoder+decoder 구성 중 SegFormer가 가장 좋은 관측 성능을 냈다. "
             "Attention/skip connection은 특징 계산 장치이며 사람이 읽는 의학적 근거 자체는 아니다."),
            ("이 실험으로 단정할 수 없는 것", "자기지도학습보다 지도 사전학습이 본질적으로 우수하다거나, "
             "CNN보다 Transformer가 항상 우수하다고 단정할 수 없다. U-Net만 사전학습이 없고 "
             "사전학습 데이터·파라미터 수·decoder가 서로 다르다.\n"
             "단일 seed·이미지 단위 단일 split이며 patient ID가 없다. Validation에서 최고 epoch와 "
             "모델을 고른 선택 편향이 있다. 원본 크기·종횡비 변화도 고려해야 한다. "
             "이 결과는 임상 안전성이나 진단 능력 검증이 아니다."),
            ("재현 및 라이선스", "일부 CUDA backward의 비결정성을 warn_only로 허용했다. "
             "완전한 bitwise 재현은 보장하지 않는다. PraNet·Polyp-PVT 소스는 연구·교육용 제한이 "
             "있으며, 사용자는 별도 제출 허가 확보를 확인했다. 추가 공유·상업 이용 권한은 별도 확인 대상이다."),
        ])
        text_page(pdf, "12. 구현·결과 및 참고 자료", [
            ("직접 확인한 구현·결과", "작업 폴더: Kaggle/Polyp/\n"
             "polyp_benchmark.py: UNet, DinoSegmentation, SegFormer, segmentation_loss, "
             "evaluate, write_submission, _train_model\n"
             "결과 폴더: /home/Data_Drive_8TB_2/kykim/polyp/benchmark/seed42_448/\n"
             "config.json / comparison.csv / 모델별 result.json / "
             "validation_public_barplot_values.csv / sources.json\n"
             "설치된 transformers 4.57.6의 SegformerDecodeHead.forward도 대조했다."),
            ("원본 구조 참고", "U-Net: https://arxiv.org/abs/1505.04597\n"
             "PraNet: https://github.com/DengPingFan/PraNet\n"
             "HarDNet-MSEG: https://github.com/james128333/HarDNet-MSEG\n"
             "DINOv2: https://arxiv.org/abs/2304.07193\n"
             "SegFormer: https://arxiv.org/abs/2105.15203\n"
             "Polyp-PVT: https://github.com/DengPingFan/Polyp-PVT"),
        ])
    print(f"Generated {page_number} pages: {DEST}")


if __name__ == "__main__":
    main()
