"""Build a consistently formatted CardioCare final report (Korean, A4)."""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT / "artifacts"
MONITOR = ARTIFACTS / "monitor"
REPORT_FIGS = ARTIFACTS / "report"
OUTPUT = PROJECT_ROOT / "report.pdf"

AUTHOR_NAME = "천잉지에"
STUDENT_ID = "202324037"
COURSE = "기계학습 · CardioCare 기말 프로젝트"

ETHICS_KO = (
    "본 시스템은 심장 전문의의 의사결정을 보조하는 도구이며, 독립적인 진단 자격이 없습니다. "
    "모든 최종 의료 결정은 면허를 가진 의사가 내려야 합니다. "
    "모델은 편향이 있을 수 있으며, 전문 의료 진단을 대체할 수 없습니다."
)

AI_DISCLOSURE_KO = (
    "본 프로젝트는 ChatGPT를 코드 템플릿 작성 및 디버깅에 사용했으며, "
    "모든 핵심 로직과 실험 결과는 본인이 독립적으로 완성했습니다. "
    "scikit-learn 공식 문서 및 UCI 데이터셋 페이지를 참고했으며, "
    "관련 소스 파일에 출처 링크를 주석으로 표기했습니다."
)

# --- Layout constants (figure coordinates, 0-1) ---
PAGE_W, PAGE_H = 8.27, 11.69
LEFT, RIGHT = 0.08, 0.92
TOP = 0.90
BOTTOM = 0.08
CONTENT_W = RIGHT - LEFT
LINE_H = 0.030
WRAP = 52


@dataclass
class Block:
    kind: str  # heading | body | spacer | ethics
    text: str = ""


def load_metadata() -> dict:
    meta_path = ARTIFACTS / "model_metadata.json"
    mon_path = MONITOR / "monitor_summary.json"
    data: dict = {}
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    if mon_path.exists():
        data["monitor_summary"] = json.loads(mon_path.read_text(encoding="utf-8"))
    return data


def font(size: int, bold: bool = False) -> font_manager.FontProperties:
    paths = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if bold else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return font_manager.FontProperties(fname=path, size=size)
    return font_manager.FontProperties(size=size, weight="bold" if bold else "normal")


def wrap(text: str, width: int = WRAP) -> list[str]:
    if not text.strip():
        return []
    return textwrap.wrap(text, width=width)


class ReportBuilder:
    """Unified A4 report renderer with header, footer, and section styles."""

    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.page_no = 0

    def _new_figure(self) -> plt.Figure:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=150)
        fig.patch.set_facecolor("white")
        return fig

    def _draw_header(self, fig: plt.Figure, title: str) -> None:
        fig.patches.append(
            mpatches.Rectangle((0, 0.955), 1, 0.045, transform=fig.transFigure, color="#1F4E79", zorder=0)
        )
        fig.text(LEFT, 0.968, "CardioCare", fontproperties=font(9, bold=True), color="white", va="center")
        fig.text(RIGHT, 0.968, COURSE, fontproperties=font(8), color="white", ha="right", va="center")
        fig.text(LEFT, TOP, title, fontproperties=font(13, bold=True), va="top")
        fig.add_artist(
            Line2D([LEFT, RIGHT], [TOP - 0.015, TOP - 0.015], transform=fig.transFigure, color="#AAAAAA", linewidth=0.8)
        )

    def _draw_footer(self, fig: plt.Figure) -> None:
        self.page_no += 1
        fig.add_artist(
            Line2D([LEFT, RIGHT], [BOTTOM + 0.02, BOTTOM + 0.02], transform=fig.transFigure, color="#CCCCCC", linewidth=0.6)
        )
        fig.text(
            0.5,
            BOTTOM,
            f"- {self.page_no} -",
            fontproperties=font(8),
            ha="center",
            va="bottom",
            color="#666666",
        )

    def _save(self, fig: plt.Figure) -> None:
        self.pdf.savefig(fig, dpi=150)
        plt.close(fig)

    def cover(self) -> None:
        fig = self._new_figure()
        fig.patches.append(
            mpatches.Rectangle((0, 0.55), 1, 0.45, transform=fig.transFigure, color="#1F4E79", zorder=0)
        )
        fig.text(0.5, 0.72, "CardioCare", fontproperties=font(28, bold=True), ha="center", color="white")
        fig.text(0.5, 0.64, "기말 프로젝트 보고서", fontproperties=font(16), ha="center", color="white")
        fig.text(0.5, 0.58, "End-to-End ML System for Heart Disease Prediction", fontproperties=font(11), ha="center", color="#D9E8F5")

        y = 0.46
        for line in [
            f"작성자: {AUTHOR_NAME}",
            f"학번: {STUDENT_ID}",
            f"과목: {COURSE}",
            "",
            "GitHub: https://github.com/touyoupo/cardiocare",
        ]:
            fig.text(0.5, y, line, fontproperties=font(11), ha="center", va="top")
            y -= 0.045

        box = mpatches.FancyBboxPatch(
            (LEFT, 0.12),
            CONTENT_W,
            0.22,
            boxstyle="round,pad=0.012",
            linewidth=1,
            edgecolor="#1F4E79",
            facecolor="#F5F8FC",
            transform=fig.transFigure,
        )
        fig.patches.append(box)
        fig.text(LEFT + 0.02, 0.30, "윤리 선언", fontproperties=font(11, bold=True), va="top")
        cy = 0.265
        for ln in wrap(ETHICS_KO, width=WRAP + 4):
            fig.text(LEFT + 0.02, cy, ln, fontproperties=font(9), va="top")
            cy -= 0.028

        self._draw_footer(fig)
        self._save(fig)

    def section(self, section_title: str, blocks: list[Block]) -> None:
        """Render a section; auto-paginate when content overflows."""
        fig = self._new_figure()
        self._draw_header(fig, section_title)
        y = TOP - 0.05

        def flush_page() -> None:
            nonlocal fig, y
            self._draw_footer(fig)
            self._save(fig)
            fig = self._new_figure()
            self._draw_header(fig, section_title + " (계속)")
            y = TOP - 0.05

        for block in blocks:
            if block.kind == "spacer":
                y -= LINE_H * 0.6
                continue

            if block.kind == "heading":
                fnt = font(11, bold=True)
                lines = [block.text]
                color = "#1F4E79"
            elif block.kind == "ethics":
                fnt = font(9)
                lines = wrap(block.text, WRAP)
                color = "#333333"
            else:
                fnt = font(10)
                lines = wrap(block.text, WRAP)
                color = "#222222"

            for line in lines:
                if y < BOTTOM + 0.06:
                    flush_page()
                fig.text(LEFT, y, line, fontproperties=fnt, va="top", color=color)
                y -= LINE_H if block.kind != "heading" else LINE_H * 1.1

        self._draw_footer(fig)
        self._save(fig)

    def figure_page(self, fig_no: int, title: str, image_path: Path, caption: str) -> None:
        self._figure_axes_page(f"그림 {fig_no}. {title}", [(str(fig_no), title, image_path, caption)])

    def dual_figure_page(
        self,
        items: list[tuple[int, str, Path, str]],
        header: str,
        slot_heights: list[float] | None = None,
    ) -> None:
        """Render two figures on one page (top / bottom)."""
        labels = [(str(n), t, p, c) for n, t, p, c in items]
        self._figure_axes_page(header, labels, slot_heights=slot_heights)

    def _figure_axes_page(
        self,
        header: str,
        items: list[tuple[str, str, Path, str]],
        slot_heights: list[float] | None = None,
    ) -> None:
        fig = self._new_figure()
        self._draw_header(fig, header)

        n = len(items)
        default_h = 0.36 if n == 2 else 0.56
        heights = slot_heights or [default_h] * n
        y_cursor = 0.82

        for index, (label, title, image_path, caption) in enumerate(items):
            slot_h = heights[index] if index < len(heights) else default_h
            y_top = y_cursor
            fig.text(LEFT, y_top, f"그림 {label}. {title}", fontproperties=font(9, bold=True), va="top", color="#1F4E79")
            box_y = y_top - slot_h - 0.02
            if image_path.exists():
                ax = fig.add_axes([LEFT, box_y, CONTENT_W, slot_h])
                image = mpimg.imread(image_path)
                ax.imshow(image, aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#CCCCCC")
            else:
                fig.text(
                    LEFT,
                    box_y + slot_h / 2,
                    f"[이미지 없음: {image_path.name}]",
                    fontproperties=font(9),
                    va="center",
                    color="#999999",
                )

            cap_y = box_y - 0.02
            for line in wrap(caption, WRAP + 2)[:2]:
                fig.text(LEFT, cap_y, line, fontproperties=font(8), va="top", color="#555555")
                cap_y -= 0.022
            y_cursor = cap_y - 0.04

        self._draw_footer(fig)
        self._save(fig)

    def model_table_page(self, comparison: list[dict], metrics: dict, best_model: str) -> None:
        fig = self._new_figure()
        self._draw_header(fig, "4. 모델 비교 및 선정")

        fig.text(
            LEFT,
            TOP - 0.05,
            "세 가지 모델 계열을 MLflow로 기록하고 재현율(Recall) 기준으로 최종 모델을 선정했습니다.",
            fontproperties=font(10),
            va="top",
        )

        col_labels = ["Model", "Accuracy", "Recall", "Precision", "F1", "AUC", "Bal.Acc"]
        rows = []
        for row in comparison:
            rows.append(
                [
                    row.get("model_family", ""),
                    f"{row.get('accuracy', 0):.3f}",
                    f"{row.get('recall', 0):.3f}",
                    f"{row.get('precision', 0):.3f}",
                    f"{row.get('f1', 0):.3f}",
                    f"{row.get('auc', 0):.3f}",
                    f"{row.get('balanced_accuracy', 0):.3f}",
                ]
            )

        ax = fig.add_axes([LEFT, 0.42, CONTENT_W, 0.30])
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1F4E79")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F5F8FC")

        display_name = best_model.replace("_", " ").title()
        summary = [
            f"최종 모델: {display_name} (Recall 최고, {metrics.get('recall', 0):.3f})",
            f"홀드아웃: accuracy={metrics.get('accuracy', 0):.3f}, "
            f"auc={metrics.get('auc', 0):.3f}, f1={metrics.get('f1', 0):.3f}",
            "선정 근거: 위음성(FN) 위험을 줄이기 위해 재현율(Recall) 최대화.",
            "MLflow 3개 모델 계열 실험 기록 — 그림 7 참조.",
        ]
        y = 0.34
        for line in summary:
            for ln in wrap(line, WRAP):
                fig.text(LEFT, y, ln, fontproperties=font(10), va="top")
                y -= LINE_H

        self._draw_footer(fig)
        self._save(fig)


def build_mlops_flowchart(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        ("Commit", "#4C78A8"),
        ("CI", "#72B7B2"),
        ("CD", "#F58518"),
        ("CM", "#E45756"),
        ("CT", "#54A24B"),
        ("Review", "#B279A2"),
        ("Update", "#FF9DA6"),
    ]
    for index, (label, color) in enumerate(steps):
        x = 0.3 + index * 1.35
        rect = mpatches.FancyBboxPatch(
            (x, 1.0), 1.1, 1.0, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black"
        )
        ax.add_patch(rect)
        ax.text(x + 0.55, 1.5, label, ha="center", va="center", color="white", fontsize=9, weight="bold")
        if index < len(steps) - 1:
            ax.annotate("", xy=(x + 1.15, 1.5), xytext=(x + 1.1, 1.5), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("MLOps Closed Loop", fontsize=11, weight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def pick_best_model(comparison: list[dict]) -> str:
    if not comparison:
        return "logistic_regression"
    return max(comparison, key=lambda row: row.get("recall", 0)).get("model_family", "unknown")


def build_report(output_path: Path = OUTPUT) -> None:
    meta = load_metadata()
    metrics = meta.get("final_metrics", {})
    comparison = meta.get("comparison_table", [])
    best_model = pick_best_model(comparison)
    monitor = meta.get("monitor_summary", {})
    feature_store = meta.get("feature_store", {})
    model_registry = meta.get("model_registry", {})
    flagged = monitor.get("flagged_features", ["chol", "oldpeak"])
    base_ba = monitor.get("baseline_metrics", {}).get("balanced_accuracy", 0)
    drift_ba = monitor.get("drifted_metrics", {}).get("balanced_accuracy", 0)
    base_recall = monitor.get("baseline_metrics", {}).get("recall", 0)
    drift_recall = monitor.get("drifted_metrics", {}).get("recall", 0)
    mlflow_png = REPORT_FIGS / "mlflow_comparison.png"

    REPORT_FIGS.mkdir(parents=True, exist_ok=True)
    mlops_png = build_mlops_flowchart(REPORT_FIGS / "mlops_flowchart.png")

    with PdfPages(output_path) as pdf:
        doc = ReportBuilder(pdf)

        doc.cover()

        doc.section(
            "1-2. 문제 정의 · EDA · 전처리",
            [
                Block("heading", "1.1 문제 정의 및 데이터셋"),
                Block(
                    "body",
                    "CardioCare는 임상 특성으로 심장병 위험을 예측하여 의사 판단을 보조합니다 "
                    "(inform, not decide). UCI Cleveland 303행, seed=42.",
                ),
                Block("body", "위음성(FN)이 위양성보다 임상적으로 더 위험하여 recall 중심 평가."),
                Block("spacer"),
                Block("heading", "1.2 윤리 선언"),
                Block("ethics", ETHICS_KO),
                Block("spacer"),
                Block("heading", "2.1 EDA 및 전처리"),
                Block("body", "불균형 클래스, '?' 결측, chol/trestbps/oldpeak 이상치 확인."),
                Block("body", "중앙값/최빈값 대치, IQR 클리핑, train_test_split 후 fit(누수 방지)."),
                Block("body", "그림 1-2: notebook EDA — 클래스 분포 및 연속 특성 박스플롯."),
            ],
        )

        doc.dual_figure_page(
            [
                (
                    1,
                    "타깃 클래스 분포",
                    REPORT_FIGS / "eda_target_distribution.png",
                    "질병(1) 비율이 건강(0)보다 높아 불균형 확인.",
                ),
                (
                    2,
                    "연속 특성 박스플롯",
                    REPORT_FIGS / "eda_boxplot.png",
                    "chol, oldpeak 등 이상치·분포 차이 확인.",
                ),
            ],
            "1-2. EDA",
        )

        doc.model_table_page(comparison, metrics, best_model)

        doc.dual_figure_page(
            [
                (3, "혼동 행렬", ARTIFACTS / "confusion_matrix.png", "홀드아웃 테스트셋 혼동 행렬."),
                (4, "특성 중요도", ARTIFACTS / "feature_importance.png", "thal, ca, oldpeak 등 주요 변수."),
            ],
            "3-4. 모델 평가",
        )

        doc.section(
            "5. 테스트 및 MLOps 메타데이터",
            [
                Block("heading", "5.1 단위 테스트"),
                Block(
                    "body",
                    "tests/test_pipeline.py — (1) 예측 shape (2) 확률 [0,1] 및 합=1 "
                    "(3) 임상 범위 검증 (4) seed=42 결정론 재현.",
                ),
                Block("spacer"),
                Block("heading", "5.2 Docker 및 CI"),
                Block(
                    "body",
                    "Dockerfile(python:3.10-slim) + src/inference.py 배치 추론. "
                    "GitHub Actions: dvc repro 및 단위 테스트 자동 실행.",
                ),
                Block("body", "저장소: https://github.com/touyoupo/cardiocare"),
                Block("spacer"),
                Block("heading", "5.3 Feature Store"),
                Block(
                    "body",
                    f"등록 특성: {feature_store.get('registered_feature', 'chol')}. "
                    f"{feature_store.get('rationale_ko', '')}",
                ),
                Block("spacer"),
                Block("heading", "5.4 Model Registry"),
                Block(
                    "body",
                    f"메타데이터 필드: {model_registry.get('metadata_field', 'recall_on_holdout')} "
                    f"= {model_registry.get('value', metrics.get('recall', 0)):.3f}. "
                    f"{model_registry.get('rationale_ko', '')}",
                ),
            ],
        )

        doc.dual_figure_page(
            [
                (
                    5,
                    "KS 드리프트 (chol)",
                    MONITOR / "drift_hist_chol.png",
                    "훈련 분포 vs 드리프트 테스트 분포; p<0.05 플래그.",
                ),
                (
                    6,
                    "모니터링 시계열",
                    MONITOR / "metric_timeseries.png",
                    "balanced accuracy / recall 시계열.",
                ),
            ],
            "5-6. 드리프트 모니터링",
        )

        mlflow_items = [
            (
                7,
                "MLflow 실험 기록",
                mlflow_png if mlflow_png.exists() else REPORT_FIGS / "mlflow_placeholder.png",
                "3개 모델 계열(logistic_regression, svc, random_forest) 실험 비교.",
            ),
            (
                8,
                "MLOps 폐루프",
                mlops_png,
                "Commit → CI → CD → CM → CT → Review → Model Update.",
            ),
        ]
        doc.dual_figure_page(mlflow_items, "7-8. MLflow · MLOps", slot_heights=[0.42, 0.22])

        doc.section(
            "6-7. 드리프트 · DVC · 서빙",
            [
                Block("heading", "6. 드리프트 감지 및 재학습"),
                Block(
                    "body",
                    f"KS 검정 기준: 훈련 분포 vs 드리프트 분포. "
                    f"플래그 특성: {', '.join(flagged)}.",
                ),
                Block(
                    "body",
                    f"성능 변화 — recall {base_recall:.3f} → {drift_recall:.3f}, "
                    f"balanced accuracy {base_ba:.3f} → {drift_ba:.3f}.",
                ),
                Block(
                    "body",
                    "재학습 정책: KS p-value < 0.05 이고 recall 5%p 이상 하락 시 트리거, "
                    "배포 전 심장 전문의 검토 필수.",
                ),
                Block("spacer"),
                Block("heading", "7. DVC · 모델 서빙 · 경량화"),
                Block(
                    "body",
                    "DVC: .dvc/ 초기화, dvc.yaml 3단계(download→train→monitor), "
                    "dvc.lock 해시 추적, params.yaml(seed=42). Git=코드, DVC=데이터·모델.",
                ),
                Block(
                    "body",
                    "서빙: Dockerfile + inference.py 배치 API, MaaS 배포 및 로드밸런서 수평 확장.",
                ),
                Block(
                    "body",
                    "경량화: 상위 5개 특성 pruning 검토, joblib 직렬화, "
                    "추론 시 임상 범위 검증으로 안전성 확보.",
                ),
            ],
        )

        doc.section(
            "8. 한계 · 윤리 · 부록",
            [
                Block("heading", "8.1 한계 및 향후 과제"),
                Block("body", "소규모 단일 기관 데이터, 편향 가능, SHAP/외부 검증 필요."),
                Block("spacer"),
                Block("heading", "8.2 윤리 선언 (마무리)"),
                Block("ethics", ETHICS_KO),
                Block("spacer"),
                Block("heading", "부록: AI 도구 사용 공개"),
                Block("body", AI_DISCLOSURE_KO),
            ],
        )

    print(f"Wrote {output_path} ({doc.page_no} pages)")


if __name__ == "__main__":
    build_report()
