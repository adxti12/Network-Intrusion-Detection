"""Fill the ICT-4442 template's cover page + Part B (Interim) from the interim snapshot.

    PYTHONPATH= .venv/bin/python reports/build_interim_report.py
Reads results_interim/ and data/processed_interim/meta.json; writes
reports/Interim_Report_Team_NIDS.docx.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "ICT_4442_Mini_Project_Report_Template.docx")
OUT = os.path.join(ROOT, "reports", "Interim_Report_Team_NIDS.docx")
RES = os.path.join(ROOT, "results_interim")
META = json.load(open(os.path.join(ROOT, "data", "processed_interim", "meta.json")))
M = {k: json.load(open(os.path.join(RES, k, "metrics.json"))) for k in ("mlp", "cnn")}

BLACK = '<w:color w:val="000000" w:themeColor="text1"/>'
BODY_SZ = 21   # half-points, as used in the template's Part B
TABLE_SZ = 17
TEXT_W = 9640  # table width used by the template


# ---------------------------------------------------------------- XML helpers
def run(text, bold=False, italic=False, size=BODY_SZ, color=None):
    rpr = "<w:b/><w:bCs/>" if bold else ""
    rpr += "<w:i/><w:iCs/>" if italic else ""
    rpr += f'<w:color w:val="{color}"/>' if color else BLACK
    rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich(text, size=BODY_SZ, color=None):
    """**bold** segments inside a string."""
    out = []
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            out.append(run(part, bold=i % 2 == 1, size=size, color=color))
    return "".join(out)


def para(content, after=100, before=0, jc="both", style=None, keep_next=False, numbered=None):
    ppr = f'<w:pStyle w:val="{style}"/>' if style else ""
    ppr += "<w:keepNext/>" if keep_next else ""
    if numbered is not None:
        ppr += f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{numbered}"/></w:numPr>'
    ppr += f'<w:spacing w:before="{before}" w:after="{after}"/><w:jc w:val="{jc}"/>'
    body = content if content.startswith("<w:r") else rich(content)
    return f"<w:p><w:pPr>{ppr}</w:pPr>{body}</w:p>"


def label(text):
    return para(run(text, bold=True), after=60, before=60, keep_next=True)


def heading2(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:keepNext/><w:spacing w:before="200" w:after="80"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/>{BLACK}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')


def bullet(text):
    return para(rich(text), after=40, numbered=BULLET_NUM_ID)


def caption(text):
    return para(run(text, italic=True, size=17), after=120, jc="center")


def table(headers, rows, widths, size=TABLE_SZ, center_cols=()):
    assert sum(widths) == TEXT_W, sum(widths)
    border = "".join(f'<w:{b} w:val="single" w:color="auto" w:sz="4" w:space="0"/>'
                     for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
    x = (f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{border}</w:tblBorders>'
         f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="30" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
         f'<w:bottom w:w="30" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
         + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>")

    def cell(text, w, head, jc):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="1F4E78"/>' if head else ""
        body = (run(text, bold=True, size=size, color="FFFFFF") if head else rich(text, size=size))
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}</w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:jc w:val="{jc}"/></w:pPr>{body}</w:p></w:tc>')

    x += '<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' + "".join(
        cell(h, w, True, "center") for h, w in zip(headers, widths)) + "</w:tr>"
    for r in rows:
        x += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(
            cell(str(v), w, False, "center" if i in center_cols else "left")
            for i, (v, w) in enumerate(zip(r, widths))) + "</w:tr>"
    return x + "</w:tbl>" + para("", after=60)


def image(rid, cx_in, cy_in, pid, name):
    cx, cy = int(cx_in * 914400), int(cy_in * 914400)
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="60" w:after="40"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{pid}" name="{name}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def f3(x):
    return f"{x:.3f}"


BULLET_NUM_ID = 901  # added to numbering.xml below


# ------------------------------------------------------------------- content
def part_b():
    raw, cc = META["raw_label_counts"], META["class_counts"]
    tr, va, te = META["split_train"], META["split_val"], META["split_test"]
    classes = META["class_names"]
    x = []
    x.append('<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:spacing w:before="0" w:after="120"/></w:pPr>'
             f'<w:r><w:rPr><w:b/><w:bCs/>{BLACK}</w:rPr><w:t xml:space="preserve">PART B : INTERIM REPORT  (Phase 2)</w:t></w:r></w:p>')

    x.append(label("Team No. and Names (with Registration Numbers):"))
    x.append(para("Team No.: To be assigned", after=0))
    x.append(para("1. Aditi Arun – Registration No. 230911108", after=0))
    x.append(para("2. Anushka Thakur – Registration No. 230911048", after=100))
    x.append(label("Title of the Project (confirmed / updated from Synopsis):"))
    x.append(para("Comparative Deep Learning Approaches for Network Intrusion Detection (confirmed, unchanged from the Synopsis)."))
    x.append(label("GitHub Repository Link:"))
    x.append(para("<GitHub repository URL>"))

    # 1. Literature review ---------------------------------------------------
    x.append(heading2("1. Literature Review"))
    x.append(para("Ten papers were reviewed, covering the dataset itself, feed-forward, recurrent, convolutional, hybrid, "
                  "and attention-based detectors, and known pitfalls of the dataset. They are summarised below; full "
                  "references are listed at the end of this part."))
    lit = [
        ("[1] Sharafaldin et al., 2018", "CICFlowMeter flow-feature extraction; benchmark of KNN, RF, ID3, NB, MLP, etc.; RF-based feature ranking",
         "CIC-IDS2017", "Tree-based classifiers (ID3, RF) performed best, ≈0.98 weighted precision",
         "Source of our dataset and its 78 flow features; defines the classical baseline to beat"),
        ("[2] Shone et al., 2018", "Stacked non-symmetric deep autoencoders (NDAE) for unsupervised features + Random Forest",
         "KDD Cup '99, NSL-KDD", "97.85% 5-class accuracy on KDD Cup '99; weaker on rare classes in NSL-KDD",
         "Deep feature learning on flow statistics; highlights the rare-class problem we also face"),
        ("[3] Vinayakumar et al., 2019", "Deep feed-forward DNN (1–5 hidden layers) vs. classical ML; scale-hybrid-IDS-AlertNet",
         "KDDCup99, NSL-KDD, UNSW-NB15, Kyoto, WSN-DS, CIC-IDS2017", "DNNs outperformed classical classifiers across the benchmark datasets",
         "Feed-forward DNN on CIC-IDS2017 – direct reference for our MLP baseline"),
        ("[4] Yin et al., 2017", "RNN-IDS: recurrent neural network classifier",
         "NSL-KDD", "83.28% binary and 81.29% 5-class accuracy on KDDTest+, above J48, NB, RF, SVM",
         "Recurrent modelling of traffic records – reference for our LSTM"),
        ("[5] Kim et al., 2016", "LSTM-RNN classifier",
         "KDD Cup '99", "96.93% accuracy and 98.88% detection rate, with a slightly higher false-alarm rate",
         "LSTM for IDS; shows why false-alarm rate must be reported alongside accuracy"),
        ("[6] Wang et al., 2018 (HAST-IDS)", "CNN learns spatial features of packets, LSTM learns temporal features across packets",
         "DARPA1998, ISCX2012", "Higher accuracy and detection rate, lower false-alarm rate than prior published methods",
         "Hybrid CNN/LSTM; motivates evaluating CNN and LSTM separately under one protocol"),
        ("[7] Wu & Guo, 2019 (LuNet)", "Hierarchical CNN + RNN with gradually increasing granularity",
         "NSL-KDD, UNSW-NB15", "NSL-KDD binary: 99.42% DR, 0.53% FPR; UNSW-NB15 multi-class: 84.98% accuracy",
         "CNN-based local feature extraction on flow features; emphasises low false-positive rate"),
        ("[8] Yang et al., 2019", "Modified density-peak clustering (MDPCA) + ensemble of sub-DBNs with fuzzy aggregation",
         "NSL-KDD, UNSW-NB15", "Higher accuracy, recall, precision and F1 than well-known methods; clustering eases class imbalance",
         "Imbalance handling – relevant to our rare Bot / Web-attack classes"),
        ("[9] Gorishniy et al., 2021", "FT-Transformer (each feature becomes a token) and ResNet for tabular data",
         "11 public tabular benchmarks", "FT-Transformer outperformed ResNet on most tasks; tuned MLP/ResNet are strong baselines",
         "Basis for our Transformer design: every flow feature is a token"),
        ("[10] Engelen et al., 2021", "Audit of CIC-IDS2017 traffic generation, flow construction, feature extraction and labelling",
         "CIC-IDS2017", ">25% of flows are meaningless artefacts (up to 50% in some attack classes); >20% of traces relabelled or rebuilt",
         "Motivates our cleaning (de-duplication) and caution about near-perfect scores"),
    ]
    x.append(table(["Paper (Author, Year)", "Method", "Dataset", "Key Result", "Relevance to Project"],
                   lit, [1500, 2200, 1500, 2300, 2140]))

    # 2. Dataset & preprocessing ---------------------------------------------
    x.append(heading2("2. Dataset Acquisition & Preprocessing"))
    x.append(label("Description of preprocessing pipeline (cleaning, augmentation, encoding, normalization, train/val/test split):"))
    n_raw = META["raw_rows"]
    x.append(para(f"**Acquisition.** We use the CIC-IDS2017 MachineLearningCVE flow files [1]: eight per-day files "
                  f"(Monday–Friday) containing {n_raw:,} flows, each with 78 numeric flow features (durations, packet and byte "
                  f"counts, inter-arrival-time statistics, TCP flag counts, etc.) and one of 15 labels. Rather than a "
                  f"pre-cleaned Kaggle derivative, we download the original files (parquet mirror, ~270 MB, script "
                  f"scripts/download_data.sh) and perform every cleaning step in our own code (src/data.py), so each "
                  f"decision is visible and reproducible."))
    x.append(para("**Pipeline** (all steps in src/data.py, run with python -m src.data):"))
    steps = [
        f"**Cleaning:** removed {META['dropped_nan_inf']:,} rows containing NaN or ±∞ (in Flow Bytes/s and Flow Packets/s), "
        f"{META['dropped_duplicates']:,} exact duplicate rows, a duplicated column (Fwd Header Length) and "
        f"{len(META['dropped_constant_columns'])} constant columns (e.g. Bwd PSH Flags and all six bulk-rate features).",
        "**Shortcut removal:** Destination Port was dropped. It identifies the service, not the behaviour of the traffic "
        "(e.g. port 21/22 ⇒ FTP/SSH-Patator), and would let a model ‘detect’ attacks without learning traffic patterns.",
        f"**Label encoding:** the 15 raw labels were grouped into {len(classes)} classes – Benign, DoS (Hulk, GoldenEye, "
        "slowloris, Slowhttptest), DDoS, PortScan, BruteForce (FTP-/SSH-Patator), WebAttack (brute force, XSS, SQL injection) "
        "and Bot. Infiltration (36 flows) and Heartbleed (11 flows) were removed because they are too small to learn or "
        "evaluate. Classes are integer-encoded.",
        f"**Normalisation:** {META['n_features']} features remain. Because they span many orders of magnitude (bytes/s "
        "up to ~10⁹) and a few contain negatives (Init_Win_bytes = −1), a signed log transform sign(x)·log(1+|x|) is applied, "
        "followed by z-score standardisation fitted on the training set only, and clipping to ±10.",
        "**Split:** a stratified 70 / 15 / 15 train / validation / test split (seed 42), shared by every model. Validation and "
        "test keep the natural class distribution, so reported metrics reflect realistic traffic.",
        f"**Imbalance handling (training set only):** majority classes are capped at {max(tr.values()):,} flows by random "
        "undersampling, and the loss uses inverse-square-root class weights. No synthetic augmentation is used, so that no "
        "artificial flows enter the comparison.",
    ]
    x += [bullet(s) for s in steps]
    x.append(para(f"**Notable finding from EDA.** De-duplication after removing Destination Port shrinks PortScan from "
                  f"{raw['PortScan']:,} to {cc['PortScan']:,} flows: a port scan sends the same tiny probe to many ports, so "
                  f"the flows differ only in the port number. Keeping these copies would place identical vectors in both train "
                  f"and test and inflate scores. Only 270 of ~2.23 million unique feature vectors carry conflicting labels "
                  f"(negligible). The resulting class distribution is shown below (Benign : Bot ≈ "
                  f"{cc['Benign'] // cc['Bot']:,} : 1)."))
    rows = [[c, f"{cc[c]:,}", f"{tr[c]:,}", f"{va[c]:,}", f"{te[c]:,}"] for c in classes]
    rows.append(["**Total**", f"{sum(cc.values()):,}", f"{sum(tr.values()):,}", f"{sum(va.values()):,}", f"{sum(te.values()):,}"])
    x.append(para(run("Table I. Class counts after cleaning and in each split", italic=True, size=17), after=40, jc="center", keep_next=True))
    x.append(table(["Class", "After cleaning", "Train (capped)", "Validation", "Test"], rows,
                   [2200, 1860, 1860, 1860, 1860], size=18, center_cols=(1, 2, 3, 4)))
    x.append(image("rIdEda", 3.9, 2.23, 501, "class_distribution"))
    x.append(caption("Fig. 1. Class distribution after cleaning and grouping (log scale)."))

    # 3. Models -------------------------------------------------------------
    x.append(heading2("3. Models Implemented So Far (minimum 2 of the proposed 3–4)"))
    mlp, cnn = M["mlp"], M["cnn"]
    x.append(table(
        ["Model", "Owner (Member)", "Status", "Preliminary Metric (test)", "Notes"],
        [["MLP (baseline)", "Joint (Aditi Arun & Anushka Thakur)", "Implemented; preliminary run done",
          f"Acc {pct(mlp['accuracy'])}, macro-F1 {f3(mlp['macro_f1'])}", "1 epoch, untuned; 60k params"],
         ["1D CNN", "Aditi Arun", "Implemented; preliminary run done",
          f"Acc {pct(cnn['accuracy'])}, macro-F1 {f3(cnn['macro_f1'])}", "1 epoch, untuned; 40k params"],
         ["LSTM", "Anushka Thakur", "Implemented; first training run in progress", "–",
          "~5 min/epoch on CPU; architecture being slimmed"],
         ["Transformer", "Joint (Aditi Arun & Anushka Thakur)", "Architecture coded; not yet trained", "–",
          "Scheduled per work plan (05/10)"]],
        [1400, 1900, 2000, 2140, 2200]))
    x.append(para("**Architectures.** The **MLP** baseline has three hidden layers (256-128-64) with batch normalisation, "
                  "ReLU and dropout 0.2. The **1D CNN** treats the 68-feature vector as a one-channel sequence: three "
                  "convolution blocks (32/64/128 filters, kernel 3, batch norm, ReLU, max-pooling) followed by global average "
                  "pooling and a dense head, so its filters see windows of neighbouring features. The **LSTM** reads the "
                  "features as a 68-step sequence (value + learned feature-identity embedding) with a bidirectional LSTM. "
                  "The **Transformer** follows FT-Transformer [9]: each feature becomes a token, a [CLS] token is "
                  "prepended, and self-attention relates any feature to any other regardless of column order."))
    x.append(para("**Common training protocol.** All models use the same splits, cross-entropy with class weights, AdamW "
                  "(learning rate 10⁻³, weight decay 10⁻⁴), batch size 512, gradient clipping at 1.0 and seed 42 "
                  "(src/train.py). The preliminary runs below are deliberately **unoptimised**: a single training epoch, "
                  "with no hyperparameter tuning, early stopping or repeated seeds."))

    def row(name, m):
        b, pc = m["binary"], m["per_class"]
        return [name, pct(m["accuracy"]), f3(m["macro_precision"]), f3(m["macro_recall"]), f3(m["macro_f1"]),
                pct(b["attack"]["recall"]), pct(b["false_positive_rate"]),
                f3(pc["PortScan"]["f1-score"]), f3(pc["WebAttack"]["f1-score"]), f3(pc["Bot"]["f1-score"])]
    x.append(para(run(f"Table II. Preliminary test-set results ({sum(te.values()):,} flows, 1 training epoch)",
                      italic=True, size=17), after=40, jc="center", keep_next=True))
    x.append(table(["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Attack recall", "False-pos. rate",
                    "F1 PortScan", "F1 WebAttack", "F1 Bot"],
                   [row("MLP", mlp), row("1D CNN", cnn)],
                   [920, 960, 900, 900, 900, 1040, 1060, 960, 1000, 1000], size=17, center_cols=tuple(range(1, 10))))
    x.append(para("**Preliminary observations.**"))
    x += [bullet(s) for s in [
        f"Both models already separate benign from attack traffic well: the MLP detects {pct(mlp['binary']['attack']['recall'])} "
        f"and the 1D CNN {pct(cnn['binary']['attack']['recall'])} of attack flows. Their false-positive rates are "
        f"{pct(mlp['binary']['false_positive_rate'])} and {pct(cnn['binary']['false_positive_rate'])}.",
        "Accuracy is dominated by the benign class (85% of test flows) and therefore hides weaknesses. Macro-F1 "
        f"({f3(mlp['macro_f1'])} vs. {f3(cnn['macro_f1'])}) is limited by the rare classes – PortScan, WebAttack and "
        "especially Bot (215 test flows) – where even a small fraction of benign flows misclassified as an attack destroys precision.",
        "After one epoch the 1D CNN is ahead of the MLP on accuracy, macro-F1 and false-positive rate, while the MLP has the "
        "higher attack recall. Whether this reflects the convolutional inductive bias or only training noise must be "
        "tested with full training and repeated seeds before drawing conclusions.",
    ]]

    # 4. Contribution log -----------------------------------------------------
    x.append(heading2("4. Individual Contribution Log (to date)"))
    x.append(table(["Member Name", "Reg. No.", "Task(s) Completed", "Signature"],
                   [["Aditi Arun", "230911108",
                     "Dataset acquisition script; cleaning, encoding, normalisation and split pipeline; 1D CNN implementation "
                     "and preliminary evaluation; joint MLP baseline; literature review (papers [1], [3], [6], [7], [9])", ""],
                    ["Anushka Thakur", "230911048",
                     "EDA and class-distribution analysis; LSTM implementation and first training runs; joint MLP baseline; "
                     "common training/evaluation script; literature review (papers [2], [4], [5], [8], [10])", ""]],
                   [1700, 1200, 5040, 1700]))
    x.append(para("Commit history evidencing individual contributions is available in the GitHub repository linked above."))

    # 5. Risks / plan -------------------------------------------------------------
    x.append(heading2("5. Risk / Plan for Remaining Work"))
    x.append(label("Remaining models to implement, expected challenges, and timeline to Final submission:"))
    x.append(para("**Remaining work.** Complete LSTM training and evaluation, train the Transformer, then train all four "
                  "models to convergence under the common protocol, tune the key hyperparameters, and carry out the "
                  "comparative and error analysis for the Final report."))
    x.append(para(run("Table III. Risks and mitigations", italic=True, size=17), after=40, jc="center", keep_next=True))
    x.append(table(["Risk / challenge", "Impact", "Mitigation"], [
        ["No GPU: LSTM (~5 min/epoch) and Transformer (attention over 69 tokens) are slow on CPU",
         "Delayed training; fewer tuning runs", "Slimmer recurrent/attention models, cap majority classes in training, "
         "run independent experiments in parallel"],
        ["Severe class imbalance (Benign : Bot ≈ 1,300 : 1)", "Unstable rare-class precision; accuracy misleading",
         "Class-weighted loss; report macro-F1, per-class F1 and threshold-free PR-AUC alongside accuracy"],
        ["Noisy validation signal for rare classes", "Early stopping may pick an unrepresentative epoch",
         "Validate on the natural class ratio, use patience-based early stopping, repeat runs over 3 seeds"],
        ["Dataset artefacts and label noise in CIC-IDS2017 [10]", "Over-optimistic or non-generalisable scores",
         "De-duplication, shortcut-feature removal, and discussion as a limitation in the Final report"],
        ["Differences caused by setup rather than architecture", "Invalid comparison",
         "Single shared split, preprocessing, loss, optimiser and stopping rule for all models (src/config.py)"],
    ], [3400, 2600, 3640]))
    x.append(para(run("Table IV. Timeline to Final submission", italic=True, size=17), after=40, jc="center", keep_next=True))
    x.append(table(["Sl. No.", "Component / Work Element / Milestone", "Expected Completion"], [
        ["1", "LSTM training and preliminary evaluation (Anushka Thakur)", "28/09/2026"],
        ["2", "Transformer / attention-based model implementation and training (joint)", "05/10/2026"],
        ["3", "Hyperparameter tuning and standardised evaluation of all four models (multiple seeds)", "13/10/2026"],
        ["4", "Comparative performance analysis, error analysis and Final report writing", "23/10/2026"],
        ["5", "Presentation preparation, viva practice and in-class Final Presentations", "26–29/10/2026"],
        ["6", "Final Report and Code Repository Submission", "31/10/2026"],
    ], [1000, 6440, 2200], center_cols=(0, 2)))

    # References ----------------------------------------------------------------
    x.append(heading2("References"))
    refs = [
        "I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward generating a new intrusion detection dataset and intrusion traffic characterization,” in Proc. 4th Int. Conf. Inf. Syst. Secur. Privacy (ICISSP), 2018, pp. 108–116.",
        "N. Shone, T. N. Ngoc, V. D. Phai, and Q. Shi, “A deep learning approach to network intrusion detection,” IEEE Trans. Emerg. Topics Comput. Intell., vol. 2, no. 1, pp. 41–50, 2018.",
        "R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, and S. Venkatraman, “Deep learning approach for intelligent intrusion detection system,” IEEE Access, vol. 7, pp. 41525–41550, 2019.",
        "C. Yin, Y. Zhu, J. Fei, and X. He, “A deep learning approach for intrusion detection using recurrent neural networks,” IEEE Access, vol. 5, pp. 21954–21961, 2017.",
        "J. Kim, J. Kim, H. L. T. Thu, and H. Kim, “Long short term memory recurrent neural network classifier for intrusion detection,” in Proc. Int. Conf. Platform Technol. Service (PlatCon), 2016, pp. 1–5.",
        "W. Wang, Y. Sheng, J. Wang, X. Zeng, X. Ye, Y. Huang, and M. Zhu, “HAST-IDS: Learning hierarchical spatial-temporal features using deep neural networks to improve intrusion detection,” IEEE Access, vol. 6, pp. 1792–1806, 2018.",
        "P. Wu and H. Guo, “LuNet: A deep neural network for network intrusion detection,” in Proc. IEEE Symp. Series Comput. Intell. (SSCI), 2019, pp. 617–624.",
        "Y. Yang, K. Zheng, C. Wu, X. Niu, and Y. Yang, “Building an effective intrusion detection system using the modified density peak clustering algorithm and deep belief networks,” Appl. Sci., vol. 9, no. 2, art. 238, 2019.",
        "Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for tabular data,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, pp. 18932–18943.",
        "G. Engelen, V. Rimmer, and W. Joosen, “Troubleshooting an intrusion detection dataset: the CICIDS2017 case study,” in Proc. IEEE Secur. Privacy Workshops (SPW), 2021, pp. 7–12.",
    ]
    for i, r in enumerate(refs, 1):
        x.append(f'<w:p><w:pPr><w:spacing w:before="0" w:after="40"/><w:ind w:left="400" w:hanging="400"/><w:jc w:val="both"/></w:pPr>'
                 f'{run(f"[{i}]  ", size=18)}{run(r, size=18)}</w:p>')
    return "".join(x)


# ------------------------------------------------------------------ assemble
def build():
    tmp = tempfile.mkdtemp()
    subprocess.run(["unzip", "-q", TEMPLATE, "-d", tmp], check=True)
    subprocess.run(["find", tmp, "-type", "l", "-delete"], check=True)
    doc_p = os.path.join(tmp, "word", "document.xml")
    s = open(doc_p, encoding="utf-8").read()

    # Cover page: keep everything before the Part A heading paragraph.
    a = s.find("PART A")
    cut = s.rfind("<w:p ", 0, a)
    cover, sect = s[:cut], s[s.rfind("<w:sectPr"):]
    cover = cover.replace("&lt;TITLE OF THE PROJECT&gt;",
                          "COMPARATIVE DEEP LEARNING APPROACHES FOR NETWORK INTRUSION DETECTION")
    cover = cover.replace(" Synopsis/Interim/Final report on ", "Interim Report on")
    cover = cover.replace("&lt;NAME OF THE STUDENT-1 &amp; Reg no.&gt;", "Aditi Arun (Reg. No. 230911108)")
    cover = cover.replace("&lt;NAME OF THE STUDENT-2 &amp; Reg no.&gt;", "Anushka Thakur (Reg. No. 230911048)")
    cover = re.sub(r"<w:r [^>]*>(?:(?!</w:r>).)*<w:t>===</w:t></w:r>", "", cover, flags=re.S)
    # remove the (unused) third-student paragraph
    i = cover.find("&lt;NAME OF THE STUDENT-3")
    cover = cover[:cover.rfind("<w:p ", 0, i)] + cover[cover.find("</w:p>", i) + 6:]
    cover = cover.replace("&lt; MONTH AND YEAR OF SUBMISSION&gt;", "SEPTEMBER 2026")
    for ph in ("&lt;", "STUDENT-3", "==="):
        assert ph not in cover, ph

    open(doc_p, "w", encoding="utf-8").write(cover + part_b() + sect + "</w:body></w:document>"
                                             if not sect.endswith("</w:document>") else cover + part_b() + sect)

    # Figure: relationship + media + content type
    shutil.copy(os.path.join(RES, "figures", "eda_class_counts.png"), os.path.join(tmp, "word", "media", "eda_class_counts.png"))
    rels_p = os.path.join(tmp, "word", "_rels", "document.xml.rels")
    rels = open(rels_p, encoding="utf-8").read()
    rels = rels.replace("</Relationships>", '<Relationship Id="rIdEda" Type="http://schemas.openxmlformats.org/officeDocument/'
                        '2006/relationships/image" Target="media/eda_class_counts.png"/></Relationships>')
    open(rels_p, "w", encoding="utf-8").write(rels)
    ct_p = os.path.join(tmp, "[Content_Types].xml")
    ct = open(ct_p, encoding="utf-8").read()
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Types ", "<Types ", 1).replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
        open(ct_p, "w", encoding="utf-8").write(ct)

    # Bullet list definition
    num_p = os.path.join(tmp, "word", "numbering.xml")
    num = open(num_p, encoding="utf-8").read()
    absdef = ('<w:abstractNum w:abstractNumId="901"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0">'
              '<w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/>'
              '<w:pPr><w:ind w:left="360" w:hanging="240"/></w:pPr></w:lvl></w:abstractNum>')
    first_num = num.find("<w:num ")
    num = num[:first_num] + absdef + num[first_num:]
    num = num.replace("</w:numbering>", f'<w:num w:numId="{BULLET_NUM_ID}"><w:abstractNumId w:val="901"/></w:num></w:numbering>')
    open(num_p, "w", encoding="utf-8").write(num)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    subprocess.run(f'cd "{tmp}" && zip -qXr "{OUT}" .', shell=True, check=True)
    shutil.rmtree(tmp)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
