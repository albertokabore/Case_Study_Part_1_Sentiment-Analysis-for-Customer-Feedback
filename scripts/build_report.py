"""Build the research paper (Word .docx) from the files in ``results/``.

Usage (from the project root, after ``run_pipeline.py`` has produced the results)::

    python scripts/build_report.py
    python scripts/build_report.py --author "Jane Doe" --course "DSC 500" --instructor "Dr. Smith"

Every number, table and figure in the paper is read from ``results/`` at build
time, so re-running the pipeline (for example after the Transformer stage
finishes) and then this script keeps the paper consistent with the code. If
DistilRoBERTa predictions exist they are included automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

from src import config  # noqa: E402

FIG, MET = config.FIGURES_DIR, config.METRICS_DIR
OUT = ROOT / "reports" / "Sentiment_Analysis_Research_Paper.docx"
FONT = "Calibri"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def load_results() -> dict:
    r = {
        "cleaning": json.loads((MET / "cleaning_report.json").read_text()),
        "overview": pd.read_csv(MET / "summary_overview.csv", index_col=0)["value"],
        "sentiment": pd.read_csv(MET / "summary_sentiment.csv", index_col=0),
        "by_airline": pd.read_csv(MET / "summary_by_airline.csv", index_col=0),
        "reasons": pd.read_csv(MET / "summary_negative_reasons.csv", index_col=0),
        "prep": pd.read_csv(MET / "summary_preprocessing_effect.csv", index_col=0)["value"],
        "ablation": pd.read_csv(MET / "preprocessing_ablation.csv"),
        "terms": pd.read_csv(MET / "distinctive_terms.csv"),
        "comparison": pd.read_csv(MET / "model_comparison.csv"),
        "per_class": pd.read_csv(MET / "per_class_metrics.csv"),
        "mcnemar": pd.read_csv(MET / "mcnemar_tests.csv"),
        "kpis": pd.read_csv(MET / "airline_kpis.csv", index_col=0),
        "triage": pd.read_csv(MET / "negative_triage_operating_points.csv"),
        "by_reason": pd.read_csv(MET / "recall_by_negative_reason.csv", index_col=0),
        "by_conf": pd.read_csv(MET / "accuracy_by_confidence.csv"),
        "infos": {p.stem: json.loads(p.read_text()) for p in (MET / "models").glob("*.json")},
    }
    r["best"] = r["comparison"].iloc[0]
    from sklearn.metrics import roc_auc_score

    bp = pd.read_csv(config.PREDICTIONS_DIR / f"{r['best']['slug']}_test.csv")
    r["best_auc"] = {lab: roc_auc_score(bp["y_true"] == i, bp[f"score_{lab}"]) for i, lab in enumerate(config.LABELS)}
    r["has_transformer"] ="distilroberta" in set(r["comparison"]["slug"])
    raw = pd.read_csv(config.DATA_PATH, usecols=["airline", "text"])
    delta = raw.loc[raw["airline"] == "Delta", "text"]
    r["delta_to_jetblue"] = delta.str.contains("@jetblue", case=False).mean()
    return r


# --------------------------------------------------------------------------- #
# Document helpers
# --------------------------------------------------------------------------- #
class Paper:
    def __init__(self) -> None:
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        self.labels: dict[str, int] = {}  # "fig:<stem>" / "tab:<key>" → number, resolved in save()
        self._styles()

    def _styles(self) -> None:
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Inches(1))
        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = FONT, Pt(11)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = normal.paragraph_format
        pf.space_after, pf.line_spacing = Pt(6), 1.15
        for level, size in ((1, 15), (2, 12.5), (3, 11)):
            st = self.doc.styles[f"Heading {level}"]
            st.font.name, st.font.size, st.font.bold = FONT, Pt(size), True
            st.font.color.rgb = ACCENT
            st.element.rPr.rFonts.set(qn("w:asciiTheme"), "")  # let the explicit font win
            st.paragraph_format.space_before = Pt(14 if level == 1 else 10)
            st.paragraph_format.space_after = Pt(4)
            st.paragraph_format.keep_with_next = True
        cap = self.doc.styles["Caption"]
        cap.font.name, cap.font.size, cap.font.italic = FONT, Pt(9.5), False
        cap.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        # Page numbers in the footer.
        p = sec.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._field(p, "PAGE")

    @staticmethod
    def _field(paragraph, code: str) -> None:
        run = paragraph.add_run()
        for tag, text in (("begin", None), (None, code), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)

    # -- text ----------------------------------------------------------------
    def h(self, text: str, level: int = 1) -> None:
        self.doc.add_heading(text, level)

    def p(self, text: str = "", style: str | None = None, align=None, italic=False):
        """Paragraph with **bold** and *italic* inline markup."""
        para = self.doc.add_paragraph(style=style)
        if align is not None:
            para.alignment = align
        self._runs(para, text, italic)
        return para

    @staticmethod
    def _runs(para, text: str, italic=False) -> None:
        import re

        for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("**"):
                run = para.add_run(part[2:-2])
                run.bold = True
            elif part.startswith("`"):
                run = para.add_run(part[1:-1])
                run.font.name = "Consolas"
                run.font.size = Pt(9.5)
            elif part.startswith("*"):
                run = para.add_run(part[1:-1])
                run.italic = True
            else:
                run = para.add_run(part)
            if italic:
                run.italic = True

    def bullets(self, items: list[str], numbered: bool = False) -> None:
        style = "List Number" if numbered else "List Bullet"
        for item in items:
            para = self.p(item, style=style)
            para.paragraph_format.space_after = Pt(3)

    def code(self, text: str) -> None:
        for line in text.strip("\n").split("\n"):
            para = self.doc.add_paragraph()
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.line_spacing = 1.0
            para.paragraph_format.left_indent = Inches(0.25)
            run = para.add_run(line or " ")
            run.font.name, run.font.size = "Consolas", Pt(8.5)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # -- figures and tables -----------------------------------------------------
    def figure(self, path: Path, caption: str, width: float = 6.0) -> None:
        """Insert a figure; cite it in text as ``[[fig:<file stem>]]``."""
        if not path.exists():
            self.p(f"[Figure not available: {path.name}. Run the pipeline to generate it.]", italic=True)
            return
        self.fig_no += 1
        self.labels[f"fig:{path.stem}"] = self.fig_no
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Inches(width))
        cap = self.p(f"**Figure {self.fig_no}.** {caption}", style="Caption")
        cap.paragraph_format.space_after = Pt(10)

    def table(self, df: pd.DataFrame, caption: str, widths: list[float] | None = None,
              note: str | None = None, font_size: float = 9, key: str | None = None) -> None:
        """Insert a table; cite it in text as ``[[tab:<key>]]``."""
        self.tab_no += 1
        if key:
            self.labels[f"tab:{key}"] = self.tab_no
        cap = self.p(f"**Table {self.tab_no}.** {caption}", style="Caption")
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_after = Pt(3)
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = widths or [6.5 / len(df.columns)] * len(df.columns)
        for i, col in enumerate(df.columns):
            self._cell(t.rows[0].cells[i], str(col), font_size, bold=True, shade="DCE6F2", width=widths[i])
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, val in enumerate(row):
                numeric = isinstance(val, (int, float)) and not isinstance(val, bool)
                self._cell(cells[i], str(val), font_size, width=widths[i],
                           align=WD_ALIGN_PARAGRAPH.RIGHT if numeric else None)
        # Keep the table (and its note) on one page; repeat the header if it is ever split.
        rows = t.rows if note else t.rows[:-1]
        for row in rows:
            for c in row.cells:
                for para in c.paragraphs:
                    para.paragraph_format.keep_with_next = True
        tr_pr = t.rows[0]._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader")
        el.set(qn("w:val"), "true")
        tr_pr.append(el)
        if note:
            n = self.p(note, style="Caption")
            n.paragraph_format.space_before = Pt(2)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    @staticmethod
    def _cell(cell, text, size, bold=False, shade=None, width=None, align=None) -> None:
        cell.text = ""
        para = cell.paragraphs[0]
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = 1.0
        if align is not None:
            para.alignment = align
        run = para.add_run(text)
        run.font.size, run.bold = Pt(size), bold
        if width:
            cell.width = Inches(width)
        if shade:
            sh = OxmlElement("w:shd")
            sh.set(qn("w:val"), "clear")
            sh.set(qn("w:color"), "auto")
            sh.set(qn("w:fill"), shade)
            cell._tc.get_or_add_tcPr().append(sh)

    def page_break(self) -> None:
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def _resolve_references(self) -> None:
        import re

        def sub(match):
            kind, key = match.group(1), match.group(2)
            label = f"{kind}:{key}"
            if label not in self.labels:
                raise KeyError(f"Unresolved cross-reference [[{label}]]")
            return f"{'Figure' if kind == 'fig' else 'Table'} {self.labels[label]}"

        for para in self.doc.paragraphs:
            for run in para.runs:
                if "[[" in run.text:
                    run.text = re.sub(r"\[\[(fig|tab):([\w-]+)\]\]", sub, run.text)

    def save(self, path: Path) -> Path:
        self._resolve_references()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.core_properties.author = ""
        self.doc.core_properties.last_modified_by = ""
        self.doc.core_properties.comments = ""
        self.doc.core_properties.keywords = ""
        self.doc.save(path)
        return path


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def f3(x: float) -> str:
    return f"{x:.3f}"


# --------------------------------------------------------------------------- #
# Paper content
# --------------------------------------------------------------------------- #
def build(args) -> Path:
    r = load_results()
    P = Paper()
    cmp_, best, infos = r["comparison"], r["best"], r["infos"]
    by_slug = cmp_.set_index("slug")
    n_models = len(cmp_)
    sent = r["sentiment"]
    ov, cl, prep = r["overview"], r["cleaning"], r["prep"]
    n_clean = int(cl["clean_rows"])
    neg_share, neu_share, pos_share = (sent.loc[k, "percent"] for k in config.LABELS)
    abl = r["ablation"].set_index("variant")["cv_macro_f1_mean"]
    abl_vocab = r["ablation"].set_index("variant")["vocabulary_size"]
    v_tok = "Normalised + tokenised (keep stop words, no lemmatisation)"
    v_full = "+ lemmatisation (full pipeline – used)"
    v_neg = "Full pipeline but negations removed as stop words"
    v_stop = "+ stop-word removal (negations kept)"
    pc = r["per_class"].set_index(["model", "class"])
    best_pc = pc.loc[best["model"]]
    sig = r["mcnemar"]
    tied = sig.loc[~sig["significant_holm_at_0.05"], "comparison"].str.split(" vs ").str[1].tolist()
    classical_slugs = ["nb_tfidf", "lr_tfidf", "svm_tfidf", "svm_word_char", "lr_word2vec"]
    best_classical = cmp_[cmp_["slug"].isin(classical_slugs)].iloc[0]
    neural = cmp_[cmp_["slug"].isin(["bilstm", "distilroberta"])]
    kpis, triage = r["kpis"], r["triage"].set_index("target_recall")
    conf = r["by_conf"][r["by_conf"]["slug"] == best["slug"]].set_index("confidence_band")["accuracy"]
    reasons_recall = r["by_reason"]["recall"]
    cant_tell_recall = reasons_recall.get("Can't Tell", float("nan"))
    swapped = kpis.index[kpis["actual_rank"] != kpis["predicted_rank"]].tolist()
    tr_info = infos.get("distilroberta")
    model_word = "seven" if r["has_transformer"] else "six"

    # ---------------------------------------------------------------- title page
    for _ in range(6):
        P.doc.add_paragraph()
    t = P.p("**Sentiment Analysis for Customer Feedback:**", align=WD_ALIGN_PARAGRAPH.CENTER)
    t.runs[0].font.size = Pt(22)
    t.runs[0].font.color.rgb = ACCENT
    t = P.p("Classifying Airline Customer Feedback with Classical Machine Learning and a Bidirectional LSTM",
            align=WD_ALIGN_PARAGRAPH.CENTER)
    t.runs[0].font.size = Pt(15)
    P.doc.add_paragraph()
    P.p("Research Paper · Case Study Part 1", align=WD_ALIGN_PARAGRAPH.CENTER)
    P.doc.add_paragraph()
    for label, value in (("Author", args.author), ("Course", args.course), ("Instructor", args.instructor),
                         ("Date", args.date)):
        if (label, value) in (("Course", "[Course Name and Number]"),
                              ("Instructor", "[Instructor Name]")):
            continue
        para = P.p(f"{label}: ", align=WD_ALIGN_PARAGRAPH.CENTER)
        run = para.add_run(value)
        if value.startswith("["):
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    P.page_break()

    # ---------------------------------------------------------------- abstract
    P.h("Abstract")
    tr_sentence = (
        f"a fine-tuned DistilRoBERTa Transformer reached macro-F1 {f3(by_slug.loc['distilroberta', 'f1_macro'])}"
        if r["has_transformer"] else
        "DistilRoBERTa training code is included as an optional extension, but completed evaluation results are unavailable"
    )
    P.p(
        "This study evaluates three-class sentiment classification of airline customer feedback. The corpus is "
        "predominantly negative. The evaluated models assign customer tweets "
        "to negative, neutral or positive classes. Using the "
        f"Twitter US Airline Sentiment dataset ({n_clean:,} tweets after cleaning; {neg_share:.0f}% negative, "
        f"{neu_share:.0f}% neutral, {pos_share:.0f}% positive), tweets were normalised, tokenised, lemmatised and "
        "filtered for stop words and punctuation, then represented with TF-IDF word and character n-grams and "
        f"Word2Vec embeddings. The experiment compared {model_word} configurations: Multinomial Naive Bayes and Logistic "
        "Regression on word TF-IDF, Linear Support Vector Machines (SVM) on word TF-IDF and combined "
        "word-and-character TF-IDF, Logistic Regression on Word2Vec, and a bidirectional LSTM recurrent network"
        + (", and a fine-tuned DistilRoBERTa Transformer" if r["has_transformer"] else "")
        + ". Classical models were tuned by stratified 5-fold cross-validation, and every model was evaluated "
        "on the same test set with accuracy, macro-averaged precision, recall and F1, ROC-AUC, bootstrap "
        f"confidence intervals and McNemar's test. The highest test macro-F1 was obtained by the **{best['model']}**, with accuracy "
        f"{f3(best['accuracy'])} and macro-F1 {f3(best['f1_macro'])} (95% CI {f3(best['f1_macro_ci_low'])}–"
        f"{f3(best['f1_macro_ci_high'])}); "
        + (f"the BiLSTM reached macro-F1 {f3(by_slug.loc['bilstm', 'f1_macro'])}; " if "bilstm" in by_slug.index else "")
        + f"{tr_sentence}. Neutral tweets had the lowest class-specific F1 (F1 {f3(best_pc.loc['neutral', 'f1'])}). "
        f"The mean absolute discrepancy between predicted and annotated airline Net Sentiment Scores was "
        f"{kpis['abs_error_nss'].mean():.1f} points. At an exploratory threshold selected to recover 95% of negative "
        f"tweets, {pct(triage.loc[0.95, 'precision'], 0)} of flagged tweets were negative. These estimates indicate "
        "possible applications in complaint triage and sentiment monitoring. The triage "
        "thresholds were selected on the test set and are exploratory; deployment requires independent validation."
    )
    P.p("*Keywords:* sentiment analysis, natural language processing, customer feedback, airline industry, "
        "TF-IDF, support vector machine, LSTM, Transformer")

    # ---------------------------------------------------------------- introduction
    P.h("1. Introduction")
    P.h("1.1 Background", 2)
    P.p(
        "Sentiment analysis is the automatic identification of the opinion or attitude expressed in text "
        "(Pang & Lee, 2008; Liu, 2012). Lexicon-based methods assign sentiment using predefined word lists. Supervised classifiers "
        "learn decision functions from labelled text represented as bag-of-words features. Recurrent "
        "networks encode token sequences (Hochreiter & Schmidhuber, 1997), while pretrained "
        "Transformer language models learn contextual representations that can be fine-tuned for classification (Devlin et al., 2019; Liu et al., 2019)."
    )
    P.h("1.2 Relevance to the airline industry", 2)
    P.p(
        "Airline customer feedback includes reports of delays, cancellations, baggage problems and service "
        "interactions. Classifying these messages can help service teams prioritize review and summarize "
        "patterns in feedback. This study examines those potential uses through a historical Twitter dataset. "
        "Sentiment classification identifies the expressed attitude; it does not establish the causes of "
        "dissatisfaction or the effects of a business intervention."
    )
    P.h("1.3 Research objectives", 2)
    P.bullets([
        "Collect and describe a labelled customer-feedback dataset from the airline industry that covers all three sentiment classes.",
        "Design and justify a preprocessing pipeline for noisy social-media text, and measure the effect of each step.",
        "Compare sparse TF-IDF features with dense Word2Vec representations and a BiLSTM sequence model.",
        "Train and tune Naive Bayes, Logistic Regression, SVM and BiLSTM models; document Transformer code as an optional extension.",
        "Evaluate all models on the same held-out test set with accuracy, precision, recall, F1 and ROC analysis, and test whether differences are statistically significant.",
        "Assess the implications of classification errors for complaint routing and aggregate sentiment reporting.",
    ], numbered=True)

    # ---------------------------------------------------------------- data
    P.h("2. Data Collection")
    P.h("2.1 Source and description", 2)
    P.p(
        "The study uses the **Twitter US Airline Sentiment** dataset, released by CrowdFlower (now Figure Eight) "
        "and distributed on Kaggle (Figure Eight, 2015). It contains "
        f"{int(cl['raw_rows']):,} tweets addressed to six major US airlines between {ov['First tweet (UTC)']} "
        f"and {ov['Last tweet (UTC)']}. Human annotators labelled each tweet as negative, neutral or positive and "
        "recorded a confidence score. For negative tweets they also chose a reason from ten categories, such as "
        "*Late Flight* or *Customer Service Issue*. The fields used in this study are the tweet text, the "
        "sentiment label and its confidence, the negative reason, the airline and the timestamp."
    )
    P.p(
        "**Industry focus.** The dataset represents the **airline (hospitality and travel) industry**. It was chosen "
        "because it is publicly available, labelled by humans, includes all three sentiment classes, and comes with "
        "complaint reasons that make it possible to link model output to operational decisions."
    )
    P.h("2.2 Data cleaning", 2)
    P.p("The data audit identified duplicate identifiers, inconsistent airline assignments and text artefacts:")
    P.bullets([
        f"**Duplicate tweets.** {cl['duplicate_rows_removed']} rows repeated an existing tweet ID, and "
        f"{cl['duplicate_ids_with_conflicting_labels']} duplicate identifiers were associated with conflicting sentiment labels. "
        "One row per tweet was kept, choosing the annotation with the highest confidence.",
        f"**Airline assignment.** Of the {cl['airline_rows_relabelled']:,} rows labelled *Delta*, "
        f"{pct(r['delta_to_jetblue'])} contain @JetBlue. These rows were reassigned to JetBlue for airline-level "
        "analysis. This is a dataset-level assumption rather than verification of the intended carrier for "
        "every tweet. Tweet text and sentiment labels were retained.",
        "**Text artefacts.** The distributed text contains unusual strings such as \"Late Flightr\" and "
        "\"Cancelled Flightled\", consistent with substitution artefacts; their origin cannot be established "
        "from this CSV. These strings were left in place rather than applying an unverified repair. They "
        "show up in the vocabulary (e.g. the token *flightled*).",
    ])
    P.p(f"After cleaning, **{n_clean:,} unique tweets** remain.")
    P.h("2.3 Summary statistics", 2)
    P.table(pd.DataFrame({"Statistic": ov.index, "Value": ov.values}), "Dataset overview after cleaning.",
            widths=[3.5, 2.0], font_size=9.5)
    sent_tab = sent.reset_index().rename(columns={"airline_sentiment": "Sentiment", "count": "Tweets",
                                                  "percent": "Share (%)"})
    P.table(sent_tab, "Distribution of sentiment labels.", widths=[2.0, 1.5, 1.5], font_size=9.5)
    P.p(
        f"The class distribution is **imbalanced**: {neg_share}% of tweets are negative, {neu_share}% neutral and "
        f"{pos_share}% positive ([[fig:eda_sentiment_distribution]]). A classifier that always predicted *negative* would already reach "
        f"{neg_share:.0f}% accuracy, so macro-averaged F1, which weights the three classes equally, was chosen as "
        "the main evaluation metric. Sentiment also differs strongly by airline ([[tab:airline]] and [[fig:eda_sentiment_by_airline]]): "
        f"{r['by_airline']['% negative'].idxmax()} has the highest share of negative tweets "
        f"({r['by_airline']['% negative'].max()}%) and {r['by_airline']['% negative'].idxmin()} the lowest "
        f"({r['by_airline']['% negative'].min()}%)."
    )
    P.figure(FIG / "eda_sentiment_distribution.png", "Distribution of sentiment labels in the cleaned dataset.", 4.8)
    ba = r["by_airline"].reset_index().rename(columns={"airline": "Airline", "total": "Total"})
    ba.columns = [c.capitalize() if c in config.LABELS else c for c in ba.columns]
    P.table(ba, "Sentiment counts by airline (sorted by tweet volume).", widths=[1.5, 1, 1, 1, 0.9, 1.1], key="airline")
    P.figure(FIG / "eda_sentiment_by_airline.png", "Sentiment mix per airline.", 5.6)
    rs = r["reasons"]
    P.p(
        f"Among negative tweets, **{rs.index[0]}** ({rs['percent'].iloc[0]}%) and **{rs.index[1]}** "
        f"({rs['percent'].iloc[1]}%) are the most frequent complaint reasons, followed by {rs.index[2]} "
        f"({rs['percent'].iloc[2]}%) and {rs.index[3]} ({rs['percent'].iloc[3]}%) ([[fig:eda_negative_reasons]]). Tweets average "
        f"{ov['Mean words per tweet']} words. Annotator confidence is below 1.0 for "
        f"{ov['% tweets with confidence < 1']}% of tweets, which means a sizeable part of the labels are "
        "themselves uncertain."
    )
    P.figure(FIG / "eda_negative_reasons.png", "Reasons given by annotators for negative tweets.", 5.6)

    # ---------------------------------------------------------------- preprocessing
    P.h("3. Data Preprocessing")
    P.h("3.1 Cleaning steps", 2)
    P.p(
        "All classical models share one preprocessing pipeline, implemented in `src/preprocessing.py`. It applies "
        "the steps required by the assignment (tokenisation, lemmatisation, stop-word and punctuation removal) "
        "following tweet-specific normalisation:"
    )
    P.bullets([
        "**Normalisation:** decode HTML entities; remove URLs and @mentions; split hashtags into words "
        "(#BadService → bad service); convert emoji and emoticons into word tokens (the angry-face emoji → "
        "emo_pouting_face, "
        ":) → emo_smile); lower-case; expand contractions (can't → can not); shorten character elongations "
        "(soooo → soo).",
        "**Tokenisation:** NLTK's TweetTokenizer (Bird et al., 2009), which is designed for social-media text.",
        "**Lemmatisation:** WordNet lemmatisation guided by part-of-speech tags (delayed → delay, "
        "flights → flight). Tagging is done before stop words are removed, so the tagger sees full sentences.",
        "**Punctuation and number removal:** keep only alphabetic tokens of at least two characters.",
        "**Stop-word removal:** the NLTK English stop-word list, *except* negations and intensifiers "
        "(not, no, never, nothing, very, too, …), which are kept.",
    ])
    P.p(
        "Example: \"@united I wasn't happy, flights were DELAYED!!! #BadService\" becomes "
        "[not, happy, flight, delay, bad, service]."
    )
    P.h("3.2 Justification", 2)
    P.bullets([
        "**Mentions and URLs** were removed to limit reliance on airline identifiers and linked content. "
        "Retaining them could allow a classifier to exploit airline-specific class frequencies.",
        "**Emoji** were mapped to lexical tokens to retain affective information. Observed positive-class "
        "associations include emo_smile and emo_folded_hands.",
        "**Negations are kept** because removing *not* turns \"not happy\" into \"happy\" and reverses the meaning.",
        "**Lemmatisation and stop-word removal** reduce the vocabulary and the sparsity of the feature space, "
        "which makes models smaller and easier to interpret.",
        "**The Transformer receives lightly cleaned text** (mentions → @user, links → http, otherwise "
        "unchanged). Pre-trained language models learned from natural text and use stop words, punctuation and "
        "casing, so aggressive cleaning would remove information they rely on.",
        "**BiLSTM preprocessing** retains stop words to preserve sequence structure and potential cues to "
        "negation scope.",
    ])
    P.p(
        f"Preprocessing reduced the corpus from {int(prep['Raw whitespace tokens']):,} raw tokens to "
        f"{int(prep['Tokens after preprocessing']):,}, and the vocabulary from "
        f"{int(prep['Raw vocabulary (lower-cased)']):,} to {int(prep['Vocabulary after preprocessing']):,} "
        f"distinct terms. Only {int(prep['Tweets empty after preprocessing'])} tweets became empty. [[fig:eda_distinctive_terms]] shows "
        "the most distinctive terms per class after preprocessing, measured by weighted log-odds with an "
        "informative Dirichlet prior (Monroe et al., 2008). The identified terms include waiting and "
        "delays (*hour, hold, delay, cancel*) for negative tweets, gratitude for positive tweets, and requests "
        "and information (*dm, follow, please, tomorrow*) for neutral tweets."
    )
    P.figure(FIG / "eda_distinctive_terms.png", "Most distinctive terms per sentiment class (weighted log-odds z-scores).", 6.2)
    P.h("3.3 Preprocessing ablation", 2)
    P.p(
        "Each preprocessing step was added one at a time and scored with the tuned Linear SVM, using 5-fold "
        "cross-validated macro-F1 on the combined training and validation data ([[tab:ablation]])."
    )
    ab = r["ablation"].assign(**{
        "CV macro-F1": lambda d: d["cv_macro_f1_mean"].map(f3),
        "± SD": lambda d: d["cv_macro_f1_std"].map(f3),
        "Vocabulary": lambda d: d["vocabulary_size"].map("{:,}".format),
    })[["variant", "CV macro-F1", "± SD", "Vocabulary"]].rename(columns={"variant": "Preprocessing variant"})
    P.table(ab, "Preprocessing ablation (Linear SVM, 5-fold CV on train + validation).", widths=[3.9, 1.0, 0.7, 0.9], key="ablation")
    P.p(
        f"The full pipeline reduced the vocabulary by "
        f"{pct(1 - abl_vocab[v_full] / abl_vocab[v_tok], 0)} but did **not** improve macro-F1: "
        f"macro-F1 fell from {f3(abl[v_tok])} (tokenised, stop words kept) to {f3(abl[v_stop])} after stop-word "
        f"removal and {f3(abl[v_full])} after lemmatisation. Removing negations as well lowered it further to "
        f"{f3(abl[v_neg])}, consistent with retaining negation terms in this configuration. Short function words "
        "such as *but*, *so*, *why* and *again* carry tone in complaints, and bigrams that contain them (\"why not\", "
        "\"again\") are lost when these tokens are removed. The main comparison retained the full pipeline "
        "to meet the preprocessing specification and reduce vocabulary size. The lower macro-F1 is a "
        "limitation of this choice. These descriptive cross-validation results use fixed tuned hyperparameters "
        "and do not isolate preprocessing effects under independent model selection."
    )

    # ---------------------------------------------------------------- features
    P.h("4. Feature Extraction")
    P.p("Three numerical representations were compared (`src/features.py`):")
    P.bullets([
        "**TF-IDF over word unigrams and bigrams** (Salton & Buckley, 1988). Each tweet is a sparse vector of "
        "term frequency × inverse document frequency. Sub-linear term frequency (1 + log tf) was used, and terms "
        "seen in fewer than two tweets were dropped. Bigrams capture short phrases such as *not happy* and "
        "*customer service*.",
        "**TF-IDF over word and character n-grams.** The word features above combined with character 2–5-grams "
        "(up to 60,000). These are robust to misspellings, elongations and creative spelling, which are common "
        "in tweets.",
        "**Word2Vec mean embeddings** (Mikolov et al., 2013). A 100-dimensional skip-gram model trained only on "
        "the training tweets in each fold; a tweet is the IDF-weighted average of its word vectors.",
        "**Neural representations.** The BiLSTM starts with context-independent Word2Vec token embeddings "
        "and produces contextual hidden states. The optional Transformer uses contextual sub-word states; "
        "Transformer evaluation results are unavailable for the present experiment.",
    ])
    P.p(
        "All vectorisers are fitted inside scikit-learn pipelines, so vocabulary and IDF weights are learned only "
        "from the training part of each cross-validation fold and never from held-out data. [[fig:feature_space]] projects "
        "the TF-IDF space to two dimensions for a balanced sample of 3,000 tweets. The projections show class-specific "
        "regions with overlap between neutral tweets and both polar classes. The sample contains equal class counts, so spatial extent does not represent class prevalence. This overlap "
        "is descriptive; classification difficulty is assessed using held-out metrics rather than the two-dimensional projection."
    )
    P.figure(FIG / "feature_space.png", "TF-IDF feature space: truncated SVD (left) and t-SNE (right) projections.", 6.3)

    # ---------------------------------------------------------------- models
    P.h("5. Model Development")
    P.h("5.1 Algorithms and rationale", 2)
    model_rows = [
        ("Naive Bayes", "Multinomial NB + word TF-IDF", "Fast probabilistic baseline; strong on short texts (McCallum & Nigam, 1998)"),
        ("Logistic Regression", "Multinomial LR + word TF-IDF", "Class probabilities (calibration not assessed); interpretable coefficients"),
        ("Linear SVM", "Linear SVM + word TF-IDF", "Max-margin classifier; strong baseline for sparse text (Joachims, 1998)"),
        ("Linear SVM (word+char)", "Linear SVM + word & char TF-IDF", "Adds robustness to spelling variation"),
        ("LR (Word2Vec)", "Multinomial LR + Word2Vec", "Tests dense embeddings against sparse TF-IDF"),
        ("BiLSTM (RNN)", "Bidirectional LSTM", "Reads words in order; can model negation scope (Schuster & Paliwal, 1997)"),
        ("DistilRoBERTa", "Fine-tuned Transformer", "Transfers language knowledge from pre-training (Sanh et al., 2019)"),
    ]
    P.table(pd.DataFrame(model_rows, columns=["Model", "Algorithm / features", "Rationale"]),
            "Model specifications and selection rationale; the Transformer is an unevaluated extension.", widths=[1.4, 1.9, 3.2])
    P.h("5.2 Data splitting and training strategy", 2)
    P.p(
        "The cleaned data were split once into **70% training, 15% validation and 15% test**, stratified by "
        f"sentiment with a fixed random seed ({config.SEED}). The test set ({len(pd.read_csv(config.PREDICTIONS_DIR / f'{best.slug}_test.csv')):,} "
        "tweets) was excluded from parameter fitting, and every evaluated model was scored on the same tweets. "
        "The test set was also reused for exploratory model ranking and business diagnostics."
    )
    P.bullets([
        "**Classical models:** grid search over the hyper-parameters in [[tab:grid]] with stratified 5-fold "
        "cross-validation on training + validation data, using macro-F1 as the selection criterion. The grid "
        "includes class weighting (none vs. balanced) to address imbalance. The best configuration was then "
        "refitted on all training + validation data.",
        "**BiLSTM:** trained on the training set; the validation set was used for early stopping.",
        "**Optional Transformer:** the implementation specifies training on the training partition and "
        "checkpoint selection by validation macro-F1. No completed run is included.",
    ])
    P.p("These training strategies use unequal fitting budgets: classical models are refitted on 85% of "
        "the corpus, whereas neural models fit on 70% and select checkpoints on 15%. Comparisons describe "
        "these complete pipelines and do not isolate an algorithm-only advantage.")
    grid_rows = []
    for slug, grid in (
        ("nb_tfidf", "n-gram range {(1,1), (1,2)}; α ∈ {0.05, 0.1, 0.3, 0.5, 1}; fit prior ∈ {yes, no}"),
        ("lr_tfidf", "n-gram range {(1,1), (1,2)}; C ∈ {0.1, 0.25, 0.5, 1, 2, 5}; class weight"),
        ("svm_tfidf", "n-gram range {(1,1), (1,2)}; C ∈ {0.05, 0.1, 0.25, 0.5, 1}; class weight"),
        ("svm_word_char", "C ∈ {0.01, 0.025, 0.05, 0.1, 0.25}; class weight"),
        ("lr_word2vec", "C ∈ {0.01, 0.1, 1}; class weight"),
    ):
        if slug in infos:
            i = infos[slug]
            chosen = ", ".join(f"{k.split('__')[1]}={tuple(v) if isinstance(v, list) else v}"
                               for k, v in i["best_params"].items())
            grid_rows.append((i["name"], grid, chosen,
                              f"{f3(i['cv_macro_f1_mean'])} ± {f3(i['cv_macro_f1_std'])}"))
    P.table(pd.DataFrame(grid_rows, columns=["Model", "Current code grid", "Saved selected parameters", "Saved CV macro-F1"]),
            "Current search definitions and saved tuning results (5-fold CV).", widths=[1.3, 2.6, 1.6, 1.0], font_size=8.5, key="grid")
    P.p("Search provenance limitation: the saved Logistic Regression TF-IDF search records 20 candidates "
        "whereas the current grid defines 24; word+character SVM records 8 whereas the current grid defines 10. "
        "The original grids were not saved, so the exact historical searches cannot be reconstructed from "
        "candidate counts. Selected parameters and test metrics are verifiable; future searches save their grids.")
    P.p(
        "Balanced class weights were selected for saved LR and SVM models; the best Naive Bayes configuration "
        "instead ignores the class prior (fit prior = no), which has a similar effect. This is consistent with the hypothesis that "
        "correcting for the imbalance helps the minority classes."
    )
    P.h("5.3 Recurrent neural network (BiLSTM)", 2)
    bl = infos.get("bilstm", {})
    blc = bl.get("config", {})
    P.p(
        "Architecture: an embedding layer (100 dimensions, initialised from Word2Vec trained on the training "
        "tweets) → dropout → one bidirectional LSTM layer with 128 units in each direction → max- and mean-pooling "
        "over time → dropout (0.4) → a linear output layer with softmax over the three classes. "
        + (f"The model has {bl['n_parameters']:,} parameters and a vocabulary of {bl['vocab_size']:,} words. "
           if bl else "")
        + f"Training used Adam (learning rate {blc.get('lr', 0.001)}), batch size {blc.get('batch_size', 64)}, "
        "class-weighted cross-entropy, gradient clipping at 1.0 and sequences of up to "
        f"{blc.get('max_len', 40)} tokens. Training stopped early once validation macro-F1 had not improved for "
        f"{blc.get('patience', 3)} epochs"
        + (f"; it ran for {bl['epochs_trained']} epochs and the best epoch was {bl['best_epoch']} "
           f"(validation macro-F1 {f3(bl['best_val_macro_f1'])}), taking {bl['training_seconds'] / 60:.0f} "
           "minutes on a CPU ([[fig:training_bilstm]])." if bl else ".")
    )
    P.figure(FIG / "training_bilstm.png", "BiLSTM training curves: the validation loss rises after a few epochs, "
             "and early stopping keeps the best epoch.", 5.6)
    P.h("5.4 Transformer (DistilRoBERTa)", 2)
    P.p(
        "`distilroberta-base` (82 million parameters) is a 6-layer distilled version of RoBERTa (Liu et al., 2019; "
        "Sanh et al., 2019). Its reduced depth motivates evaluation under constrained computational resources; "
        "relative accuracy and inference cost were not measured in this study. The implementation adds a "
        "classification head with three outputs and uses "
        "AdamW (learning rate 2e-5, weight decay 0.01), batch size 32, 3 epochs, a "
        "linear schedule with 10% warm-up, and a maximum of 64 sub-word tokens per tweet (Wolf et al., 2020)."
    )
    if r["has_transformer"]:
        P.p(f"Fine-tuning took {tr_info['training_seconds'] / 60:.0f} minutes; the best validation macro-F1 was "
            f"{f3(tr_info['best_val_macro_f1'])} (epoch {tr_info['best_epoch']}).")
        P.figure(FIG / "training_distilroberta.png", "DistilRoBERTa fine-tuning curves.", 5.6)
    else:
        P.p(
            "The optional Transformer implementation (`src/transformer_model.py`) can be run with "
            "`python scripts/run_pipeline.py --stages transformer evaluate insights`. No completed Transformer "
            "predictions or training metadata are available in this submission, so the Transformer is **not** "
            "included in the model comparison. Completing its training and evaluation is a proposed "
            "extension (Section 7.3)."
        )

    # ---------------------------------------------------------------- evaluation
    P.h("6. Model Evaluation")
    P.h("6.1 Metrics", 2)
    P.p(
        "All models were evaluated on the same held-out test set with: **accuracy**; **precision**, **recall** "
        "and **F1** per class and macro-averaged (each class weighted equally, the primary metric); weighted "
        "F1; and **one-vs-rest ROC-AUC** (macro), computed from predicted probabilities or SVM decision scores. "
        "95% confidence intervals were estimated by bootstrapping the test set 1,000 times (Efron & Tibshirani, "
        "1993), and pairs of models were compared with **McNemar's test** (Dietterich, 1998), with Holm "
        "adjustment across comparisons. McNemar tests paired classification error rates, not macro-F1. "
        "Failure to reject does not establish equivalence; comparisons against a test-ranked winner are exploratory."
    )
    P.h("6.2 Comparison of models", 2)
    ct = cmp_.assign(**{
        "Accuracy": cmp_["accuracy"].map(f3),
        "Precision": cmp_["precision_macro"].map(f3),
        "Recall": cmp_["recall_macro"].map(f3),
        "F1": cmp_["f1_macro"].map(f3),
        "F1 95% CI": cmp_.apply(lambda x: f"{f3(x.f1_macro_ci_low)}–{f3(x.f1_macro_ci_high)}", axis=1),
        "ROC-AUC": cmp_["roc_auc_macro_ovr"].map(f3),
        "ms / 1k tweets": cmp_["slug"].map(lambda s: f"{infos.get(s, {}).get('inference_ms_per_1k', float('nan')):,.0f}"),
    })[["model", "Accuracy", "Precision", "Recall", "F1", "F1 95% CI", "ROC-AUC", "ms / 1k tweets"]]
    P.table(ct.rename(columns={"model": "Model"}),
            "Test-set performance, sorted by macro-F1. Precision, recall and F1 are macro-averaged.",
            widths=[1.75, 0.68, 0.68, 0.6, 0.55, 0.9, 0.65, 0.69], font_size=8.5, key="comparison",
            note="ms / 1k tweets = CPU prediction time for 1,000 tweets, including feature extraction.")
    second = cmp_.iloc[1]
    P.p(
        f"The **{best['model']}** achieved the highest macro-F1 ({f3(best['f1_macro'])}) and accuracy "
        f"({f3(best['accuracy'])}). {second['model']} was second (macro-F1 {f3(second['f1_macro'])})"
        + (f" and had the highest ROC-AUC ({f3(cmp_['roc_auc_macro_ovr'].max())})"
           if cmp_['roc_auc_macro_ovr'].idxmax() == 1 else "")
        + ". "
        + (f"Holm-adjusted McNemar tests did not detect a difference in error rates against "
           f"{', '.join(tied)} (adjusted p ≥ 0.05); differences against the remaining models were detected "
           "([[tab:mcnemar]]). " if tied else "Holm-adjusted McNemar tests detected error-rate differences "
           "against every other model ([[tab:mcnemar]]). ")
        + "Adding character n-grams improved the SVM slightly, and dense Word2Vec averages performed worst: "
        f"macro-F1 {f3(by_slug.loc['lr_word2vec', 'f1_macro'])}. One possible explanation is that averaging word vectors loses word "
        "order and the negation cues that TF-IDF bigrams keep. Naive Bayes was competitive on accuracy but weaker "
        "on the minority classes. Prediction cost also differs: the word-level linear models score 1,000 tweets "
        f"in about {infos['svm_tfidf']['inference_ms_per_1k']:.0f} ms on a CPU, compared with "
        f"{infos['bilstm']['inference_ms_per_1k']:.0f} ms for the BiLSTM ([[tab:comparison]]). [[fig:model_comparison]] shows the confidence intervals."
    )
    P.figure(FIG / "model_comparison.png", "Accuracy and macro-F1 per model with 95% bootstrap confidence intervals.", 6.0)
    mc = sig.assign(**{
        "Only best correct": sig["a_only_correct"], "Only other correct": sig["b_only_correct"],
        "χ²": sig["statistic"].map("{:.2f}".format),
        "p-value": sig["p_value_holm"].map(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.3f}"),
        "Significant": sig["significant_holm_at_0.05"].map({True: "Yes", False: "No"}),
    })[["comparison", "Only best correct", "Only other correct", "χ²", "p-value", "Significant"]]
    P.table(mc.rename(columns={"comparison": "Comparison"}),
            "Exploratory McNemar comparisons of error rates; p-values and significance use Holm adjustment.",
            widths=[2.6, 0.85, 0.85, 0.6, 0.8, 0.8], font_size=8.5, key="mcnemar")
    P.h("6.3 Per-class results and confusion matrices", 2)
    pcr = r["per_class"].copy()
    for c in ("precision", "recall", "f1"):
        pcr[c] = pcr[c].map(f3)
    P.table(pcr.rename(columns=str.capitalize), "Precision, recall and F1 per class for every model.",
            widths=[2.4, 1.0, 0.9, 0.8, 0.7, 0.7], font_size=8.5)
    P.p(
        f"For the best model, F1 is {f3(best_pc.loc['negative', 'f1'])} for negative, "
        f"{f3(best_pc.loc['positive', 'f1'])} for positive and {f3(best_pc.loc['neutral', 'f1'])} for "
        "neutral tweets. The confusion matrices ([[fig:confusion_matrices]]) indicate that many errors involve neutral tweets classified as "
        "negative and negative tweets classified as neutral. Questions about an operational issue, such as \"is flight 1234 "
        "on time?\", may share vocabulary with complaints; this is a plausible explanation rather than a separately tested mechanism. The BiLSTM has the highest neutral recall "
        f"({f3(pc.loc[('BiLSTM (RNN)', 'neutral'), 'recall'])}) but lower negative recall, which illustrates a "
        "trade-off between the classes rather than a uniformly better model."
        if "BiLSTM (RNN)" in pc.index.get_level_values(0) else
        "The confusion matrices ([[fig:confusion_matrices]]) show that most errors are between the neutral and negative classes."
    )
    P.figure(FIG / "per_class_f1.png", "F1 score per class and model.", 5.8)
    P.figure(FIG / "confusion_matrices.png", "Row-normalised confusion matrices on the test set (each row "
             "shows how tweets of that true class were predicted).", 6.5)
    P.h("6.4 ROC curves", 2)
    P.p(
        "The best model's one-vs-rest ROC curves ([[fig:roc_best_model]]) give AUC values of "
        + ", ".join(f"{f3(v)} for {k}" for k, v in r["best_auc"].items())
        + f" (macro {f3(best['roc_auc_macro_ovr'])}). Neutral-vs-rest is the hardest separation, consistent with "
        "the per-class F1 scores. "
        "ROC curves for all models are shown in Appendix B."
    )
    P.figure(FIG / "roc_best_model.png", f"One-vs-rest ROC curves for the {best['model']}.", 4.6)

    # ---------------------------------------------------------------- interpretation
    P.h("7. Interpretation and Discussion")
    P.h("7.1 Language associated with sentiment", 2)
    P.p(
        "The Logistic Regression coefficients ([[fig:insight_top_coefficients]]) identify terms associated with "
        "each sentiment class in that model. Negative-class terms include waiting and delays "
        "(*hour, delay, hold, wait, cancel*), baggage (*bag, lose, luggage*) and service contact (*call, customer*). "
        "Positive-class terms include gratitude and praise (*thanks, great, love, awesome, kudos*). "
        "These associations help explain the Logistic Regression representation; they do not directly explain "
        "the leading SVM or establish causal drivers of customer satisfaction."
    )
    P.figure(FIG / "insight_top_coefficients.png", "Words with the largest Logistic Regression coefficients per class.", 6.2)
    P.h("7.2 Business insights", 2)
    t95, t90 = triage.loc[0.95], triage.loc[0.90]
    P.bullets([
        f"**Exploratory complaint triage.** With the threshold selected using test labels, the {best['model']} catches 95% of "
        f"negative tweets while {pct(t95['precision'], 0)} of the flagged tweets are truly negative; the team "
        f"would review {pct(t95['share_of_tweets_flagged'], 0)} of these test tweets. At 90% recall, precision "
        f"rises to {pct(t90['precision'], 0)} and the review load drops to {pct(t90['share_of_tweets_flagged'], 0)} "
        "([[fig:insight_negative_pr]]). These operating points quantify the observed trade-off between complaint recall and review volume in the test sample.",
        f"**Airline-level sentiment estimates.** Net Sentiment Score (% positive − % negative) calculated from model "
        f"predictions differs from the human-labelled score by {kpis['abs_error_nss'].mean():.1f} points on "
        f"average (maximum {kpis['abs_error_nss'].max():.1f}) ([[tab:nss]], [[fig:insight_airline_nss]]). "
        + (f"The airline ranking is reproduced except for {' and '.join(swapped)}, whose true scores differ by only "
           f"{kpis.loc[swapped, 'actual_nss'].max() - kpis.loc[swapped, 'actual_nss'].min():.1f} points. "
           if swapped else "The airline ranking is reproduced exactly. ")
        + "Airline comparisons therefore require uncertainty estimates, validation on recent data and periodic review of classification errors.",
        f"**Operational priorities.** {rs.index[0]} and {rs.index[1]} together make up "
        f"{rs['percent'].iloc[0] + rs['percent'].iloc[1]:.0f}% of complaint reasons. Investment in call-centre "
        "capacity, proactive delay notifications and faster rebooking could be investigated for the most frequent categories of "
        "negative feedback.",
        f"**Performance by annotation confidence.** On tweets with annotation confidence equal to 1.0, the best model is "
        f"{pct(conf.get('1.00 (maximum score)', float('nan')), 0)} accurate; on tweets with annotator confidence "
        f"between 0.60 and 0.80 accuracy is {pct(conf.get('0.60–0.80', float('nan')), 0)} ([[fig:insight_accuracy_by_confidence]]). "
        f"Among complaint types, recall is highest for clear operational events such as cancelled flights "
        f"({pct(reasons_recall.get('Cancelled Flight', float('nan')), 0)}) and lowest for complaints the annotators "
        f"themselves labelled *Can't Tell* ({pct(cant_tell_recall, 0)}). "
        "The association between annotation confidence and predictive accuracy motivates human review of ambiguous cases; performance on sarcasm was not evaluated separately.",
    ])
    kt = kpis.reset_index()[["airline", "n_test_tweets", "actual_nss", "predicted_nss", "abs_error_nss",
                             "actual_rank", "predicted_rank"]]
    kt.columns = ["Airline", "Test tweets", "Actual NSS", "Predicted NSS", "Abs. error", "Actual rank", "Predicted rank"]
    P.table(kt, "Net Sentiment Score per airline on the test set: human labels vs. model predictions.",
            widths=[1.3, 0.8, 0.85, 0.95, 0.8, 0.8, 0.95], font_size=8.5, key="nss")
    P.figure(FIG / "insight_negative_pr.png", "Precision–recall trade-off for detecting negative tweets, with the "
             "90/95/98% recall operating points marked.", 4.8)
    P.figure(FIG / "insight_airline_nss.png", "Net Sentiment Score per airline: human-labelled vs. predicted.", 5.6)
    P.figure(FIG / "insight_accuracy_by_confidence.png", "Model accuracy by annotator confidence band.", 5.0)
    P.h("7.3 Limitations and potential improvements", 2)
    P.bullets([
        "**Split dependence.** Although tweet IDs are disjoint, 13 distinct lowercased texts occur in both "
        "the test and development sets. Retweets, templates and repeated users may also induce dependence. "
        "Reported results retain the original split; grouped text/user or chronological evaluation is future work. "
        "Bootstrap intervals assume independent tweets and may understate uncertainty.",
        "**Exploratory business analysis.** Triage thresholds were selected using test labels and are not "
        "independent estimates of future operational performance. Select thresholds on validation or out-of-fold "
        "scores and evaluate on fresh data. SVM margins are not calibrated probabilities. Complaint reason "
        "summaries use existing annotations, rather than a trained reason classifier. This convenience sample "
        "cannot establish population customer satisfaction or causal effects of operational changes.",
        "**Temporal and source restrictions.** All tweets come from Twitter in February 2015. Language, platforms and "
        "customer issues change over time, so a deployed model should be retrained periodically and monitored "
        "for drift.",
        f"**Label noise.** {ov['% tweets with confidence < 1']}% of labels have annotator confidence below 1, "
        "which indicates annotation uncertainty. Confidence-based weighting and additional annotation review "
        "are potential improvements whose effects require evaluation.",
        "**Preprocessing sensitivity.** The ablation shows that stop-word removal and lemmatisation "
        "lower macro-F1 slightly for linear models. Preprocessing should be treated as a hyper-parameter "
        "tuned for each model.",
        "**Incomplete Transformer evaluation.** "
        + ("Transformer fine-tuning on a CPU is slow. "
           if r["has_transformer"] else
           "Completed Transformer training and evaluation results are unavailable. ")
        + "A future experiment could evaluate a Transformer on tweet classification, using benchmarks such as "
        "TweetEval (Barbieri et al., 2020) to inform the design. Any improvement would need to be measured.",
        "**Label granularity.** One label per tweet cannot represent mixed opinions (\"great crew, terrible delay\"). "
        "Aspect-based sentiment analysis would separate opinions about crew, punctuality, baggage and service.",
        "**Dataset artefacts.** The unusual strings described in Section 2.2 may introduce lexical noise. An independently verified "
        "version of the source text would permit evaluation of their effect.",
    ])

    # ---------------------------------------------------------------- conclusion
    P.h("8. Conclusion")
    P.p(
        f"This study compared {model_word} sentiment-classification configurations on {n_clean:,} airline customer tweets, "
        "using a documented preprocessing pipeline, three feature representations and a single held-out test set. "
        f"The **{best['model']}** performed best (accuracy {f3(best['accuracy'])}, macro-F1 {f3(best['f1_macro'])})"
        + (f", ahead of the BiLSTM ({f3(by_slug.loc['bilstm', 'f1_macro'])})" if "bilstm" in by_slug.index else "")
        + (f" and DistilRoBERTa ({f3(by_slug.loc['distilroberta', 'f1_macro'])})" if r["has_transformer"] else "")
        + ". The results favour TF-IDF linear classifiers as computationally economical candidates for short customer feedback; they were "
        "competitive with the evaluated BiLSTM in this experiment. Neutral-class performance remains weaker than performance on "
        "the other classes, making ambiguous messages a priority for further evaluation."
    )
    P.p(
        "The measured performance warrants further evaluation of complaint routing and sentiment monitoring. Exploratory "
        "thresholds demonstrate the tradeoff between missed complaints and review workload, while aggregate "
        "scores show how prediction errors can affect airline comparisons. Existing complaint annotations "
        "identify themes for operational investigation. Before deployment, thresholds should be selected on "
        "development data and assessed on fresh data, with human review and ongoing performance monitoring."
    )

    # ---------------------------------------------------------------- references
    P.h("References")
    refs = [
        "Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. In *Findings of the Association for Computational Linguistics: EMNLP 2020* (pp. 1644–1650). Association for Computational Linguistics.",
        "Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python*. O'Reilly Media.",
        "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186). Association for Computational Linguistics.",
        "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923.",
        "Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.",
        "Figure Eight. (2015). *Twitter US airline sentiment* [Data set]. Kaggle. https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment",
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.",
        "Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In *Machine Learning: ECML-98* (pp. 137–142). Springer.",
        "Liu, B. (2012). *Sentiment analysis and opinion mining*. Morgan & Claypool.",
        "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.",
        "McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. In *AAAI-98 Workshop on Learning for Text Categorization* (pp. 41–48).",
        "Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.",
        "Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and evaluation for identifying the content of political conflict. *Political Analysis, 16*(4), 372–403.",
        "Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval, 2*(1–2), 1–135.",
        "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035). https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html",
        "Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks* (pp. 45–50). ELRA.",
        "Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.",
        "Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.",
        "Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*(11), 2673–2681.",
        "Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., ... Rush, A. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of EMNLP 2020: System Demonstrations* (pp. 38–45). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-demos.6",
    ]
    for ref in refs:
        para = P.p(ref)
        para.paragraph_format.left_indent = Inches(0.5)
        para.paragraph_format.first_line_indent = Inches(-0.5)

    # ---------------------------------------------------------------- appendices
    P.page_break()
    P.h("Appendix A. Additional Exploratory Figures")
    P.figure(FIG / "eda_reasons_by_airline.png", "Complaint reasons by airline (share of each airline's negative tweets).", 6.2)
    P.figure(FIG / "eda_length_by_sentiment.png", "Tweet length by sentiment.", 5.4)
    P.figure(FIG / "eda_daily_volume.png", "Daily tweet volume by sentiment.", 5.8)
    P.h("Appendix B. ROC Curves for All Models")
    P.figure(FIG / "roc_all_models.png", "Macro-average one-vs-rest ROC curves for every model.", 4.8)
    P.h("Appendix C. Codebase and Reproducibility")
    P.p(
        "The complete, documented code is submitted with this paper. `README.md` explains installation and how to "
        "reproduce every result; `notebooks/sentiment_analysis.ipynb` walks through each step with explanations. "
        f"Stochastic components use a common seed ({config.SEED}), and the test-set membership of every tweet is "
        "recorded in `results/metrics/split_assignment.csv`. Set PYTHONHASHSEED before interpreter startup and "
        "use consistent library versions; seeds alone do not guarantee identical results across environments."
    )
    P.code("""
python scripts/run_pipeline.py          # all stages: eda, classical, ablation, rnn,
                                        # transformer, evaluate, insights
python scripts/build_report.py          # regenerate this paper from results/
jupyter notebook notebooks/sentiment_analysis.ipynb
""")
    P.p("Core of the preprocessing pipeline (`src/preprocessing.py`):")
    P.code('''
def process_corpus(self, texts):
    tokenized = [self.tokenize(t) for t in texts]          # normalise + TweetTokenizer
    if self.lemmatize:
        tagged = nltk.pos_tag_sents(tokenized)              # POS tags before stop-word removal
        tokenized = [[self._lemma(tok, tag) for tok, tag in sent] for sent in tagged]
    return [self._filter(tokens) for tokens in tokenized]  # drop punctuation/stop words
''')
    P.p("Model tuning with feature fitting within cross-validation folds (`src/classical_models.py`):")
    P.code('''
pipeline = Pipeline([("tfidf", word_tfidf()), ("clf", LinearSVC(random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=cv, refit=True)
search.fit(train_val_texts, train_val_labels)
''')
    return P.save(Path(args.output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--author", default="Albert Kabore, PhD Student in AI")
    parser.add_argument("--course", default="[Course Name and Number]")
    parser.add_argument("--instructor", default="[Instructor Name]")
    parser.add_argument("--date", default=date.today().strftime("%B %d, %Y"))
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    print(f"Wrote {build(args)}")


if __name__ == "__main__":
    main()
