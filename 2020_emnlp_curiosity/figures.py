# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Tuple, Optional, Set
import typer
import subprocess
import math
from itertools import zip_longest
from datetime import datetime
from collections import defaultdict, Counter
from pprint import pformat
import pickle
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import krippendorff
from jinja2 import Environment, FileSystemLoader
from statsmodels.stats.proportion import proportions_ztest
from plotnine import (
    ggplot,
    aes,
    facet_wrap,
    facet_grid,
    xlab,
    ylab,
    labs,
    theme,
    theme_light,
    geom_histogram,
    geom_smooth,
    geom_boxplot,
    geom_bar,
    geom_text,
    geom_col,
    scale_x_date,
    scale_y_continuous,
    element_line,
    element_blank,
    element_rect,
    element_text,
    coord_flip,
    guides,
    guide_legend,
    annotate,
)

from mizani.formatters import date_format, percent_format
from db import verify_checksum, CuriosityStore

STATS_PATH = "2020_emnlp_curiosity/auto_fig/"
BOUNDARY = datetime(year=2019, month=9, day=12)

# For Paper, PDF=True, PNG=False, but its easier to view PNG
# in web/terminal/vs code
PNG_OUT = False
PDF_OUT = True

DATASET_ROOT = "2020_emnlp_curiosity/data/"
DATASET_PATHS = {
    "full": os.path.join(DATASET_ROOT, "curiosity_dialogs.json"),
    "train": os.path.join(DATASET_ROOT, "curiosity_dialogs.train.json"),
    "val": os.path.join(DATASET_ROOT, "curiosity_dialogs.val.json"),
    "test": os.path.join(DATASET_ROOT, "curiosity_dialogs.test.json"),
    "testzero": os.path.join(DATASET_ROOT, "curiosity_dialogs.test_zero.json"),
    "wiki": os.path.join(DATASET_ROOT, "wiki_sql.sqlite.db"),
}

AUTOFIG_PATH = "2020_emnlp_curiosity/auto_fig/"
KRIPP_PATH = "2020_emnlp_curiosity/data/agreement-krippendorff.pkl"
SOURCE_CATS = ["Known-Aspect", "Known-General", "Aspect", "General"]

INFORM_RESPONSE = "inform_response"
INFORM_RELATED = "inform_related"
INFORM_UNRELATED = "inform_unrelated"
INFORM = {INFORM_RESPONSE, INFORM_RELATED, INFORM_UNRELATED}
REQUEST_FOLLOWUP = "request_followup"
REQUEST_TOPIC = "request_topic"
REQUEST_ASPECT = "request_aspect"
REQUEST_OTHER = "request_other"
REQUEST = {REQUEST_FOLLOWUP, REQUEST_TOPIC, REQUEST_ASPECT, REQUEST_OTHER}
FEEDBACK = {"feedback_negative", "feedback_positive", "feedback_ask"}
OFFER = {
    "offer_topic",
    "offer_aspect",
    "offer_followup",
    "offer_other",
    "offer_accept",
    "offer_decline",
}
# SOCIAL = {'offer_topic', 'offer_aspect', 'offer_followup', 'offer_other',
#    'offer_accept', 'offer_decline'}
# We decided not to annotate with social, so leaving it out
DIALOG_ACT_SET = INFORM | REQUEST | FEEDBACK | OFFER

HTTP_PATH = "https://obj.umiacs.umd.edu/curiosity/"
DATA_PATH = Path("2020_emnlp_curiosity/data/")
FILES = [
    (HTTP_PATH + "curiosity_dialogs.json", DATA_PATH / "curiosity_dialogs.json"),
    (HTTP_PATH + "wiki_sql.sqlite.db", DATA_PATH / "wiki_sql.sqlite.db"),
]
for fold in ("train", "val", "test", "test_zero"):
    FILES.append(
        (
            HTTP_PATH + f"curiosity_dialogs.{fold}.json",
            DATA_PATH / f"curiosity_dialogs.{fold}.json",
        )
    )


def download(remote_file: str, local_file: str):
    eprint(f"Downloading {remote_file} to {local_file}")
    subprocess.run(
        f"curl -f --create-dirs -o {local_file} {remote_file}", shell=True, check=True
    )


def download_all(overwrite=False):
    os.makedirs(DATA_PATH, exist_ok=True)
    for remote_file, local_file in FILES:
        if os.path.exists(local_file):
            if overwrite:
                download(remote_file, local_file)
            else:
                eprint(f"File exists, skipping download of: {local_file}")
        else:
            download(remote_file, local_file)


class theme_curio(theme_light):
    """
    A theme similar to :class:`theme_linedraw` but with light grey
    lines and axes to direct more attention towards the data.
    Parameters
    ----------
    base_size : int, optional
        Base font size. All text sizes are a scaled versions of
        the base font size. Default is 11.
    base_family : str, optional
        Base font family.
    """

    def __init__(self, base_size=11, base_family="DejaVu Sans"):
        theme_light.__init__(self, base_size, base_family)
        self.add_theme(
            theme(
                text=element_text(size=14),
                axis_ticks=element_line(color="#DDDDDD", size=0.5),
                panel_border=element_rect(fill="None", color="#838383", size=1),
                panel_spacing=0.40,
                strip_background=element_rect(fill="#DDDDDD", color="#838383", size=1),
                strip_text_x=element_text(color="black"),
                strip_text_y=element_text(color="black", angle=-90),
                legend_key=element_blank(),
                legend_position="top",
                plot_margin=0,
            ),
            inplace=True,
        )


def save(plot, filename, themeables: Optional[List] = None, **kwargs):
    plot = plot + theme_curio()
    if themeables is not None:
        for t in themeables:
            plot = plot + t
    if PNG_OUT:
        plot.save(os.path.join(STATS_PATH, filename) + ".png", **kwargs)
    if PDF_OUT:
        plot.save(os.path.join(STATS_PATH, filename) + ".pdf", **kwargs)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def to_precision(x, p):
    """
    returns a string representation of x formatted with a precision of p
    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.0:
        return "0." + "0" * (p - 1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x / tens)

    if n < math.pow(10, p - 1):
        e = e - 1
        tens = math.pow(10, e - p + 1)
        n = math.floor(x / tens)

    if abs((n + 1.0) * tens - x) <= abs(n * tens - x):
        n = n + 1

    if n >= math.pow(10, p):
        n = n / 10.0
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append("e")
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[: e + 1])
        if e + 1 < len(m):
            out.append(".")
            out.extend(m[e + 1 :])
    else:
        out.append("0.")
        out.extend(["0"] * -(e + 1))
        out.append(m)

    return "".join(out)


def used_facts_in_msg(msg: Dict):
    n = 0
    for f in msg["facts"]:
        if f["used"]:
            n += 1
    return n


def compute_used_fact_sources(messages: List) -> List[str]:
    sources = []
    for msg in messages:
        for f in msg["facts"]:
            if f["used"]:
                sources.append(f["source"])
    return sources


def compute_shown_fact_sources(messages: List) -> List[str]:
    sources = []
    for msg in messages:
        for f in msg["facts"]:
            sources.append(f["source"])
    return sources


def to_booktabs(rows: List, col_spec: str):
    start = ["\\begin{tabular}{" + col_spec + "}", "\\toprule"]
    end = ["\\bottomrule", "\\end{tabular}"]
    contents = []
    for r in rows:
        contents.append("&".join(r))

    return "\n".join(start + contents + end)


def histogram_plots(dialog_df: pd.DataFrame):
    p = (
        ggplot(dialog_df)
        + aes(x="n_msgs")
        + facet_wrap("created_date_str")
        + geom_histogram(binwidth=1)
        + xlab("Dialog Length")
    )
    save(p, "dialog_length")

    p = (
        ggplot(dialog_df)
        + aes(x="n_used_facts")
        + facet_wrap("created_date_str")
        + geom_histogram(binwidth=1)
        + xlab("Number of Used Facts per Dialog")
    )
    save(p, "n_used_facts")

    p = (
        ggplot(dialog_df)
        + aes(x="n_likes")
        + facet_wrap("created_date_str")
        + geom_histogram(binwidth=1)
        + xlab("Number of Likes per Dialog")
    )
    save(p, "n_likes")


def trend_plots(df: pd.DataFrame):
    melted = df[
        [
            "created_date_str",
            "dialog_id",
            "n_likes",
            "n_used_facts",
            "n_msgs",
            "n_known_entities",
        ]
    ].melt(id_vars=["dialog_id", "created_date_str"])
    p = (
        ggplot(melted)
        + aes(x="created_date_str", y="value")
        # + scale_x_date(breaks=date_breaks('1 day'), labels=date_format('%m-%d'))
        + geom_boxplot()
        + facet_wrap("variable", scales="free_y")
    )
    save(p, "trend", width=15, height=5)


def source_plots_dialog(dialog_df: pd.DataFrame):
    """
    This computes across the entire dataset, the breakdown of
    how often each fact source occurs, divided by day
    """
    source_rows = []
    for r in dialog_df.itertuples():
        for s in r.shown_sources:
            source_rows.append(
                {
                    "dialog_id": r.dialog_id,
                    "shown_fact_source": s,
                    "created_date_str": r.created_date_str,
                }
            )
    source_df = pd.DataFrame(source_rows)
    p = (
        ggplot(source_df)
        + aes(x="shown_fact_source")
        + geom_bar()
        + facet_wrap("created_date_str")
    )
    save(p, "fact_source_shown_counts")


def exposure_plots(dialogs: List[Dict]):
    """
    Since not all sources are *actually shown equally, a
    better measure of engagement with facts is based on null choice
    upon exposure
    (we do best to show equal proportions, but sometimes that isn't possible).
    This computes for every time a fact source is shown, did the user
    fail to pick it. This number being low, is a good indicator that
    the fact source is valuable, even if its less frequent.

    Under certain conditions in early collection source can be none.
    In collecting the dataset handling this was helpful, but
    that data is not included as part of the released dataset
    """
    rows = []
    for d in dialogs:
        created_date = datetime.fromtimestamp(d["created_time"])
        if created_date <= BOUNDARY:
            raise ValueError("Invalid date due to boundary")

        for msg in d["messages"]:
            if msg["sender"] == "assistant":
                used_sources = set()
                for f in msg["facts"]:
                    # source isn't in f when inferred_step=False and
                    # there was an off by one annotation
                    # TODO: For release, remove the source check
                    if "source" in f and f["used"]:
                        used_sources.add(f["source"])

                for f in msg["facts"]:
                    # Same as above, catch when source is missing
                    # TODO: For release, remove the source check
                    if "source" in f:
                        if f["source"] in used_sources:
                            used = True
                        else:
                            used = False
                        source = f["source"]
                    else:
                        used = False
                        source = "unknown"
                    rows.append(
                        {
                            "used": used,
                            "source": source,
                            "liked": msg["liked"],
                            "created_date_str": created_date.strftime("%m-%d"),
                        }
                    )
    df = pd.DataFrame.from_records(rows)
    p = (
        ggplot(df)
        + aes(x="source", fill="liked")
        + geom_bar()
        + facet_grid(["used", "created_date_str"], labeller="label_both")
        + ylab("Exposure Count")
    )
    save(p, "exposure_counts")

    prop_df = create_prop_df(df, ["created_date_str", "liked", "source"], "used")
    p = (
        ggplot(prop_df)
        + aes(x="source", y="frac", fill="used")
        + geom_bar(stat="identity")
        + facet_grid(["liked", "created_date_str"], labeller="label_both")
    )
    save(p, "exposure_props")


def create_prop_df(df: pd.DataFrame, count_cols: List[str], frac_col: str):
    df["_prop_n"] = 1
    counts = df.groupby(count_cols).count()
    prop_df = (
        (df.groupby(count_cols + [frac_col]).count() / counts)
        .drop(columns=[frac_col])
        .reset_index()
        .rename(columns={"_prop_n": "frac"})
    )
    return prop_df


def fact_use_per_msg_plots(msg_df: pd.DataFrame):
    eprint("Breakdown of Facts used per Message")
    eprint(str(msg_df.groupby(["created_date_str", "n_used_facts"]).count()))
    p = (
        ggplot(msg_df)
        + aes(x="n_used_facts.map(str)")
        + geom_bar()
        + facet_wrap("created_date_str")
        + xlab("Number of Used Facts per Message")
    )
    save(p, "facts_per_message")


def is_used(msg, source=None):
    for f in msg["facts"]:
        if f["used"]:
            if source is None:
                return True
            elif "source" not in f:
                return False
            elif f["source"] == source:
                return True
    return False


def _check_round_followups(
    curr_msg, next_msg, created_date_str
) -> Tuple[Optional[Dict], bool]:
    if curr_msg["sender"] != "assistant":
        raise ValueError("Odd turns should be assistant turns")
    if next_msg["sender"] != "user":
        raise ValueError("even turns should be user turns")
    known_used = is_used(curr_msg, "known")
    section_used = is_used(curr_msg, "section")
    random_used = is_used(curr_msg, "random")
    if known_used:
        source = "known"
    elif section_used:
        source = "section"
    elif random_used:
        source = "random"
    else:
        # source = 'unknown'
        return None, 0

    curr_acts = set(curr_msg["dialog_acts"])
    next_acts = set(next_msg["dialog_acts"])

    has_followup = REQUEST_FOLLOWUP in next_acts
    if len(curr_acts.intersection(INFORM)) > 0:
        if REQUEST_FOLLOWUP in next_acts:
            return (
                {
                    "assist_act": "inform",
                    "user_act": REQUEST_FOLLOWUP,
                    "created_date": created_date_str,
                    "source": source,
                },
                has_followup,
            )
        elif REQUEST_ASPECT in next_acts:
            return (
                {
                    "assist_act": "inform",
                    "user_act": REQUEST_ASPECT,
                    "created_date": created_date_str,
                    "source": source,
                },
                has_followup,
            )
        elif REQUEST_TOPIC in next_acts:
            return (
                {
                    "assist_act": "inform",
                    "user_act": REQUEST_TOPIC,
                    "created_date": created_date_str,
                    "source": source,
                },
                has_followup,
            )
        elif REQUEST_OTHER in next_acts:
            return (
                {
                    "assist_act": "inform",
                    "user_act": REQUEST_OTHER,
                    "created_date": created_date_str,
                    "source": source,
                },
                has_followup,
            )
        else:
            return (
                {
                    "assist_act": "inform",
                    "user_act": "non_request",
                    "created_date": created_date_str,
                    "source": source,
                },
                has_followup,
            )
    else:
        return None, has_followup


def followup_act_plots(dataset):
    seqs = []
    n_followups = 0
    n_annotated = 0
    for d in dataset["dialogs"]:
        created_date = datetime.fromtimestamp(d["created_time"])
        if not d["inferred_steps"]:
            raise ValueError("inferred steps forced in final dataset")
        if created_date <= BOUNDARY:
            raise ValueError("invalid boundary date")
        if d["annotated"]:
            n_annotated += 1
            created_date_str = created_date.strftime("%m-%d")
            messages = d["messages"]
            # Start on the assistant turn checking for
            # an inform action
            idx = 1
            # Only compute if there is a user turn after the assistant turn
            while idx < len(messages) - 1:
                curr_msg = messages[idx]
                next_msg = messages[idx + 1]
                record, has_followup = _check_round_followups(
                    curr_msg, next_msg, created_date_str
                )
                if record is not None:
                    seqs.append(record)
                if has_followup:
                    n_followups += 1
                # skip to next assistant turn
                idx += 2
    df = pd.DataFrame(seqs)
    counts = df.groupby("user_act").count()
    eprint(f"N Followups total: {n_followups} N Dialog Annotated: {n_annotated}")
    eprint(f"Requests\n{counts}")
    p = (
        ggplot(df)
        + aes(x="source", fill="user_act")
        + geom_bar()
        + facet_wrap("created_date")
        + coord_flip()
    )
    prop_df = create_prop_df(df, ["created_date", "source"], "user_act")
    eprint(f"Request Props\n{prop_df}")
    p = (
        ggplot(prop_df)
        + aes(x="source", y="frac", fill="user_act")
        + geom_bar(stat="identity")
        + facet_wrap("created_date")
        + coord_flip()
    )
    save(p, "followup_acts_prop")


def compute_krippendorff(path, weighted=False):
    """
    Krippendorff does not naturally support multi-label settings.
    There are two reasonable adaptations to do in this case:
    1. Compute Krippendorff per (binary) label and show results in table
        and/or average somehow
    2. Convert one example to multiple examples, eg:
        ex1: ann1 = { a }, ann2 = {a,b}
        ex2: ann1 = {a,c}, ann2 = {c}
        map to a single matrix where rows are examples, columns are annotators:
        ex1a = [ yes  yes ]
        ex1b = [ no   yes ]
        ex1c = [ no no ]
        ex2a = [ yes no ]
        ex2b = [ no no ]
        ex2c = [ yes yes ]
    
    The advantage of the first case is that it treats each class separately so
    the per class agreement score can be weighted by any mechanism, so this is
    like a macro average. The advantage of the second case is that it checks
    for agreement on everything, the disadvantage being that there will be
    high agreement on negative cases when the number of classes is high
    (eg, if there are 20 classes and typically only 2 labels are chosen, even with
    some disagreement there will be very imbalanced agreement on the negative cases).
    
    However, this can be alleviated by marking rows with no/no as missing data or excluding
    them entirely.
    """
    with open(path, "rb") as f:
        # Dictionary from (dial_id, msg_id) -> List of annotations
        # 1 Annotation -> {'annotator_id': string, 'acts': <list/set of dialog act strings>}
        data = pickle.load(f)
    rows = []
    n_total = 0
    row_by_annotator = defaultdict(dict)
    for (dial_id, msg_id), annotations in data.items():
        # Agreement scores only make sense if there is more than one example
        combined = str(dial_id) + str(msg_id)
        unique_annotator_ids = {annot["annotator_id"] for annot in annotations}
        if len(unique_annotator_ids) > 1:
            n_total += 1
            for annot in annotations:
                acts = annot["acts"]
                annotator_id = annot["annotator_id"]
                for a in DIALOG_ACT_SET:
                    if a in acts:
                        label = 2  # 'yes'
                    else:
                        label = 1  # 'no'
                    rows.append(
                        {
                            "annotator_id": annotator_id,
                            "label": label,
                            "ex_id": combined,
                            "act": a,
                            "ex_id_act": combined + a,
                        }
                    )
                    row_by_annotator[annotator_id][combined + a] = label

    ordered_ex = list({r["ex_id"] for r in rows})
    # Compute Krippendorff using (1)
    krip_by_act = {}
    count_by_act = defaultdict(int)
    for act in DIALOG_ACT_SET:
        filtered = [r for r in rows if r["act"] == act]
        act_rows_by_annotator = defaultdict(dict)
        for r in filtered:
            annotator_id = r["annotator_id"]
            label = r["label"]
            if label == 2:
                count_by_act[act] += 1
            ex_id = r["ex_id"]
            act_rows_by_annotator[annotator_id][ex_id] = label
        act_krip_alpha = krip_from_rows(act_rows_by_annotator, ordered_ex)
        krip_by_act[act] = act_krip_alpha

    # Compute Krippendorff using (2)
    # Rows represent one annotators labels
    # Columns represents labels for an example
    # https://github.com/pln-fing-udelar/fast-krippendorff/blob/master/sample.py
    ordered_ex_act = list({r["ex_id_act"] for r in rows})
    krip_alpha = krip_from_rows(row_by_annotator, ordered_ex_act)
    eprint(f"N Messages with >1 annotations: {n_total}")
    eprint(f"Krip Alpha: {krip_alpha}")
    with open("2020_emnlp_curiosity/auto_fig/krip-score.tex", "w") as f:
        f.write(renew_command("kripscore", to_precision(krip_alpha, 3)))

    if weighted:
        total = sum(count_by_act.values())
        ratio_by_act = {}
        for act, count in count_by_act.items():
            ratio_by_act[act] = count / total
        eprint(pformat(count_by_act))
        eprint(pformat(ratio_by_act))
        weighted_krip = 0
        for act in krip_by_act:
            weighted_krip = krip_by_act[act] * ratio_by_act[act]

        eprint(f"Act Krip Alphas: {pformat(krip_by_act)}")
        eprint(f"Weighted Krip Alpha: {weighted_krip}")


def krip_from_rows(row_by_annotator, ordered_ex):
    krip_matrix = []
    for annotator_id in row_by_annotator:
        annot_rows = row_by_annotator[annotator_id]
        krip_row = []
        for ex_id in ordered_ex:
            if ex_id in annot_rows:
                label = annot_rows[ex_id]
            else:
                label = np.nan
            krip_row.append(label)
        krip_matrix.append(krip_row)

    return krippendorff.alpha(
        reliability_data=krip_matrix, level_of_measurement="nominal"
    )


def chunk(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def multicol(entry, width, align):
    return "\multicolumn{" f"{width}" "}{" f"{align}" "}{" f"{entry}" "}"


def load_key_from_json(path, key):
    with open(path) as f:
        return json.load(f)[key]


def from_json(path):
    with open(path) as f:
        return json.load(f)


EVAL_FOLDS = ["val", "test"]
EVAL_METRICS = ["fact_mrr", "da_micro_f1", "policy_micro_f1", "like_accuracy"]
EMPTY_CELL = r"\abr{n/a}"


class Experiments:
    def __init__(self, exp_path="2020_emnlp_curiosity/data/experiments/"):
        self._exp_path = exp_path
        self._like_majority = self._load_like_majority()
        self._mrr_tfidf = self._load_mrr_tfidf()
        self._utter_act_majority = self._load_utter_act_majority()
        self._policy_act_majority = self._load_policy_act_majority()

    def _load_like_majority(self):
        metrics = {}
        for fold in EVAL_FOLDS:
            accuracy = load_key_from_json(
                os.path.join(self._exp_path, f"like_majority_{fold}_metrics.json"),
                "best_validation_like_accuracy",
            )
            metrics[fold] = to_precision(accuracy, 3)
        return metrics

    def _load_mrr_tfidf(self):
        metrics = {}
        for fold in EVAL_FOLDS:
            mrr = load_key_from_json(
                os.path.join(self._exp_path, f"mrr_tfidf_{fold}_metrics.json"),
                "best_validation_fact_mrr",
            )
            metrics[fold] = to_precision(mrr, 3)
        return metrics

    def _load_utter_act_majority(self):
        metrics = {}
        for fold in EVAL_FOLDS:
            accuracy = load_key_from_json(
                os.path.join(self._exp_path, f"da_majority_{fold}_metrics.json"),
                "best_validation_da_micro_f1",
            )
            metrics[fold] = to_precision(accuracy, 3)
        return metrics

    def _load_policy_act_majority(self):
        metrics = {}
        for fold in EVAL_FOLDS:
            accuracy = load_key_from_json(
                os.path.join(self._exp_path, f"policy_majority_{fold}_metrics.json"),
                "best_validation_policy_micro_f1",
            )
            metrics[fold] = to_precision(accuracy, 3)
        return metrics

    def like_majority_row(self):
        like_val = self._like_majority["val"]
        like_test = self._like_majority["test"]
        # zero = self._like_majority['zero']
        utter_act_val = self._utter_act_majority["val"]
        utter_act_test = self._utter_act_majority["test"]
        policy_act_val = self._policy_act_majority["val"]
        policy_act_test = self._policy_act_majority["test"]
        cells = [
            "Majority Class",
            EMPTY_CELL,
            EMPTY_CELL,  # MRR cells
            f"{utter_act_val}",
            f"{utter_act_test}",
            f"{policy_act_val}",
            f"{policy_act_test}",
            f"{like_val}",
            f"{like_test}",
        ]
        return "&".join(cells)

    def mrr_tfidf_row(self):
        val = self._mrr_tfidf["val"]
        test = self._mrr_tfidf["test"]
        # zero = self._mrr_tfidf['zero']
        return (
            "\\abr{tf-idf} &"
            + f"{val} & {test}"
            + EMPTY_CELL.join(6 * ["&"])
            + EMPTY_CELL
        )

    def glove_rows(self) -> List[str]:
        models = [
            ("\\abr{e2e} \\bert{}", "e2e_bert"),
            ("\\charm{}", "glove_bilstm"),
            (" $-$ context", "glove_distributed"),
            # ("\\multicolumn{1}{l}{- acts}", "glove_bilstm-da"),
            # ("\\multicolumn{1}{l}{- facts}", "glove_bilstm-facts"),
            # ("\\multicolumn{1}{l}{- known}", "glove_bilstm-known"),
            # ("\\multicolumn{1}{l}{- likes}", "glove_bilstm-like"),
            # ("\\hre{}$+$\\bert{}", "bert"),
            # ("\\multicolumn{1}{l}{- known}", "bert-known"),
        ]
        all_rows = []
        for table_name, name in models:
            val_path = os.path.join(self._exp_path, f"{name}_val_metrics.json")
            test_path = os.path.join(self._exp_path, f"{name}_test_metrics.json")
            # zero_path = os.path.join(self._exp_path, f'{name}_zero_metrics.json')
            metrics = {}
            metrics["val"] = from_json(val_path)
            metrics["test"] = from_json(test_path)
            # metrics['zero'] = from_json(zero_path)
            row = [table_name]
            for metric_name in EVAL_METRICS:
                for fold in EVAL_FOLDS:
                    if metrics[fold][metric_name] != 0:
                        row.append(to_precision(metrics[fold][metric_name], 3))
                    else:
                        row.append(EMPTY_CELL)
            all_rows.append("&".join(row))
        return all_rows

    def generate_table(self, jinja_env: Environment):
        mrr = r"\textsc{mrr}"
        fone = "\\fone{}"
        tasks = [
            multicol(f"Fact Rank ({mrr})", 2, "c"),  # Span val/test
            multicol(f"Utt. Act ({fone})", 2, "c"),  # span val/test for 3 metrics
            multicol(f"Policy Act ({fone})", 2, "c"),
            multicol("Like (Accuracy)", 2, "c"),
        ]
        # One for normal and one for zero shot
        tasks_row = "&".join(tasks)
        metric_names = ["Val", "Test"]
        n_metrics = len(metric_names)
        metrics = [
            multicol(mrr, n_metrics, "c"),
            multicol(fone, n_metrics, "c"),
            # multicol('P@1', 2, 'c'),
            # multicol('R@1', 2, 'c'),
            multicol(fone, n_metrics, "c"),
            # multicol('P@1', 2, 'c'),
            # multicol('R@1', 2, 'c'),
            multicol("Accuracy", n_metrics, "c"),
        ]
        metrics_row = "&".join(metrics)
        scores = 4 * metric_names
        scores_row = "&".join(scores)
        results_rows = []
        results_rows.append(self.like_majority_row())
        # results_rows.append(self.mrr_tfidf_row())
        results_rows.extend(self.glove_rows())
        tex = jinja_env.get_template("table_template.tex.jinja2").render(
            col_spec="l " + " ".join(12 * ["r"]),
            tasks_row=tasks_row,
            cmidrules=r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            # metrics_row=metrics_row,
            scores_row=scores_row,
            result_rows=results_rows,
        )
        with open("2020_emnlp_curiosity/auto_fig/experiment-table.tex", "w") as f:
            f.write(tex)


def paraphrase_table(env: Environment):
    with open("2020_emnlp_curiosity/commit_data/paraphrases.json") as f:
        counts = json.load(f)

    mapping = {
        "verbatim": "Copy",
        "cherrypick": "Copy",
        "paraphrase-correct": "Paraphrase",
        "paraphrase-error": "Error",
        "paraphrase-multiple": "Paraphrase",
        "context": "Copy",
        "unrelated": "Unrelated",
    }
    summary = defaultdict(Counter)
    total = 0
    for label, value in counts.items():
        category = mapping[label]
        summary[category][label] += value
        total += value

    rows = []
    for category, counts in summary.items():
        category_total = 0
        for label, label_count in counts.items():
            fraction = to_precision(100 * label_count / total, 3)
            category_total += label_count
            rows.append(f"{category} & {label} & {label_count} & ${fraction}\%$")
        category_fraction = to_precision(100 * category_total / total, 3)
        if len(counts) > 1:
            rows.append(
                r"\midrule {category} & Total & {category_total} & ${category_fraction}\%$\\\midrule".format(
                    category=category,
                    category_total=category_total,
                    category_fraction=category_fraction,
                )
            )
    rows.append(r"\bottomrule Total && {total} & $100\%$".format(total=total))

    tex = env.get_template("paraphrases.tex.jinja2").render(
        col_spec="l l r r", result_rows=rows
    )
    with open("2020_emnlp_curiosity/auto_fig/paraphrase-table.tex", "w") as f:
        f.write(tex)


def renew_command(name: str, value: str):
    return "\\renewcommand{\\" + name + "}{" + value + "}"


def rename(name):
    """
    Rename a variety of variables from their name in the raw data to the
    name shown in paper plots
    """
    if name == "random":
        return "General"
    elif name == "section":
        return "Aspect"
    elif name == "known":
        return "Known"
    else:
        raise ValueError(f"Unknown name: {name}")


def label_thousands(labels: List[int]) -> List[str]:
    new = []
    for l in labels:
        if l == 0:
            new.append("0")
        else:
            new.append(str(int(l // 1000)) + "K")
    return new


class Dataset:
    def __init__(self, fold: str, tag: Optional[str] = None):
        dataset_path = DATASET_PATHS[fold]
        self._tag = tag
        self._wiki_sql_path = DATASET_PATHS["wiki"]
        self.fold = fold
        self.dataset = self.load_dialogs(dataset_path, self._wiki_sql_path)
        self.fact_sections: Dict[int, str] = CuriosityStore(
            self._wiki_sql_path
        ).get_fact_sections()
        self.dialogs = self.dataset["dialogs"]
        self.dialog_df, self.msg_df, self.topics = self.dataset_to_dfs()

        shown_fids = set()
        used_fids = set()
        # Need some breakdown of fact source
        for d in self.dialogs:
            for msg in d["messages"]:
                for f in msg["facts"]:
                    shown_fids.add(f["fid"])
                    if f["used"]:
                        used_fids.add(f["fid"])
        self._n_shown = len(shown_fids)
        self._n_used = len(used_fids)

        # Source plot vars
        self._source_rows = []
        self._expose_rows = []
        self._n_known = 0
        self._n_known_used = 0
        self._n_random = 0
        self._n_random_used = 0
        self._n_section = 0
        self._n_section_used = 0
        self._n_total_used = 0
        self._n_total = 0
        self._source_count_df = None
        self._used_count_df = None

    def load_dialogs(self, dataset_path: str, wiki_db_path: str):
        with open(dataset_path) as f:
            dataset = json.load(f)

        if self._tag is not None:
            dialogs = [d for d in dataset["dialogs"] if d["tag"] == self._tag]
            dataset["dialogs"] = dialogs

        verify_checksum(dataset["db_checksum"], wiki_db_path)
        return dataset

    def dataset_to_dfs(self) -> pd.DataFrame:
        dialog_rows = []
        msg_rows = []
        topics = set()
        for d in self.dataset["dialogs"]:
            # Skip data with mis-aligned fact uses
            if not d["inferred_steps"]:
                raise ValueError("Inferred steps is forced for final dataset")
            created_date = datetime.fromtimestamp(d["created_time"])
            if created_date <= BOUNDARY:
                raise ValueError("Final dataset must use valid dates")
            topics.add(d["focus_entity"])
            record = {}
            messages = d["messages"]
            record["n_msgs"] = len(messages)
            record["n_likes"] = sum(1 for m in messages if m["liked"])
            record["n_used_facts"] = sum(used_facts_in_msg(m) for m in messages)
            record["used_sources"] = compute_used_fact_sources(messages)
            record["shown_sources"] = compute_shown_fact_sources(messages)
            record["used_aspects"] = d["first_aspect"] is not None
            record["created_date"] = created_date
            record["created_date_str"] = created_date.strftime("%m-%d")
            record["dialog_id"] = d["dialog_id"]
            record["n_known_entities"] = len(d["known_entities"])
            dialog_rows.append(record)
            for m in messages:
                msg_rows.append(
                    {
                        "liked": m["liked"],
                        "n_used_facts": used_facts_in_msg(m),
                        "created_date": record["created_date"],
                        "created_date_str": record["created_date_str"],
                        "dialog_id": record["dialog_id"],
                        "used_sources": compute_used_fact_sources([m]),
                        "shown_sources": compute_shown_fact_sources([m]),
                        "sender": m["sender"],
                    }
                )

        dialog_df = pd.DataFrame.from_records(dialog_rows)
        msg_df = pd.DataFrame.from_records(msg_rows)
        return dialog_df, msg_df, topics

    @property
    def n_facts(self):
        n_facts = subprocess.run(
            f"sqlite3 {self._wiki_sql_path} 'select count(*) from fact;'",
            stdout=subprocess.PIPE,
            shell=True,
            check=True,
        ).stdout
        return int(n_facts)

    @property
    def n_likes(self):
        return self.dialog_df.n_likes.sum()

    def render_stats(self, jinja_env: Environment) -> str:
        commands = [
            renew_command(f"nfacts{self.fold}", f"{self.n_facts:,}"),
            renew_command(f"ndialogs{self.fold}", f"{len(self.dialog_df):,}"),
            renew_command(f"nutter{self.fold}", f"{len(self.msg_df):,}"),
            renew_command(f"ntopics{self.fold}", f"{len(self.topics):,}"),
            renew_command(f"nshown{self.fold}", f"{self._n_shown:,}"),
            renew_command(f"nused{self.fold}", f"{self._n_used:,}"),
            renew_command(f"nliked{self.fold}", f"{self.n_likes:,}"),
        ]
        return jinja_env.get_template("stats.tex.jinja2").render(commands=commands)

    def save_stats(self, jinja_env: Environment):
        """
        Computes stats and saves them for use in the paper
        """
        tex = self.render_stats(jinja_env)
        with open(
            os.path.join(AUTOFIG_PATH, f"dataset-stats-{self.fold}.tex"), "w"
        ) as f:
            f.write(tex)

    def save_prefs(self, jinja_env: Environment, grouped_df: pd.DataFrame):
        commands = []
        for r in (
            grouped_df[grouped_df.event == "Dialog Act Followup"]
            .dropna(axis=0)
            .itertuples()
        ):
            source = r.source.replace("-", "")
            outcome = r.Outcome.replace(" ", "").replace("-", "")
            commands.append(renew_command(f"da{source}{outcome}", f"{int(r.n):,}"))
        for r in (
            grouped_df[grouped_df.event == "Like Button"].dropna(axis=0).itertuples()
        ):
            source = r.source.replace("-", "")
            outcome = r.Outcome.replace(" ", "").replace("-", "")
            commands.append(renew_command(f"like{source}{outcome}", f"{int(r.n):,}"))
        tex = jinja_env.get_template("stats.tex.jinja2").render(commands=commands)
        with open(os.path.join(AUTOFIG_PATH, "student-prefs.tex"), "w") as f:
            f.write(tex)

    def save_topics(self):
        with open(os.path.join(AUTOFIG_PATH, f"topics-{self.fold}.txt"), "w") as f:
            rows = []
            f.write("\n".join(sorted(self.topics)))

    def dataset_table(self):
        # TODO: Write to latex
        n_dialogs = len(self.dialog_df)
        n_turns = self.dialog_df.n_msgs.sum()
        avg_likes = self.dialog_df.n_likes.mean()
        avg_used_facts = self.dialog_df.n_used_facts.mean()
        avg_known_entities = self.dialog_df.n_known_entities.mean()
        eprint(f"N Dialogs: {n_dialogs}")
        eprint(f"N Messages: {n_turns}")
        eprint(f"Avg Messages per Dialog: {self.dialog_df.n_msgs.mean()}")
        eprint(f"Avg Likes per Dialog: {avg_likes}")
        eprint(f"Avg Used Facts per Dialog: {avg_used_facts}")
        eprint(f"Avg Known Entities per Dialog: {avg_known_entities}")
        eprint(f"Avg Likes per Message: {self.msg_df.liked.mean()}")
        eprint(f"Avg Facts per Message: {self.msg_df.n_used_facts.mean()}")
        eprint(self.dialog_df.groupby("created_date_str").mean())

    def _prepare_source_data(self):
        self._source_rows = []
        self._expose_rows = []
        self._n_known = 0
        self._n_known_used = 0
        self._n_random = 0
        self._n_random_used = 0
        self._n_section = 0
        self._n_section_used = 0
        for d in self.dialogs:
            dialog_aspects = d["aspects"]
            messages = d["messages"]
            if messages[0]["sender"] != "user":
                raise ValueError("first message shouold be user")
            idx = 1
            while idx < len(messages) - 1:
                msg = messages[idx]
                user_msg = messages[idx + 1]
                if msg["sender"] != "assistant":
                    raise ValueError("Must be assistant msg")
                for f in msg["facts"]:
                    user_acts = user_msg["dialog_acts"]
                    if user_acts is None:
                        followup = None
                    elif REQUEST_FOLLOWUP in user_acts:
                        followup = "Yes"
                    else:
                        followup = "No"

                    source = f["source"]
                    source_name = rename(source)
                    if source == "known":
                        section = self.fact_sections[f["fid"]]
                        if section in dialog_aspects:
                            source_name = "Known-Aspect"
                        else:
                            source_name = "Known-General"
                        self._n_known += 1
                    elif source == "section":
                        self._n_section += 1
                    elif source == "random":
                        self._n_random += 1
                    if f["used"]:
                        if source == "known":
                            self._n_known_used += 1
                        elif source == "section":
                            self._n_section_used += 1
                        elif source == "random":
                            self._n_random_used += 1
                        self._expose_rows.append(
                            {
                                "source": source_name,
                                "event": "Like Button",
                                "Outcome": "Yes" if msg["liked"] else "No",
                                "n": 1,
                            }
                        )
                        self._expose_rows.append(
                            {
                                "source": source_name,
                                "event": "Dialog Act Followup",
                                "Outcome": followup,
                                "n": 1,
                            }
                        )
                    self._source_rows.append(
                        {
                            "Used": "Yes" if f["used"] else "No",
                            "source": source_name,
                            "dialog_id": d["dialog_id"],
                            "liked": "Yes" if msg["liked"] else "No",
                            "has_followup": followup,
                            "n": 1,
                        }
                    )
                idx += 2
        eprint(f"N Random: {self._n_random} Used: {self._n_random_used}")
        eprint(f"N Section: {self._n_section} Used: {self._n_section_used}")
        eprint(f"N Known: {self._n_known} Used: {self._n_known_used}")
        self._n_total_used = (
            self._n_known_used + self._n_section_used + self._n_random_used
        )
        self._n_total = len(self._source_rows)
        source_df = pd.DataFrame(self._source_rows)
        source_df["source"] = pd.Categorical(
            source_df["source"], categories=SOURCE_CATS
        )
        self._source_count_df = source_df.groupby("source").count().reset_index()
        self._source_count_df["frac"] = self._source_count_df["n"] / self._n_total
        self._source_count_df["dummie"] = ""
        self._used_count_df = source_df.groupby("Used").count().reset_index()
        self._used_count_df["frac"] = self._used_count_df["n"] / self._n_total
        self._used_count_df["dummie"] = ""

    def fact_source_plots(self, jinja_env: Environment):
        """
        This computes across the dataset:
        - Broken down by day, liked, and fact source
        - The counts of each
        - The proportion of each
        """
        # TODO: Add double axes, one with raw counts to complement percent
        # TODO: Merge these two breakdown plots
        # This computes a breakdown of fact source in a single bar
        p = (
            ggplot(self._source_count_df)
            + aes(x="dummie", y="frac", fill="source")
            + geom_col()
            + annotate(
                "text",
                label=to_precision((self._n_known / self._n_total) * 100, 3) + "%",
                x=1,
                y=(self._n_random + self._n_section + self._n_known / 2)
                / self._n_total,
            )
            + annotate(
                "text",
                label=to_precision((self._n_section / self._n_total) * 100, 3) + "%",
                x=1,
                y=(self._n_random + self._n_section / 2) / self._n_total,
            )
            + annotate(
                "text",
                label=to_precision((self._n_random / self._n_total) * 100, 3) + "%",
                x=1,
                y=(self._n_random / 2) / self._n_total,
            )
            + scale_y_continuous(
                labels=percent_format(), breaks=[0, 0.2, 0.4, 0.6, 0.8, 1]
            )
            # No label since this stacks on top of the used plot, which has labels
            + labs(x="", y="", fill="Fact Source")
            + coord_flip()
        )
        # defaults: 6.4, 4.8
        save(p, "fact_source_counts", width=6.4, height=1.2)

        # Breakdown by Used or not
        p = (
            ggplot(self._used_count_df)
            + aes(x="dummie", y="frac", fill="Used")
            + geom_col()
            + annotate(
                "text",
                label=to_precision((self._n_total_used / self._n_total) * 100, 3) + "%",
                x=1,
                y=(self._n_total_used / 2) / self._n_total,
            )
            + annotate(
                "text",
                label=to_precision((1 - self._n_total_used / self._n_total) * 100, 3)
                + "%",
                x=1,
                y=(self._n_total_used + (self._n_total - self._n_total_used) / 2)
                / self._n_total,
            )
            + scale_y_continuous(
                labels=percent_format(), breaks=[0, 0.2, 0.4, 0.6, 0.8, 1]
            )
            + labs(x="", y="Percent of Fact Shown Events", fill="Fact Used")
            + coord_flip()
            + guides(fill=guide_legend(reverse=True))
        )
        # defaults: 6.4, 4.8
        save(p, "fact_used_counts", width=6.4, height=1.2)

    def plot_user_prefs(self, jinja_env: Environment):
        # There are two rows per used event
        # The first is for the value of the like
        # The second for the value of the act followup
        # Need to normalize grouped by this event type
        expose_df = pd.DataFrame(self._expose_rows)
        expose_df = expose_df.dropna(axis=0)
        expose_df["source"] = pd.Categorical(
            expose_df["source"], categories=SOURCE_CATS
        )
        counts = expose_df.groupby(["event", "source"]).count()
        grouped = expose_df.groupby(["event", "source", "Outcome"]).count()
        self.save_prefs(jinja_env, grouped.reset_index())
        prop_df = (
            (grouped / counts)
            .drop(columns="Outcome")
            .reset_index()
            .rename(columns={"n": "frac"})
        )
        eprint("Preference dataframes")
        eprint(counts)
        eprint(grouped)
        eprint(prop_df)

        def is_preferred(outcome):
            if outcome == "Followup" or outcome == "Like":
                return "Prefer"
            else:
                return "No Prefer"

        def is_rooted(source):
            if source.startswith("Known"):
                return "Rooted"
            else:
                return "Not Rooted"

        def fact_type(source):
            if "Aspect" in source:
                return "Aspect"
            elif "General" in source:
                return "General"
            else:
                raise ValueError("Unexpected")

        prop_df["Preferred"] = prop_df["Outcome"].map(is_preferred)
        prop_df["rooted"] = prop_df["source"].map(is_rooted)
        prop_df["fact_source"] = prop_df["source"].map(fact_type)

        prop_df["source"] = pd.Categorical(prop_df["source"], categories=SOURCE_CATS)
        prop_df["Outcome"] = pd.Categorical(
            prop_df["Outcome"], categories=["Yes", "No",]
        )
        prop_df["rooted"] = pd.Categorical(
            prop_df["rooted"], categories=["Rooted", "Not Rooted"]
        )
        # After computing the fraction, we can toss nos since plotting yes/no ratios is repetetive
        prop_df = prop_df[prop_df.Outcome == "Yes"]
        p = (
            ggplot(prop_df)
            + aes(x="fact_source", y="frac", fill="rooted")
            + facet_wrap("event", scales="free_y")
            + geom_bar(stat="identity", position="dodge")
            + labs(x="Fact Source", y="Proportion Preferred", fill="Fact Source")
        )
        # Defaults
        dpi = 100
        width, height = (640 / dpi, 480 / dpi)
        width = 6.8 / 1.5
        height = 4.8 / 1.5
        save(
            p,
            "student_prefs",
            themeables=[
                theme(
                    legend_title=element_blank(),
                    legend_direction="horizontal",
                    legend_position="top",
                    legend_box_margin=0,
                    panel_spacing_x=0.45,
                )
            ],
            width=width,
            height=height,
        )


def print_zscores():
    """
    For dialog act followups and like buttons, dependent on rooted
    or not

    expose_df['rooted'] = expose_df['source'].map(is_rooted)
    expose_df['fact_source'] = expose_df['source'].map(fact_type)
    expose_df.groupby(['event', 'fact_source', 'rooted', 'Outcome']).count()

                                                            source      n
    event               fact_source rooted     Outcome               
    Dialog Act Followup Aspect      Not Rooted No        28218  28218
                                               Yes        2498   2498
                                    Rooted     No         7043   7043
                                               Yes         747    747
                        General     Not Rooted No        11754  11754
                                               Yes         540    540
                                    Rooted     No         2440   2440
                                               Yes         117    117
    Like Button         Aspect      Not Rooted No         6180   6180
                                               Yes       24536  24536
                                    Rooted     No         1250   1250
                                               Yes        6540   6540
                        General     Not Rooted No         3404   3404
                                               Yes        8890   8890
                                    Rooted     No          576    576
                                               Yes        1981   1981
    """
    eprint(proportions_ztest([28218, 7043], [28218 + 2498, 7043 + 747]))
    eprint(proportions_ztest([11754, 2440], [11754 + 540, 2440 + 117]))
    eprint(proportions_ztest([6180, 1250], [6180 + 24536, 1250 + 6540]))
    eprint(proportions_ztest([3404, 576], [3404 + 8890, 576 + 1981]))


def print_like_comparison():
    """
    This takes predictions from the first 30 dialogs, searches
    for instances where charm and bert disagree on like prediction,
    and prints the corresponding messages with predictions
    """
    with open("2020_emnlp_curiosity/commit_data/results_charm.json") as f:
        charm = json.load(f)
    with open("2020_emnlp_curiosity/commit_data/results_e2e_bert.json") as f:
        bert = json.load(f)

    with open("2020_emnlp_curiosity/data/curiosity_dialogs.test.json") as f:
        dialogs = json.load(f)["dialogs"]
        dialogs = {d["dialog_id"]: d for d in dialogs}

    b_correct = 0
    c_correct = 0
    total = 0
    disagree = 0
    if len(charm) != len(bert):
        raise ValueError("differing number of predictions")
    eprint("Model order: charm bert")
    for charm_pred, bert_pred in zip(charm, bert):
        if charm_pred["dialog_id"] != bert_pred["dialog_id"]:
            raise ValueError("mismatched dialogs")
        dialog_id = charm_pred["dialog_id"]
        conv = dialogs[dialog_id]
        for idx, (charm_logit, bert_logit, message) in enumerate(
            zip(charm_pred["like_logits"], bert_pred["like_logits"], conv["messages"])
        ):
            if message["sender"] == "assistant":
                total += 1
                if bool(charm_logit) == message["liked"]:
                    c_correct += 1
                if bool(bert_logit) == message["liked"]:
                    b_correct += 1
                if charm_logit != bert_logit:
                    disagree += 1
                    # eprint(
                    #     dialog_id,
                    #     charm_logit,
                    #     bert_logit,
                    #     message["liked"],
                    #     message["message"],
                    # )
                    if bool(charm_logit) == message["liked"]:
                        correct_model = r"\charm{}"
                    else:
                        correct_model = r"\bert{}"
                    eprint(
                        "Yes" if message["liked"] else "No",
                        "&",
                        correct_model,
                        "&",
                        message["message"],
                        r"\\",
                    )
    eprint(
        total,
        "charm",
        c_correct,
        c_correct / total,
        "bert",
        b_correct,
        b_correct / total,
    )
    eprint(len(charm), disagree, disagree / total)


def main(
    krip_weighted: bool = False,
    create_all: bool = False,
    create_tables: bool = False,
    create_plots: bool = False,
):
    """
    Only run these if the dataset exists
    """
    download_all()
    loader = FileSystemLoader("2020_emnlp_curiosity/figures")
    jinja_env = Environment(loader=loader)
    experiments = Experiments()
    if create_tables or create_all:
        paraphrase_table(jinja_env)
        experiments.generate_table(jinja_env)

    dataset_exists = True
    for path in DATASET_PATHS.values():
        if not os.path.exists(path):
            dataset_exists = False
            eprint(f"Warning: no file: {path}")
            break
    print_zscores()
    print_like_comparison()
    if dataset_exists:
        full_dataset = Dataset("full")
        train_dataset = Dataset("train")
        val_dataset = Dataset("val")
        test_dataset = Dataset("test")
        zero_dataset = Dataset("testzero")
        if create_all or create_tables:
            full_dataset.save_stats(jinja_env)
            full_dataset.save_topics()
            full_dataset.dataset_table()
            train_dataset.save_stats(jinja_env)
            val_dataset.save_stats(jinja_env)
            test_dataset.save_stats(jinja_env)
            zero_dataset.save_stats(jinja_env)

        if create_plots or create_all:
            full_dataset._prepare_source_data()
            full_dataset.fact_source_plots(jinja_env)
            full_dataset.plot_user_prefs(jinja_env)

            followup_act_plots(full_dataset.dataset)
            histogram_plots(full_dataset.dialog_df)
            trend_plots(full_dataset.dialog_df)
            exposure_plots(full_dataset.dialogs)
            fact_use_per_msg_plots(full_dataset.msg_df)
    else:
        eprint("Datasets missing, skipping autofig")

    if os.path.exists(KRIPP_PATH):
        eprint("Computing krippendorff")
        compute_krippendorff(KRIPP_PATH, weighted=krip_weighted)
    else:
        eprint("Skipping krippendorff")


if __name__ == "__main__":
    typer.run(main)
