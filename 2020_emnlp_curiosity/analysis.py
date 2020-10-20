# Copyright (c) Facebook, Inc. and its affiliates.
import random
from collections import Counter
import json

from db import CuriosityStore


def main():
    with open("2020_emnlp_curiosity/data/curiosity_dialogs.json") as f:
        dataset = json.load(f)
    store = CuriosityStore("2020_emnlp_curiosity/data/wiki_sql.sqlite.db")
    fact_lookup = store.get_fact_lookup()
    paraphrases = []
    n_facts = Counter()
    for dialog in dataset["dialogs"]:
        for msg in dialog["messages"]:
            if msg["sender"] != "assistant":
                continue
            used_facts = []
            for fact in msg["facts"]:
                if fact["used"]:
                    fact["text"] = fact_lookup[fact["fid"]]
                    used_facts.append(fact)
            # Keep things simple for now
            if len(used_facts) != 1:
                continue

            paraphrases.append(
                {
                    "message_id": msg["message_id"],
                    "msg_text": msg["message"],
                    "dialog_acts": msg["dialog_acts"],
                    "liked": msg["liked"],
                    "source": used_facts[0]["source"],
                    "fact_text": used_facts[0]["text"],
                }
            )
            n_facts[len(used_facts)] += 1

    random.seed(42)
    random.shuffle(paraphrases)
    print("fact distribution: ", n_facts)
    with open("/tmp/paraphrases.json", "w") as f:
        json.dump(paraphrases, f)

    with open("/tmp/paraphrases.tsv", "w") as f:
        f.write("message_id\tlabel\ttext\n\n")
        for para in paraphrases[:500]:
            message_id = para["message_id"]
            msg_text = para["msg_text"].replace("\t", " ").strip()
            fact_text = para["fact_text"].replace("\t", " ").strip()
            example = f"{message_id}\t\t{msg_text}\n\t\t{fact_text}\n\n"
            f.write(example)


if __name__ == "__main__":
    main()
