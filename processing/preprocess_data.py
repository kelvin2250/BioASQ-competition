import os
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_context(snippets):
    return " ".join(s["text"].strip() for s in snippets if "text" in s)

def preprocess_bioasq(json_path, output_dir, dev_ratio=0.2, seed=42):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)["questions"]

    yesno, factoid, list_q, summary = [], [], [], []

    for entry in tqdm(raw_data, desc="Preprocessing"):
        qtype = entry.get("type")
        question = entry.get("body", "").strip()
        context = extract_context(entry.get("snippets", []))

        if qtype == "yesno":
            label = 1 if entry.get("exact_answer", "").strip().lower() == "yes" else 0
            yesno.append({"question": question, "context": context, "label": label})

        elif qtype == "factoid":
            answers = entry.get("exact_answer", [])
            flat_answers = []

            if isinstance(answers, list):
                for ans in answers:
                    if isinstance(ans, list):
                        flat_answers.extend([a.strip() for a in ans if isinstance(a, str) and len(a.strip()) > 1])
                    elif isinstance(ans, str) and len(ans.strip()) > 1:
                        flat_answers.append(ans.strip())

            if flat_answers:
                factoid.append({
                    "question": question,
                    "context": context,
                    "answers": flat_answers  # ✅ Lưu tất cả thực thể vào đây
                })


        elif qtype == "list":
            answers = entry.get("exact_answer", [])
            list_items = [item[0].strip() for item in answers if isinstance(item, list) and item and len(item[0]) > 1]
            if list_items:
                list_q.append({"question": question, "context": context, "answers": list_items})

        elif qtype == "summary":
            ideal = entry.get("ideal_answer", [])
            if ideal and isinstance(ideal[0], str):
                summary.append({"question": question, "context": context, "ideal_answer": ideal[0].strip()})

    os.makedirs(output_dir, exist_ok=True)

    for name, dataset in zip(["yesno", "factoid", "list", "summary"], [yesno, factoid, list_q, summary]):
        if len(dataset) == 0:
            print(f"⚠️ Skipped {name}: No valid samples found.")
            continue

        train_set, dev_set = train_test_split(dataset, test_size=dev_ratio, random_state=seed)

        with open(os.path.join(output_dir, f"{name}_train.jsonl"), "w", encoding="utf-8") as ftrain:
            for item in train_set:
                ftrain.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(os.path.join(output_dir, f"{name}_dev.jsonl"), "w", encoding="utf-8") as fdev:
            for item in dev_set:
                fdev.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n✅ Preprocessing done. Files saved in:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to BioASQ-train-taskb.json")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    preprocess_bioasq(args.input_path, args.output_dir)