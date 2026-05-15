"""Fit the two LDA topic models used by TAPJFNN's β and δ attention."""

import argparse

from ..config import APP_CFG, TAPJFNN_CFG
from ..data.dataset import load_hf_dataset
from ..utils.topic_model import fit_and_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset", type=str, default=APP_CFG.dataset_name)
    args = ap.parse_args()

    train = load_hf_dataset(split="train", limit=args.limit, name=args.dataset)
    job_docs = [ex.job_text for ex in train]
    resume_docs = [ex.resume_text for ex in train]
    print(f"Fitting LDA on {len(job_docs)} JDs and {len(resume_docs)} resumes...")
    fit_and_save(
        job_docs=job_docs,
        resume_docs=resume_docs,
        num_topics_jd=TAPJFNN_CFG.topic_dim_jd,
        num_topics_resume=TAPJFNN_CFG.topic_dim_resume,
    )
    print("Saved LDA models.")


if __name__ == "__main__":
    main()
