import argparse, os, json, sys
import pandas as pd
from .cleaner import Cleaner, load_config, make_ai_suggestion



def main():
    p= argparse.ArgumentParser(description="data qulity & cleaning CLI")
    p.add_argument("--input", required=True ,help="path to input data file csv")
    p.add_argument("--config", required=True ,help="path to oYAML config")
    p.add_argument("--outdir", default="run/run_1" ,help="output directory")
    p.add_argument("--ai", action="store_true", help="Call LLM to generate a suggestions markdown")
    p.add_argument("--ai-model", default="gpt-4o-mini", help="LLM model name")

    args= p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df=pd.read_csv(args.input)
    cfg=load_config(args.config)  
    cleaner=Cleaner(cfg)
    before=cleaner.profile(df.copy())
    if args.ai:
        md = make_ai_suggestion(before, df, model=args.ai_model, cfg=cfg)
        md_path = os.path.join(args.outdir, "ai_suggestions.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
    df_clean=cleaner.run(df)
    after= cleaner.profile(df_clean.copy())

    df_clean.to_csv(os.path.join(args.outdir, "cleaned.csv"), index=False)
    with open(os.path.join(args.outdir, "before_profile.json"), "w", encoding="utf-8") as f:
        json.dump(before, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "after_profile.json"), "w", encoding="utf-8") as f:
        json.dump(after, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "transform_log.json"), "w", encoding="utf-8") as f:
        json.dump([sr.__dict__ for sr in cleaner.log], f, ensure_ascii=False, indent=2)

    
    print(f"[OK] rows: {before['rows']} -> {after['rows']}, cols: {before['cols']} -> {after['cols']}")
    print(f"[OK] artifacts saved to: {args.outdir}")

if __name__ == "__main__":
    main()
