import pandas as pd
from pathlib import Path

from config.base import DEVICE, FOLDS, SAMPLE_SUB
from registry import MODEL_REGISTRY
from config.experiments import build_experiments

from runner.fold_runner import run_single_fold

from ensemble.select_oof_seed import pick_best_seed
from ensemble.global_weighted import weighted_average_logits, logits_to_pred
from ensemble.classwise_weighted import classwise_weighted_logits

from inference.predict_test import predict_test_for_seed
from inference.io import save_submission


def main():
    device = DEVICE
    print("Device:", device)

    # =====================================================
    # 0. Load test IDs
    # =====================================================
    ids = pd.read_csv(SAMPLE_SUB)["ID"].tolist()

    # =====================================================
    # 1. TRAIN: model × seed × fold
    # =====================================================
    for model_name, builder in MODEL_REGISTRY.items():
        experiments = build_experiments(model_name)
        print(f"\n=== {model_name.upper()} | {len(experiments)} experiments ===")

        for exp_cfg in experiments:
            base_seed = int(exp_cfg["base_seed"])
            for fold in FOLDS:
                run_single_fold(
                    model_name,
                    builder,
                    exp_cfg,
                    fold,
                    device
                )

    # =====================================================
    # 2. OOF: select best seed per model
    # =====================================================
    best_seed = {}
    oof_acc = {}

    print("\n=== OOF SEED SELECTION ===")
    for model_name in MODEL_REGISTRY.keys():
        seed, scores = pick_best_seed(model_name)
        best_seed[model_name] = seed
        oof_acc[model_name] = scores[seed]

        print(f"[{model_name}] best_seed={seed}  OOF acc={scores[seed]:.4f}")
        for s, a in scores.items():
            print(f"    seed {s}: {a:.4f}")

    # =====================================================
    # 3. TEST: predict logits using best seed
    # =====================================================
    print("\n=== TEST LOGITS (BEST SEED ONLY) ===")
    test_logits = {}

    for model_name, seed in best_seed.items():
        builder = MODEL_REGISTRY[model_name]
        exp_cfg = [
            e for e in build_experiments(model_name)
            if int(e["base_seed"]) == int(seed)
        ][0]

        print(f"[{model_name}] predict test (seed={seed})")

        logits = predict_test_for_seed(
            model_name=model_name,
            builder=builder,
            seed=seed,
            device=device,
            DatasetCls=exp_cfg["dataset_cls"],
            batch_size=int(exp_cfg.get("batch_size", 32)),
            temperature=float(exp_cfg.get("temperature", 1.0)),
        )

        test_logits[model_name] = logits

    # =====================================================
    # 4A. GLOBAL WEIGHTED ENSEMBLE
    # =====================================================
    print("\n=== GLOBAL WEIGHTED ENSEMBLE ===")
    final_global = weighted_average_logits(test_logits, oof_acc)
    preds_global = logits_to_pred(final_global)

    out_global = save_submission(
        ids,
        preds_global,
        out_path="submission_global.csv"
    )
    print("Saved:", out_global)

    # =====================================================
    # 4B. CLASS-WISE WEIGHTED ENSEMBLE (Mixture-of-Experts)
    # =====================================================
    print("\n=== CLASS-WISE WEIGHTED ENSEMBLE ===")

    # class_weights[model][class] should already be built
    # from OOF per-class accuracy
    from ensemble.class_weights import build_class_weights_from_oof

    class_weights = build_class_weights_from_oof(
        model_names=list(test_logits.keys()),
        alpha=1.5,
        eps=1e-3
    )

    final_classwise = classwise_weighted_logits(test_logits, class_weights)
    preds_classwise = logits_to_pred(final_classwise)

    out_classwise = save_submission(
        ids,
        preds_classwise,
        out_path="submission_classwise.csv"
    )
    print("Saved:", out_classwise)

    print("\n=== ALL DONE ===")
    print("Global  -> submission_global.csv")
    print("Classwise -> submission_classwise.csv")


if __name__ == "__main__":
    main()
