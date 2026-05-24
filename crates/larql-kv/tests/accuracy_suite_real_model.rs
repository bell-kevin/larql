//! End-to-end accuracy-suite test against a real model.
//!
//! Drives `evaluate_parametric` + `evaluate_in_context` +
//! `evaluate_conflict` through `StandardEngine` on a real vindex, then
//! aggregates with `compute_strategy_split`. Verifies the runners
//! return sensible per-prompt structures and that the parametric vs
//! in-context vs conflict columns split.
//!
//! Opt in with:
//!
//! ```sh
//! LARQL_MODEL=<path-or-hf-id> \
//!   cargo test -p larql-kv --test accuracy_suite_real_model \
//!   --release -- --ignored --nocapture
//! ```
//!
//! Skipping when `LARQL_MODEL` is unset (or when load fails) keeps CI
//! green; the failure cases that matter for correctness are exercised
//! by the in-module unit tests against synthetic logits.

use larql_inference::cpu_engine_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::InferenceModel;
use larql_kv::accuracy_suite::conflict::conflict_quick;
use larql_kv::accuracy_suite::needle::needle_tests;
use larql_kv::accuracy_suite::prompts::{quick_20, KnowledgeSource};
use larql_kv::accuracy_suite::runner::{
    compute_strategy_split, evaluate_conflict, evaluate_in_context, evaluate_parametric,
    format_strategy_split, EvalLabels,
};
use larql_kv::EngineKind;

fn load_or_skip(label: &str) -> Option<InferenceModel> {
    let mid = std::env::var("LARQL_MODEL").ok()?;
    match InferenceModel::load(&mid) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("skip {label}: {e}");
            None
        }
    }
}

#[test]
#[ignore = "real-model end-to-end; set LARQL_MODEL and run with --ignored"]
fn parametric_corpus_runs_through_standard_engine() {
    let Some(model) = load_or_skip("parametric") else {
        return;
    };
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let ffn = WeightFfn { weights };

    // 4-prompt slice — keeps the test fast on real models while still
    // exercising the driver shape (categories, knowledge-source tag,
    // bits column).
    let prompts = quick_20();
    let scores = evaluate_parametric(
        || EngineKind::Standard { window_size: None }.build(cpu_engine_backend()),
        weights,
        &ffn,
        tokenizer,
        EvalLabels::for_kv_engine("Standard"),
        &prompts[..4],
    );

    assert_eq!(scores.len(), 4);
    for s in &scores {
        assert_eq!(s.strategy, "Standard");
        assert_eq!(s.knowledge_source, KnowledgeSource::Parametric);
        // Standard engine never skips a real prompt — every row must
        // be Served and carry the score fields.
        assert!(
            s.outcome.is_served(),
            "standard engine must serve, got {:?}",
            s.outcome
        );
        let predicted = s
            .predicted_top1
            .as_deref()
            .expect("served row has predicted_top1");
        assert!(!predicted.is_empty(), "empty predicted_top1");
        let bits = s.bits_per_token.expect("served row has bits_per_token");
        // Real models on real prompts should produce finite bits.
        assert!(
            bits.is_finite(),
            "bits should be finite on a real model, got {} for prompt {:?}",
            bits,
            s.prompt
        );
    }
    eprintln!(
        "parametric scores (first 4 prompts): {:#?}",
        scores
            .iter()
            .map(|s| (&s.prompt, s.top1_match, s.bits_per_token))
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "real-model end-to-end; set LARQL_MODEL and run with --ignored"]
fn in_context_needle_runs_through_standard_engine() {
    let Some(model) = load_or_skip("needle") else {
        return;
    };
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let ffn = WeightFfn { weights };

    // Two shortest needles (512, 1024 tokens) — full 32K is too slow
    // for CI even on opt-in `--ignored`.
    let needles: Vec<_> = needle_tests().into_iter().take(2).collect();
    let scores = evaluate_in_context(
        || EngineKind::Standard { window_size: None }.build(cpu_engine_backend()),
        weights,
        &ffn,
        tokenizer,
        EvalLabels::for_kv_engine("Standard"),
        &needles,
    );
    assert_eq!(scores.len(), 2);
    for s in &scores {
        assert_eq!(s.knowledge_source, KnowledgeSource::InContext);
        assert_eq!(s.category, "needle");
        assert!(s.outcome.is_served(), "standard engine must serve needles");
        let bits = s.bits_per_token.expect("served row has bits_per_token");
        assert!(bits.is_finite() || bits.is_nan());
    }
}

#[test]
#[ignore = "real-model end-to-end; set LARQL_MODEL and run with --ignored"]
fn conflict_corpus_runs_through_standard_engine() {
    let Some(model) = load_or_skip("conflict") else {
        return;
    };
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let ffn = WeightFfn { weights };

    let prompts = conflict_quick();
    let scores = evaluate_conflict(
        || EngineKind::Standard { window_size: None }.build(cpu_engine_backend()),
        weights,
        &ffn,
        tokenizer,
        EvalLabels::for_kv_engine("Standard"),
        &prompts,
    );
    assert_eq!(scores.len(), prompts.len());
    for s in &scores {
        assert!(
            s.outcome.is_served(),
            "standard engine must serve, got {:?}",
            s.outcome
        );
        let followed = s.followed_context.expect("served row has followed_context");
        let fallback = s
            .parametric_fallback
            .expect("served row has parametric_fallback");
        assert!(
            !(followed && fallback),
            "followed and fallback are mutually exclusive by construction"
        );
    }
    eprintln!(
        "conflict scores: follow={} fallback={} other={}",
        scores
            .iter()
            .filter(|s| s.followed_context == Some(true))
            .count(),
        scores
            .iter()
            .filter(|s| s.parametric_fallback == Some(true))
            .count(),
        scores
            .iter()
            .filter(|s| {
                s.outcome.is_served()
                    && s.followed_context != Some(true)
                    && s.parametric_fallback != Some(true)
            })
            .count(),
    );
}

#[test]
#[ignore = "real-model end-to-end; set LARQL_MODEL and run with --ignored"]
fn split_table_renders_for_real_model_run() {
    let Some(model) = load_or_skip("split") else {
        return;
    };
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let ffn = WeightFfn { weights };

    let prompts = quick_20();
    let needles: Vec<_> = needle_tests().into_iter().take(1).collect();
    let conflicts = conflict_quick();

    let mut all_scores = evaluate_parametric(
        || EngineKind::Standard { window_size: None }.build(cpu_engine_backend()),
        weights,
        &ffn,
        tokenizer,
        EvalLabels::for_kv_engine("Standard"),
        &prompts[..4],
    );
    all_scores.extend(evaluate_in_context(
        || EngineKind::Standard { window_size: None }.build(cpu_engine_backend()),
        weights,
        &ffn,
        tokenizer,
        EvalLabels::for_kv_engine("Standard"),
        &needles,
    ));
    let conflict_scores = evaluate_conflict(
        || EngineKind::Standard { window_size: None }.build(cpu_engine_backend()),
        weights,
        &ffn,
        tokenizer,
        EvalLabels::for_kv_engine("Standard"),
        &conflicts,
    );

    let splits = compute_strategy_split(&all_scores, &conflict_scores);
    let table = format_strategy_split(&splits);
    eprintln!("{table}");

    // Structural assertions: one row per engine, parametric column
    // populated (we ran ≥1 parametric prompt).
    assert_eq!(splits.len(), 1);
    let split = &splits[0];
    assert_eq!(split.strategy, "Standard");
    assert!(split.parametric_n > 0);
    assert!(split.parametric_match_rate.is_finite());
}
