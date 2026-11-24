use crate::schema_ast::SchemaStats;

#[derive(Debug, Clone)]
pub struct ComputedMetrics {
    pub risk_score: f64,
    pub is_elevated: bool,
    pub forced_literal_ratio: f64,
    pub single_option_enum_count: usize,
    pub max_low_entropy_depth: usize,
    pub has_lookup_dictionary: bool,
    pub has_refusal_branch: bool,
    pub triggered_rules: Vec<String>,
}

pub fn evaluate_metrics(stats: SchemaStats) -> ComputedMetrics {
    let mut triggered_rules = stats.triggered_rules;
    
    let total_elements = (stats.total_properties + stats.multi_option_enums + stats.forced_literals).max(1);
    let forced_literal_ratio = (stats.forced_literals as f64) / (total_elements as f64);

    let mut score = 0.0;

    // 1. Forced literal component (up to 0.40)
    score += forced_literal_ratio * 0.40;

    // 2. Single-option enum component (EnumAttack signature)
    if stats.single_option_enums > 0 {
        score += 0.35;
        if stats.single_option_enums > 1 {
            score += ((stats.single_option_enums - 1) as f64 * 0.08).min(0.25);
        }
    }

    // 3. Lookup dictionary topology component (DictAttack signature)
    if stats.has_lookup_dictionary {
        score += 0.35;
    }

    // 4. Low-entropy path depth
    if stats.max_low_entropy_depth >= 2 {
        let depth_penalty = ((stats.max_low_entropy_depth - 1) as f64 * 0.05).min(0.20);
        score += depth_penalty;
        triggered_rules.push(format!("LOW_ENTROPY_CHAIN_DEPTH_{}", stats.max_low_entropy_depth));
    }

    // 5. Absence of refusal branch
    if !stats.has_refusal_branch && (stats.single_option_enums > 0 || stats.has_lookup_dictionary || forced_literal_ratio > 0.2) {
        score += 0.15;
        triggered_rules.push("ABSENCE_OF_REFUSAL_UNION_BRANCH".to_string());
    }

    // Clamp score to [0.0, 1.0]
    let risk_score = score.clamp(0.0, 1.0);

    // Elevated threshold: triggers Layer 3 rolling monitor
    let is_elevated = risk_score >= 0.50
        || stats.single_option_enums > 0
        || stats.has_lookup_dictionary
        || stats.max_low_entropy_depth >= 2;

    // Deduplicate triggered rules
    triggered_rules.sort();
    triggered_rules.dedup();

    ComputedMetrics {
        risk_score,
        is_elevated,
        forced_literal_ratio,
        single_option_enum_count: stats.single_option_enums,
        max_low_entropy_depth: stats.max_low_entropy_depth,
        has_lookup_dictionary: stats.has_lookup_dictionary,
        has_refusal_branch: stats.has_refusal_branch,
        triggered_rules,
    }
}
