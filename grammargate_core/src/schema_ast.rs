use crate::json_parser::JsonValue;

#[derive(Debug, Clone, Default)]
pub struct SchemaStats {
    pub total_nodes: usize,
    pub single_option_enums: usize,
    pub multi_option_enums: usize,
    pub const_nodes: usize,
    pub max_low_entropy_depth: usize,
    pub has_lookup_dictionary: bool,
    pub has_refusal_branch: bool,
    pub total_properties: usize,
    pub forced_literals: usize,
    pub triggered_rules: Vec<String>,
}

pub fn inspect_schema_value(root: &JsonValue) -> SchemaStats {
    let mut stats = SchemaStats {
        has_refusal_branch: detect_refusal_branch(root),
        ..Default::default()
    };
    traverse_node(root, 1, &mut stats);

    if stats.total_nodes > 0 {
        stats.forced_literals = stats.single_option_enums + stats.const_nodes;
    }

    stats
}

fn detect_refusal_branch(val: &JsonValue) -> bool {
    let branches = val.get("anyOf").or_else(|| val.get("oneOf"));
    if let Some(JsonValue::Array(items)) = branches {
        for branch in items {
            if let Some(props) = branch.get("properties").and_then(|p| p.as_object()) {
                for (key, _) in props {
                    let k = key.to_lowercase();
                    if k.contains("refusal") || k.contains("error") || k.contains("denial") {
                        return true;
                    }
                }
            }
            if let Some(title) = branch.get("title").and_then(|t| t.as_str()) {
                let t = title.to_lowercase();
                if t.contains("refusal") || t.contains("error") {
                    return true;
                }
            }
        }
    }
    false
}

fn traverse_node(node: &JsonValue, current_depth: usize, stats: &mut SchemaStats) {
    if current_depth > 128 {
        return;
    }
    stats.total_nodes += 1;
    let mut is_forced_step = false;

    if node.get("const").is_some() {
        stats.const_nodes += 1;
        is_forced_step = true;
    }

    if let Some(JsonValue::Array(enum_arr)) = node.get("enum") {
        if enum_arr.len() == 1 {
            stats.single_option_enums += 1;
            is_forced_step = true;
            stats.triggered_rules.push("SINGLE_OPTION_ENUM".to_string());
        } else if !enum_arr.is_empty() {
            stats.multi_option_enums += 1;
        }
    }

    if is_forced_step && current_depth > stats.max_low_entropy_depth {
        stats.max_low_entropy_depth = current_depth;
    }

    if let Some(props) = node.get("properties").and_then(|p| p.as_object()) {
        stats.total_properties += props.len();

        let mut short_keys_count = 0;
        let mut literal_values = 0;

        for (k, v) in props {
            let k_str = k.to_lowercase();
            let is_dict_key = k_str.len() <= 6
                || k_str.chars().all(|c| c.is_ascii_digit())
                || k_str.starts_with("key")
                || k_str.starts_with("slot")
                || k_str.starts_with("var")
                || k_str.starts_with("sub")
                || k_str.starts_with("node")
                || k_str.starts_with("idx")
                || k_str.starts_with('k')
                || k_str.starts_with('p')
                || k_str.starts_with('c')
                || k_str.starts_with('t');

            if is_dict_key {
                short_keys_count += 1;
            }

            if v.get("enum").is_some() || v.get("const").is_some() || v.get("type").and_then(|t| t.as_str()) == Some("string") {
                literal_values += 1;
            }

            let next_depth = if is_forced_step { current_depth + 1 } else { 1 };
            traverse_node(v, next_depth, stats);
        }

        if props.len() >= 4 && short_keys_count >= 3 && literal_values >= 3 {
            stats.has_lookup_dictionary = true;
            stats.triggered_rules.push("LOOKUP_DICTIONARY_TOPOLOGY".to_string());
        }
    }

    if let Some(items) = node.get("items") {
        let next_depth = if is_forced_step { current_depth + 1 } else { 1 };
        traverse_node(items, next_depth, stats);
    }

    for combinator in &["anyOf", "oneOf", "allOf"] {
        if let Some(JsonValue::Array(branches)) = node.get(combinator) {
            for b in branches {
                traverse_node(b, current_depth + 1, stats);
            }
        }
    }
}
