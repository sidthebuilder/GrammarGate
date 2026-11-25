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
        has_refusal_branch: detect_refusal_branch(root, root, 0),
        ..Default::default()
    };
    let mut visited_refs: Vec<String> = Vec::new();
    traverse_node(root, root, 1, &mut stats, &mut visited_refs);

    if stats.total_nodes > 0 {
        stats.forced_literals = stats.single_option_enums + stats.const_nodes;
    }

    stats
}

fn resolve_local_ref<'a>(root: &'a JsonValue, ref_str: &str) -> Option<&'a JsonValue> {
    if !ref_str.starts_with("#/") {
        return None;
    }
    let mut cur = root;
    for raw_seg in ref_str[2..].split('/') {
        let seg = raw_seg.replace("~1", "/").replace("~0", "~");
        match cur {
            JsonValue::Object(_) => {
                cur = cur.get(&seg)?;
            }
            JsonValue::Array(arr) => {
                let idx: usize = seg.parse().ok()?;
                cur = arr.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(cur)
}

fn is_rigid_pattern_literal(pattern: &str) -> bool {
    let trimmed = pattern.strip_prefix('^').unwrap_or(pattern);
    let trimmed = trimmed.strip_suffix('$').unwrap_or(trimmed);
    if trimmed.len() < 8 {
        return false;
    }
    // Literal prefix/string forcing without regex alternation, repetition, or character classes
    let mut chars = trimmed.chars().peekable();
    let mut literal_len = 0usize;
    while let Some(c) = chars.next() {
        if c == '\\' {
            if chars.next().is_some() {
                literal_len += 1;
                continue;
            }
            return false;
        }
        if matches!(c, '[' | ']' | '*' | '+' | '?' | '|' | '(' | ')' | '{' | '}') {
            return false;
        }
        literal_len += 1;
    }
    literal_len >= 8
}

fn is_dict_lookup_key(key: &str) -> bool {
    let k_str = key.to_lowercase();
    if k_str.len() <= 2 || k_str.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    for prefix in &["key", "slot", "var", "sub", "node", "idx", "map"] {
        if let Some(rest) = k_str.strip_prefix(prefix) {
            let trimmed = rest.strip_prefix('_').unwrap_or(rest);
            if trimmed.is_empty() || trimmed.len() <= 2 || trimmed.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    let bytes = k_str.as_bytes();
    if matches!(bytes[0], b'k' | b'p' | b'c' | b't' | b'f' | b'h')
        && bytes[1..].iter().all(|b| b.is_ascii_digit())
    {
        return true;
    }
    false
}

fn branch_has_forced_exploit(root: &JsonValue, val: &JsonValue, depth: usize) -> bool {
    if depth > 16 {
        return false;
    }
    if let Some(ref_str) = val.get("$ref").and_then(|r| r.as_str()) {
        if let Some(target) = resolve_local_ref(root, ref_str) {
            return branch_has_forced_exploit(root, target, depth + 1);
        }
    }
    if val.get("const").is_some() {
        return true;
    }
    if let Some(JsonValue::Array(arr)) = val.get("enum") {
        if arr.len() == 1 {
            return true;
        }
    }
    if let Some(props) = val.get("properties").and_then(|p| p.as_object()) {
        for (_, v) in props {
            if branch_has_forced_exploit(root, v, depth + 1) {
                return true;
            }
        }
    }
    false
}

fn detect_refusal_branch(root: &JsonValue, val: &JsonValue, depth: usize) -> bool {
    if depth > 16 {
        return false;
    }
    let branches = val.get("anyOf").or_else(|| val.get("oneOf"));
    if let Some(JsonValue::Array(items)) = branches {
        for branch in items {
            let resolved = branch
                .get("$ref")
                .and_then(|r| r.as_str())
                .and_then(|s| resolve_local_ref(root, s))
                .unwrap_or(branch);
            if branch_has_forced_exploit(root, resolved, depth + 1) {
                continue;
            }
            if let Some(props) = resolved.get("properties").and_then(|p| p.as_object()) {
                for (key, _) in props {
                    let k = key.to_lowercase();
                    if k.contains("refusal") || k.contains("error") || k.contains("denial") {
                        return true;
                    }
                }
            }
            if let Some(title) = resolved.get("title").and_then(|t| t.as_str()) {
                let t = title.to_lowercase();
                if t.contains("refusal") || t.contains("error") {
                    return true;
                }
            }
        }
    }
    false
}

fn traverse_node(
    root: &JsonValue,
    node: &JsonValue,
    current_depth: usize,
    stats: &mut SchemaStats,
    visited_refs: &mut Vec<String>,
) {
    if current_depth > 128 {
        return;
    }
    stats.total_nodes += 1;
    let mut is_forced_step = false;

    if let Some(ref_str) = node.get("$ref").and_then(|r| r.as_str()) {
        if !visited_refs.iter().any(|r| r == ref_str) {
            if let Some(target) = resolve_local_ref(root, ref_str) {
                visited_refs.push(ref_str.to_string());
                traverse_node(root, target, current_depth + 1, stats, visited_refs);
                visited_refs.pop();
            }
        }
    }

    if node.get("const").is_some() {
        stats.const_nodes += 1;
        is_forced_step = true;
    }

    if let Some(pat) = node.get("pattern").and_then(|p| p.as_str()) {
        if is_rigid_pattern_literal(pat) {
            stats.const_nodes += 1;
            is_forced_step = true;
            stats.triggered_rules.push("FORCED_PATTERN_LITERAL".to_string());
        }
    }

    if let Some(JsonValue::Array(enum_arr)) = node.get("enum") {
        if enum_arr.len() == 1 {
            stats.single_option_enums += 1;
            is_forced_step = true;
            stats.triggered_rules.push("SINGLE_OPTION_ENUM".to_string());
        } else if !enum_arr.is_empty() {
            let is_long_prefix_set = enum_arr.len() <= 3
                && enum_arr.iter().all(|item| {
                    item.as_str()
                        .map(|s| s.len() >= 18 && s.contains(' '))
                        .unwrap_or(false)
                });
            if is_long_prefix_set {
                stats.single_option_enums += 1;
                is_forced_step = true;
                stats.triggered_rules.push("MULTI_OPTION_FORCED_PREFIX".to_string());
            } else {
                stats.multi_option_enums += 1;
            }
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
            if is_dict_lookup_key(k) {
                short_keys_count += 1;
            }

            if v.get("enum").is_some()
                || v.get("const").is_some()
                || v.get("pattern").is_some()
                || v.get("$ref").is_some()
                || v.get("type").and_then(|t| t.as_str()) == Some("string")
            {
                literal_values += 1;
            }

            let next_depth = if is_forced_step { current_depth + 1 } else { 1 };
            traverse_node(root, v, next_depth, stats, visited_refs);
        }

        if props.len() >= 4 && short_keys_count >= 3 && literal_values >= 3 {
            stats.has_lookup_dictionary = true;
            stats.triggered_rules.push("LOOKUP_DICTIONARY_TOPOLOGY".to_string());
        }
    }

    if let Some(items) = node.get("items") {
        let next_depth = if is_forced_step { current_depth + 1 } else { 1 };
        traverse_node(root, items, next_depth, stats, visited_refs);
    }

    if let Some(JsonValue::Array(prefix_items)) = node.get("prefixItems") {
        for item in prefix_items {
            let next_depth = if is_forced_step { current_depth + 1 } else { 1 };
            traverse_node(root, item, next_depth, stats, visited_refs);
        }
    }

    for combinator in &["anyOf", "oneOf", "allOf"] {
        if let Some(JsonValue::Array(branches)) = node.get(combinator) {
            for b in branches {
                traverse_node(root, b, current_depth + 1, stats, visited_refs);
            }
        }
    }
}
