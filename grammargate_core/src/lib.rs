pub mod analyzer;
pub mod json_parser;
pub mod metrics;
pub mod schema_ast;

use analyzer::{analyze_grammar_impl, analyze_schema_impl};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
pub struct CSchemaRiskReport {
    pub risk_score: f64,
    pub is_elevated: bool,
    pub forced_literal_ratio: f64,
    pub single_option_enum_count: usize,
    pub max_low_entropy_depth: usize,
    pub has_lookup_dictionary: bool,
    pub has_refusal_branch: bool,
}

/// Analyzes a JSON schema AST and writes the risk assessment into `out_report`.
///
/// # Safety
/// - `schema_json` must point to a valid null-terminated C string.
/// - `out_report` must point to writable memory large enough to hold `CSchemaRiskReport`.
/// - `out_rules_buf`, if non-null, must point to a buffer of at least `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn grammargate_analyze_schema(
    schema_json: *const c_char,
    out_report: *mut CSchemaRiskReport,
    out_rules_buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if schema_json.is_null() || out_report.is_null() {
            return -1;
        }

        let c_str = match CStr::from_ptr(schema_json).to_str() {
            Ok(s) => s,
            Err(_) => return -2,
        };

        let report = match analyze_schema_impl(c_str) {
            Ok(r) => r,
            Err(_) => return -3,
        };

        (*out_report) = CSchemaRiskReport {
            risk_score: report.risk_score,
            is_elevated: report.is_elevated,
            forced_literal_ratio: report.forced_literal_ratio,
            single_option_enum_count: report.single_option_enum_count,
            max_low_entropy_depth: report.max_low_entropy_depth,
            has_lookup_dictionary: report.has_lookup_dictionary,
            has_refusal_branch: report.has_refusal_branch,
        };

        if !out_rules_buf.is_null() && buf_len > 0 {
            let rules_joined = report.triggered_rules.join(",");
            if let Ok(c_rules) = CString::new(rules_joined) {
                let bytes = c_rules.as_bytes();
                let copy_len = bytes.len().min(buf_len.saturating_sub(1));
                if copy_len > 0 {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out_rules_buf, copy_len);
                }
                *out_rules_buf.add(copy_len) = 0;
            } else {
                *out_rules_buf = 0;
            }
        }

        0
    })).unwrap_or(-99)
}

/// Analyzes a context-free grammar string (GBNF / EBNF) for absence of refusal productions.
///
/// # Safety
/// - `grammar_str` must point to a valid null-terminated C string.
/// - `syntax_str` may be null (defaults to "gbnf") or a valid null-terminated C string.
/// - `out_report` must point to writable memory large enough to hold `CSchemaRiskReport`.
#[no_mangle]
pub unsafe extern "C" fn grammargate_analyze_grammar(
    grammar_str: *const c_char,
    syntax_str: *const c_char,
    out_report: *mut CSchemaRiskReport,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if grammar_str.is_null() || out_report.is_null() {
            return -1;
        }

        let g_str = match CStr::from_ptr(grammar_str).to_str() {
            Ok(s) => s,
            Err(_) => return -2,
        };

        let s_str = if syntax_str.is_null() {
            "gbnf"
        } else {
            CStr::from_ptr(syntax_str).to_str().unwrap_or("gbnf")
        };

        let report = match analyze_grammar_impl(g_str, s_str) {
            Ok(r) => r,
            Err(_) => return -3,
        };

        (*out_report) = CSchemaRiskReport {
            risk_score: report.risk_score,
            is_elevated: report.is_elevated,
            forced_literal_ratio: report.forced_literal_ratio,
            single_option_enum_count: report.single_option_enum_count,
            max_low_entropy_depth: report.max_low_entropy_depth,
            has_lookup_dictionary: report.has_lookup_dictionary,
            has_refusal_branch: report.has_refusal_branch,
        };

        0
    })).unwrap_or(-99)
}
