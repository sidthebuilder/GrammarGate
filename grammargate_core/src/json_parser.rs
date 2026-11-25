#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(pairs) => {
                for (k, v) in pairs {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(o) => Some(o.as_slice()),
            _ => None,
        }
    }
}

pub struct Parser<'a> {
    chars: &'a [u8],
    idx: usize,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            chars: input.as_bytes(),
            idx: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.idx < self.chars.len() {
            match self.chars[self.idx] {
                b' ' | b'\t' | b'\r' | b'\n' => self.idx += 1,
                _ => break,
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        if self.idx < self.chars.len() {
            Some(self.chars[self.idx])
        } else {
            None
        }
    }

    pub fn parse(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        let val = self.parse_value()?;
        self.skip_whitespace();
        if self.idx < self.chars.len() {
            return Err(format!("Unexpected trailing content at pos {}", self.idx));
        }
        Ok(val)
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c.is_ascii_digit() || c == b'-' => self.parse_number(),
            Some(c) => Err(format!("Unexpected character '{}' at pos {}", c as char, self.idx)),
            None => Err("Unexpected end of input".to_string()),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.idx += 1; // skip '{'
        let mut pairs = Vec::new();

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.idx += 1;
                break;
            }

            let key = self.parse_string()?;
            self.skip_whitespace();

            if self.peek() != Some(b':') {
                return Err(format!("Expected ':' at pos {}", self.idx));
            }
            self.idx += 1; // skip ':'

            let val = self.parse_value()?;
            pairs.push((key, val));

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.idx += 1;
                }
                Some(b'}') => {
                    self.idx += 1;
                    break;
                }
                _ => return Err(format!("Expected ',' or '}}' at pos {}", self.idx)),
            }
        }

        Ok(JsonValue::Object(pairs))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.idx += 1; // skip '['
        let mut items = Vec::new();

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.idx += 1;
                break;
            }

            let val = self.parse_value()?;
            items.push(val);

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.idx += 1;
                }
                Some(b']') => {
                    self.idx += 1;
                    break;
                }
                _ => return Err(format!("Expected ',' or ']' at pos {}", self.idx)),
            }
        }

        Ok(JsonValue::Array(items))
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        if self.idx + 4 > self.chars.len() {
            return Err("Truncated \\u escape sequence".to_string());
        }
        let hex_str = std::str::from_utf8(&self.chars[self.idx..self.idx + 4])
            .map_err(|e| e.to_string())?;
        let val = u32::from_str_radix(hex_str, 16)
            .map_err(|_| format!("Invalid hex escape '\\u{}'", hex_str))?;
        self.idx += 4;
        Ok(val)
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        if self.idx >= self.chars.len() || self.chars[self.idx] != b'"' {
            return Err(format!("Expected '\"' at pos {}", self.idx));
        }
        self.idx += 1;

        let mut s = String::new();
        while self.idx < self.chars.len() {
            let b = self.chars[self.idx];
            self.idx += 1;
            if b == b'"' {
                return Ok(s);
            }
            if b == b'\\' {
                if self.idx >= self.chars.len() {
                    return Err("Unfinished escape sequence".to_string());
                }
                let esc = self.chars[self.idx];
                self.idx += 1;
                match esc {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    b'b' => s.push('\x08'),
                    b'f' => s.push('\x0C'),
                    b'u' => {
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if self.idx + 1 < self.chars.len()
                                && self.chars[self.idx] == b'\\'
                                && self.chars[self.idx + 1] == b'u'
                            {
                                self.idx += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err("Invalid UTF-16 low surrogate".to_string());
                                }
                                let combined = 0x10000 + (((code - 0xD800) << 10) | (low - 0xDC00));
                                let ch = char::from_u32(combined)
                                    .ok_or_else(|| "Invalid surrogate scalar".to_string())?;
                                s.push(ch);
                            } else {
                                return Err("Unpaired UTF-16 high surrogate".to_string());
                            }
                        } else {
                            let ch = char::from_u32(code)
                                .ok_or_else(|| "Invalid unicode scalar".to_string())?;
                            s.push(ch);
                        }
                    }
                    _ => s.push(esc as char),
                }
            } else {
                s.push(b as char);
            }
        }
        Err("Unterminated string".to_string())
    }

    fn parse_bool(&mut self) -> Result<JsonValue, String> {
        if self.chars[self.idx..].starts_with(b"true") {
            self.idx += 4;
            Ok(JsonValue::Bool(true))
        } else if self.chars[self.idx..].starts_with(b"false") {
            self.idx += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err("Expected boolean".to_string())
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.chars[self.idx..].starts_with(b"null") {
            self.idx += 4;
            Ok(JsonValue::Null)
        } else {
            Err("Expected null".to_string())
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.idx;
        if self.chars[self.idx] == b'-' {
            self.idx += 1;
        }
        while self.idx < self.chars.len() {
            let b = self.chars[self.idx];
            if b.is_ascii_digit() || b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
                self.idx += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.chars[start..self.idx])
            .map_err(|e| e.to_string())?;
        let num: f64 = s.parse().map_err(|_| format!("Invalid number '{}'", s))?;
        Ok(JsonValue::Number(num))
    }
}
