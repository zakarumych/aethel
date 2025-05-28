use core::str::{Bytes, CharIndices, Chars, FromStr};

use alloc::rc::Rc;

use smol_str::{SmolStr, SmolStrBuilder};

use crate::span::{Location, Span};

/// A word starting with lowercase or underscore is an identifier.
/// It may contain letters, digits, and underscores.
///
/// # Examples
///
/// "foo" is an identifier.
/// "foo_bar" is an identifier.
/// "foo_bar123" is an identifier.
///
/// Some of the identifiers are keywords.
/// Those are always contextual.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ident {
    pub name: SmolStr,
    pub span: Span,
}

const fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_lowercase() || ch == '_'
}

const fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

/// A word starting with uppercase and containing only letters, digits, and underscores is an Atom.
/// Atom is a value that is only equal to itself.
/// Unlike other types, Atoms are never declared.
///
/// # Examples
///
/// "Foo" is an Atom.
/// "Foo_Bar" is an Atom.
/// "Foo_Bar123" is an Atom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atom {
    pub name: SmolStr,
    pub span: Span,
}

const fn is_atom_start(ch: char) -> bool {
    ch.is_ascii_uppercase()
}

const fn is_atom_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

/// Specifies if the punctuation character is followed by another punctuation character or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spacing {
    Joint,
    Alone,
}

/// A single punctuation characters is parsed as a Punct token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Punct {
    pub char: char,
    pub spacing: Spacing,
    pub span: Span,
}

/// Allowed punctuation characters in identifiers.
///
/// This is a subset of ascii punctuation characters.
///
/// Characters included in this set are:
/// ! # $ % & * + , - . / : ; < = > ? @ ^ | ~
///
/// ( ) [ ] { } are not included in this set. They are used for grouping.
/// ' and " are not included in this set. They are used for string and character literals.
const fn is_punctuation_char(ch: char) -> bool {
    matches!(
        ch,
        '!' | '#'
            | '$'
            | '%'
            | '&'
            | '*'
            | '+'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '^'
            | '|'
            | '~'
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LitNumKind {
    Binary,
    Octal,
    Hexadecimal,
    Decimal {
        point: Option<u32>,
        exponent: Option<u32>,
    },
}

/// A number literal.
/// A sequence of non-whitespace characters that starts with a digit or plus/minus sign followed by digit.
/// Must follow one of the following patterns:
///
/// - A decimal number (base 10) with an optional decimal point and exponent.
/// - `0x` followed by hex digits
/// - `0o` followed by octal digits
/// - `0b` followed by binary digits
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LitNum {
    pub literal: SmolStr,
    pub kind: LitNumKind,
    pub span: Span,
}

impl LitNum {
    pub fn is_integer(&self) -> bool {
        match self.kind {
            LitNumKind::Binary | LitNumKind::Octal | LitNumKind::Hexadecimal => true,
            LitNumKind::Decimal {
                point: None,
                exponent: None,
            } => true,
            LitNumKind::Decimal {
                point: None,
                exponent: Some(exponent_pos),
            } => {
                // If there is an exponent, it must be followed by at least one digit.
                // So the mantissa must not be empty.
                !self.literal[exponent_pos as usize + 1..].starts_with('-')
            }
            LitNumKind::Decimal { point: Some(_), .. } => false,
        }
    }

    pub fn parse_integer(&self) -> Option<u64> {
        match self.kind {
            LitNumKind::Binary => u64::from_str_radix(&self.literal[2..], 2).ok(),
            LitNumKind::Octal => u64::from_str_radix(&self.literal[2..], 8).ok(),
            LitNumKind::Hexadecimal => u64::from_str_radix(&self.literal[2..], 16).ok(),
            LitNumKind::Decimal {
                exponent: None,
                point: None,
            } => u64::from_str_radix(&self.literal, 10).ok(),
            LitNumKind::Decimal {
                exponent: Some(exponent_pos),
                point: None,
            } => {
                let mantissa = &self.literal[..exponent_pos as usize];
                let exponent = &self.literal[exponent_pos as usize + 1..];

                let exponent = i32::from_str_radix(exponent, 10).ok()?;

                if exponent < 0 {
                    return None;
                }

                let mantissa = u64::from_str_radix(mantissa, 10).ok()?;
                mantissa.checked_mul(10u64.checked_pow(exponent as u32)?)
            }
            LitNumKind::Decimal { point: Some(_), .. } => None,
        }
    }

    pub fn parse_float(&self) -> Option<f64> {
        match self.kind {
            LitNumKind::Binary => u128::from_str_radix(&self.literal[2..], 2)
                .ok()
                .map(|n| n as f64),
            LitNumKind::Octal => u128::from_str_radix(&self.literal[2..], 8)
                .ok()
                .map(|n| n as f64),
            LitNumKind::Hexadecimal => u128::from_str_radix(&self.literal[2..], 16)
                .ok()
                .map(|n| n as f64),
            LitNumKind::Decimal { exponent: None, .. } => f64::from_str(&self.literal).ok(),
            LitNumKind::Decimal {
                exponent: Some(exponent_pos),
                ..
            } => {
                let mantissa = &self.literal[..exponent_pos as usize];
                let exponent = &self.literal[exponent_pos as usize + 1..];

                let mantissa = f64::from_str(mantissa).ok()?;
                let exponent = f64::from_str(exponent).ok()?;

                Some(mantissa * 10.0f64.powf(exponent))
            }
        }
    }
}

/// A single character literal.
/// A sequence of non-whitespace characters that starts with a single quote and ends with a single quote.
/// Between the quotes, there may be a single character or single escape sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LitChar {
    pub literal: SmolStr,
    pub span: Span,
}

/// A string literal.
/// A sequence of characters that starts with a double quote and ends with a double quote.
/// Between the quotes, there may be a sequence of characters or escape sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LitStr {
    pub literal: SmolStr,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lit {
    Num(LitNum),
    Char(LitChar),
    Str(LitStr),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delim {
    /// An open parenthesis `(` token.
    ///
    /// This token is always first in a group.
    /// Closing parenthesis `)` is not a token, but closes the group when encountered.
    Paren,

    /// An open bracket `[` token.
    ///
    /// This token is always first in a group.
    /// Closing bracket `]` is not a token, but closes the group when encountered.
    Bracket,

    /// An open brace `{` token.
    ///
    /// This token is always first in a group.
    /// Closing brace `}` is not a token, but closes the group when encountered.
    Brace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub delim: Delim,
    pub stream: TokenStream,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenTree {
    Ident(Ident),
    Atom(Atom),
    Punct(Punct),
    Lit(Lit),
    Group(Group),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenStream {
    trees: Rc<[TokenTree]>,
}

pub struct IntoIter {
    trees: Rc<[TokenTree]>, // But it is know to be unique
    idx: usize,
}

impl Iterator for IntoIter {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        if self.idx >= self.trees.len() {
            return None;
        }
        let item = match Rc::get_mut(&mut self.trees) {
            None => self.trees[self.idx].clone(),
            Some(trees) => core::mem::replace(
                &mut trees[self.idx],
                TokenTree::Punct(Punct {
                    char: '!',
                    spacing: Spacing::Alone,
                    span: Span::new(0),
                }),
            ),
        };
        self.idx += 1;
        Some(item)
    }
}

pub struct Iter<'a> {
    inner: core::slice::Iter<'a, TokenTree>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a TokenTree;

    fn next(&mut self) -> Option<&'a TokenTree> {
        self.inner.next()
    }
}

impl TokenStream {
    pub fn new() -> Self {
        TokenStream {
            trees: Rc::from([]),
        }
    }

    pub fn iter(&self) -> Iter {
        Iter {
            inner: self.trees.iter(),
        }
    }

    pub fn into_iter(self) -> IntoIter {
        IntoIter {
            trees: Rc::clone(&self.trees),
            idx: 0,
        }
    }
}

impl IntoIterator for TokenStream {
    type Item = TokenTree;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexError {
    span: Span,
}

pub fn parse(s: &str) -> Result<TokenStream, LexError> {
    TokenStream::from_str(s)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cursor<'a> {
    buffer: &'a str,
    off: u32,
}

impl<'a> Cursor<'a> {
    #[must_use]
    fn advance(&self, bytes: usize) -> Cursor<'a> {
        let (_front, rest) = self.buffer.split_at(bytes);

        Cursor {
            buffer: rest,
            off: self.off + _front.chars().count() as u32,
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[must_use]
    fn skip_whitespace(&self) -> Cursor<'a> {
        let rest = self.buffer.trim_start();
        let bytes = self.buffer.len() - rest.len();
        self.advance(bytes)
    }

    fn chars(&self) -> Chars<'_> {
        self.buffer.chars()
    }

    fn char_indices(&self) -> CharIndices<'_> {
        self.buffer.char_indices()
    }

    fn bytes(&self) -> Bytes<'_> {
        self.buffer.bytes()
    }

    fn starts_with(&self, prefix: &str) -> bool {
        self.buffer.starts_with(prefix)
    }

    fn starts_with_char(&self, first: char) -> bool {
        self.buffer.starts_with(first)
    }

    fn starts_with_fn(&self, f: impl Fn(char) -> bool) -> bool {
        self.buffer.starts_with(f)
    }
}

impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(source: &str) -> Result<Self, LexError> {
        assert!(source.len() < u32::MAX as usize, "Source is too long");

        parse_impl(Cursor {
            buffer: source,
            off: 0,
        })
    }
}

fn parse_impl(mut cursor: Cursor) -> Result<TokenStream, LexError> {
    let mut trees = Vec::new();
    let mut stack = Vec::<(Delim, u32, Vec<_>)>::new();

    loop {
        cursor = cursor.skip_whitespace();

        let pos = cursor.off;

        let Some(first) = cursor.bytes().next() else {
            match stack.last() {
                None => {
                    return Ok(TokenStream {
                        trees: Rc::from(trees),
                    });
                }
                Some(&(_, pos, _)) => {
                    return Err(LexError {
                        span: Span::new(pos),
                    });
                }
            }
        };

        if let Some(open_delim) = match first {
            b'(' => Some(Delim::Paren),
            b'[' => Some(Delim::Bracket),
            b'{' => Some(Delim::Brace),
            _ => None,
        } {
            cursor = cursor.advance(1);
            stack.push((open_delim, pos, trees));
            trees = Vec::new();
            continue;
        } else if let Some(close_delim) = match first {
            b')' => Some(Delim::Paren),
            b']' => Some(Delim::Bracket),
            b'}' => Some(Delim::Brace),
            _ => None,
        } {
            let Some((open_delim, open_pos, outer)) = stack.pop() else {
                return Err(LexError {
                    span: Span::new(pos),
                });
            };

            if open_delim != close_delim {
                return Err(LexError {
                    span: Span::new(pos),
                });
            }

            cursor = cursor.advance(1);

            let g = Group {
                delim: close_delim,
                stream: TokenStream {
                    trees: Rc::from(trees),
                },
                span: Span::range(open_pos, pos),
            };

            trees = outer;
            trees.push(TokenTree::Group(g));
            continue;
        }

        // Parse leaf tokens in the following order:
        // LitStr
        // LitChar
        // LitNum
        // Punct
        // Ident
        // Atom

        if let Some((lit_str, next)) = parse_str(cursor)? {
            cursor = next;
            trees.push(TokenTree::Lit(Lit::Str(lit_str)));
            continue;
        }

        if let Some((lit_char, next)) = parse_char(cursor)? {
            cursor = next;
            trees.push(TokenTree::Lit(Lit::Char(lit_char)));
            continue;
        }

        if let Some((lit_num, next)) = parse_num(cursor)? {
            cursor = next;
            trees.push(TokenTree::Lit(Lit::Num(lit_num)));
            continue;
        }

        if let Some((punct, next)) = parse_punct(cursor)? {
            cursor = next;
            trees.push(TokenTree::Punct(punct));
            continue;
        }

        if let Some((ident, next)) = parse_ident(cursor)? {
            cursor = next;
            trees.push(TokenTree::Ident(ident));
            continue;
        }

        if let Some((atom, next)) = parse_atom(cursor)? {
            cursor = next;
            trees.push(TokenTree::Atom(atom));
            continue;
        }

        return Err(LexError {
            span: Span::new(pos),
        });
    }
}

type PResult<'a, T> = Result<Option<(T, Cursor<'a>)>, LexError>;

fn parse_str(cursor: Cursor) -> PResult<LitStr> {
    let mut chars = cursor.char_indices();

    match chars.next() {
        Some((_, '"')) => {}
        _ => return Ok(None),
    }

    let mut end = 0;
    for (i, ch) in chars {
        if ch == '"' {
            end = i;
            break;
        }
    }

    if end == 0 {
        return Ok(None);
    }

    let next = cursor.advance(end + 1);
    let literal = unescape(&cursor.buffer[1..end], cursor.off + 1)?;
    let span = Span::range(cursor.off, next.off);

    Ok(Some((LitStr { literal, span }, next)))
}

fn parse_char(cursor: Cursor) -> PResult<LitChar> {
    let mut chars = cursor.char_indices();

    match chars.next() {
        Some((_, '\'')) => {}
        _ => return Ok(None),
    }

    let mut end = 0;
    for (i, ch) in chars {
        if ch == '\'' {
            end = i;
            break;
        }
    }

    if end == 0 {
        return Ok(None);
    }

    let next = cursor.advance(end + 1);
    let literal = unescape(&cursor.buffer[1..end], cursor.off + 1)?;
    let span = Span::range(cursor.off, next.off);

    Ok(Some((LitChar { literal, span }, next)))
}

fn parse_num(cursor: Cursor) -> PResult<LitNum> {
    let mut chars = cursor.char_indices();

    let mut kind = LitNumKind::Decimal {
        point: None,
        exponent: None,
    };

    match chars.next() {
        Some((_, '0')) => match chars.next() {
            Some((_, 'b' | 'B')) => kind = LitNumKind::Binary,
            Some((_, 'o' | 'O')) => kind = LitNumKind::Octal,
            Some((_, 'x' | 'X')) => kind = LitNumKind::Hexadecimal,
            Some((pos, '.')) => {
                kind = LitNumKind::Decimal {
                    point: Some(pos as u32),
                    exponent: None,
                };
            }
            Some((pos, 'e' | 'E')) => {
                kind = LitNumKind::Decimal {
                    point: None,
                    exponent: Some(pos as u32),
                };
            }
            Some((_, '0'..='9')) => {}
            _ => {
                let next = cursor.advance(1);
                let literal = SmolStr::new(&cursor.buffer[..1]);
                let span = Span::range(cursor.off, next.off);

                return Ok(Some((
                    LitNum {
                        literal,
                        kind,
                        span,
                    },
                    next,
                )));
            }
        },
        Some((_, '1'..='9')) => {}
        _ => return Ok(None),
    }

    let mut end = cursor.len();
    match &mut kind {
        LitNumKind::Binary => {
            for (i, ch) in chars {
                if matches!(ch, '0'..='1') {
                    continue;
                }

                end = i;
                break;
            }
        }
        LitNumKind::Octal => {
            for (i, ch) in chars {
                if matches!(ch, '0'..='7') {
                    continue;
                }

                end = i;
                break;
            }
        }
        LitNumKind::Hexadecimal => {
            for (i, ch) in chars {
                if matches!(ch, '0'..='9' | 'a'..='f' | 'A'..='F') {
                    continue;
                }

                end = i;
                break;
            }
        }
        LitNumKind::Decimal { point, exponent } => {
            let mut exponent_sign = None;
            for (i, ch) in chars {
                if exponent_sign.is_none() && matches!(ch, '+' | '-') {
                    exponent_sign = Some(i as u32);
                    continue;
                }

                if point.is_none() && exponent.is_none() && ch == '.' {
                    *point = Some(i as u32);
                    continue;
                }

                if exponent.is_none() && matches!(ch, 'e' | 'E') {
                    *exponent = Some(i as u32);
                    continue;
                }

                if matches!(ch, '0'..='9') {
                    continue;
                }

                if *exponent == Some(i as u32) || exponent_sign == Some(i as u32) {
                    // If we are at the exponent position, we must have at least one digit after it.
                    return Err(LexError {
                        span: Span::new(cursor.off + i as u32),
                    });
                }

                end = i;
                break;
            }
        }
    }

    debug_assert_ne!(end, 0, "Parsed at least one character");

    if end < 2 && !matches!(kind, LitNumKind::Decimal { .. }) {
        return Err(LexError {
            span: Span::new(cursor.off + end as u32),
        });
    }

    if let LitNumKind::Decimal { exponent, .. } = kind {
        if let Some(exponent) = exponent {
            if exponent as usize >= end - 1 {
                // Exponent symbol may not be the last character
                // and must be followed by at least one digit.
                return Err(LexError {
                    span: Span::new(cursor.off + exponent),
                });
            }
        }
    }

    let next = cursor.advance(end);
    let literal = SmolStr::new(&cursor.buffer[..end]);
    let span = Span::range(cursor.off, next.off);

    Ok(Some((
        LitNum {
            literal,
            kind,
            span,
        },
        next,
    )))
}

fn parse_punct(cursor: Cursor) -> PResult<Punct> {
    let mut chars = cursor.char_indices();

    let (ch, spacing) = match chars.next() {
        Some((_, ch)) if is_punctuation_char(ch) => {
            let spacing = match chars.next() {
                Some((_, next_ch)) if is_punctuation_char(next_ch) => Spacing::Joint,
                _ => Spacing::Alone,
            };
            (ch, spacing)
        }
        _ => return Ok(None),
    };

    let next = cursor.advance(1);
    let char = ch;
    let span = Span::range(cursor.off, next.off);
    Ok(Some((
        Punct {
            char,
            spacing,
            span,
        },
        next,
    )))
}

fn parse_ident(cursor: Cursor) -> PResult<Ident> {
    let mut chars = cursor.char_indices();

    match chars.next() {
        Some((_, ch)) if is_ident_start(ch) => {}
        _ => return Ok(None),
    }

    let mut end = cursor.len();
    for (i, ch) in chars {
        if !is_ident_continue(ch) {
            end = i;
            break;
        }
    }

    let next = cursor.advance(end);
    let name = SmolStr::new(&cursor.buffer[..end]);
    let span = Span::range(cursor.off, next.off);

    Ok(Some((Ident { name, span }, next)))
}

fn parse_atom(cursor: Cursor) -> PResult<Atom> {
    let mut chars = cursor.char_indices();

    match chars.next() {
        Some((_, ch)) if is_atom_start(ch) => {}
        _ => return Ok(None),
    }

    let mut end = cursor.len();
    for (i, ch) in chars {
        if !is_atom_continue(ch) {
            end = i;
            break;
        }
    }

    let next = cursor.advance(end);
    let name = SmolStr::new(&cursor.buffer[..end]);
    let span = Span::range(cursor.off, next.off);

    Ok(Some((Atom { name, span }, next)))
}

fn unescape(s: &str, start: u32) -> Result<SmolStr, LexError> {
    let mut builder = SmolStrBuilder::new();
    let mut chars = s.char_indices().peekable();

    while let Some((_, ch)) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some((_, 'n')) => builder.push('\n'),
                Some((_, 't')) => builder.push('\t'),
                Some((_, 'r')) => builder.push('\r'),
                Some((_, '\\')) => builder.push('\\'),
                Some((_, '"')) => builder.push('"'),
                Some((_, '\'')) => builder.push('\''),
                Some((pos, _other)) => {
                    return Err(LexError {
                        span: Span::new(start + pos as u32),
                    });
                }
                None => {
                    return Err(LexError {
                        span: Span::new(start + s.len() as u32),
                    });
                }
            }
        } else {
            builder.push(ch);
        }
    }

    Ok(builder.finish())
}
