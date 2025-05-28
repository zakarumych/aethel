#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod span;
mod token;

#[cfg(test)]
mod tests;

pub use self::{
    span::Span,
    token::{
        Atom, Ident, LexError, Lit, LitChar, LitNum, LitNumKind, LitStr, Punct, TokenStream,
        TokenTree, parse,
    },
};
