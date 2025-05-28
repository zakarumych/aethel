#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod span;
mod token;

#[cfg(test)]
mod tests;
