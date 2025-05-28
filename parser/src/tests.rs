mod tokenizer {
    use crate::token::{Lit, TokenTree};

    #[test]
    fn ident() {
        let test_string = "hello_world";

        let stream = crate::token::parse(&test_string).expect("Failed to parse identifier");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Ident(_)));

        match token {
            TokenTree::Ident(ident) => {
                assert_eq!(ident.name, test_string);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn atom() {
        let test_string = "HelloWorld";

        let stream = crate::token::parse(&test_string).expect("Failed to parse atom");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Atom(_)));

        match token {
            TokenTree::Atom(atom) => {
                assert_eq!(atom.name, test_string);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn punct() {
        let test_string = ",";

        let stream = crate::token::parse(&test_string).expect("Failed to parse atom");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Punct(_)));

        match token {
            TokenTree::Punct(punct) => {
                assert_eq!(punct.char, ',');
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn integer() {
        let test_string = "12345";

        let stream = crate::token::parse(&test_string).expect("Failed to parse number");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Num(_))));

        match token {
            TokenTree::Lit(Lit::Num(literal)) => {
                assert_eq!(literal.parse_integer(), Some(12345));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn integer_with_exponent() {
        let test_string = "12345e4";

        let stream = crate::token::parse(&test_string).expect("Failed to parse number");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Num(_))));

        match token {
            TokenTree::Lit(Lit::Num(literal)) => {
                assert_eq!(literal.parse_integer(), Some(123450000));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn float() {
        let test_string = "123.45";

        let stream = crate::token::parse(&test_string).expect("Failed to parse number");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Num(_))));

        match token {
            TokenTree::Lit(Lit::Num(literal)) => {
                assert_eq!(literal.parse_float(), Some(123.45));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn float_with_exponent() {
        let test_string = "123.45e4";

        let stream = crate::token::parse(&test_string).expect("Failed to parse number");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Num(_))));

        match token {
            TokenTree::Lit(Lit::Num(literal)) => {
                assert_eq!(literal.parse_float(), Some(1234500.0));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn float_with_neg_exponent() {
        let test_string = "123.45e-4";

        let stream = crate::token::parse(&test_string).expect("Failed to parse number");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Num(_))));

        match token {
            TokenTree::Lit(Lit::Num(literal)) => {
                assert_eq!(literal.parse_float(), Some(0.012345));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn boolean_true() {
        let test_string = "true";

        let stream = crate::token::parse(&test_string).expect("Failed to parse boolean");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Bool(_))));

        match token {
            TokenTree::Lit(Lit::Bool(literal)) => {
                assert!(literal.value());
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn boolean_false() {
        let test_string = "false";

        let stream = crate::token::parse(&test_string).expect("Failed to parse boolean");
        let tokens = stream.into_iter().collect::<Vec<_>>();

        assert_eq!(tokens.len(), 1);
        let token = &tokens[0];

        assert!(matches!(token, TokenTree::Lit(Lit::Bool(_))));

        match token {
            TokenTree::Lit(Lit::Bool(literal)) => {
                assert!(!literal.value());
            }
            _ => unreachable!(),
        }
    }
}
