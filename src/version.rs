#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub rev: u32,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.rev)
    }
}

impl Version {
    pub fn new(mut s: &str) -> Version {
        if let Some(p) = s.find('+') {
            s = &s[..p];
        }
        if let Some(p) = s.find('-') {
            s = &s[..p];
        }

        let mut pieces = s.split('.');
        let major = pieces.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let minor = pieces.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let rev = pieces.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        Version { major, minor, rev }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_version() {
        assert_eq!(
            Version::new("1.2.3"),
            Version {
                major: 1,
                minor: 2,
                rev: 3
            }
        );
    }

    #[test]
    fn test_version_with_plus_suffix() {
        assert_eq!(
            Version::new("1.2.3+build"),
            Version {
                major: 1,
                minor: 2,
                rev: 3
            }
        );
    }

    #[test]
    fn test_version_with_dash_suffix() {
        assert_eq!(
            Version::new("1.2.3-beta"),
            Version {
                major: 1,
                minor: 2,
                rev: 3
            }
        );
    }

    #[test]
    fn test_partial_version_major() {
        assert_eq!(
            Version::new("1"),
            Version {
                major: 1,
                minor: 0,
                rev: 0
            }
        );
    }

    #[test]
    fn test_partial_version_major_minor() {
        assert_eq!(
            Version::new("1.2"),
            Version {
                major: 1,
                minor: 2,
                rev: 0
            }
        );
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(
            Version::new(""),
            Version {
                major: 0,
                minor: 0,
                rev: 0
            }
        );
    }

    #[test]
    fn test_malformed_input() {
        assert_eq!(
            Version::new("a.b.c"),
            Version {
                major: 0,
                minor: 0,
                rev: 0
            }
        );
    }

    #[test]
    fn test_too_many_numbers() {
        assert_eq!(
            Version::new("1.2.3.4"),
            Version {
                major: 1,
                minor: 2,
                rev: 3
            }
        );
    }
}
