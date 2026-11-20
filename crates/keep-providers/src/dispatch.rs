//! The guard that stands between the harness and any remote model.
//!
//! Remote providers take a [`DispatchPermit`] in their constructors. The only
//! way to obtain one is [`DispatchPolicy::permit`], which refuses unless the
//! manifest enables dispatch, the operator passed the explicit CLI flag, and
//! the caller reports no unresolved configuration problems (unpinned models,
//! zero budgets and so on).

use std::fmt;

#[derive(Clone, Debug, Default)]
pub struct DispatchPolicy {
    /// `dispatch.allow_model_calls` in the run manifest.
    pub manifest_allows: bool,
    /// `--allow-model-calls` on the command line.
    pub operator_allows: bool,
}

/// Proof that dispatch was explicitly authorised for one run. It cannot be
/// constructed outside this module.
#[derive(Debug)]
pub struct DispatchPermit {
    run_label: String,
}

impl DispatchPermit {
    pub fn run_label(&self) -> &str {
        &self.run_label
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub struct DispatchRefused {
    pub reasons: Vec<String>,
}

impl fmt::Display for DispatchRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "model calls are disabled for this run:")?;
        for reason in &self.reasons {
            writeln!(f, "  - {reason}")?;
        }
        write!(
            f,
            "Remote model calls need `dispatch.allow_model_calls = true` in the manifest, \
             the `--allow-model-calls` flag, pinned models and non-zero budget caps. \
             Use `dry-run` or the mock providers to work offline."
        )
    }
}

impl DispatchPolicy {
    /// Grant a permit only if every condition holds. `config_problems` lists
    /// anything else the caller found that must block dispatch.
    pub fn permit(
        &self,
        run_label: &str,
        config_problems: Vec<String>,
    ) -> Result<DispatchPermit, DispatchRefused> {
        let mut reasons = Vec::new();
        if !self.manifest_allows {
            reasons.push("the manifest does not set dispatch.allow_model_calls = true".into());
        }
        if !self.operator_allows {
            reasons.push("the --allow-model-calls flag was not passed".into());
        }
        reasons.extend(config_problems);
        if reasons.is_empty() {
            Ok(DispatchPermit {
                run_label: run_label.to_owned(),
            })
        } else {
            Err(DispatchRefused { reasons })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_is_refused_by_default_and_needs_both_switches() {
        let default = DispatchPolicy::default().permit("r", vec![]).unwrap_err();
        assert_eq!(default.reasons.len(), 2);

        let manifest_only = DispatchPolicy {
            manifest_allows: true,
            operator_allows: false,
        };
        assert!(manifest_only.permit("r", vec![]).is_err());

        let flag_only = DispatchPolicy {
            manifest_allows: false,
            operator_allows: true,
        };
        assert!(flag_only.permit("r", vec![]).is_err());

        let both = DispatchPolicy {
            manifest_allows: true,
            operator_allows: true,
        };
        let blocked = both
            .permit("r", vec!["answer model revision is not pinned".into()])
            .unwrap_err();
        assert!(blocked.to_string().contains("not pinned"));
        assert_eq!(both.permit("r", vec![]).unwrap().run_label(), "r");
    }
}
