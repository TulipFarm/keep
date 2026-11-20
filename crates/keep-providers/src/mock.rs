//! Deterministic offline chat models.

use async_trait::async_trait;

use crate::ProviderDescriptor;
use crate::chat::{ChatModel, ChatRequest, ChatResponse, ProviderError, Usage};
use crate::tokenizer::{ApproxTokenizer, Tokenizer};

type Responder = dyn Fn(&ChatRequest) -> Result<String, ProviderError> + Send + Sync;

/// A chat model whose reply is computed by a pure function of the request.
/// Usage is estimated with [`ApproxTokenizer`] and marked as not
/// provider-reported.
pub struct FnChat {
    descriptor: ProviderDescriptor,
    respond: Box<Responder>,
}

impl FnChat {
    pub fn new(
        model: &str,
        respond: impl Fn(&ChatRequest) -> Result<String, ProviderError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            descriptor: ProviderDescriptor {
                provider: "mock".into(),
                model: model.into(),
                revision: "offline".into(),
                mock: true,
            },
            respond: Box::new(respond),
        }
    }
}

#[async_trait]
impl ChatModel for FnChat {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let text = (self.respond)(request)?;
        let tokenizer = ApproxTokenizer;
        let input: usize = request
            .messages
            .iter()
            .map(|m| tokenizer.count(&m.content))
            .sum();
        let mut output = tokenizer.count(&text) as u64;
        let mut text = text;
        let mut finish = "stop";
        if output > u64::from(request.max_output_tokens) {
            // Mirror a real server: cut the reply at the output allowance.
            let keep_chars = request.max_output_tokens as usize * 4;
            text = text.chars().take(keep_chars).collect();
            output = u64::from(request.max_output_tokens);
            finish = "length";
        }
        Ok(ChatResponse {
            text,
            usage: Usage {
                input_tokens: input as u64,
                output_tokens: output,
                provider_reported: false,
            },
            finish_reason: Some(finish.into()),
            billing: None,
            served_model: None,
            system_fingerprint: None,
        })
    }
}
