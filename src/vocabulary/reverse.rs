use std::collections::BTreeMap;
use std::sync::Arc;

use crate::error::{CompileError, CompileStage};
use crate::primitives::TokenId;
use crate::schema::CompileLimits;
use crate::{Error, Result};

use super::Vocabulary;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DerivedVocabulary {
    pub trie: Arc<TokenTrie>,
    pub reverse: Vec<Option<Arc<[u8]>>>,
    pub eos_token_id: TokenId,
    pub vocab_size: usize,
    pub mask_words: usize,
    pub total_token_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TokenTrie {
    pub nodes: Vec<TrieNode>,
    pub edges: Vec<TrieEdge>,
    pub terminal_ids: Vec<TokenId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrieNode {
    pub edge_start: u32,
    pub edge_len: u32,
    pub token_start: u32,
    pub token_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrieEdge {
    pub byte: u8,
    pub child: u32,
}

#[derive(Default)]
struct BuildNode {
    children: BTreeMap<u8, usize>,
    terminal_ids: Vec<TokenId>,
}

impl DerivedVocabulary {
    pub fn build(vocabulary: &Vocabulary, limits: &CompileLimits) -> Result<Self> {
        let entry_count = vocabulary.tokens().len();
        enforce(
            CompileStage::VocabularyProjection,
            entry_count,
            limits.max_trie_nodes,
        )?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(entry_count)
            .map_err(|_| allocation_limit(entry_count, limits.max_trie_nodes))?;
        for (bytes, ids) in vocabulary.tokens() {
            entries.push((bytes.as_slice(), ids.as_slice()));
        }
        entries.sort_by(|left, right| left.0.cmp(right.0));

        let mut max_id = vocabulary.eos_token_id();
        let mut total_token_bytes = 0usize;
        let mut alias_count = 0usize;
        for (bytes, ids) in &entries {
            if bytes.is_empty() {
                return Err(Error::EmptyTokenDisallowed);
            }
            enforce(
                CompileStage::VocabularyProjection,
                bytes.len(),
                limits.max_token_bytes,
            )?;
            total_token_bytes = total_token_bytes.checked_add(bytes.len()).ok_or(
                CompileError::ResourceLimitExceeded {
                    stage: CompileStage::VocabularyProjection,
                    observed: usize::MAX,
                    limit: limits.max_total_token_bytes,
                },
            )?;
            enforce(
                CompileStage::VocabularyProjection,
                total_token_bytes,
                limits.max_total_token_bytes,
            )?;
            alias_count =
                alias_count
                    .checked_add(ids.len())
                    .ok_or(CompileError::ResourceLimitExceeded {
                        stage: CompileStage::VocabularyProjection,
                        observed: usize::MAX,
                        limit: limits.max_trie_terminal_ids,
                    })?;
            enforce(
                CompileStage::VocabularyProjection,
                alias_count,
                limits.max_trie_terminal_ids,
            )?;
            if let Some(id) = ids.iter().max() {
                max_id = max_id.max(*id);
            }
        }

        let vocab_size = usize::try_from(max_id)
            .ok()
            .and_then(|id| id.checked_add(1))
            .ok_or(CompileError::ResourceLimitExceeded {
                stage: CompileStage::VocabularyProjection,
                observed: usize::MAX,
                limit: limits.max_vocabulary_slots,
            })?;
        enforce(
            CompileStage::VocabularyProjection,
            vocab_size,
            limits.max_vocabulary_slots,
        )?;
        let mask_words = vocab_size
            .checked_add(31)
            .ok_or(CompileError::ResourceLimitExceeded {
                stage: CompileStage::VocabularyProjection,
                observed: usize::MAX,
                limit: limits.max_vocabulary_slots,
            })?
            / 32;

        let mut reverse: Vec<Option<Arc<[u8]>>> = Vec::new();
        reverse
            .try_reserve_exact(vocab_size)
            .map_err(|_| CompileError::ResourceLimitExceeded {
                stage: CompileStage::VocabularyProjection,
                observed: vocab_size,
                limit: limits.max_vocabulary_slots,
            })?;
        reverse.resize_with(vocab_size, || None);

        let mut build_nodes = vec![BuildNode::default()];
        for (bytes, ids) in entries {
            let shared: Arc<[u8]> = Arc::from(bytes);
            let mut node = 0usize;
            for &byte in bytes {
                if let Some(&child) = build_nodes[node].children.get(&byte) {
                    node = child;
                } else {
                    let child = build_nodes.len();
                    enforce(
                        CompileStage::VocabularyProjection,
                        child.saturating_add(1),
                        limits.max_trie_nodes,
                    )?;
                    build_nodes
                        .try_reserve(1)
                        .map_err(|_| allocation_limit(child + 1, limits.max_trie_nodes))?;
                    build_nodes.push(BuildNode::default());
                    build_nodes[node].children.insert(byte, child);
                    node = child;
                }
            }
            let mut sorted_ids = Vec::new();
            sorted_ids
                .try_reserve_exact(ids.len())
                .map_err(|_| allocation_limit(ids.len(), limits.max_trie_terminal_ids))?;
            sorted_ids.extend_from_slice(ids);
            sorted_ids.sort_unstable();
            sorted_ids.dedup();
            for id in sorted_ids {
                let slot =
                    usize::try_from(id).map_err(|_| CompileError::ResourceLimitExceeded {
                        stage: CompileStage::VocabularyProjection,
                        observed: usize::MAX,
                        limit: limits.max_vocabulary_slots,
                    })?;
                match &reverse[slot] {
                    Some(existing) if existing.as_ref() != bytes => {
                        return Err(Error::AmbiguousTokenId { token_id: id });
                    }
                    Some(_) => {}
                    None => reverse[slot] = Some(Arc::clone(&shared)),
                }
                build_nodes[node]
                    .terminal_ids
                    .try_reserve(1)
                    .map_err(|_| allocation_limit(alias_count, limits.max_trie_terminal_ids))?;
                build_nodes[node].terminal_ids.push(id);
            }
        }

        let edge_count = build_nodes.iter().try_fold(0usize, |total, node| {
            total
                .checked_add(node.children.len())
                .ok_or(CompileError::ResourceLimitExceeded {
                    stage: CompileStage::VocabularyProjection,
                    observed: usize::MAX,
                    limit: limits.max_trie_edges,
                })
        })?;
        enforce(
            CompileStage::VocabularyProjection,
            edge_count,
            limits.max_trie_edges,
        )?;

        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(build_nodes.len())
            .map_err(|_| allocation_limit(build_nodes.len(), limits.max_trie_nodes))?;
        let mut edges = Vec::new();
        edges
            .try_reserve_exact(edge_count)
            .map_err(|_| allocation_limit(edge_count, limits.max_trie_edges))?;
        let mut terminal_ids = Vec::new();
        terminal_ids
            .try_reserve_exact(alias_count)
            .map_err(|_| allocation_limit(alias_count, limits.max_trie_terminal_ids))?;

        for node in &mut build_nodes {
            node.terminal_ids.sort_unstable();
            node.terminal_ids.dedup();
            let edge_start = checked_u32(edges.len(), limits.max_trie_edges)?;
            for (&byte, &child) in &node.children {
                edges.push(TrieEdge {
                    byte,
                    child: checked_u32(child, limits.max_trie_nodes)?,
                });
            }
            let token_start = checked_u32(terminal_ids.len(), limits.max_trie_terminal_ids)?;
            terminal_ids.extend_from_slice(&node.terminal_ids);
            nodes.push(TrieNode {
                edge_start,
                edge_len: checked_u32(node.children.len(), limits.max_trie_edges)?,
                token_start,
                token_len: checked_u32(node.terminal_ids.len(), limits.max_trie_terminal_ids)?,
            });
        }

        Ok(Self {
            trie: Arc::new(TokenTrie {
                nodes,
                edges,
                terminal_ids,
            }),
            reverse,
            eos_token_id: vocabulary.eos_token_id(),
            vocab_size,
            mask_words,
            total_token_bytes,
        })
    }

    pub fn shared_bytes(&self, id: TokenId) -> Option<Arc<[u8]>> {
        let slot = usize::try_from(id).ok()?;
        self.reverse
            .get(slot)
            .and_then(Option::as_ref)
            .map(Arc::clone)
    }
}

fn checked_u32(value: usize, limit: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| allocation_limit(value, limit).into())
}

fn allocation_limit(observed: usize, limit: usize) -> CompileError {
    CompileError::ResourceLimitExceeded {
        stage: CompileStage::VocabularyProjection,
        observed,
        limit,
    }
}

fn enforce(stage: CompileStage, observed: usize, limit: usize) -> Result<(), CompileError> {
    if observed > limit {
        Err(CompileError::ResourceLimitExceeded {
            stage,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_trie_preserves_aliases_and_sparse_width() {
        let mut vocabulary = Vocabulary::new(100);
        vocabulary.try_insert("ab", 5).unwrap();
        vocabulary.try_insert("a", 0).unwrap();
        vocabulary.try_insert("ab", 19).unwrap();
        let first = DerivedVocabulary::build(&vocabulary, &CompileLimits::default()).unwrap();
        let second = DerivedVocabulary::build(&vocabulary, &CompileLimits::default()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.vocab_size, 101);
        assert_eq!(first.mask_words, 4);
        assert_eq!(first.shared_bytes(5).as_deref(), Some(b"ab".as_slice()));
        assert_eq!(first.trie.terminal_ids, vec![0, 5, 19]);
    }

    #[test]
    fn conflicting_ids_and_empty_tokens_fail() {
        let mut conflict = Vocabulary::new(2);
        conflict.try_insert("a", 1).unwrap();
        conflict.try_insert("b", 1).unwrap();
        assert!(matches!(
            DerivedVocabulary::build(&conflict, &CompileLimits::default()),
            Err(Error::AmbiguousTokenId { token_id: 1 })
        ));

        let mut empty = Vocabulary::new(1);
        empty.try_insert(Vec::<u8>::new(), 0).unwrap();
        assert!(matches!(
            DerivedVocabulary::build(&empty, &CompileLimits::default()),
            Err(Error::EmptyTokenDisallowed)
        ));
    }

    #[test]
    fn sparse_ids_respect_the_slot_limit_before_allocation() {
        let mut vocabulary = Vocabulary::new(u32::MAX);
        vocabulary.try_insert("x", 0).unwrap();
        assert!(matches!(
            DerivedVocabulary::build(&vocabulary, &CompileLimits::default()),
            Err(Error::Compile(CompileError::ResourceLimitExceeded {
                stage: CompileStage::VocabularyProjection,
                ..
            }))
        ));
    }
}
