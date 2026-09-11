//! Incremental Earley recognition with nullable closure and optional Leo completion.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;

use crate::error::{CompileError, CompileStage, RuntimeError, RuntimeResource};
use crate::grammar::{
    NonterminalId, ProductionId, ResidualGrammar, Symbol, TerminalCursorState, TerminalId,
};
use crate::schema::{CompileLimits, RuntimeLimits};

pub type DottedId = u32;

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DottedRule {
    pub production: ProductionId,
    pub dot: u32,
    pub next: Option<Symbol>,
    pub advanced: Option<DottedId>,
    pub lhs: NonterminalId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EarleyItem {
    pub dotted: DottedId,
    pub origin: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct EarleyStats {
    pub columns: u64,
    pub items_inserted: u64,
    pub duplicate_items: u64,
    pub predictions: u64,
    pub scans: u64,
    pub completions: u64,
    pub nullable_advances: u64,
    pub leo_candidates: u64,
    pub leo_hits: u64,
    pub max_column_items: u64,
    pub max_active_scans: u64,
    pub estimated_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChartColumn {
    pub position: u32,
    pub items: Vec<EarleyItem>,
    membership: FxHashSet<u64>,
    waiting_nonterminal: FxHashMap<NonterminalId, Vec<EarleyItem>>,
    waiting_terminal: FxHashMap<TerminalId, Vec<EarleyItem>>,
    leo_by_postdot: FxHashMap<NonterminalId, EarleyItem>,
}

impl ChartColumn {
    fn new(position: u32) -> Self {
        Self {
            position,
            items: Vec::new(),
            membership: FxHashSet::default(),
            waiting_nonterminal: FxHashMap::default(),
            waiting_terminal: FxHashMap::default(),
            leo_by_postdot: FxHashMap::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActiveScan {
    pub terminal: TerminalId,
    pub start: u32,
    pub cursor: TerminalCursorState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EarleyProduction {
    lhs: NonterminalId,
    rhs: Vec<Symbol>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledEarley {
    grammar: Arc<ResidualGrammar>,
    productions: Vec<EarleyProduction>,
    dotted: Vec<DottedRule>,
    dotted_offsets: Vec<DottedId>,
    starts_by_lhs: Vec<Vec<DottedId>>,
    augmented_production: ProductionId,
    augmented_complete: DottedId,
    pub nullable_nonterminals: usize,
    pub leo_eligible_rules: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EarleyCheckpoint {
    generation: u64,
    column_count: usize,
    active_scans: Vec<ActiveScan>,
    byte_position: u32,
    total_items: usize,
    leo_items: usize,
    stats: EarleyStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EarleyRecognizer {
    compiled: Arc<CompiledEarley>,
    limits: RuntimeLimits,
    leo_enabled: bool,
    generation: u64,
    chart: Vec<ChartColumn>,
    active_scans: Vec<ActiveScan>,
    byte_position: u32,
    total_items: usize,
    leo_items: usize,
    stats: EarleyStats,
}

impl CompiledEarley {
    pub fn build(
        grammar: Arc<ResidualGrammar>,
        limits: &CompileLimits,
    ) -> Result<Self, CompileError> {
        let augmented = u32::try_from(grammar.nonterminals.len()).map_err(|_| {
            CompileError::ResourceLimitExceeded {
                stage: CompileStage::EarleyPreparation,
                observed: grammar.nonterminals.len(),
                limit: u32::MAX as usize,
            }
        })?;
        let mut productions: Vec<_> = grammar
            .productions
            .iter()
            .map(|production| EarleyProduction {
                lhs: production.lhs,
                rhs: production.rhs.clone(),
            })
            .collect();
        let augmented_production =
            u32::try_from(productions.len()).map_err(|_| CompileError::ResourceLimitExceeded {
                stage: CompileStage::EarleyPreparation,
                observed: productions.len(),
                limit: u32::MAX as usize,
            })?;
        productions.push(EarleyProduction {
            lhs: augmented,
            rhs: vec![Symbol::Nonterminal(grammar.start)],
        });

        let mut dotted = Vec::new();
        let mut dotted_offsets = Vec::new();
        let mut starts_by_lhs = vec![Vec::new(); grammar.nonterminals.len() + 1];
        for (production_id, production) in productions.iter().enumerate() {
            enforce_compile_limit(
                CompileStage::EarleyPreparation,
                dotted
                    .len()
                    .saturating_add(production.rhs.len())
                    .saturating_add(1),
                limits.max_lr_items,
            )?;
            let offset =
                u32::try_from(dotted.len()).map_err(|_| CompileError::ResourceLimitExceeded {
                    stage: CompileStage::EarleyPreparation,
                    observed: dotted.len(),
                    limit: u32::MAX as usize,
                })?;
            dotted_offsets.push(offset);
            starts_by_lhs[production.lhs as usize].push(offset);
            let production_id =
                u32::try_from(production_id).map_err(|_| CompileError::ResourceLimitExceeded {
                    stage: CompileStage::EarleyPreparation,
                    observed: production_id,
                    limit: u32::MAX as usize,
                })?;
            for dot in 0..=production.rhs.len() {
                let id = u32::try_from(dotted.len()).map_err(|_| {
                    CompileError::ResourceLimitExceeded {
                        stage: CompileStage::EarleyPreparation,
                        observed: dotted.len(),
                        limit: u32::MAX as usize,
                    }
                })?;
                dotted.push(DottedRule {
                    production: production_id,
                    dot: u32::try_from(dot).map_err(|_| CompileError::InternalInvariant {
                        message: "production length does not fit u32",
                    })?,
                    next: production.rhs.get(dot).copied(),
                    advanced: (dot < production.rhs.len()).then_some(id.saturating_add(1)),
                    lhs: production.lhs,
                });
            }
        }
        let augmented_complete = dotted_offsets[augmented_production as usize]
            .checked_add(1)
            .ok_or(CompileError::InternalInvariant {
                message: "augmented dotted rule overflowed",
            })?;
        let leo_eligible_rules = productions
            .iter()
            .filter(|production| matches!(production.rhs.last(), Some(Symbol::Nonterminal(_))))
            .count();
        Ok(Self {
            nullable_nonterminals: grammar.nullable.iter().filter(|value| **value).count(),
            grammar,
            productions,
            dotted,
            dotted_offsets,
            starts_by_lhs,
            augmented_production,
            augmented_complete,
            leo_eligible_rules,
        })
    }

    pub fn recognizer(
        self: &Arc<Self>,
        limits: RuntimeLimits,
        leo_enabled: bool,
    ) -> Result<EarleyRecognizer, RuntimeError> {
        enforce_runtime_limit(RuntimeResource::ChartColumns, 1, limits.max_chart_columns)?;
        let mut recognizer = EarleyRecognizer {
            compiled: Arc::clone(self),
            limits,
            leo_enabled,
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            chart: vec![ChartColumn::new(0)],
            active_scans: Vec::new(),
            byte_position: 0,
            total_items: 0,
            leo_items: 0,
            stats: EarleyStats::default(),
        };
        let start = self.dotted_offsets[self.augmented_production as usize];
        recognizer.insert_item(
            0,
            EarleyItem {
                dotted: start,
                origin: 0,
            },
        )?;
        let scans = recognizer.saturate(0)?;
        enforce_runtime_limit(
            RuntimeResource::ActiveScans,
            scans.len(),
            recognizer.limits.max_active_scans,
        )?;
        recognizer.active_scans = scans;
        recognizer.finish_column_stats(0);
        Ok(recognizer)
    }
}

impl EarleyRecognizer {
    pub fn advance_byte(&mut self, byte: u8) -> Result<bool, RuntimeError> {
        let next_position = self.byte_position as usize + 1;
        enforce_runtime_limit(
            RuntimeResource::InputBytes,
            next_position,
            self.limits.max_input_bytes,
        )?;
        enforce_runtime_limit(
            RuntimeResource::ChartColumns,
            self.chart.len().saturating_add(1),
            self.limits.max_chart_columns,
        )?;
        let next_position_u32 =
            u32::try_from(next_position).map_err(|_| RuntimeError::ResourceLimitExceeded {
                resource: RuntimeResource::InputBytes,
                observed: next_position,
                limit: u32::MAX as usize,
            })?;
        self.chart.push(ChartColumn::new(next_position_u32));
        let column = self.chart.len() - 1;
        let mut continuing = Vec::new();
        let scans = std::mem::take(&mut self.active_scans);
        for scan in scans {
            let terminal = &self.compiled.grammar.terminals[scan.terminal as usize];
            let Some(cursor) = terminal.advance(scan.cursor, byte) else {
                continue;
            };
            let accepting = terminal.is_accepting(cursor);
            let live = terminal.is_live(cursor);
            self.stats.scans = self.stats.scans.saturating_add(1);
            if accepting {
                let waiters = self.chart[scan.start as usize]
                    .waiting_terminal
                    .get(&scan.terminal)
                    .cloned()
                    .unwrap_or_default();
                for waiter in waiters {
                    let advanced = self.advance_item(waiter)?;
                    self.insert_item(column, advanced)?;
                }
            }
            if live {
                continuing.push(ActiveScan { cursor, ..scan });
            }
        }
        let created = self.saturate(column)?;
        continuing.extend(created);
        continuing.sort_by_key(|scan| (scan.terminal, scan.start, cursor_key(scan.cursor)));
        continuing.dedup();
        enforce_runtime_limit(
            RuntimeResource::ActiveScans,
            continuing.len(),
            self.limits.max_active_scans,
        )?;
        self.active_scans = continuing;
        self.byte_position = next_position_u32;
        self.finish_column_stats(column);
        Ok(self.is_live())
    }

    fn saturate(&mut self, column: usize) -> Result<Vec<ActiveScan>, RuntimeError> {
        let mut agenda = 0;
        let mut scans = Vec::new();
        while agenda < self.chart[column].items.len() {
            let item = self.chart[column].items[agenda];
            agenda += 1;
            let dotted = self.dotted(item)?;
            match dotted.next {
                Some(Symbol::Nonterminal(nonterminal)) => {
                    self.chart[column]
                        .waiting_nonterminal
                        .entry(nonterminal)
                        .or_default()
                        .push(item);
                    let starts = self.compiled.starts_by_lhs[nonterminal as usize].clone();
                    for dotted in starts {
                        self.stats.predictions = self.stats.predictions.saturating_add(1);
                        self.insert_item(
                            column,
                            EarleyItem {
                                dotted,
                                origin: self.chart[column].position,
                            },
                        )?;
                    }
                    if self.compiled.grammar.nullable[nonterminal as usize] {
                        self.stats.nullable_advances =
                            self.stats.nullable_advances.saturating_add(1);
                        let advanced = self.advance_item(item)?;
                        self.insert_item(column, advanced)?;
                    }
                }
                Some(Symbol::Terminal(terminal)) => {
                    self.chart[column]
                        .waiting_terminal
                        .entry(terminal)
                        .or_default()
                        .push(item);
                    scans.push(ActiveScan {
                        terminal,
                        start: self.chart[column].position,
                        cursor: self.compiled.grammar.terminals[terminal as usize].start(),
                    });
                }
                None => self.complete(column, item)?,
            }
        }
        scans.sort_by_key(|scan| (scan.terminal, scan.start, cursor_key(scan.cursor)));
        scans.dedup();
        if self.leo_enabled {
            self.finish_leo_column(column)?;
        }
        Ok(scans)
    }

    fn complete(&mut self, column: usize, item: EarleyItem) -> Result<(), RuntimeError> {
        let dotted = self.dotted(item)?;
        if dotted.production == self.compiled.augmented_production {
            return Ok(());
        }
        if self.leo_enabled {
            if let Some(top) = self.leo_completion(item)? {
                self.stats.leo_hits = self.stats.leo_hits.saturating_add(1);
                self.insert_item(column, top)?;
                return Ok(());
            }
        }
        let waiters = self.chart[item.origin as usize]
            .waiting_nonterminal
            .get(&dotted.lhs)
            .cloned()
            .unwrap_or_default();
        for waiter in waiters {
            self.stats.completions = self.stats.completions.saturating_add(1);
            let advanced = self.advance_item(waiter)?;
            self.insert_item(column, advanced)?;
        }
        Ok(())
    }

    fn leo_completion(&mut self, item: EarleyItem) -> Result<Option<EarleyItem>, RuntimeError> {
        let dotted = self.dotted(item)?;
        let origin = usize::try_from(item.origin).map_err(|_| RuntimeError::InternalInvariant {
            message: "Earley item origin does not fit usize",
        })?;
        let origin_column = self
            .chart
            .get(origin)
            .ok_or(RuntimeError::InternalInvariant {
                message: "Earley item origin column does not exist",
            })?;
        Ok(origin_column.leo_by_postdot.get(&dotted.lhs).copied())
    }

    fn finish_leo_column(&mut self, column: usize) -> Result<(), RuntimeError> {
        let mut postdots: Vec<_> = self.chart[column]
            .waiting_nonterminal
            .keys()
            .copied()
            .collect();
        postdots.sort_unstable();
        for postdot in postdots {
            let waiters = &self.chart[column].waiting_nonterminal[&postdot];
            if waiters.len() != 1 {
                continue;
            }
            let waiter = waiters[0];
            let waiter_dotted = self.dotted(waiter)?;
            let production = &self.compiled.productions[waiter_dotted.production as usize];
            if waiter_dotted.dot as usize + 1 != production.rhs.len() {
                continue;
            }

            let advanced = self.advance_item(waiter)?;
            let advanced_dotted = self.dotted(advanced)?;
            let advanced_origin =
                usize::try_from(advanced.origin).map_err(|_| RuntimeError::InternalInvariant {
                    message: "Leo predecessor origin does not fit usize",
                })?;
            let target = if advanced_origin < column {
                self.chart[advanced_origin]
                    .leo_by_postdot
                    .get(&advanced_dotted.lhs)
                    .copied()
                    .unwrap_or(advanced)
            } else {
                advanced
            };
            let next_count =
                self.leo_items
                    .checked_add(1)
                    .ok_or(RuntimeError::ResourceLimitExceeded {
                        resource: RuntimeResource::LeoItems,
                        observed: usize::MAX,
                        limit: self.limits.max_leo_items,
                    })?;
            enforce_runtime_limit(
                RuntimeResource::LeoItems,
                next_count,
                self.limits.max_leo_items,
            )?;
            self.chart[column].leo_by_postdot.insert(postdot, target);
            self.leo_items = next_count;
            self.stats.leo_candidates = self.stats.leo_candidates.saturating_add(1);
        }
        Ok(())
    }

    fn insert_item(&mut self, column: usize, item: EarleyItem) -> Result<(), RuntimeError> {
        let key = (u64::from(item.dotted) << 32) | u64::from(item.origin);
        if !self.chart[column].membership.insert(key) {
            self.stats.duplicate_items = self.stats.duplicate_items.saturating_add(1);
            return Ok(());
        }
        let next_column_items = self.chart[column].items.len().saturating_add(1);
        enforce_runtime_limit(
            RuntimeResource::ItemsPerColumn,
            next_column_items,
            self.limits.max_items_per_column,
        )?;
        let total = self
            .total_items
            .checked_add(1)
            .ok_or(RuntimeError::ResourceLimitExceeded {
                resource: RuntimeResource::TotalItems,
                observed: usize::MAX,
                limit: self.limits.max_total_items,
            })?;
        enforce_runtime_limit(
            RuntimeResource::TotalItems,
            total,
            self.limits.max_total_items,
        )?;
        self.chart[column].items.push(item);
        self.total_items = total;
        self.stats.items_inserted = self.stats.items_inserted.saturating_add(1);
        Ok(())
    }

    fn advance_item(&self, item: EarleyItem) -> Result<EarleyItem, RuntimeError> {
        let advanced = self
            .dotted(item)?
            .advanced
            .ok_or(RuntimeError::InternalInvariant {
                message: "completed Earley item cannot advance",
            })?;
        Ok(EarleyItem {
            dotted: advanced,
            origin: item.origin,
        })
    }

    fn dotted(&self, item: EarleyItem) -> Result<DottedRule, RuntimeError> {
        self.compiled
            .dotted
            .get(item.dotted as usize)
            .copied()
            .ok_or(RuntimeError::InternalInvariant {
                message: "Earley dotted rule does not exist",
            })
    }

    fn finish_column_stats(&mut self, column: usize) {
        self.stats.columns = usize_to_u64(self.chart.len());
        self.stats.max_column_items = self
            .stats
            .max_column_items
            .max(usize_to_u64(self.chart[column].items.len()));
        self.stats.max_active_scans = self
            .stats
            .max_active_scans
            .max(usize_to_u64(self.active_scans.len()));
        let item_bytes = self
            .total_items
            .saturating_mul(std::mem::size_of::<EarleyItem>());
        let column_bytes = self
            .chart
            .len()
            .saturating_mul(std::mem::size_of::<ChartColumn>());
        let scan_bytes = self
            .active_scans
            .capacity()
            .saturating_mul(std::mem::size_of::<ActiveScan>());
        self.stats.estimated_bytes = usize_to_u64(
            item_bytes
                .saturating_add(column_bytes)
                .saturating_add(scan_bytes),
        );
    }

    #[must_use]
    pub fn is_accepting(&self) -> bool {
        let key = u64::from(self.compiled.augmented_complete) << 32;
        self.chart
            .last()
            .is_some_and(|column| column.membership.contains(&key))
    }

    #[must_use]
    pub fn is_live(&self) -> bool {
        self.is_accepting() || !self.active_scans.is_empty()
    }

    #[must_use]
    pub fn checkpoint(&self) -> EarleyCheckpoint {
        EarleyCheckpoint {
            generation: self.generation,
            column_count: self.chart.len(),
            active_scans: self.active_scans.clone(),
            byte_position: self.byte_position,
            total_items: self.total_items,
            leo_items: self.leo_items,
            stats: self.stats.clone(),
        }
    }

    pub fn restore(&mut self, checkpoint: &EarleyCheckpoint) -> Result<(), RuntimeError> {
        if checkpoint.generation != self.generation {
            return Err(RuntimeError::InvalidCheckpoint {
                expected_generation: self.generation,
                found_generation: checkpoint.generation,
            });
        }
        if checkpoint.column_count == 0 || checkpoint.column_count > self.chart.len() {
            return Err(RuntimeError::InternalInvariant {
                message: "Earley checkpoint column count is invalid",
            });
        }
        self.chart.truncate(checkpoint.column_count);
        self.active_scans.clone_from(&checkpoint.active_scans);
        self.byte_position = checkpoint.byte_position;
        self.total_items = checkpoint.total_items;
        self.leo_items = checkpoint.leo_items;
        self.stats.clone_from(&checkpoint.stats);
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> &EarleyStats {
        &self.stats
    }
}

fn cursor_key(cursor: TerminalCursorState) -> (u8, u32) {
    match cursor {
        TerminalCursorState::Literal(offset) => (0, offset),
        TerminalCursorState::Dfa(state) => (1, state),
    }
}

fn enforce_compile_limit(
    stage: CompileStage,
    observed: usize,
    limit: usize,
) -> Result<(), CompileError> {
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

fn enforce_runtime_limit(
    resource: RuntimeResource,
    observed: usize,
    limit: usize,
) -> Result<(), RuntimeError> {
    if observed > limit {
        Err(RuntimeError::ResourceLimitExceeded {
            resource,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::prepare_structural_for_test;
    use crate::grammar::{CompiledTerminal, ResidualNonterminal, ResidualProduction};
    use crate::schema::{CompileOptions, Provenance, ResourceId, SchemaPointer};

    fn compile(schema: &[u8]) -> Arc<CompiledEarley> {
        let residual = prepare_structural_for_test(schema, &CompileOptions::default()).unwrap();
        Arc::new(CompiledEarley::build(Arc::new(residual), &CompileLimits::default()).unwrap())
    }

    fn provenance() -> Provenance {
        Provenance {
            resource: ResourceId(0),
            pointer: SchemaPointer(String::new()),
            keyword: None,
        }
    }

    fn grammar(
        nonterminal_count: usize,
        productions: Vec<(NonterminalId, Vec<Symbol>)>,
        literals: &[&[u8]],
    ) -> Arc<CompiledEarley> {
        let provenance = provenance();
        let productions: Vec<_> = productions
            .into_iter()
            .enumerate()
            .map(|(id, (lhs, rhs))| ResidualProduction {
                id: id as u32,
                original: id as u32,
                lhs,
                rhs,
                provenance: provenance.clone(),
            })
            .collect();
        let mut productions_by_lhs = vec![Vec::new(); nonterminal_count];
        for production in &productions {
            productions_by_lhs[production.lhs as usize].push(production.id);
        }
        let mut nullable = vec![false; nonterminal_count];
        loop {
            let mut changed = false;
            for production in &productions {
                if !nullable[production.lhs as usize]
                    && production.rhs.iter().all(|symbol| match symbol {
                        Symbol::Nonterminal(id) => nullable[*id as usize],
                        Symbol::Terminal(_) => false,
                    })
                {
                    nullable[production.lhs as usize] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let productive_suffixes = productions
            .iter()
            .map(|production| {
                let mut suffixes = vec![false; production.rhs.len() + 1];
                suffixes[production.rhs.len()] = true;
                for index in (0..production.rhs.len()).rev() {
                    suffixes[index] = suffixes[index + 1]
                        && matches!(production.rhs[index], Symbol::Nonterminal(id) if nullable[id as usize]);
                }
                suffixes
            })
            .collect();
        let residual = ResidualGrammar {
            start: 0,
            nonterminals: (0..nonterminal_count)
                .map(|id| ResidualNonterminal {
                    id: id as u32,
                    original: id as u32,
                    name: format!("n{id}"),
                    provenance: provenance.clone(),
                })
                .collect(),
            terminals: literals
                .iter()
                .map(|bytes| CompiledTerminal::Literal(Arc::from(*bytes)))
                .collect(),
            terminal_provenance: literals.iter().map(|_| provenance.clone()).collect(),
            productions,
            productions_by_lhs,
            nullable,
            productive_suffixes,
            collapsed_nonterminals: 0,
            collapsed_regions: 0,
            terminal_dfa_states: 0,
            terminal_dfa_bytes: 0,
        };
        Arc::new(CompiledEarley::build(Arc::new(residual), &CompileLimits::default()).unwrap())
    }

    fn recognize(compiled: &Arc<CompiledEarley>, bytes: &[u8], leo: bool) -> EarleyRecognizer {
        let mut recognizer = compiled.recognizer(RuntimeLimits::default(), leo).unwrap();
        for byte in bytes {
            assert!(recognizer.advance_byte(*byte).unwrap());
        }
        assert!(recognizer.is_accepting());
        recognizer
    }

    #[test]
    fn recursive_array_recognizes_nested_values() {
        let grammar = compile(br##"{
          "$defs":{"node":{"anyOf":[{"type":"null"},{"type":"array","prefixItems":[{"$ref":"#/$defs/node"}],"items":false}]}},
          "$ref":"#/$defs/node"
        }"##);
        for value in ["null", "[null]", "[[null]]", "[[[null]]]"] {
            let mut recognizer = grammar.recognizer(RuntimeLimits::default(), false).unwrap();
            for byte in value.bytes() {
                assert!(recognizer.advance_byte(byte).unwrap(), "{value}");
            }
            assert!(recognizer.is_accepting(), "{value}");
        }
    }

    #[test]
    fn rejected_trial_can_be_restored() {
        let grammar = compile(br##"{
          "$defs":{"node":{"anyOf":[{"type":"null"},{"type":"array","prefixItems":[{"$ref":"#/$defs/node"}],"items":false}]}},
          "$ref":"#/$defs/node"
        }"##);
        let mut recognizer = grammar.recognizer(RuntimeLimits::default(), true).unwrap();
        let checkpoint = recognizer.checkpoint();
        assert!(!recognizer.advance_byte(b'x').unwrap());
        recognizer.restore(&checkpoint).unwrap();
        for byte in b"null" {
            assert!(recognizer.advance_byte(*byte).unwrap());
        }
        assert!(recognizer.is_accepting());
    }

    #[test]
    fn nullable_chain_and_cycle_terminate() {
        let chain = grammar(
            4,
            vec![
                (0, vec![Symbol::Nonterminal(1), Symbol::Terminal(0)]),
                (1, vec![Symbol::Nonterminal(2)]),
                (2, vec![Symbol::Nonterminal(3)]),
                (3, vec![]),
            ],
            &[b"x"],
        );
        recognize(&chain, b"x", false);

        let cycle = grammar(
            3,
            vec![
                (0, vec![Symbol::Nonterminal(1), Symbol::Terminal(0)]),
                (1, vec![Symbol::Nonterminal(2)]),
                (1, vec![]),
                (2, vec![Symbol::Nonterminal(1)]),
            ],
            &[b"x"],
        );
        let recognized = recognize(&cycle, b"x", false);
        assert!(recognized.stats().items_inserted < 32);
    }

    #[test]
    fn left_recursion_is_recognized_without_leo() {
        let list = grammar(
            1,
            vec![
                (0, vec![Symbol::Nonterminal(0), Symbol::Terminal(0)]),
                (0, vec![]),
            ],
            &[b"a"],
        );
        recognize(&list, &vec![b'a'; 256], false);
    }

    #[test]
    fn leo_preserves_recognition_and_linearizes_right_recursion() {
        let list = grammar(
            1,
            vec![
                (0, vec![Symbol::Terminal(0), Symbol::Nonterminal(0)]),
                (0, vec![]),
            ],
            &[b"a"],
        );
        let input = vec![b'a'; 2_000];
        let ordinary = recognize(&list, &input, false);
        let optimized = recognize(&list, &input, true);
        assert!(optimized.stats().leo_hits > 1_900);
        assert!(optimized.stats().leo_candidates < 4_100);
        assert!(optimized.stats().items_inserted < 10_100);
        assert!(ordinary.stats().items_inserted > optimized.stats().items_inserted * 10);
    }

    #[test]
    fn leo_work_respects_runtime_limit() {
        let list = grammar(
            1,
            vec![
                (0, vec![Symbol::Terminal(0), Symbol::Nonterminal(0)]),
                (0, vec![]),
            ],
            &[b"a"],
        );
        let limits = RuntimeLimits {
            max_leo_items: 0,
            ..RuntimeLimits::default()
        };
        assert!(matches!(
            list.recognizer(limits, true),
            Err(RuntimeError::ResourceLimitExceeded {
                resource: RuntimeResource::LeoItems,
                observed: 1,
                limit: 0,
            })
        ));
    }

    #[test]
    fn leo_matches_baseline_on_bounded_ambiguous_inputs() {
        let grammar = grammar(
            1,
            vec![
                (0, vec![Symbol::Terminal(0), Symbol::Nonterminal(0)]),
                (0, vec![Symbol::Terminal(0)]),
                (0, vec![]),
            ],
            &[b"a"],
        );
        for length in 0..=8 {
            for bits in 0..(1_usize << length) {
                let input: Vec<_> = (0..length)
                    .map(|index| if bits & (1 << index) == 0 { b'a' } else { b'b' })
                    .collect();
                let outcomes = [false, true].map(|leo| {
                    let mut recognizer = grammar.recognizer(RuntimeLimits::default(), leo).unwrap();
                    let live = input
                        .iter()
                        .all(|byte| recognizer.advance_byte(*byte).unwrap());
                    (live, recognizer.is_accepting(), recognizer.is_live())
                });
                assert_eq!(outcomes[0], outcomes[1], "input={input:?}");
            }
        }
    }

    #[test]
    fn ambiguous_terminal_lengths_preserve_both_segmentations() {
        let choices = grammar(
            1,
            vec![
                (0, vec![Symbol::Terminal(0)]),
                (0, vec![Symbol::Terminal(1)]),
            ],
            &[b"a", b"aa"],
        );
        recognize(&choices, b"a", false);
        recognize(&choices, b"aa", false);
    }

    #[test]
    fn initial_runtime_limits_return_typed_errors() {
        let compiled = grammar(1, vec![(0, vec![Symbol::Terminal(0)])], &[b"x"]);
        let cases = [
            (
                RuntimeLimits {
                    max_chart_columns: 0,
                    ..RuntimeLimits::default()
                },
                RuntimeResource::ChartColumns,
            ),
            (
                RuntimeLimits {
                    max_items_per_column: 0,
                    ..RuntimeLimits::default()
                },
                RuntimeResource::ItemsPerColumn,
            ),
            (
                RuntimeLimits {
                    max_total_items: 0,
                    ..RuntimeLimits::default()
                },
                RuntimeResource::TotalItems,
            ),
            (
                RuntimeLimits {
                    max_active_scans: 0,
                    ..RuntimeLimits::default()
                },
                RuntimeResource::ActiveScans,
            ),
        ];

        for (limits, resource) in cases {
            assert!(matches!(
                compiled.recognizer(limits, false),
                Err(RuntimeError::ResourceLimitExceeded {
                    resource: found,
                    ..
                }) if found == resource
            ));
        }
    }

    #[test]
    fn preparation_item_limit_returns_a_typed_error() {
        let compiled = grammar(1, vec![(0, vec![Symbol::Terminal(0)])], &[b"x"]);
        let limits = CompileLimits {
            max_lr_items: 0,
            ..CompileLimits::default()
        };
        assert!(matches!(
            CompiledEarley::build(Arc::clone(&compiled.grammar), &limits),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::EarleyPreparation,
                ..
            })
        ));
    }
}
