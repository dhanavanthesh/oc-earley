use std::collections::{BTreeMap, VecDeque};
use std::mem;
use std::time::{Duration, Instant};

use regex_automata::dfa::{dense, Automaton};
use regex_automata::nfa::thompson::{self, State, Transition, WhichCaptures};
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::{Anchored, MatchKind};
use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::error::{CompileError, CompileStage};
use crate::json_schema::{BOOLEAN, INTEGER, NULL, NUMBER, STRING};
use crate::schema::{
    AdditionalProperties, ArrayConstraints, CompileLimits, NormalizedSchema, ObjectConstraints,
    Provenance, SchemaArena, SchemaId,
};

pub type NonterminalId = u32;
pub type TerminalId = u32;
pub type ProductionId = u32;
pub type SccId = u32;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Nonterminal {
    pub id: NonterminalId,
    pub name: String,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum RegularTerminalKind {
    Literal(Vec<u8>),
    Pattern {
        forward: String,
        reverse: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RegularTerminal {
    pub id: TerminalId,
    pub kind: RegularTerminalKind,
    pub provenance: Provenance,
}

impl RegularTerminal {
    pub fn forward_pattern(&self) -> String {
        match &self.kind {
            RegularTerminalKind::Literal(bytes) => regex::escape(&String::from_utf8_lossy(bytes)),
            RegularTerminalKind::Pattern { forward, .. } => forward.clone(),
        }
    }

    pub fn reverse_pattern(&self) -> Option<String> {
        match &self.kind {
            RegularTerminalKind::Literal(bytes) => {
                if !bytes.is_ascii() {
                    return None;
                }
                let reversed: Vec<_> = bytes.iter().rev().copied().collect();
                Some(regex::escape(&String::from_utf8_lossy(&reversed)))
            }
            RegularTerminalKind::Pattern { reverse, .. } => reverse.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Symbol {
    Nonterminal(NonterminalId),
    Terminal(TerminalId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Production {
    pub id: ProductionId,
    pub lhs: NonterminalId,
    pub rhs: Vec<Symbol>,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Grammar {
    pub start: NonterminalId,
    pub nonterminals: Vec<Nonterminal>,
    pub terminals: Vec<RegularTerminal>,
    pub productions: Vec<Production>,
}

impl Grammar {
    pub fn productions_for(&self, nonterminal: NonterminalId) -> impl Iterator<Item = &Production> {
        self.productions
            .iter()
            .filter(move |production| production.lhs == nonterminal)
    }
}

pub fn lower(arena: &SchemaArena, limits: &CompileLimits) -> Result<Grammar, CompileError> {
    let mut builder = GrammarBuilder::new(limits);
    let mut schema_nonterminals = Vec::new();
    schema_nonterminals
        .try_reserve(arena.nodes.len())
        .map_err(|_| {
            resource_error(
                CompileStage::Lowering,
                arena.nodes.len(),
                limits.max_symbols,
            )
        })?;

    for node in &arena.nodes {
        schema_nonterminals.push(
            builder.add_nonterminal(format!("schema_{}", node.id.0), node.provenance.clone())?,
        );
    }

    for node in &arena.nodes {
        let lhs = schema_nonterminals[node.id.0 as usize];
        lower_node(
            arena,
            &schema_nonterminals,
            &mut builder,
            lhs,
            node.id,
            &node.kind,
            &node.provenance,
        )?;
    }

    Ok(Grammar {
        start: schema_nonterminals[arena.root.0 as usize],
        nonterminals: builder.nonterminals,
        terminals: builder.terminals,
        productions: builder.productions,
    })
}

#[allow(clippy::too_many_arguments)]
fn lower_node(
    arena: &SchemaArena,
    schema_nonterminals: &[NonterminalId],
    builder: &mut GrammarBuilder<'_>,
    lhs: NonterminalId,
    schema_id: SchemaId,
    kind: &NormalizedSchema,
    provenance: &Provenance,
) -> Result<(), CompileError> {
    match kind {
        NormalizedSchema::Never => {}
        NormalizedSchema::Any => {
            return Err(CompileError::UnsupportedCombination {
                location: provenance.location(),
                reason: "unconstrained JSON objects require canonical key-order memory".to_owned(),
            });
        }
        NormalizedSchema::Null => {
            let terminal = builder.add_pattern(NULL, Some(NULL), provenance.clone())?;
            builder.add_production(lhs, vec![Symbol::Terminal(terminal)], provenance.clone())?;
        }
        NormalizedSchema::Boolean => {
            let terminal = builder.add_pattern(BOOLEAN, None, provenance.clone())?;
            builder.add_production(lhs, vec![Symbol::Terminal(terminal)], provenance.clone())?;
        }
        NormalizedSchema::String(constraints) => {
            if constraints.min_length != 0 || constraints.max_length.is_some() {
                return Err(CompileError::UnsupportedCombination {
                    location: provenance.location(),
                    reason: "Unicode length constraints require a certified scalar-count automaton"
                        .to_owned(),
                });
            }
            let terminal = builder.add_pattern(STRING, None, provenance.clone())?;
            builder.add_production(lhs, vec![Symbol::Terminal(terminal)], provenance.clone())?;
        }
        NormalizedSchema::Number => {
            let terminal = builder.add_pattern(NUMBER, None, provenance.clone())?;
            builder.add_production(lhs, vec![Symbol::Terminal(terminal)], provenance.clone())?;
        }
        NormalizedSchema::Integer => {
            let terminal = builder.add_pattern(INTEGER, None, provenance.clone())?;
            builder.add_production(lhs, vec![Symbol::Terminal(terminal)], provenance.clone())?;
        }
        NormalizedSchema::Const(value) => {
            let terminal = builder.add_literal(value.json.as_bytes(), provenance.clone())?;
            builder.add_production(lhs, vec![Symbol::Terminal(terminal)], provenance.clone())?;
        }
        NormalizedSchema::Enum(values) => {
            for value in values {
                let terminal = builder.add_literal(value.json.as_bytes(), provenance.clone())?;
                builder.add_production(
                    lhs,
                    vec![Symbol::Terminal(terminal)],
                    provenance.clone(),
                )?;
            }
        }
        NormalizedSchema::Union(branches) => {
            for branch in branches {
                builder.add_production(
                    lhs,
                    vec![Symbol::Nonterminal(schema_nonterminals[branch.0 as usize])],
                    provenance.clone(),
                )?;
            }
        }
        NormalizedSchema::Ref(target) => {
            builder.add_production(
                lhs,
                vec![Symbol::Nonterminal(schema_nonterminals[target.0 as usize])],
                provenance.clone(),
            )?;
        }
        NormalizedSchema::Array(array) => lower_array(
            arena,
            schema_nonterminals,
            builder,
            lhs,
            schema_id,
            array,
            provenance,
        )?,
        NormalizedSchema::Object(object) => lower_object(
            schema_nonterminals,
            builder,
            lhs,
            schema_id,
            object,
            provenance,
        )?,
    }
    Ok(())
}

fn lower_object(
    schema_nonterminals: &[NonterminalId],
    builder: &mut GrammarBuilder<'_>,
    lhs: NonterminalId,
    schema_id: SchemaId,
    object: &ObjectConstraints,
    provenance: &Provenance,
) -> Result<(), CompileError> {
    if object.additional_properties != AdditionalProperties::Forbidden {
        return Err(CompileError::UnsupportedCombination {
            location: provenance.location(),
            reason: "open objects require canonical key-order memory".to_owned(),
        });
    }
    if object
        .required
        .iter()
        .any(|name| !object.properties.contains_key(name))
    {
        return Ok(());
    }

    let entries: Vec<_> = object.properties.iter().collect();
    let mut members = Vec::new();
    for index in 0..=entries.len() {
        let absent = builder.add_nonterminal(
            format!("object_{}_members_{}_empty", schema_id.0, index),
            keyword_provenance(provenance, "properties"),
        )?;
        let present = builder.add_nonterminal(
            format!("object_{}_members_{}_present", schema_id.0, index),
            keyword_provenance(provenance, "properties"),
        )?;
        members.push([absent, present]);
    }

    for &member in &members[entries.len()] {
        builder.add_production(
            member,
            Vec::new(),
            keyword_provenance(provenance, "properties"),
        )?;
    }

    for (index, (name, value)) in entries.iter().enumerate() {
        let required = object.required.contains(*name);
        for emitted in 0..=1 {
            let current = members[index][emitted];
            if !required {
                builder.add_production(
                    current,
                    vec![Symbol::Nonterminal(members[index + 1][emitted])],
                    keyword_provenance(provenance, "properties"),
                )?;
            }
            let mut rhs = Vec::new();
            if emitted == 1 {
                let comma = builder.add_literal(b",", provenance.clone())?;
                rhs.push(Symbol::Terminal(comma));
            }
            let key = serde_json::to_string(name).map_err(|error| CompileError::InvalidJson {
                message: error.to_string(),
            })?;
            let key = builder.add_literal(key.as_bytes(), provenance.clone())?;
            let colon = builder.add_literal(b":", provenance.clone())?;
            rhs.extend([
                Symbol::Terminal(key),
                Symbol::Terminal(colon),
                Symbol::Nonterminal(schema_nonterminals[value.0 as usize]),
                Symbol::Nonterminal(members[index + 1][1]),
            ]);
            builder.add_production(current, rhs, keyword_provenance(provenance, "properties"))?;
        }
    }

    let open = builder.add_literal(b"{", provenance.clone())?;
    let close = builder.add_literal(b"}", provenance.clone())?;
    builder.add_production(
        lhs,
        vec![
            Symbol::Terminal(open),
            Symbol::Nonterminal(members[0][0]),
            Symbol::Terminal(close),
        ],
        provenance.clone(),
    )?;
    Ok(())
}

fn lower_array(
    arena: &SchemaArena,
    schema_nonterminals: &[NonterminalId],
    builder: &mut GrammarBuilder<'_>,
    lhs: NonterminalId,
    schema_id: SchemaId,
    array: &ArrayConstraints,
    provenance: &Provenance,
) -> Result<(), CompileError> {
    let item_is_never = array
        .items
        .and_then(|id| arena.node(id))
        .is_some_and(|node| node.kind == NormalizedSchema::Never);
    let effective_max = if item_is_never {
        Some(
            array
                .max_items
                .unwrap_or(array.prefix_items.len())
                .min(array.prefix_items.len()),
        )
    } else {
        array.max_items
    };
    if effective_max.is_some_and(|maximum| maximum < array.min_items) {
        return Ok(());
    }
    if effective_max.is_none() && array.items.is_none() {
        return Err(CompileError::UnsupportedCombination {
            location: provenance.location(),
            reason: "unbounded unconstrained array items require a structural backend".to_owned(),
        });
    }
    if effective_max.is_some_and(|maximum| maximum > array.prefix_items.len())
        && array.items.is_none()
    {
        return Err(CompileError::UnsupportedCombination {
            location: provenance.location(),
            reason: "unconstrained remainder items require a structural backend".to_owned(),
        });
    }

    let position_count = effective_max.unwrap_or(array.prefix_items.len());
    let state_count = position_count
        .checked_add(1)
        .ok_or(CompileError::ResourceLimitExceeded {
            stage: CompileStage::Lowering,
            observed: usize::MAX,
            limit: builder.limits.max_symbols,
        })?;
    let total_symbols = builder.nonterminals.len().checked_add(state_count).ok_or(
        CompileError::ResourceLimitExceeded {
            stage: CompileStage::Lowering,
            observed: usize::MAX,
            limit: builder.limits.max_symbols,
        },
    )?;
    if total_symbols > builder.limits.max_symbols {
        return Err(CompileError::ResourceLimitExceeded {
            stage: CompileStage::Lowering,
            observed: total_symbols,
            limit: builder.limits.max_symbols,
        });
    }
    let mut states = Vec::new();
    states
        .try_reserve_exact(state_count)
        .map_err(|_| CompileError::ResourceLimitExceeded {
            stage: CompileStage::Lowering,
            observed: state_count,
            limit: builder.limits.max_symbols,
        })?;
    for index in 0..state_count {
        states.push(builder.add_nonterminal(
            format!("array_{}_position_{}", schema_id.0, index),
            keyword_provenance(provenance, "items"),
        )?);
    }

    for index in 0..state_count {
        if index >= array.min_items {
            builder.add_production(
                states[index],
                Vec::new(),
                keyword_provenance(provenance, "minItems"),
            )?;
        }
        if effective_max.is_some_and(|maximum| index >= maximum) {
            continue;
        }
        let item = if index < array.prefix_items.len() {
            Some(array.prefix_items[index])
        } else {
            array.items
        };
        let Some(item) = item else {
            continue;
        };
        if arena
            .node(item)
            .is_some_and(|node| node.kind == NormalizedSchema::Never)
        {
            continue;
        }
        let next = if index + 1 < states.len() {
            states[index + 1]
        } else {
            states[index]
        };
        let mut rhs = Vec::new();
        if index != 0 {
            let comma = builder.add_literal(b",", provenance.clone())?;
            rhs.push(Symbol::Terminal(comma));
        }
        rhs.extend([
            Symbol::Nonterminal(schema_nonterminals[item.0 as usize]),
            Symbol::Nonterminal(next),
        ]);
        builder.add_production(states[index], rhs, keyword_provenance(provenance, "items"))?;
    }

    let open = builder.add_literal(b"[", provenance.clone())?;
    let close = builder.add_literal(b"]", provenance.clone())?;
    builder.add_production(
        lhs,
        vec![
            Symbol::Terminal(open),
            Symbol::Nonterminal(states[0]),
            Symbol::Terminal(close),
        ],
        provenance.clone(),
    )?;
    Ok(())
}

struct GrammarBuilder<'a> {
    limits: &'a CompileLimits,
    nonterminals: Vec<Nonterminal>,
    terminals: Vec<RegularTerminal>,
    productions: Vec<Production>,
    rhs_symbols: usize,
    literal_cache: BTreeMap<(Vec<u8>, String, Option<String>), TerminalId>,
}

impl<'a> GrammarBuilder<'a> {
    fn new(limits: &'a CompileLimits) -> Self {
        Self {
            limits,
            nonterminals: Vec::new(),
            terminals: Vec::new(),
            productions: Vec::new(),
            rhs_symbols: 0,
            literal_cache: BTreeMap::new(),
        }
    }

    fn add_nonterminal(
        &mut self,
        name: String,
        provenance: Provenance,
    ) -> Result<NonterminalId, CompileError> {
        enforce_limit(
            CompileStage::Lowering,
            self.nonterminals.len() + self.terminals.len() + 1,
            self.limits.max_symbols,
        )?;
        let id = to_u32(self.nonterminals.len(), CompileStage::Lowering)?;
        self.nonterminals.push(Nonterminal {
            id,
            name,
            provenance,
        });
        Ok(id)
    }

    fn add_literal(
        &mut self,
        bytes: &[u8],
        provenance: Provenance,
    ) -> Result<TerminalId, CompileError> {
        let key = (
            bytes.to_vec(),
            provenance.pointer.0.clone(),
            provenance.keyword.clone(),
        );
        if let Some(id) = self.literal_cache.get(&key) {
            return Ok(*id);
        }
        let id = self.add_terminal(RegularTerminalKind::Literal(bytes.to_vec()), provenance)?;
        self.literal_cache.insert(key, id);
        Ok(id)
    }

    fn add_pattern(
        &mut self,
        forward: &str,
        reverse: Option<&str>,
        provenance: Provenance,
    ) -> Result<TerminalId, CompileError> {
        self.add_terminal(
            RegularTerminalKind::Pattern {
                forward: forward.to_owned(),
                reverse: reverse.map(str::to_owned),
            },
            provenance,
        )
    }

    fn add_terminal(
        &mut self,
        kind: RegularTerminalKind,
        provenance: Provenance,
    ) -> Result<TerminalId, CompileError> {
        enforce_limit(
            CompileStage::Lowering,
            self.nonterminals.len() + self.terminals.len() + 1,
            self.limits.max_symbols,
        )?;
        let id = to_u32(self.terminals.len(), CompileStage::Lowering)?;
        self.terminals.push(RegularTerminal {
            id,
            kind,
            provenance,
        });
        Ok(id)
    }

    fn add_production(
        &mut self,
        lhs: NonterminalId,
        rhs: Vec<Symbol>,
        provenance: Provenance,
    ) -> Result<ProductionId, CompileError> {
        enforce_limit(
            CompileStage::Lowering,
            self.productions.len() + 1,
            self.limits.max_productions,
        )?;
        let next_rhs = self.rhs_symbols.checked_add(rhs.len()).ok_or_else(|| {
            resource_error(
                CompileStage::Lowering,
                usize::MAX,
                self.limits.max_rhs_symbols,
            )
        })?;
        enforce_limit(
            CompileStage::Lowering,
            next_rhs,
            self.limits.max_rhs_symbols,
        )?;
        let id = to_u32(self.productions.len(), CompileStage::Lowering)?;
        self.rhs_symbols = next_rhs;
        self.productions.push(Production {
            id,
            lhs,
            rhs,
            provenance,
        });
        Ok(id)
    }
}

fn keyword_provenance(provenance: &Provenance, keyword: &str) -> Provenance {
    Provenance {
        resource: provenance.resource,
        pointer: provenance.pointer.clone(),
        keyword: Some(keyword.to_owned()),
    }
}

fn enforce_limit(stage: CompileStage, observed: usize, limit: usize) -> Result<(), CompileError> {
    if observed > limit {
        Err(resource_error(stage, observed, limit))
    } else {
        Ok(())
    }
}

fn resource_error(stage: CompileStage, observed: usize, limit: usize) -> CompileError {
    CompileError::ResourceLimitExceeded {
        stage,
        observed,
        limit,
    }
}

fn to_u32(value: usize, stage: CompileStage) -> Result<u32, CompileError> {
    let limit = usize::try_from(u32::MAX).unwrap_or(usize::MAX);
    u32::try_from(value).map_err(|_| resource_error(stage, value, limit))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ReductionStats {
    pub removed_nonterminals: usize,
    pub removed_terminals: usize,
    pub removed_productions: usize,
}

pub fn reduce(grammar: &mut Grammar) -> Result<ReductionStats, CompileError> {
    let original_nonterminals = grammar.nonterminals.len();
    let original_terminals = grammar.terminals.len();
    let original_productions = grammar.productions.len();
    let productive = productive_nonterminals(grammar)?;

    if !productive[grammar.start as usize] {
        let start = grammar.nonterminals[grammar.start as usize].clone();
        grammar.start = 0;
        grammar.nonterminals = vec![Nonterminal { id: 0, ..start }];
        grammar.terminals.clear();
        grammar.productions.clear();
        return Ok(ReductionStats {
            removed_nonterminals: original_nonterminals.saturating_sub(1),
            removed_terminals: original_terminals,
            removed_productions: original_productions,
        });
    }

    let reachable = reachable_nonterminals(grammar);
    let retained: Vec<bool> = productive
        .iter()
        .zip(reachable.iter())
        .map(|(productive, reachable)| *productive && *reachable)
        .collect();

    let mut nonterminal_map = vec![None; grammar.nonterminals.len()];
    let mut nonterminals = Vec::new();
    for nonterminal in &grammar.nonterminals {
        if retained[nonterminal.id as usize] {
            let id = to_u32(nonterminals.len(), CompileStage::GrammarReduction)?;
            nonterminal_map[nonterminal.id as usize] = Some(id);
            nonterminals.push(Nonterminal {
                id,
                name: nonterminal.name.clone(),
                provenance: nonterminal.provenance.clone(),
            });
        }
    }

    let retained_productions: Vec<_> = grammar
        .productions
        .iter()
        .filter(|production| {
            retained[production.lhs as usize]
                && production.rhs.iter().all(|symbol| match symbol {
                    Symbol::Nonterminal(id) => retained[*id as usize],
                    Symbol::Terminal(_) => true,
                })
        })
        .collect();
    let mut terminal_used = vec![false; grammar.terminals.len()];
    for production in &retained_productions {
        for symbol in &production.rhs {
            if let Symbol::Terminal(id) = symbol {
                terminal_used[*id as usize] = true;
            }
        }
    }
    let mut terminal_map = vec![None; grammar.terminals.len()];
    let mut terminals = Vec::new();
    for terminal in &grammar.terminals {
        if terminal_used[terminal.id as usize] {
            let id = to_u32(terminals.len(), CompileStage::GrammarReduction)?;
            terminal_map[terminal.id as usize] = Some(id);
            terminals.push(RegularTerminal {
                id,
                kind: terminal.kind.clone(),
                provenance: terminal.provenance.clone(),
            });
        }
    }

    let mut productions = Vec::new();
    productions
        .try_reserve(retained_productions.len())
        .map_err(|_| {
            resource_error(
                CompileStage::GrammarReduction,
                retained_productions.len(),
                retained_productions.len().saturating_sub(1),
            )
        })?;
    for production in retained_productions {
        let lhs =
            nonterminal_map[production.lhs as usize].ok_or(CompileError::InternalInvariant {
                message: "retained production has removed lhs",
            })?;
        let mut rhs = Vec::new();
        rhs.try_reserve(production.rhs.len()).map_err(|_| {
            resource_error(
                CompileStage::GrammarReduction,
                production.rhs.len(),
                production.rhs.len().saturating_sub(1),
            )
        })?;
        for symbol in &production.rhs {
            rhs.push(match symbol {
                Symbol::Nonterminal(id) => Symbol::Nonterminal(
                    nonterminal_map[*id as usize].ok_or(CompileError::InternalInvariant {
                        message: "retained production references removed nonterminal",
                    })?,
                ),
                Symbol::Terminal(id) => Symbol::Terminal(terminal_map[*id as usize].ok_or(
                    CompileError::InternalInvariant {
                        message: "retained production references removed terminal",
                    },
                )?),
            });
        }
        productions.push(Production {
            id: to_u32(productions.len(), CompileStage::GrammarReduction)?,
            lhs,
            rhs,
            provenance: production.provenance.clone(),
        });
    }

    grammar.start =
        nonterminal_map[grammar.start as usize].ok_or(CompileError::InternalInvariant {
            message: "productive reachable start was removed",
        })?;
    grammar.nonterminals = nonterminals;
    grammar.terminals = terminals;
    grammar.productions = productions;

    Ok(ReductionStats {
        removed_nonterminals: original_nonterminals - grammar.nonterminals.len(),
        removed_terminals: original_terminals - grammar.terminals.len(),
        removed_productions: original_productions - grammar.productions.len(),
    })
}

fn productive_nonterminals(grammar: &Grammar) -> Result<Vec<bool>, CompileError> {
    let mut remaining = Vec::new();
    remaining
        .try_reserve(grammar.productions.len())
        .map_err(|_| {
            resource_error(
                CompileStage::GrammarReduction,
                grammar.productions.len(),
                grammar.productions.len().saturating_sub(1),
            )
        })?;
    let mut waiting = vec![Vec::new(); grammar.nonterminals.len()];
    for (index, production) in grammar.productions.iter().enumerate() {
        let mut count = 0usize;
        for symbol in &production.rhs {
            if let Symbol::Nonterminal(id) = symbol {
                count = count.checked_add(1).ok_or_else(|| {
                    resource_error(CompileStage::GrammarReduction, usize::MAX, usize::MAX - 1)
                })?;
                waiting[*id as usize].push(index);
            }
        }
        remaining.push(count);
    }

    let mut productive = vec![false; grammar.nonterminals.len()];
    let mut queue = std::collections::VecDeque::new();
    for (index, production) in grammar.productions.iter().enumerate() {
        if remaining[index] == 0 && !productive[production.lhs as usize] {
            productive[production.lhs as usize] = true;
            queue.push_back(production.lhs);
        }
    }
    while let Some(nonterminal) = queue.pop_front() {
        for &production_index in &waiting[nonterminal as usize] {
            remaining[production_index] -= 1;
            if remaining[production_index] == 0 {
                let lhs = grammar.productions[production_index].lhs;
                if !productive[lhs as usize] {
                    productive[lhs as usize] = true;
                    queue.push_back(lhs);
                }
            }
        }
    }
    Ok(productive)
}

fn reachable_nonterminals(grammar: &Grammar) -> Vec<bool> {
    let mut productions_by_lhs = vec![Vec::new(); grammar.nonterminals.len()];
    for production in &grammar.productions {
        productions_by_lhs[production.lhs as usize].push(production);
    }
    let mut reachable = vec![false; grammar.nonterminals.len()];
    let mut stack = vec![grammar.start];
    reachable[grammar.start as usize] = true;
    while let Some(nonterminal) = stack.pop() {
        for production in &productions_by_lhs[nonterminal as usize] {
            for symbol in &production.rhs {
                if let Symbol::Nonterminal(target) = symbol {
                    if !reachable[*target as usize] {
                        reachable[*target as usize] = true;
                        stack.push(*target);
                    }
                }
            }
        }
    }
    reachable
}

pub fn dependency_graph(grammar: &Grammar) -> Vec<Vec<NonterminalId>> {
    let mut graph = vec![Vec::new(); grammar.nonterminals.len()];
    for production in &grammar.productions {
        for symbol in &production.rhs {
            if let Symbol::Nonterminal(target) = symbol {
                graph[production.lhs as usize].push(*target);
            }
        }
    }
    for edges in &mut graph {
        edges.sort_unstable();
        edges.dedup();
    }
    graph
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Scc {
    pub id: SccId,
    pub members: Vec<NonterminalId>,
    pub recursive: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SccAnalysis {
    pub components: Vec<Scc>,
    pub component_of: Vec<SccId>,
}

pub fn analyze_sccs(grammar: &Grammar) -> Result<SccAnalysis, CompileError> {
    let graph = dependency_graph(grammar);
    let node_count = graph.len();
    let mut next_index = 0usize;
    let mut indices = vec![None; node_count];
    let mut lowlink = vec![0usize; node_count];
    let mut on_stack = vec![false; node_count];
    let mut tarjan_stack = Vec::new();
    let mut components = Vec::new();

    for root in 0..node_count {
        if indices[root].is_some() {
            continue;
        }
        discover(
            root,
            &mut next_index,
            &mut indices,
            &mut lowlink,
            &mut on_stack,
            &mut tarjan_stack,
        )?;
        let mut frames = vec![(root, 0usize)];
        while let Some((node, edge_index)) = frames.last_mut() {
            if *edge_index < graph[*node].len() {
                let target = graph[*node][*edge_index] as usize;
                *edge_index += 1;
                if indices[target].is_none() {
                    discover(
                        target,
                        &mut next_index,
                        &mut indices,
                        &mut lowlink,
                        &mut on_stack,
                        &mut tarjan_stack,
                    )?;
                    frames.push((target, 0));
                } else if on_stack[target] {
                    lowlink[*node] = lowlink[*node].min(indices[target].unwrap());
                }
                continue;
            }

            let completed = *node;
            frames.pop();
            if let Some((parent, _)) = frames.last() {
                lowlink[*parent] = lowlink[*parent].min(lowlink[completed]);
            }
            if lowlink[completed] == indices[completed].unwrap() {
                let mut members = Vec::new();
                loop {
                    let member = tarjan_stack.pop().ok_or(CompileError::InternalInvariant {
                        message: "Tarjan stack became empty while closing a component",
                    })?;
                    on_stack[member] = false;
                    members.push(member as NonterminalId);
                    if member == completed {
                        break;
                    }
                }
                members.sort_unstable();
                components.push(members);
            }
        }
    }

    components.sort_by_key(|members| members[0]);
    let mut component_of = vec![0; node_count];
    let mut reports = Vec::new();
    for (index, members) in components.into_iter().enumerate() {
        let id = to_u32(index, CompileStage::SccAnalysis)?;
        for member in &members {
            component_of[*member as usize] = id;
        }
        let recursive = members.len() > 1
            || graph[members[0] as usize]
                .binary_search(&members[0])
                .is_ok();
        reports.push(Scc {
            id,
            members,
            recursive,
        });
    }
    Ok(SccAnalysis {
        components: reports,
        component_of,
    })
}

fn discover(
    node: usize,
    next_index: &mut usize,
    indices: &mut [Option<usize>],
    lowlink: &mut [usize],
    on_stack: &mut [bool],
    tarjan_stack: &mut Vec<usize>,
) -> Result<(), CompileError> {
    indices[node] = Some(*next_index);
    lowlink[node] = *next_index;
    *next_index = next_index
        .checked_add(1)
        .ok_or_else(|| resource_error(CompileStage::SccAnalysis, usize::MAX, usize::MAX - 1))?;
    tarjan_stack.push(node);
    on_stack[node] = true;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RegularCertificateKind {
    Acyclic,
    RightLinear,
    LeftLinear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CertificateFailureReason {
    MultipleRecursiveOccurrences,
    MixedLinearOrientation,
    RecursiveSymbolInInterior,
    UncertifiedDependency,
    UnsupportedRegularOperation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SccCertificate {
    pub scc: SccId,
    pub kind: Option<RegularCertificateKind>,
    pub failure_reason: Option<CertificateFailureReason>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegularExpression {
    Empty,
    Epsilon,
    Atom {
        forward: String,
        reverse: Option<String>,
    },
    Concat(Vec<RegularExpression>),
    Union(Vec<RegularExpression>),
    Star(Box<RegularExpression>),
}

impl RegularExpression {
    pub fn pattern(&self) -> Result<Option<String>, CompileError> {
        self.pattern_with_limit(usize::MAX)
    }

    fn pattern_with_limit(&self, limit: usize) -> Result<Option<String>, CompileError> {
        enum Frame<'a> {
            Expression(&'a RegularExpression),
            Text(&'a str),
        }

        fn reserve_frames(
            stack: &mut Vec<Frame<'_>>,
            additional: usize,
            byte_limit: usize,
        ) -> Result<(), CompileError> {
            let frames = stack.len().checked_add(additional).ok_or_else(|| {
                resource_error(CompileStage::NfaConstruction, usize::MAX, byte_limit)
            })?;
            let bytes = frames
                .checked_mul(mem::size_of::<Frame<'_>>())
                .ok_or_else(|| {
                    resource_error(CompileStage::NfaConstruction, usize::MAX, byte_limit)
                })?;
            enforce_limit(CompileStage::NfaConstruction, bytes, byte_limit)?;
            stack
                .try_reserve_exact(additional)
                .map_err(|_| resource_error(CompileStage::NfaConstruction, bytes, byte_limit))
        }

        if self == &Self::Empty {
            return Ok(None);
        }
        let mut pattern = String::new();
        let mut stack = vec![Frame::Expression(self)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Text(text) => append_pattern(&mut pattern, text, limit)?,
                Frame::Expression(Self::Empty) => {
                    return Err(CompileError::InternalInvariant {
                        message: "empty expression remained inside a regular expression",
                    });
                }
                Frame::Expression(Self::Epsilon) => {}
                Frame::Expression(Self::Atom { forward, .. }) => {
                    append_pattern(&mut pattern, forward, limit)?;
                }
                Frame::Expression(Self::Concat(parts)) => {
                    let additional = parts.len().checked_mul(3).ok_or_else(|| {
                        resource_error(CompileStage::NfaConstruction, usize::MAX, limit)
                    })?;
                    reserve_frames(&mut stack, additional, limit)?;
                    for part in parts.iter().rev() {
                        stack.push(Frame::Text(")"));
                        stack.push(Frame::Expression(part));
                        stack.push(Frame::Text("(?:"));
                    }
                }
                Frame::Expression(Self::Union(branches)) => {
                    let additional = branches
                        .len()
                        .checked_mul(2)
                        .and_then(|frames| frames.checked_add(1))
                        .ok_or_else(|| {
                            resource_error(CompileStage::NfaConstruction, usize::MAX, limit)
                        })?;
                    reserve_frames(&mut stack, additional, limit)?;
                    stack.push(Frame::Text(")"));
                    for (index, branch) in branches.iter().enumerate().rev() {
                        stack.push(Frame::Expression(branch));
                        if index != 0 {
                            stack.push(Frame::Text("|"));
                        }
                    }
                    stack.push(Frame::Text("(?:"));
                }
                Frame::Expression(Self::Star(expression)) => {
                    reserve_frames(&mut stack, 4, limit)?;
                    stack.push(Frame::Text("*"));
                    stack.push(Frame::Text(")"));
                    stack.push(Frame::Expression(expression));
                    stack.push(Frame::Text("(?:"));
                }
            }
        }
        Ok(Some(pattern))
    }

    fn reversed(&self) -> Option<Self> {
        match self {
            Self::Empty => Some(Self::Empty),
            Self::Epsilon => Some(Self::Epsilon),
            Self::Atom { forward, reverse } => Some(Self::Atom {
                forward: reverse.clone()?,
                reverse: Some(forward.clone()),
            }),
            Self::Concat(parts) => {
                let mut reversed = Vec::new();
                for part in parts.iter().rev() {
                    reversed.push(part.reversed()?);
                }
                Some(concat(reversed))
            }
            Self::Union(branches) => {
                let mut reversed = Vec::new();
                for branch in branches {
                    reversed.push(branch.reversed()?);
                }
                Some(union(reversed))
            }
            Self::Star(expression) => Some(star(expression.reversed()?)),
        }
    }
}

fn append_pattern(pattern: &mut String, text: &str, limit: usize) -> Result<(), CompileError> {
    let required = pattern
        .len()
        .checked_add(text.len())
        .ok_or_else(|| resource_error(CompileStage::NfaConstruction, usize::MAX, limit))?;
    if required > limit {
        return Err(resource_error(
            CompileStage::NfaConstruction,
            required,
            limit,
        ));
    }
    pattern
        .try_reserve_exact(text.len())
        .map_err(|_| resource_error(CompileStage::NfaConstruction, required, limit))?;
    enforce_limit(CompileStage::NfaConstruction, pattern.capacity(), limit)?;
    pattern.push_str(text);
    Ok(())
}

pub const DEAD_DFA_STATE: u32 = u32::MAX;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledByteDfa {
    transitions: Vec<[u32; 256]>,
    accepting: Vec<bool>,
    live: Vec<bool>,
    nfa_states: usize,
    nfa_transitions: usize,
    memory_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AutomatonTimings {
    pub nfa: Duration,
    pub dfa: Duration,
}

impl CompiledByteDfa {
    pub fn compile(
        expression: &RegularExpression,
        limits: &CompileLimits,
        pointer: &crate::schema::SchemaPointer,
    ) -> Result<Self, CompileError> {
        Self::compile_configured(expression, limits, pointer, false)
    }

    pub(crate) fn compile_profiled(
        expression: &RegularExpression,
        limits: &CompileLimits,
        pointer: &crate::schema::SchemaPointer,
    ) -> Result<(Self, AutomatonTimings), CompileError> {
        Self::compile_measured(expression, limits, pointer, false)
    }

    fn compile_configured(
        expression: &RegularExpression,
        limits: &CompileLimits,
        pointer: &crate::schema::SchemaPointer,
        minimize: bool,
    ) -> Result<Self, CompileError> {
        Self::compile_measured(expression, limits, pointer, minimize).map(|(dfa, _)| dfa)
    }

    fn compile_measured(
        expression: &RegularExpression,
        limits: &CompileLimits,
        pointer: &crate::schema::SchemaPointer,
        minimize: bool,
    ) -> Result<(Self, AutomatonTimings), CompileError> {
        let nfa_started = Instant::now();
        if expression == &RegularExpression::Empty {
            let nfa = nfa_started.elapsed();
            let dfa_started = Instant::now();
            let dfa = Self::empty(limits)?;
            return Ok((
                dfa,
                AutomatonTimings {
                    nfa,
                    dfa: dfa_started.elapsed(),
                },
            ));
        }

        let nfa_byte_limit = limits
            .max_nfa_states
            .checked_mul(mem::size_of::<State>())
            .and_then(|states| {
                limits
                    .max_nfa_transitions
                    .checked_mul(mem::size_of::<Transition>())
                    .and_then(|transitions| states.checked_add(transitions))
            })
            .ok_or_else(|| {
                resource_error(
                    CompileStage::NfaConstruction,
                    usize::MAX,
                    limits.max_dfa_bytes,
                )
            })?;
        let pattern = expression.pattern_with_limit(nfa_byte_limit)?.ok_or(
            CompileError::InternalInvariant {
                message: "non-empty regular expression rendered as empty language",
            },
        )?;
        let nfa = thompson::NFA::compiler()
            .configure(
                thompson::NFA::config()
                    .which_captures(WhichCaptures::None)
                    .nfa_size_limit(Some(nfa_byte_limit)),
            )
            .build(&pattern)
            .map_err(|error| {
                if let Some(limit) = error.size_limit() {
                    resource_error(
                        CompileStage::NfaConstruction,
                        limit.saturating_add(1),
                        limit,
                    )
                } else {
                    CompileError::AutomatonBuild {
                        pointer: pointer.clone(),
                        message: error.to_string(),
                    }
                }
            })?;
        let nfa_states = nfa.states().len();
        enforce_limit(
            CompileStage::NfaConstruction,
            nfa_states,
            limits.max_nfa_states,
        )?;
        let nfa_transitions = count_nfa_transitions(&nfa, limits.max_nfa_transitions)?;
        let nfa_elapsed = nfa_started.elapsed();

        let dfa_started = Instant::now();
        let mut builder = dense::Builder::new();
        builder.configure(
            dense::Config::new()
                .match_kind(MatchKind::All)
                .minimize(minimize)
                .dfa_size_limit(Some(limits.max_dfa_bytes))
                .determinize_size_limit(Some(limits.max_dfa_bytes)),
        );
        let dfa = builder.build_from_nfa(&nfa).map_err(|error| {
            if error.is_size_limit_exceeded() {
                resource_error(
                    CompileStage::DfaDeterminization,
                    limits.max_dfa_bytes.saturating_add(1),
                    limits.max_dfa_bytes,
                )
            } else {
                CompileError::AutomatonBuild {
                    pointer: pointer.clone(),
                    message: error.to_string(),
                }
            }
        })?;
        enforce_limit(
            CompileStage::DfaDeterminization,
            dfa.memory_usage(),
            limits.max_dfa_bytes,
        )?;
        let start =
            dfa.universal_start_state(Anchored::Yes)
                .ok_or(CompileError::AutomatonBuild {
                    pointer: pointer.clone(),
                    message: "anchored DFA start state is unavailable".to_string(),
                })?;

        let mut raw_to_stable = FxHashMap::default();
        raw_to_stable.try_reserve(1).map_err(|_| {
            resource_error(CompileStage::DfaDeterminization, 1, limits.max_dfa_states)
        })?;
        raw_to_stable.insert(start, 0_u32);
        let mut queue = VecDeque::from([start]);
        let mut transitions = Vec::new();
        let mut accepting = Vec::new();
        while let Some(current) = queue.pop_front() {
            let expected_id = transitions.len();
            let actual_id = usize::try_from(raw_to_stable[&current]).map_err(|_| {
                CompileError::InternalInvariant {
                    message: "stable DFA state does not fit usize",
                }
            })?;
            if expected_id != actual_id {
                return Err(CompileError::InternalInvariant {
                    message: "DFA states were not processed in BFS order",
                });
            }
            ensure_dfa_capacity(expected_id + 1, limits)?;
            transitions.try_reserve_exact(1).map_err(|_| {
                resource_error(
                    CompileStage::DfaDeterminization,
                    expected_id + 1,
                    limits.max_dfa_states,
                )
            })?;
            accepting.try_reserve_exact(1).map_err(|_| {
                resource_error(
                    CompileStage::DfaDeterminization,
                    expected_id + 1,
                    limits.max_dfa_states,
                )
            })?;

            let mut row = [DEAD_DFA_STATE; 256];
            for byte in 0_u8..=u8::MAX {
                let next = dfa.next_state(current, byte);
                if dfa.is_dead_state(next) || dfa.is_quit_state(next) {
                    continue;
                }
                let stable = if let Some(stable) = raw_to_stable.get(&next) {
                    *stable
                } else {
                    let next_id = u32::try_from(raw_to_stable.len()).map_err(|_| {
                        resource_error(
                            CompileStage::DfaDeterminization,
                            raw_to_stable.len(),
                            limits.max_dfa_states,
                        )
                    })?;
                    enforce_limit(
                        CompileStage::DfaDeterminization,
                        raw_to_stable.len() + 1,
                        limits.max_dfa_states,
                    )?;
                    raw_to_stable.try_reserve(1).map_err(|_| {
                        resource_error(
                            CompileStage::DfaDeterminization,
                            raw_to_stable.len() + 1,
                            limits.max_dfa_states,
                        )
                    })?;
                    raw_to_stable.insert(next, next_id);
                    queue.try_reserve(1).map_err(|_| {
                        resource_error(
                            CompileStage::DfaDeterminization,
                            raw_to_stable.len(),
                            limits.max_dfa_states,
                        )
                    })?;
                    queue.push_back(next);
                    next_id
                };
                row[usize::from(byte)] = stable;
            }
            transitions.push(row);
            accepting.push(dfa.is_match_state(dfa.next_eoi_state(current)));
        }

        drop(dfa);
        drop(nfa);
        let live = compute_liveness(&transitions, &accepting, limits)?;
        let memory_bytes = allocated_dfa_bytes(&transitions, &accepting, &live)?;
        enforce_limit(
            CompileStage::DfaDeterminization,
            memory_bytes,
            limits.max_dfa_bytes,
        )?;
        Ok((
            Self {
                transitions,
                accepting,
                live,
                nfa_states,
                nfa_transitions,
                memory_bytes,
            },
            AutomatonTimings {
                nfa: nfa_elapsed,
                dfa: dfa_started.elapsed(),
            },
        ))
    }

    fn empty(limits: &CompileLimits) -> Result<Self, CompileError> {
        ensure_dfa_capacity(1, limits)?;
        Ok(Self {
            transitions: vec![[DEAD_DFA_STATE; 256]],
            accepting: vec![false],
            live: vec![false],
            nfa_states: 1,
            nfa_transitions: 0,
            memory_bytes: minimum_dfa_bytes(1)?,
        })
    }

    #[must_use]
    pub fn start_state(&self) -> u32 {
        0
    }

    #[must_use]
    pub fn next_state(&self, state: u32, byte: u8) -> u32 {
        usize::try_from(state)
            .ok()
            .and_then(|state| self.transitions.get(state))
            .map_or(DEAD_DFA_STATE, |row| row[usize::from(byte)])
    }

    #[must_use]
    pub fn is_dead(&self, state: u32) -> bool {
        state == DEAD_DFA_STATE
            || usize::try_from(state).map_or(true, |state| state >= self.transitions.len())
    }

    #[must_use]
    pub fn is_accepting(&self, state: u32) -> bool {
        usize::try_from(state)
            .ok()
            .and_then(|state| self.accepting.get(state))
            .copied()
            .unwrap_or(false)
    }

    #[must_use]
    pub fn is_live(&self, state: u32) -> bool {
        usize::try_from(state)
            .ok()
            .and_then(|state| self.live.get(state))
            .copied()
            .unwrap_or(false)
    }

    #[must_use]
    pub fn state_count(&self) -> usize {
        self.transitions.len()
    }

    #[must_use]
    pub fn nfa_state_count(&self) -> usize {
        self.nfa_states
    }

    #[must_use]
    pub fn nfa_transition_count(&self) -> usize {
        self.nfa_transitions
    }

    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.memory_bytes
    }

    #[must_use]
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut state = self.start_state();
        for byte in bytes {
            state = self.next_state(state, *byte);
            if self.is_dead(state) {
                return false;
            }
        }
        self.is_accepting(state)
    }
}

fn count_nfa_transitions(nfa: &thompson::NFA, limit: usize) -> Result<usize, CompileError> {
    let mut count = 0_usize;
    for state in nfa.states() {
        let additional = match state {
            State::ByteRange { .. } | State::Look { .. } | State::Capture { .. } => 1,
            State::Sparse(transitions) => transitions.transitions.len(),
            State::Dense(transitions) => transitions
                .transitions
                .iter()
                .filter(|state| **state != AutomataStateId::ZERO)
                .count(),
            State::Union { alternates } => alternates.len(),
            State::BinaryUnion { .. } => 2,
            State::Fail | State::Match { .. } => 0,
        };
        count = count
            .checked_add(additional)
            .ok_or_else(|| resource_error(CompileStage::NfaConstruction, usize::MAX, limit))?;
        enforce_limit(CompileStage::NfaConstruction, count, limit)?;
    }
    Ok(count)
}

fn minimum_dfa_bytes(states: usize) -> Result<usize, CompileError> {
    states
        .checked_mul(mem::size_of::<[u32; 256]>() + 2 * mem::size_of::<bool>())
        .ok_or(CompileError::ResourceLimitExceeded {
            stage: CompileStage::DfaDeterminization,
            observed: usize::MAX,
            limit: usize::MAX,
        })
}

fn allocated_dfa_bytes(
    transitions: &Vec<[u32; 256]>,
    accepting: &Vec<bool>,
    live: &Vec<bool>,
) -> Result<usize, CompileError> {
    transitions
        .capacity()
        .checked_mul(mem::size_of::<[u32; 256]>())
        .and_then(|bytes| {
            accepting
                .capacity()
                .checked_mul(mem::size_of::<bool>())
                .and_then(|accepting| bytes.checked_add(accepting))
        })
        .and_then(|bytes| {
            live.capacity()
                .checked_mul(mem::size_of::<bool>())
                .and_then(|live| bytes.checked_add(live))
        })
        .ok_or(CompileError::ResourceLimitExceeded {
            stage: CompileStage::DfaDeterminization,
            observed: usize::MAX,
            limit: usize::MAX,
        })
}

fn ensure_dfa_capacity(states: usize, limits: &CompileLimits) -> Result<(), CompileError> {
    enforce_limit(
        CompileStage::DfaDeterminization,
        states,
        limits.max_dfa_states,
    )?;
    let bytes = minimum_dfa_bytes(states)?;
    enforce_limit(
        CompileStage::DfaDeterminization,
        bytes,
        limits.max_dfa_bytes,
    )
}

fn compute_liveness(
    transitions: &Vec<[u32; 256]>,
    accepting: &Vec<bool>,
    limits: &CompileLimits,
) -> Result<Vec<bool>, CompileError> {
    if transitions.len() != accepting.len() {
        return Err(CompileError::InternalInvariant {
            message: "DFA state and acceptance tables have different lengths",
        });
    }
    let state_count = transitions.len();
    let mut predecessor_counts = Vec::new();
    predecessor_counts
        .try_reserve_exact(state_count)
        .map_err(|_| {
            resource_error(
                CompileStage::DfaDeterminization,
                state_count,
                limits.max_dfa_states,
            )
        })?;
    predecessor_counts.resize(state_count, 0_usize);
    let mut edge_count = 0_usize;
    for row in transitions {
        let mut targets = *row;
        targets.sort_unstable();
        let mut previous = DEAD_DFA_STATE;
        for target in targets {
            if target == DEAD_DFA_STATE || target == previous {
                continue;
            }
            previous = target;
            let target = usize::try_from(target).map_err(|_| CompileError::InternalInvariant {
                message: "DFA target state does not fit usize",
            })?;
            let count =
                predecessor_counts
                    .get_mut(target)
                    .ok_or(CompileError::InternalInvariant {
                        message: "DFA transition target is out of range",
                    })?;
            *count = count.checked_add(1).ok_or_else(|| {
                resource_error(
                    CompileStage::DfaDeterminization,
                    usize::MAX,
                    limits.max_dfa_bytes,
                )
            })?;
            edge_count = edge_count.checked_add(1).ok_or_else(|| {
                resource_error(
                    CompileStage::DfaDeterminization,
                    usize::MAX,
                    limits.max_dfa_bytes,
                )
            })?;
        }
    }

    let offset_count = state_count.checked_add(1).ok_or_else(|| {
        resource_error(
            CompileStage::DfaDeterminization,
            usize::MAX,
            limits.max_dfa_bytes,
        )
    })?;
    let retained_bytes = transitions
        .capacity()
        .checked_mul(mem::size_of::<[u32; 256]>())
        .and_then(|bytes| {
            accepting
                .capacity()
                .checked_mul(mem::size_of::<bool>())
                .and_then(|accepting| bytes.checked_add(accepting))
        })
        .and_then(|bytes| {
            state_count
                .checked_mul(mem::size_of::<bool>())
                .and_then(|live| bytes.checked_add(live))
        })
        .ok_or_else(|| {
            resource_error(
                CompileStage::DfaDeterminization,
                usize::MAX,
                limits.max_dfa_bytes,
            )
        })?;
    let temporary_bytes = predecessor_counts
        .capacity()
        .checked_mul(mem::size_of::<usize>())
        .and_then(|bytes| {
            offset_count
                .checked_mul(mem::size_of::<usize>())
                .and_then(|offsets| bytes.checked_add(offsets))
        })
        .and_then(|bytes| {
            edge_count
                .checked_mul(mem::size_of::<u32>())
                .and_then(|edges| bytes.checked_add(edges))
        })
        .and_then(|bytes| {
            state_count
                .checked_mul(mem::size_of::<bool>() + mem::size_of::<u32>())
                .and_then(|work| bytes.checked_add(work))
        })
        .and_then(|bytes| retained_bytes.checked_add(bytes))
        .ok_or_else(|| {
            resource_error(
                CompileStage::DfaDeterminization,
                usize::MAX,
                limits.max_dfa_bytes,
            )
        })?;
    enforce_limit(
        CompileStage::DfaDeterminization,
        temporary_bytes,
        limits.max_dfa_bytes,
    )?;

    let mut offsets = Vec::new();
    offsets.try_reserve_exact(offset_count).map_err(|_| {
        resource_error(
            CompileStage::DfaDeterminization,
            temporary_bytes,
            limits.max_dfa_bytes,
        )
    })?;
    offsets.push(0_usize);
    for count in &predecessor_counts {
        let next = offsets
            .last()
            .and_then(|offset| offset.checked_add(*count))
            .ok_or_else(|| {
                resource_error(
                    CompileStage::DfaDeterminization,
                    usize::MAX,
                    limits.max_dfa_bytes,
                )
            })?;
        offsets.push(next);
    }

    let mut predecessors = Vec::new();
    predecessors.try_reserve_exact(edge_count).map_err(|_| {
        resource_error(
            CompileStage::DfaDeterminization,
            temporary_bytes,
            limits.max_dfa_bytes,
        )
    })?;
    predecessors.resize(edge_count, 0_u32);
    predecessor_counts.fill(0);
    for (source, row) in transitions.iter().enumerate() {
        let source = to_u32(source, CompileStage::DfaDeterminization)?;
        let mut targets = *row;
        targets.sort_unstable();
        let mut previous = DEAD_DFA_STATE;
        for target in targets {
            if target == DEAD_DFA_STATE || target == previous {
                continue;
            }
            previous = target;
            let target = usize::try_from(target).map_err(|_| CompileError::InternalInvariant {
                message: "DFA target state does not fit usize",
            })?;
            let cursor = offsets[target]
                .checked_add(predecessor_counts[target])
                .ok_or(CompileError::InternalInvariant {
                    message: "DFA predecessor cursor overflowed",
                })?;
            predecessors[cursor] = source;
            predecessor_counts[target] = predecessor_counts[target].checked_add(1).ok_or(
                CompileError::InternalInvariant {
                    message: "DFA predecessor cursor overflowed",
                },
            )?;
        }
    }

    let mut live = accepting.to_vec();
    let mut stack = Vec::new();
    stack.try_reserve_exact(state_count).map_err(|_| {
        resource_error(
            CompileStage::DfaDeterminization,
            temporary_bytes,
            limits.max_dfa_bytes,
        )
    })?;
    for (state, is_accepting) in accepting.iter().copied().enumerate() {
        if is_accepting {
            stack.push(to_u32(state, CompileStage::DfaDeterminization)?);
        }
    }
    while let Some(target) = stack.pop() {
        let target = usize::try_from(target).map_err(|_| CompileError::InternalInvariant {
            message: "DFA liveness target does not fit usize",
        })?;
        for source in &predecessors[offsets[target]..offsets[target + 1]] {
            let source_id = *source;
            let source =
                usize::try_from(source_id).map_err(|_| CompileError::InternalInvariant {
                    message: "DFA liveness source does not fit usize",
                })?;
            if !live[source] {
                live[source] = true;
                stack.push(source_id);
            }
        }
    }
    Ok(live)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegularAnalysis {
    pub sccs: SccAnalysis,
    pub certificates: Vec<SccCertificate>,
    pub expressions: Vec<Option<RegularExpression>>,
    pub whole_language: Option<RegularExpression>,
}

pub fn certify_regular(grammar: &Grammar) -> Result<RegularAnalysis, CompileError> {
    let sccs = analyze_sccs(grammar)?;
    certify_regular_with_sccs(grammar, sccs)
}

pub(crate) fn certify_regular_with_sccs(
    grammar: &Grammar,
    sccs: SccAnalysis,
) -> Result<RegularAnalysis, CompileError> {
    let mut expressions = vec![None; grammar.nonterminals.len()];
    let mut outcomes: Vec<Option<Result<RegularCertificateKind, CertificateFailureReason>>> =
        vec![None; sccs.components.len()];
    let mut remaining = sccs.components.len();

    while remaining != 0 {
        let mut progress = false;
        for component in &sccs.components {
            if outcomes[component.id as usize].is_some() {
                continue;
            }
            let dependencies = external_components(grammar, &sccs, component);
            if dependencies
                .iter()
                .any(|dependency| outcomes[*dependency as usize].is_none())
            {
                continue;
            }
            if dependencies.iter().any(|dependency| {
                outcomes[*dependency as usize]
                    .as_ref()
                    .is_some_and(Result::is_err)
            }) {
                outcomes[component.id as usize] =
                    Some(Err(CertificateFailureReason::UncertifiedDependency));
                remaining -= 1;
                progress = true;
                continue;
            }

            let outcome = certify_component(grammar, &sccs, component, &mut expressions);
            outcomes[component.id as usize] = Some(outcome);
            remaining -= 1;
            progress = true;
        }
        if !progress {
            return Err(CompileError::InternalInvariant {
                message: "SCC condensation graph did not make progress",
            });
        }
    }

    let mut certificates = Vec::new();
    certificates
        .try_reserve_exact(outcomes.len())
        .map_err(|_| {
            resource_error(
                CompileStage::RegularCertification,
                outcomes.len(),
                outcomes.len(),
            )
        })?;
    for (index, outcome) in outcomes.into_iter().enumerate() {
        let outcome = outcome.ok_or(CompileError::InternalInvariant {
            message: "regular certification left an SCC unprocessed",
        })?;
        let scc = to_u32(index, CompileStage::RegularCertification)?;
        certificates.push(match outcome {
            Ok(kind) => SccCertificate {
                scc,
                kind: Some(kind),
                failure_reason: None,
            },
            Err(reason) => SccCertificate {
                scc,
                kind: None,
                failure_reason: Some(reason),
            },
        });
    }
    let whole_language = expressions[grammar.start as usize].clone();
    Ok(RegularAnalysis {
        sccs,
        certificates,
        expressions,
        whole_language,
    })
}

fn external_components(grammar: &Grammar, analysis: &SccAnalysis, component: &Scc) -> Vec<SccId> {
    let mut dependencies = Vec::new();
    for member in &component.members {
        for production in grammar.productions_for(*member) {
            for symbol in &production.rhs {
                if let Symbol::Nonterminal(target) = symbol {
                    let dependency = analysis.component_of[*target as usize];
                    if dependency != component.id {
                        dependencies.push(dependency);
                    }
                }
            }
        }
    }
    dependencies.sort_unstable();
    dependencies.dedup();
    dependencies
}

fn certify_component(
    grammar: &Grammar,
    analysis: &SccAnalysis,
    component: &Scc,
    expressions: &mut [Option<RegularExpression>],
) -> Result<RegularCertificateKind, CertificateFailureReason> {
    if !component.recursive {
        let member = component.members[0];
        let mut branches = Vec::new();
        for production in grammar.productions_for(member) {
            branches.push(rhs_expression(
                grammar,
                analysis,
                component,
                &production.rhs,
                expressions,
                false,
            )?);
        }
        expressions[member as usize] = Some(union(branches));
        return Ok(RegularCertificateKind::Acyclic);
    }

    let orientation = component_orientation(grammar, analysis, component)?;
    let reversed = orientation == LinearOrientation::Left;
    let solved = solve_linear_component(grammar, analysis, component, expressions, reversed)?;
    for (member, expression) in component.members.iter().zip(solved) {
        expressions[*member as usize] = Some(if reversed {
            expression
                .reversed()
                .ok_or(CertificateFailureReason::UnsupportedRegularOperation)?
        } else {
            expression
        });
    }
    Ok(if reversed {
        RegularCertificateKind::LeftLinear
    } else {
        RegularCertificateKind::RightLinear
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearOrientation {
    Neutral,
    Right,
    Left,
}

fn component_orientation(
    grammar: &Grammar,
    analysis: &SccAnalysis,
    component: &Scc,
) -> Result<LinearOrientation, CertificateFailureReason> {
    let mut orientation = LinearOrientation::Neutral;
    for member in &component.members {
        for production in grammar.productions_for(*member) {
            let positions: Vec<_> = production
                .rhs
                .iter()
                .enumerate()
                .filter_map(|(index, symbol)| match symbol {
                    Symbol::Nonterminal(target)
                        if analysis.component_of[*target as usize] == component.id =>
                    {
                        Some(index)
                    }
                    _ => None,
                })
                .collect();
            if positions.len() > 1 {
                return Err(CertificateFailureReason::MultipleRecursiveOccurrences);
            }
            let Some(position) = positions.first().copied() else {
                continue;
            };
            let production_orientation = if production.rhs.len() == 1 {
                LinearOrientation::Neutral
            } else if position == 0 {
                LinearOrientation::Left
            } else if position + 1 == production.rhs.len() {
                LinearOrientation::Right
            } else {
                return Err(CertificateFailureReason::RecursiveSymbolInInterior);
            };
            if production_orientation == LinearOrientation::Neutral {
                continue;
            }
            if orientation != LinearOrientation::Neutral && orientation != production_orientation {
                return Err(CertificateFailureReason::MixedLinearOrientation);
            }
            orientation = production_orientation;
        }
    }
    Ok(if orientation == LinearOrientation::Neutral {
        LinearOrientation::Right
    } else {
        orientation
    })
}

fn solve_linear_component(
    grammar: &Grammar,
    analysis: &SccAnalysis,
    component: &Scc,
    expressions: &[Option<RegularExpression>],
    reversed: bool,
) -> Result<Vec<RegularExpression>, CertificateFailureReason> {
    let state_count = component.members.len() + 1;
    let final_state = component.members.len();
    let mut local = BTreeMap::new();
    for (index, member) in component.members.iter().enumerate() {
        local.insert(*member, index);
    }
    let mut edges = vec![vec![RegularExpression::Empty; state_count]; state_count];

    for (source, member) in component.members.iter().enumerate() {
        for production in grammar.productions_for(*member) {
            let recursive = production.rhs.iter().position(|symbol| match symbol {
                Symbol::Nonterminal(target) => {
                    analysis.component_of[*target as usize] == component.id
                }
                Symbol::Terminal(_) => false,
            });
            let (target, atoms) = match recursive {
                Some(position) => {
                    let Symbol::Nonterminal(target) = production.rhs[position] else {
                        unreachable!();
                    };
                    let atoms = if reversed {
                        &production.rhs[position + 1..]
                    } else {
                        &production.rhs[..position]
                    };
                    (local[&target], atoms)
                }
                None => (final_state, production.rhs.as_slice()),
            };
            let label = rhs_expression(grammar, analysis, component, atoms, expressions, reversed)?;
            edges[source][target] = union(vec![edges[source][target].clone(), label]);
        }
    }

    for pivot in 0..component.members.len() {
        let incoming: Vec<_> = (0..state_count)
            .map(|source| edges[source][pivot].clone())
            .collect();
        let outgoing = edges[pivot].clone();
        let loop_expression = star(edges[pivot][pivot].clone());
        for source in 0..state_count {
            if incoming[source] == RegularExpression::Empty {
                continue;
            }
            for target in 0..state_count {
                if outgoing[target] == RegularExpression::Empty {
                    continue;
                }
                let path = concat(vec![
                    incoming[source].clone(),
                    loop_expression.clone(),
                    outgoing[target].clone(),
                ]);
                edges[source][target] = union(vec![edges[source][target].clone(), path]);
            }
        }
    }
    Ok((0..component.members.len())
        .map(|source| edges[source][final_state].clone())
        .collect())
}

fn rhs_expression(
    grammar: &Grammar,
    analysis: &SccAnalysis,
    component: &Scc,
    rhs: &[Symbol],
    expressions: &[Option<RegularExpression>],
    reversed: bool,
) -> Result<RegularExpression, CertificateFailureReason> {
    let symbols: Box<dyn Iterator<Item = &Symbol>> = if reversed {
        Box::new(rhs.iter().rev())
    } else {
        Box::new(rhs.iter())
    };
    let mut parts = Vec::new();
    for symbol in symbols {
        let expression = match symbol {
            Symbol::Terminal(id) => {
                let terminal = &grammar.terminals[*id as usize];
                let expression = RegularExpression::Atom {
                    forward: terminal.forward_pattern(),
                    reverse: terminal.reverse_pattern(),
                };
                if reversed {
                    expression
                        .reversed()
                        .ok_or(CertificateFailureReason::UnsupportedRegularOperation)?
                } else {
                    expression
                }
            }
            Symbol::Nonterminal(id) => {
                if analysis.component_of[*id as usize] == component.id {
                    return Err(CertificateFailureReason::RecursiveSymbolInInterior);
                }
                let expression = expressions[*id as usize]
                    .clone()
                    .ok_or(CertificateFailureReason::UncertifiedDependency)?;
                if reversed {
                    expression
                        .reversed()
                        .ok_or(CertificateFailureReason::UnsupportedRegularOperation)?
                } else {
                    expression
                }
            }
        };
        parts.push(expression);
    }
    Ok(concat(parts))
}

fn concat(parts: Vec<RegularExpression>) -> RegularExpression {
    let mut flattened = Vec::new();
    for part in parts {
        match part {
            RegularExpression::Empty => return RegularExpression::Empty,
            RegularExpression::Epsilon => {}
            RegularExpression::Concat(nested) => flattened.extend(nested),
            other => flattened.push(other),
        }
    }
    match flattened.len() {
        0 => RegularExpression::Epsilon,
        1 => flattened.pop().unwrap(),
        _ => RegularExpression::Concat(flattened),
    }
}

fn union(branches: Vec<RegularExpression>) -> RegularExpression {
    let mut unique = Vec::new();
    for branch in branches {
        match branch {
            RegularExpression::Empty => {}
            RegularExpression::Union(nested) => {
                unique.extend(nested);
            }
            other => unique.push(other),
        }
    }
    unique.sort();
    unique.dedup();
    match unique.len() {
        0 => RegularExpression::Empty,
        1 => unique.pop().unwrap(),
        _ => RegularExpression::Union(unique),
    }
}

fn star(expression: RegularExpression) -> RegularExpression {
    match expression {
        RegularExpression::Empty | RegularExpression::Epsilon => RegularExpression::Epsilon,
        RegularExpression::Star(_) => expression,
        other => RegularExpression::Star(Box::new(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{parse_and_normalize, CompileOptions};

    fn grammar(schema: &str) -> Grammar {
        let options = CompileOptions::default();
        let arena = parse_and_normalize(schema.as_bytes(), &options).unwrap();
        lower(&arena, &options.limits).unwrap()
    }

    fn byte_dfa(schema: &str) -> CompiledByteDfa {
        let options = CompileOptions::default();
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let expression = certify_regular(&grammar).unwrap().whole_language.unwrap();
        CompiledByteDfa::compile(
            &expression,
            &options.limits,
            &crate::schema::SchemaPointer(String::new()),
        )
        .unwrap()
    }

    #[test]
    fn deep_reference_chain_lowers_to_unit_productions() {
        let schema = include_str!("../testdata/regressions/deep_acyclic_ref.json");
        let grammar = grammar(schema);
        assert_eq!(grammar.productions.len(), 6);
        assert!(grammar.productions.iter().all(|production| production
            .provenance
            .pointer
            .0
            .starts_with("/$defs")
            || production.provenance.pointer.0.is_empty()));
    }

    #[test]
    fn recursive_property_remains_inside_delimiters() {
        let schema = include_str!("../testdata/regressions/recursive_optional_property.json");
        let grammar = grammar(schema);
        let node = grammar
            .nonterminals
            .iter()
            .find(|nonterminal| nonterminal.provenance.pointer.0 == "/$defs/node")
            .unwrap();
        assert!(grammar.productions_for(node.id).any(|production| {
            production.rhs.len() == 3
                && matches!(production.rhs[0], Symbol::Terminal(_))
                && matches!(production.rhs[2], Symbol::Terminal(_))
        }));
        assert!(grammar
            .nonterminals
            .iter()
            .any(|nonterminal| nonterminal.name.contains("members")));
    }

    #[test]
    fn prefix_items_uses_linear_position_states() {
        let schema = include_str!("../testdata/regressions/prefix_items.json");
        let grammar = grammar(schema);
        let positions = grammar
            .nonterminals
            .iter()
            .filter(|nonterminal| nonterminal.name.contains("array_0_position"))
            .count();
        assert_eq!(positions, 4);
    }

    #[test]
    fn closed_optional_object_does_not_enumerate_subsets() {
        let grammar = grammar(
            r#"{"type":"object","properties":{"a":{"const":1},"b":{"const":2},"c":{"const":3}},"additionalProperties":false}"#,
        );
        let member_nonterminals = grammar
            .nonterminals
            .iter()
            .filter(|nonterminal| nonterminal.name.contains("members"))
            .count();
        assert_eq!(member_nonterminals, 8);
    }

    #[test]
    fn every_production_has_provenance() {
        let grammar = grammar(r#"{"type":"array","items":{"const":1},"maxItems":2}"#);
        assert!(grammar.productions.iter().all(|production| !production
            .provenance
            .pointer
            .0
            .contains("//")));
    }

    #[test]
    fn lowering_budget_is_typed() {
        let mut options = CompileOptions::default();
        options.limits.max_symbols = 1;
        let arena = parse_and_normalize(br#"{"const":1}"#, &options).unwrap();
        assert!(matches!(
            lower(&arena, &options.limits),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::Lowering,
                ..
            })
        ));
    }

    #[test]
    fn maximum_array_bound_cannot_overflow_state_count() {
        let arena = crate::schema::parse_and_normalize(
            br#"{"type":"array","items":{"const":1},"maxItems":18446744073709551615}"#,
            &CompileOptions::default(),
        );
        match arena {
            Ok(arena) => assert!(matches!(
                lower(&arena, &CompileLimits::default()),
                Err(CompileError::ResourceLimitExceeded {
                    stage: CompileStage::Lowering,
                    ..
                })
            )),
            Err(CompileError::InvalidKeywordValue { .. }) => {}
            Err(error) => panic!("unexpected error: {error}"),
        }
    }

    #[test]
    fn reduction_removes_unreachable_and_unproductive_symbols() {
        let mut grammar = grammar(r#"{"const":1,"$defs":{"unused":{"const":2}}}"#);
        let before = grammar.nonterminals.len();
        let stats = reduce(&mut grammar).unwrap();
        assert!(grammar.nonterminals.len() < before);
        assert!(stats.removed_nonterminals > 0);
        assert_eq!(grammar.productions.len(), 1);
    }

    #[test]
    fn reduction_preserves_empty_language_start() {
        let mut grammar = grammar("false");
        reduce(&mut grammar).unwrap();
        assert_eq!(grammar.start, 0);
        assert_eq!(grammar.nonterminals.len(), 1);
        assert!(grammar.productions.is_empty());
    }

    #[test]
    fn reduction_preserves_epsilon_language() {
        let provenance = Provenance {
            resource: crate::schema::ResourceId(0),
            pointer: crate::schema::SchemaPointer(String::new()),
            keyword: None,
        };
        let mut grammar = Grammar {
            start: 0,
            nonterminals: vec![Nonterminal {
                id: 0,
                name: "start".to_owned(),
                provenance: provenance.clone(),
            }],
            terminals: Vec::new(),
            productions: vec![Production {
                id: 0,
                lhs: 0,
                rhs: Vec::new(),
                provenance,
            }],
        };
        reduce(&mut grammar).unwrap();
        assert_eq!(grammar.productions.len(), 1);
        assert!(grammar.productions[0].rhs.is_empty());
    }

    #[test]
    fn tarjan_is_deterministic_for_mutual_recursion() {
        let schema = include_str!("../testdata/regressions/recursive_required_property.json");
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let first = analyze_sccs(&grammar).unwrap();
        let second = analyze_sccs(&grammar).unwrap();
        assert_eq!(first, second);
        assert!(first.components.iter().any(|component| component.recursive));
        assert!(first
            .components
            .iter()
            .all(|component| component.members.windows(2).all(|pair| pair[0] < pair[1])));
    }

    #[test]
    fn acyclic_chain_has_no_recursive_component() {
        let schema = include_str!("../testdata/regressions/deep_acyclic_ref.json");
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let analysis = analyze_sccs(&grammar).unwrap();
        assert!(analysis
            .components
            .iter()
            .all(|component| !component.recursive));
    }

    #[test]
    fn deep_chain_has_acyclic_certificate() {
        let schema = include_str!("../testdata/regressions/deep_acyclic_ref.json");
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let analysis = certify_regular(&grammar).unwrap();
        assert!(analysis
            .certificates
            .iter()
            .all(|certificate| certificate.kind == Some(RegularCertificateKind::Acyclic)));
        assert_eq!(
            analysis.whole_language.unwrap().pattern().unwrap().unwrap(),
            r#""done""#
        );
    }

    #[test]
    fn bounded_prefix_items_certificate_is_exact() {
        let schema = include_str!("../testdata/regressions/prefix_items.json");
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let expression = certify_regular(&grammar).unwrap().whole_language.unwrap();
        let regex = regex::Regex::new(&format!(
            r"\A(?:{})\z",
            expression.pattern().unwrap().unwrap()
        ))
        .unwrap();
        for valid in ["[1]", "[1,2]", "[1,2,2]"] {
            assert!(regex.is_match(valid), "must accept {valid}");
        }
        for invalid in ["[]", "[2]", "[1,3]", "[1,2,2,2]"] {
            assert!(!regex.is_match(invalid), "must reject {invalid}");
        }
    }

    #[test]
    fn recursive_json_object_is_not_mislabeled_regular() {
        let schema = include_str!("../testdata/regressions/recursive_optional_property.json");
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let analysis = certify_regular(&grammar).unwrap();
        assert!(analysis.whole_language.is_none());
        assert!(analysis.certificates.iter().any(|certificate| {
            matches!(
                certificate.failure_reason,
                Some(
                    CertificateFailureReason::MultipleRecursiveOccurrences
                        | CertificateFailureReason::RecursiveSymbolInInterior
                )
            )
        }));
    }

    fn manual_linear_grammar(left: bool) -> Grammar {
        let provenance = Provenance {
            resource: crate::schema::ResourceId(0),
            pointer: crate::schema::SchemaPointer(String::new()),
            keyword: None,
        };
        let terminal = RegularTerminal {
            id: 0,
            kind: RegularTerminalKind::Literal(b"a".to_vec()),
            provenance: provenance.clone(),
        };
        let recursive_rhs = if left {
            vec![Symbol::Nonterminal(0), Symbol::Terminal(0)]
        } else {
            vec![Symbol::Terminal(0), Symbol::Nonterminal(0)]
        };
        Grammar {
            start: 0,
            nonterminals: vec![Nonterminal {
                id: 0,
                name: "start".to_owned(),
                provenance: provenance.clone(),
            }],
            terminals: vec![terminal],
            productions: vec![
                Production {
                    id: 0,
                    lhs: 0,
                    rhs: recursive_rhs,
                    provenance: provenance.clone(),
                },
                Production {
                    id: 1,
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                    provenance,
                },
            ],
        }
    }

    #[test]
    fn certifies_right_linear_component() {
        let grammar = manual_linear_grammar(false);
        let analysis = certify_regular(&grammar).unwrap();
        assert_eq!(
            analysis.certificates[0].kind,
            Some(RegularCertificateKind::RightLinear)
        );
        let regex = regex::Regex::new(&format!(
            r"\A(?:{})\z",
            analysis.whole_language.unwrap().pattern().unwrap().unwrap()
        ))
        .unwrap();
        assert!(regex.is_match("aaaa"));
        assert!(!regex.is_match(""));
    }

    #[test]
    fn certifies_left_linear_component() {
        let grammar = manual_linear_grammar(true);
        let analysis = certify_regular(&grammar).unwrap();
        assert_eq!(
            analysis.certificates[0].kind,
            Some(RegularCertificateKind::LeftLinear)
        );
        let regex = regex::Regex::new(&format!(
            r"\A(?:{})\z",
            analysis.whole_language.unwrap().pattern().unwrap().unwrap()
        ))
        .unwrap();
        assert!(regex.is_match("aaaa"));
        assert!(!regex.is_match(""));
    }

    #[test]
    fn byte_dfa_accepts_deep_reference_result_exactly() {
        let schema = include_str!("../testdata/regressions/deep_acyclic_ref.json");
        let dfa = byte_dfa(schema);
        assert!(dfa.accepts(br#""done""#));
        assert!(!dfa.accepts(br#""other""#));
        assert!(!dfa.accepts(b"done"));
        assert!(dfa.nfa_state_count() > 0);
        assert!(dfa.nfa_transition_count() > 0);
    }

    #[test]
    fn byte_dfa_preserves_prefix_overlapping_alternatives() {
        let expression = RegularExpression::Union(vec![
            RegularExpression::Atom {
                forward: "1".to_owned(),
                reverse: Some("1".to_owned()),
            },
            RegularExpression::Atom {
                forward: "12".to_owned(),
                reverse: Some("21".to_owned()),
            },
        ]);
        let dfa = CompiledByteDfa::compile(
            &expression,
            &CompileLimits::default(),
            &crate::schema::SchemaPointer(String::new()),
        )
        .unwrap();
        assert!(dfa.accepts(b"1"));
        assert!(dfa.accepts(b"12"));
        assert!(!dfa.accepts(b"123"));
    }

    #[test]
    fn byte_dfa_handles_empty_language() {
        let dfa = CompiledByteDfa::compile(
            &RegularExpression::Empty,
            &CompileLimits::default(),
            &crate::schema::SchemaPointer(String::new()),
        )
        .unwrap();
        assert_eq!(dfa.state_count(), 1);
        assert!(!dfa.is_accepting(dfa.start_state()));
        assert!(!dfa.is_live(dfa.start_state()));
        assert!(dfa.is_dead(dfa.next_state(dfa.start_state(), b'x')));
        assert!(!dfa.accepts(b""));
    }

    #[test]
    fn byte_dfa_handles_epsilon_language() {
        let dfa = CompiledByteDfa::compile(
            &RegularExpression::Epsilon,
            &CompileLimits::default(),
            &crate::schema::SchemaPointer(String::new()),
        )
        .unwrap();
        assert!(dfa.is_accepting(dfa.start_state()));
        assert!(dfa.is_live(dfa.start_state()));
        assert!(dfa.accepts(b""));
        assert!(!dfa.accepts(b"x"));
    }

    #[test]
    fn byte_dfa_accepts_only_canonical_literal_bytes() {
        let unicode = byte_dfa(r#"{"const":"é"}"#);
        assert!(unicode.accepts("\"é\"".as_bytes()));
        assert!(!unicode.accepts(br#""\u00e9""#));

        let control = byte_dfa("{\"const\":\"line\\nfeed\"}");
        assert!(control.accepts(br#""line\nfeed""#));
        assert!(!control.accepts(b"\"line\nfeed\""));
    }

    #[test]
    fn schema_enum_preserves_numeric_prefix_alternatives() {
        let dfa = byte_dfa(r#"{"enum":[1,12]}"#);
        assert!(dfa.accepts(b"1"));
        assert!(dfa.accepts(b"12"));
        assert!(!dfa.accepts(b"2"));
    }

    #[test]
    fn malformed_regular_atom_reports_its_pointer() {
        let expression = RegularExpression::Atom {
            forward: "(".to_owned(),
            reverse: None,
        };
        let pointer = crate::schema::SchemaPointer("/$defs/bad".to_owned());
        assert!(matches!(
            CompiledByteDfa::compile(&expression, &CompileLimits::default(), &pointer),
            Err(CompileError::AutomatonBuild {
                pointer: error_pointer,
                ..
            }) if error_pointer == pointer
        ));
    }

    #[test]
    fn non_ascii_literal_does_not_claim_a_reverse_certificate() {
        let terminal = RegularTerminal {
            id: 0,
            kind: RegularTerminalKind::Literal("é".as_bytes().to_vec()),
            provenance: Provenance {
                resource: crate::schema::ResourceId(0),
                pointer: crate::schema::SchemaPointer(String::new()),
                keyword: None,
            },
        };
        assert_eq!(terminal.reverse_pattern(), None);
    }

    #[test]
    fn byte_dfa_prefix_items_language_is_exact() {
        let schema = include_str!("../testdata/regressions/prefix_items.json");
        let dfa = byte_dfa(schema);
        for valid in ["[1]", "[1,2]", "[1,2,2]"] {
            assert!(dfa.accepts(valid.as_bytes()), "must accept {valid}");
        }
        for invalid in ["[]", "[2]", "[1,3]", "[1,2,2,2]"] {
            assert!(!dfa.accepts(invalid.as_bytes()), "must reject {invalid}");
        }
    }

    #[test]
    fn byte_dfa_liveness_is_reverse_reachability() {
        let dfa = byte_dfa(r#"{"const":"done"}"#);
        let mut state = dfa.start_state();
        assert!(dfa.is_live(state));
        for byte in br#""done""# {
            state = dfa.next_state(state, *byte);
            assert!(dfa.is_live(state));
        }
        assert!(dfa.is_accepting(state));
        let dead = dfa.next_state(dfa.start_state(), b'x');
        assert!(dfa.is_dead(dead));
        assert!(!dfa.is_live(dead));
        assert!(!dfa.is_accepting(u32::MAX - 1));
    }

    #[test]
    fn byte_dfa_numbering_is_stable_across_builds() {
        let first = byte_dfa(include_str!("../testdata/regressions/prefix_items.json"));
        let second = byte_dfa(include_str!("../testdata/regressions/prefix_items.json"));
        assert_eq!(first, second);
    }

    #[test]
    fn minimized_and_unminimized_dfas_accept_the_same_language() {
        let expression = RegularExpression::Union(vec![
            RegularExpression::Atom {
                forward: "1".to_owned(),
                reverse: Some("1".to_owned()),
            },
            RegularExpression::Atom {
                forward: "12".to_owned(),
                reverse: Some("21".to_owned()),
            },
        ]);
        let limits = CompileLimits::default();
        let pointer = crate::schema::SchemaPointer(String::new());
        let plain =
            CompiledByteDfa::compile_configured(&expression, &limits, &pointer, false).unwrap();
        let minimized =
            CompiledByteDfa::compile_configured(&expression, &limits, &pointer, true).unwrap();

        let mut candidates = vec![Vec::new()];
        for _ in 0..=3 {
            let existing = candidates.clone();
            for prefix in existing {
                for byte in b"12x" {
                    let mut candidate = prefix.clone();
                    candidate.push(*byte);
                    candidates.push(candidate);
                }
            }
        }
        for candidate in candidates {
            assert_eq!(
                plain.accepts(&candidate),
                minimized.accepts(&candidate),
                "minimization changed acceptance for {candidate:?}"
            );
        }
    }

    #[test]
    fn liveness_matches_bounded_suffix_enumeration() {
        let expression = RegularExpression::Union(vec![
            RegularExpression::Atom {
                forward: "ab".to_owned(),
                reverse: Some("ba".to_owned()),
            },
            RegularExpression::Atom {
                forward: "ac".to_owned(),
                reverse: Some("ca".to_owned()),
            },
        ]);
        let dfa = CompiledByteDfa::compile(
            &expression,
            &CompileLimits::default(),
            &crate::schema::SchemaPointer(String::new()),
        )
        .unwrap();
        let suffixes = [b"".as_slice(), b"a", b"b", b"c", b"ab", b"ac"];
        for state in 0..dfa.state_count() {
            let state = u32::try_from(state).unwrap();
            let can_finish = suffixes.iter().any(|suffix| {
                let mut trial = state;
                for byte in *suffix {
                    trial = dfa.next_state(trial, *byte);
                }
                dfa.is_accepting(trial)
            });
            assert_eq!(dfa.is_live(state), can_finish, "state {state}");
        }
    }

    #[test]
    fn byte_dfa_enforces_nfa_state_and_transition_limits() {
        let expression = RegularExpression::Atom {
            forward: "done".to_owned(),
            reverse: Some("enod".to_owned()),
        };
        let pointer = crate::schema::SchemaPointer(String::new());
        let limits = CompileLimits {
            max_nfa_states: 1,
            ..CompileLimits::default()
        };
        assert!(matches!(
            CompiledByteDfa::compile(&expression, &limits, &pointer),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::NfaConstruction,
                ..
            })
        ));

        let limits = CompileLimits {
            max_nfa_transitions: 0,
            ..CompileLimits::default()
        };
        assert!(matches!(
            CompiledByteDfa::compile(&expression, &limits, &pointer),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::NfaConstruction,
                ..
            })
        ));
    }

    #[test]
    fn byte_dfa_enforces_state_and_memory_limits() {
        let expression = RegularExpression::Atom {
            forward: "done".to_owned(),
            reverse: Some("enod".to_owned()),
        };
        let pointer = crate::schema::SchemaPointer(String::new());
        let limits = CompileLimits {
            max_dfa_states: 1,
            ..CompileLimits::default()
        };
        assert!(matches!(
            CompiledByteDfa::compile(&expression, &limits, &pointer),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::DfaDeterminization,
                ..
            })
        ));

        let limits = CompileLimits {
            max_dfa_bytes: mem::size_of::<[u32; 256]>() - 1,
            ..CompileLimits::default()
        };
        assert!(matches!(
            CompiledByteDfa::compile(&RegularExpression::Empty, &limits, &pointer),
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::DfaDeterminization,
                ..
            })
        ));
    }
}
