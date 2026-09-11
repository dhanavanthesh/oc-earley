use std::collections::BTreeMap;

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

    let state_count = effective_max
        .map(|maximum| maximum + 1)
        .unwrap_or(array.prefix_items.len() + 1);
    let mut states = Vec::new();
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
    u32::try_from(value).map_err(|_| resource_error(stage, value, u32::MAX as usize))
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

#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub fn pattern(&self) -> Option<String> {
        match self {
            Self::Empty => None,
            Self::Epsilon => Some(String::new()),
            Self::Atom { forward, .. } => Some(forward.clone()),
            Self::Concat(parts) => {
                let mut pattern = String::new();
                for part in parts {
                    pattern.push_str("(?:");
                    pattern.push_str(&part.pattern()?);
                    pattern.push(')');
                }
                Some(pattern)
            }
            Self::Union(branches) => {
                let mut pattern = String::from("(?:");
                for (index, branch) in branches.iter().enumerate() {
                    if index != 0 {
                        pattern.push('|');
                    }
                    pattern.push_str(&branch.pattern()?);
                }
                pattern.push(')');
                Some(pattern)
            }
            Self::Star(expression) => Some(format!("(?:{})*", expression.pattern()?)),
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegularAnalysis {
    pub sccs: SccAnalysis,
    pub certificates: Vec<SccCertificate>,
    pub expressions: Vec<Option<RegularExpression>>,
    pub whole_language: Option<RegularExpression>,
}

pub fn certify_regular(grammar: &Grammar) -> Result<RegularAnalysis, CompileError> {
    let sccs = analyze_sccs(grammar)?;
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

    let certificates = outcomes
        .into_iter()
        .enumerate()
        .map(
            |(index, outcome)| match outcome.expect("all components processed") {
                Ok(kind) => SccCertificate {
                    scc: index as SccId,
                    kind: Some(kind),
                    failure_reason: None,
                },
                Err(reason) => SccCertificate {
                    scc: index as SccId,
                    kind: None,
                    failure_reason: Some(reason),
                },
            },
        )
        .collect();
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
    let mut unique = BTreeMap::new();
    for branch in branches {
        match branch {
            RegularExpression::Empty => {}
            RegularExpression::Union(nested) => {
                for branch in nested {
                    unique.insert(expression_key(&branch), branch);
                }
            }
            other => {
                unique.insert(expression_key(&other), other);
            }
        }
    }
    let mut branches: Vec<_> = unique.into_values().collect();
    match branches.len() {
        0 => RegularExpression::Empty,
        1 => branches.pop().unwrap(),
        _ => RegularExpression::Union(branches),
    }
}

fn star(expression: RegularExpression) -> RegularExpression {
    match expression {
        RegularExpression::Empty | RegularExpression::Epsilon => RegularExpression::Epsilon,
        RegularExpression::Star(_) => expression,
        other => RegularExpression::Star(Box::new(other)),
    }
}

fn expression_key(expression: &RegularExpression) -> String {
    format!("{expression:?}")
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
            analysis.whole_language.unwrap().pattern().unwrap(),
            r#""done""#
        );
    }

    #[test]
    fn bounded_prefix_items_certificate_is_exact() {
        let schema = include_str!("../testdata/regressions/prefix_items.json");
        let mut grammar = grammar(schema);
        reduce(&mut grammar).unwrap();
        let expression = certify_regular(&grammar).unwrap().whole_language.unwrap();
        let regex =
            regex::Regex::new(&format!(r"\A(?:{})\z", expression.pattern().unwrap())).unwrap();
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
            analysis.whole_language.unwrap().pattern().unwrap()
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
            analysis.whole_language.unwrap().pattern().unwrap()
        ))
        .unwrap();
        assert!(regex.is_match("aaaa"));
        assert!(!regex.is_match(""));
    }
}
