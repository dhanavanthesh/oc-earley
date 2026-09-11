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
}
