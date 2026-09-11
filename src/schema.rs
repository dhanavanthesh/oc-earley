use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};

use crate::error::{CompileError, CompileStage};

pub const COMPILED_FORMAT_VERSION: u32 = 1;
pub const PROFILE_ID: &str = "K1";
pub const CANONICAL_POLICY_ID: &str = "oc-earley-json-compact-v1";
pub const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct SchemaId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct ResourceId(pub u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct SchemaPointer(pub String);

impl fmt::Display for SchemaPointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("#")
        } else {
            write!(f, "#{}", self.0)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Provenance {
    pub resource: ResourceId,
    pub pointer: SchemaPointer,
    pub keyword: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DiagnosticLocation {
    pub resource: ResourceId,
    pub pointer: SchemaPointer,
    pub keyword: Option<String>,
}

impl fmt::Display for DiagnosticLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.keyword {
            Some(keyword) => write!(f, "{} keyword `{keyword}`", self.pointer),
            None => self.pointer.fmt(f),
        }
    }
}

impl Provenance {
    pub fn location(&self) -> DiagnosticLocation {
        DiagnosticLocation {
            resource: self.resource,
            pointer: self.pointer.clone(),
            keyword: self.keyword.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompileLimits {
    pub max_schema_bytes: usize,
    pub max_schema_nodes: usize,
    pub max_ref_edges: usize,
    pub max_symbols: usize,
    pub max_productions: usize,
    pub max_rhs_symbols: usize,
    pub max_nfa_states: usize,
    pub max_nfa_transitions: usize,
    pub max_dfa_states: usize,
    pub max_dfa_bytes: usize,
}

impl Default for CompileLimits {
    fn default() -> Self {
        Self {
            max_schema_bytes: 8 * 1024 * 1024,
            max_schema_nodes: 100_000,
            max_ref_edges: 200_000,
            max_symbols: 500_000,
            max_productions: 1_000_000,
            max_rhs_symbols: 4_000_000,
            max_nfa_states: 1_000_000,
            max_nfa_transitions: 4_000_000,
            max_dfa_states: 200_000,
            max_dfa_bytes: 512 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    pub limits: CompileLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CanonicalValue {
    pub json: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum JsonType {
    Null,
    Boolean,
    String,
    Number,
    Integer,
    Array,
    Object,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StringConstraints {
    pub min_length: usize,
    pub max_length: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArrayConstraints {
    pub prefix_items: Vec<SchemaId>,
    pub items: Option<SchemaId>,
    pub min_items: usize,
    pub max_items: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum AdditionalProperties {
    Forbidden,
    Unconstrained,
    Schema(SchemaId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ObjectConstraints {
    pub properties: BTreeMap<String, SchemaId>,
    pub required: BTreeSet<String>,
    pub additional_properties: AdditionalProperties,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum NormalizedSchema {
    Any,
    Never,
    Null,
    Boolean,
    String(StringConstraints),
    Number,
    Integer,
    Const(CanonicalValue),
    Enum(Vec<CanonicalValue>),
    Array(ArrayConstraints),
    Object(ObjectConstraints),
    Union(Vec<SchemaId>),
    Ref(SchemaId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SchemaNode {
    pub id: SchemaId,
    pub provenance: Provenance,
    pub kind: NormalizedSchema,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReferenceEdge {
    pub from: SchemaId,
    pub to: SchemaId,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SchemaArena {
    pub root: SchemaId,
    pub nodes: Vec<SchemaNode>,
    pub reference_edges: Vec<ReferenceEdge>,
}

impl SchemaArena {
    pub fn node(&self, id: SchemaId) -> Option<&SchemaNode> {
        self.nodes.get(id.0 as usize)
    }
}

#[derive(Debug)]
struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(UniqueValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("__OCE_DUPLICATE__{key}")));
            }
            let UniqueValue(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

pub fn parse_and_normalize(
    schema: &[u8],
    options: &CompileOptions,
) -> Result<SchemaArena, CompileError> {
    enforce_limit(
        CompileStage::Parse,
        schema.len(),
        options.limits.max_schema_bytes,
    )?;

    let root = match serde_json::from_slice::<UniqueValue>(schema) {
        Ok(UniqueValue(value)) => value,
        Err(error) => {
            let message = error.to_string();
            if let Some(rest) = message.split("__OCE_DUPLICATE__").nth(1) {
                let key = rest.split(" at line ").next().unwrap_or(rest).to_owned();
                return Err(CompileError::DuplicateSchemaKey {
                    location: root_location(),
                    key,
                });
            }
            return Err(CompileError::InvalidJson { message });
        }
    };

    if !is_schema(&root) {
        return Err(invalid_value(
            &SchemaPointer(String::new()),
            "schema",
            "a schema must be a boolean or object",
        ));
    }

    let raw_nodes = index_schema_locations(root, &options.limits)?;
    let mut pointer_to_id = BTreeMap::new();
    for (index, (pointer, _)) in raw_nodes.iter().enumerate() {
        let id = u32::try_from(index).map_err(|_| CompileError::ResourceLimitExceeded {
            stage: CompileStage::SchemaIndex,
            observed: index,
            limit: u32::MAX as usize,
        })?;
        pointer_to_id.insert(pointer.clone(), SchemaId(id));
    }

    let mut nodes = Vec::new();
    nodes
        .try_reserve(raw_nodes.len())
        .map_err(|_| CompileError::ResourceLimitExceeded {
            stage: CompileStage::Normalization,
            observed: raw_nodes.len(),
            limit: options.limits.max_schema_nodes,
        })?;
    let mut reference_edges = Vec::new();

    for (index, (pointer, raw)) in raw_nodes.iter().enumerate() {
        let id = SchemaId(index as u32);
        let provenance = Provenance {
            resource: ResourceId(0),
            pointer: pointer.clone(),
            keyword: None,
        };
        let kind = normalize_node(
            id,
            pointer,
            raw,
            &raw_nodes,
            &pointer_to_id,
            &mut reference_edges,
            &options.limits,
        )?;
        nodes.push(SchemaNode {
            id,
            provenance,
            kind,
        });
    }

    Ok(SchemaArena {
        root: SchemaId(0),
        nodes,
        reference_edges,
    })
}

fn index_schema_locations(
    root: Value,
    limits: &CompileLimits,
) -> Result<Vec<(SchemaPointer, Value)>, CompileError> {
    let mut work = vec![(SchemaPointer(String::new()), root)];
    let mut raw_nodes = Vec::new();
    let mut seen = BTreeSet::new();

    while let Some((pointer, value)) = work.pop() {
        if !seen.insert(pointer.clone()) {
            continue;
        }
        enforce_limit(
            CompileStage::SchemaIndex,
            raw_nodes.len() + 1,
            limits.max_schema_nodes,
        )?;
        let mut children = schema_children(&pointer, &value);
        children.sort_by(|left, right| left.0.cmp(&right.0));
        for child in children.into_iter().rev() {
            work.push(child);
        }
        raw_nodes.push((pointer, value));
    }
    Ok(raw_nodes)
}

fn schema_children(pointer: &SchemaPointer, value: &Value) -> Vec<(SchemaPointer, Value)> {
    let Some(object) = value.as_object() else {
        return Vec::new();
    };
    let mut children = Vec::new();
    for keyword in ["$defs", "properties"] {
        if let Some(entries) = object.get(keyword).and_then(Value::as_object) {
            let mut names: Vec<_> = entries.keys().collect();
            names.sort();
            for name in names {
                if let Some(child) = entries.get(name).filter(|value| is_schema(value)) {
                    children.push((
                        pointer_join(&pointer_join(pointer, keyword), name),
                        child.clone(),
                    ));
                }
            }
        }
    }
    for keyword in ["additionalProperties", "items"] {
        if let Some(child) = object.get(keyword).filter(|value| is_schema(value)) {
            children.push((pointer_join(pointer, keyword), child.clone()));
        }
    }
    for keyword in ["prefixItems", "allOf", "anyOf", "oneOf"] {
        if let Some(entries) = object.get(keyword).and_then(Value::as_array) {
            for (index, child) in entries.iter().enumerate() {
                if is_schema(child) {
                    children.push((
                        pointer_join(&pointer_join(pointer, keyword), &index.to_string()),
                        child.clone(),
                    ));
                }
            }
        }
    }
    children
}

#[allow(clippy::too_many_arguments)]
fn normalize_node(
    id: SchemaId,
    pointer: &SchemaPointer,
    raw: &Value,
    raw_nodes: &[(SchemaPointer, Value)],
    pointer_to_id: &BTreeMap<SchemaPointer, SchemaId>,
    reference_edges: &mut Vec<ReferenceEdge>,
    limits: &CompileLimits,
) -> Result<NormalizedSchema, CompileError> {
    if let Some(value) = raw.as_bool() {
        return Ok(if value {
            NormalizedSchema::Any
        } else {
            NormalizedSchema::Never
        });
    }
    let object = raw
        .as_object()
        .ok_or_else(|| invalid_value(pointer, "schema", "a schema must be a boolean or object"))?;
    validate_keywords(pointer, object)?;
    validate_containers(pointer, object)?;

    if let Some(dialect) = object.get("$schema") {
        let found = dialect.as_str().ok_or_else(|| {
            invalid_value(
                pointer,
                "$schema",
                "the dialect identifier must be a string",
            )
        })?;
        if found != DRAFT_2020_12 && found != format!("{DRAFT_2020_12}#") {
            return Err(CompileError::UnsupportedDialect {
                found: found.to_owned(),
            });
        }
    }

    let semantic_count = ["$ref", "const", "enum", "type", "allOf", "anyOf", "oneOf"]
        .iter()
        .filter(|key| object.contains_key(**key))
        .count();

    if object.contains_key("allOf") {
        if semantic_count != 1 || has_structural_keywords(object) {
            return Err(unsupported_combination(
                pointer,
                "allOf with adjacent assertions is not implemented exactly",
            ));
        }
        return normalize_simple_all_of(pointer, object, raw_nodes, pointer_to_id);
    }
    if object.contains_key("oneOf") {
        return Err(unsupported_combination(
            pointer,
            "exclusive oneOf requires exact Boolean automaton construction",
        ));
    }
    if let Some(branches) = object.get("anyOf") {
        if semantic_count != 1 || has_structural_keywords(object) {
            return Err(unsupported_combination(
                pointer,
                "anyOf with adjacent assertions is not implemented exactly",
            ));
        }
        return Ok(NormalizedSchema::Union(child_ids(
            pointer,
            "anyOf",
            branches,
            pointer_to_id,
        )?));
    }
    if let Some(reference) = object.get("$ref") {
        if semantic_count != 1 || has_structural_keywords(object) {
            return Err(unsupported_combination(
                pointer,
                "$ref siblings with assertions require exact intersection",
            ));
        }
        let reference = reference
            .as_str()
            .ok_or_else(|| invalid_value(pointer, "$ref", "a reference must be a string"))?;
        let target_pointer = resolve_local_pointer(pointer, reference)?;
        let target = pointer_to_id.get(&target_pointer).copied().ok_or_else(|| {
            CompileError::ReferenceNotFound {
                location: keyword_location(pointer, "$ref"),
                reference: reference.to_owned(),
            }
        })?;
        enforce_limit(
            CompileStage::ReferenceResolution,
            reference_edges.len() + 1,
            limits.max_ref_edges,
        )?;
        reference_edges.push(ReferenceEdge {
            from: id,
            to: target,
            provenance: Provenance {
                resource: ResourceId(0),
                pointer: pointer.clone(),
                keyword: Some("$ref".to_owned()),
            },
        });
        return Ok(NormalizedSchema::Ref(target));
    }

    let declared_type = parse_declared_type(pointer, object)?;
    if let Some(constant) = object.get("const") {
        if object.contains_key("enum") || has_structural_keywords(object) {
            return Err(unsupported_combination(
                pointer,
                "const cannot be combined with other value assertions in K1",
            ));
        }
        if let Some(kind) = declared_type {
            if !value_matches_type(constant, kind) {
                return Ok(NormalizedSchema::Never);
            }
        }
        return Ok(NormalizedSchema::Const(canonical_value(constant)?));
    }
    if let Some(values) = object.get("enum") {
        if has_structural_keywords(object) {
            return Err(unsupported_combination(
                pointer,
                "enum cannot be combined with structural assertions in K1",
            ));
        }
        let values = values
            .as_array()
            .ok_or_else(|| invalid_value(pointer, "enum", "enum must be an array"))?;
        if values.is_empty() {
            return Err(invalid_value(pointer, "enum", "enum must not be empty"));
        }
        let mut canonical = BTreeMap::new();
        for value in values {
            if declared_type.is_none_or(|kind| value_matches_type(value, kind)) {
                let value = canonical_value(value)?;
                canonical.entry(value.json.clone()).or_insert(value);
            }
        }
        return if canonical.is_empty() {
            Ok(NormalizedSchema::Never)
        } else {
            Ok(NormalizedSchema::Enum(canonical.into_values().collect()))
        };
    }

    if has_object_keywords(object) {
        if declared_type != Some(JsonType::Object) {
            return Err(unsupported_combination(
                pointer,
                "object keywords require an explicit object type in K1",
            ));
        }
        return normalize_object(pointer, object, pointer_to_id);
    }
    if has_array_keywords(object) {
        if declared_type != Some(JsonType::Array) {
            return Err(unsupported_combination(
                pointer,
                "array keywords require an explicit array type in K1",
            ));
        }
        return normalize_array(pointer, object, pointer_to_id);
    }

    match declared_type {
        None => Ok(NormalizedSchema::Any),
        Some(JsonType::Null) => Ok(NormalizedSchema::Null),
        Some(JsonType::Boolean) => Ok(NormalizedSchema::Boolean),
        Some(JsonType::String) => normalize_string(pointer, object),
        Some(JsonType::Number) => Ok(NormalizedSchema::Number),
        Some(JsonType::Integer) => Ok(NormalizedSchema::Integer),
        Some(JsonType::Array) => normalize_array(pointer, object, pointer_to_id),
        Some(JsonType::Object) => normalize_object(pointer, object, pointer_to_id),
    }
}

fn validate_keywords(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
) -> Result<(), CompileError> {
    const SUPPORTED: &[&str] = &[
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "const",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "allOf",
        "anyOf",
        "oneOf",
    ];
    for keyword in object.keys() {
        if !SUPPORTED.contains(&keyword.as_str()) {
            return Err(CompileError::UnsupportedKeyword {
                location: keyword_location(pointer, keyword),
                keyword: keyword.clone(),
            });
        }
    }
    Ok(())
}

fn validate_containers(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
) -> Result<(), CompileError> {
    if let Some(definitions) = object.get("$defs") {
        let definitions = definitions
            .as_object()
            .ok_or_else(|| invalid_value(pointer, "$defs", "$defs must be an object"))?;
        for (name, schema) in definitions {
            if !is_schema(schema) {
                return Err(invalid_value(
                    &pointer_join(&pointer_join(pointer, "$defs"), name),
                    "$defs",
                    "definition must be a boolean or object schema",
                ));
            }
        }
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| invalid_value(pointer, "properties", "properties must be an object"))?;
        for (name, schema) in properties {
            if !is_schema(schema) {
                return Err(invalid_value(
                    &pointer_join(&pointer_join(pointer, "properties"), name),
                    "properties",
                    "property value must be a boolean or object schema",
                ));
            }
        }
    }
    for keyword in ["additionalProperties", "items"] {
        if let Some(value) = object.get(keyword) {
            if !is_schema(value) {
                return Err(invalid_value(
                    pointer,
                    keyword,
                    "value must be a boolean or object schema",
                ));
            }
        }
    }
    for keyword in ["prefixItems", "allOf", "anyOf", "oneOf"] {
        if let Some(value) = object.get(keyword) {
            let entries = value
                .as_array()
                .ok_or_else(|| invalid_value(pointer, keyword, "value must be an array"))?;
            if entries.is_empty() && matches!(keyword, "allOf" | "anyOf" | "oneOf") {
                return Err(invalid_value(pointer, keyword, "array must not be empty"));
            }
            if entries.iter().any(|entry| !is_schema(entry)) {
                return Err(invalid_value(
                    pointer,
                    keyword,
                    "every entry must be a boolean or object schema",
                ));
            }
        }
    }
    Ok(())
}

fn normalize_simple_all_of(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
    raw_nodes: &[(SchemaPointer, Value)],
    pointer_to_id: &BTreeMap<SchemaPointer, SchemaId>,
) -> Result<NormalizedSchema, CompileError> {
    let ids = child_ids(pointer, "allOf", &object["allOf"], pointer_to_id)?;
    let mut constant = None;
    let mut required_type = None;
    for id in ids {
        let raw = &raw_nodes[id.0 as usize].1;
        let branch = raw.as_object().ok_or_else(|| {
            unsupported_combination(pointer, "boolean allOf branches require Boolean operations")
        })?;
        if let Some(value) = branch.get("const") {
            if branch
                .keys()
                .any(|key| !is_annotation(key) && key != "const")
                || constant.is_some()
            {
                return Err(unsupported_combination(
                    pointer,
                    "allOf is supported only for one const intersected with simple types",
                ));
            }
            constant = Some(value);
        } else if let Some(kind) = parse_declared_type(&raw_nodes[id.0 as usize].0, branch)? {
            if branch
                .keys()
                .any(|key| !is_annotation(key) && key != "type")
            {
                return Err(unsupported_combination(
                    pointer,
                    "allOf simple type branches cannot contain other assertions",
                ));
            }
            if required_type
                .replace(kind)
                .is_some_and(|previous| previous != kind)
            {
                return Ok(NormalizedSchema::Never);
            }
        } else {
            return Err(unsupported_combination(
                pointer,
                "allOf requires exact intersection not available for these branches",
            ));
        }
    }
    let constant = constant.ok_or_else(|| {
        unsupported_combination(
            pointer,
            "allOf without a finite const branch requires DFA intersection",
        )
    })?;
    if required_type.is_none_or(|kind| value_matches_type(constant, kind)) {
        Ok(NormalizedSchema::Const(canonical_value(constant)?))
    } else {
        Ok(NormalizedSchema::Never)
    }
}

fn normalize_string(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
) -> Result<NormalizedSchema, CompileError> {
    let min_length = optional_usize(pointer, object, "minLength")?.unwrap_or(0);
    let max_length = optional_usize(pointer, object, "maxLength")?;
    if max_length.is_some_and(|maximum| maximum < min_length) {
        return Ok(NormalizedSchema::Never);
    }
    Ok(NormalizedSchema::String(StringConstraints {
        min_length,
        max_length,
    }))
}

fn normalize_array(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
    pointer_to_id: &BTreeMap<SchemaPointer, SchemaId>,
) -> Result<NormalizedSchema, CompileError> {
    let min_items = optional_usize(pointer, object, "minItems")?.unwrap_or(0);
    let max_items = optional_usize(pointer, object, "maxItems")?;
    if max_items.is_some_and(|maximum| maximum < min_items) {
        return Ok(NormalizedSchema::Never);
    }
    let prefix_items = match object.get("prefixItems") {
        Some(value) => child_ids(pointer, "prefixItems", value, pointer_to_id)?,
        None => Vec::new(),
    };
    let items = object
        .get("items")
        .map(|_| pointer_join(pointer, "items"))
        .map(|child| {
            pointer_to_id
                .get(&child)
                .copied()
                .ok_or(CompileError::InternalInvariant {
                    message: "indexed items schema is missing",
                })
        })
        .transpose()?;
    Ok(NormalizedSchema::Array(ArrayConstraints {
        prefix_items,
        items,
        min_items,
        max_items,
    }))
}

fn normalize_object(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
    pointer_to_id: &BTreeMap<SchemaPointer, SchemaId>,
) -> Result<NormalizedSchema, CompileError> {
    let mut properties = BTreeMap::new();
    if let Some(raw_properties) = object.get("properties").and_then(Value::as_object) {
        for name in raw_properties.keys() {
            let child = pointer_join(&pointer_join(pointer, "properties"), name);
            let id = pointer_to_id
                .get(&child)
                .copied()
                .ok_or(CompileError::InternalInvariant {
                    message: "indexed property schema is missing",
                })?;
            properties.insert(name.clone(), id);
        }
    }
    let mut required = BTreeSet::new();
    if let Some(raw_required) = object.get("required") {
        let entries = raw_required
            .as_array()
            .ok_or_else(|| invalid_value(pointer, "required", "required must be an array"))?;
        for entry in entries {
            let name = entry.as_str().ok_or_else(|| {
                invalid_value(pointer, "required", "required entries must be strings")
            })?;
            if !required.insert(name.to_owned()) {
                return Err(invalid_value(
                    pointer,
                    "required",
                    "required entries must be unique",
                ));
            }
        }
    }
    let additional_properties = match object.get("additionalProperties") {
        None | Some(Value::Bool(true)) => AdditionalProperties::Unconstrained,
        Some(Value::Bool(false)) => AdditionalProperties::Forbidden,
        Some(_) => {
            let child = pointer_join(pointer, "additionalProperties");
            AdditionalProperties::Schema(pointer_to_id.get(&child).copied().ok_or(
                CompileError::InternalInvariant {
                    message: "indexed additionalProperties schema is missing",
                },
            )?)
        }
    };
    Ok(NormalizedSchema::Object(ObjectConstraints {
        properties,
        required,
        additional_properties,
    }))
}

fn parse_declared_type(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
) -> Result<Option<JsonType>, CompileError> {
    let Some(raw_type) = object.get("type") else {
        return Ok(None);
    };
    let name = raw_type.as_str().ok_or_else(|| {
        invalid_value(
            pointer,
            "type",
            "K1 requires type to be a single string",
        )
    })?;
    let kind = match name {
        "null" => JsonType::Null,
        "boolean" => JsonType::Boolean,
        "string" => JsonType::String,
        "number" => JsonType::Number,
        "integer" => JsonType::Integer,
        "array" => JsonType::Array,
        "object" => JsonType::Object,
        _ => {
            return Err(invalid_value(pointer, "type", "unknown JSON Schema type"));
        }
    };
    Ok(Some(kind))
}

fn child_ids(
    pointer: &SchemaPointer,
    keyword: &str,
    value: &Value,
    pointer_to_id: &BTreeMap<SchemaPointer, SchemaId>,
) -> Result<Vec<SchemaId>, CompileError> {
    let entries = value
        .as_array()
        .ok_or_else(|| invalid_value(pointer, keyword, "value must be an array"))?;
    let mut ids = Vec::new();
    ids.try_reserve(entries.len())
        .map_err(|_| CompileError::ResourceLimitExceeded {
            stage: CompileStage::Normalization,
            observed: entries.len(),
            limit: entries.len().saturating_sub(1),
        })?;
    for index in 0..entries.len() {
        let child = pointer_join(&pointer_join(pointer, keyword), &index.to_string());
        ids.push(
            pointer_to_id
                .get(&child)
                .copied()
                .ok_or(CompileError::InternalInvariant {
                    message: "indexed applicator schema is missing",
                })?,
        );
    }
    Ok(ids)
}

fn resolve_local_pointer(
    source: &SchemaPointer,
    reference: &str,
) -> Result<SchemaPointer, CompileError> {
    if !reference.starts_with('#') {
        return Err(CompileError::RemoteReferenceUnsupported {
            location: keyword_location(source, "$ref"),
            reference: reference.to_owned(),
        });
    }
    let decoded =
        percent_decode(&reference[1..]).map_err(|reason| CompileError::InvalidJsonPointer {
            location: keyword_location(source, "$ref"),
            pointer: reference.to_owned(),
            reason,
        })?;
    if decoded.is_empty() {
        return Ok(SchemaPointer(String::new()));
    }
    if !decoded.starts_with('/') {
        return Err(CompileError::InvalidJsonPointer {
            location: keyword_location(source, "$ref"),
            pointer: reference.to_owned(),
            reason: "only JSON Pointer fragments are supported".to_owned(),
        });
    }
    let mut canonical = String::new();
    for raw_segment in decoded[1..].split('/') {
        let segment = decode_pointer_segment(raw_segment).map_err(|reason| {
            CompileError::InvalidJsonPointer {
                location: keyword_location(source, "$ref"),
                pointer: reference.to_owned(),
                reason,
            }
        })?;
        canonical.push('/');
        canonical.push_str(&escape_pointer_segment(&segment));
    }
    Ok(SchemaPointer(canonical))
}

fn percent_decode(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err("truncated percent escape".to_owned());
            }
            let high = hex(bytes[index + 1]).ok_or_else(|| "invalid percent escape".to_owned())?;
            let low = hex(bytes[index + 2]).ok_or_else(|| "invalid percent escape".to_owned())?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| "fragment is not valid UTF-8".to_owned())
}

fn decode_pointer_segment(value: &str) -> Result<String, String> {
    let mut decoded = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(character) = chars.next() {
        if character != '~' {
            decoded.push(character);
            continue;
        }
        match chars.next() {
            Some('0') => decoded.push('~'),
            Some('1') => decoded.push('/'),
            _ => return Err("invalid RFC 6901 escape".to_owned()),
        }
    }
    Ok(decoded)
}

fn canonical_value(value: &Value) -> Result<CanonicalValue, CompileError> {
    let mut output = String::new();
    write_canonical(value, &mut output)?;
    Ok(CanonicalValue { json: output })
}

fn write_canonical(value: &Value, output: &mut String) -> Result<(), CompileError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => {
            output.push_str(&serde_json::to_string(value).map_err(|error| {
                CompileError::InvalidJson {
                    message: error.to_string(),
                }
            })?)
        }
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_canonical(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).map_err(|error| {
                    CompileError::InvalidJson {
                        message: error.to_string(),
                    }
                })?);
                output.push(':');
                write_canonical(&values[key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn value_matches_type(value: &Value, kind: JsonType) -> bool {
    match kind {
        JsonType::Null => value.is_null(),
        JsonType::Boolean => value.is_boolean(),
        JsonType::String => value.is_string(),
        JsonType::Number => value.is_number(),
        JsonType::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
        JsonType::Array => value.is_array(),
        JsonType::Object => value.is_object(),
    }
}

fn optional_usize(
    pointer: &SchemaPointer,
    object: &Map<String, Value>,
    keyword: &str,
) -> Result<Option<usize>, CompileError> {
    let Some(value) = object.get(keyword) else {
        return Ok(None);
    };
    let value = value
        .as_u64()
        .ok_or_else(|| invalid_value(pointer, keyword, "value must be a non-negative integer"))?;
    usize::try_from(value)
        .map(Some)
        .map_err(|_| invalid_value(pointer, keyword, "value is too large for this platform"))
}

fn enforce_limit(stage: CompileStage, observed: usize, limit: usize) -> Result<(), CompileError> {
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

fn has_structural_keywords(object: &Map<String, Value>) -> bool {
    has_object_keywords(object) || has_array_keywords(object)
}

fn has_object_keywords(object: &Map<String, Value>) -> bool {
    ["properties", "required", "additionalProperties"]
        .iter()
        .any(|key| object.contains_key(*key))
}

fn has_array_keywords(object: &Map<String, Value>) -> bool {
    ["items", "prefixItems", "minItems", "maxItems"]
        .iter()
        .any(|key| object.contains_key(*key))
}

fn is_annotation(keyword: &str) -> bool {
    matches!(
        keyword,
        "$schema"
            | "$id"
            | "$defs"
            | "$comment"
            | "title"
            | "description"
            | "default"
            | "examples"
            | "deprecated"
            | "readOnly"
            | "writeOnly"
    )
}

fn is_schema(value: &Value) -> bool {
    value.is_boolean() || value.is_object()
}

fn pointer_join(pointer: &SchemaPointer, segment: &str) -> SchemaPointer {
    SchemaPointer(format!("{}/{}", pointer.0, escape_pointer_segment(segment)))
}

fn escape_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn keyword_location(pointer: &SchemaPointer, keyword: &str) -> DiagnosticLocation {
    DiagnosticLocation {
        resource: ResourceId(0),
        pointer: pointer.clone(),
        keyword: Some(keyword.to_owned()),
    }
}

fn root_location() -> DiagnosticLocation {
    DiagnosticLocation {
        resource: ResourceId(0),
        pointer: SchemaPointer(String::new()),
        keyword: None,
    }
}

fn invalid_value(pointer: &SchemaPointer, keyword: &str, reason: &str) -> CompileError {
    CompileError::InvalidKeywordValue {
        location: keyword_location(pointer, keyword),
        keyword: keyword.to_owned(),
        reason: reason.to_owned(),
    }
}

fn unsupported_combination(pointer: &SchemaPointer, reason: &str) -> CompileError {
    CompileError::UnsupportedCombination {
        location: DiagnosticLocation {
            resource: ResourceId(0),
            pointer: pointer.clone(),
            keyword: None,
        },
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(schema: &str) -> Result<SchemaArena, CompileError> {
        parse_and_normalize(schema.as_bytes(), &CompileOptions::default())
    }

    #[test]
    fn rejects_duplicate_keys() {
        let error = parse(r#"{"type":"string","type":"number"}"#).unwrap_err();
        assert!(matches!(error, CompileError::DuplicateSchemaKey { .. }));
    }

    #[test]
    fn resolves_deep_acyclic_chain_without_depth_limit() {
        let schema = include_str!("../testdata/regressions/deep_acyclic_ref.json");
        let arena = parse(schema).unwrap();
        assert_eq!(arena.reference_edges.len(), 5);
        assert!(matches!(arena.nodes[0].kind, NormalizedSchema::Ref(_)));
    }

    #[test]
    fn preserves_recursive_optional_property() {
        let schema = include_str!("../testdata/regressions/recursive_optional_property.json");
        let arena = parse(schema).unwrap();
        assert!(arena.reference_edges.iter().any(|edge| {
            arena.nodes[edge.from.0 as usize].provenance.pointer.0 == "/$defs/node/properties/next"
                && arena.nodes[edge.to.0 as usize].provenance.pointer.0 == "/$defs/node"
        }));
        let node = arena
            .nodes
            .iter()
            .find(|node| node.provenance.pointer.0 == "/$defs/node")
            .unwrap();
        let NormalizedSchema::Object(object) = &node.kind else {
            panic!("node must remain an object");
        };
        assert!(object.properties.contains_key("next"));
    }

    #[test]
    fn all_of_is_intersection_not_concatenation() {
        let schema = include_str!("../testdata/regressions/allof_intersection.json");
        let arena = parse(schema).unwrap();
        assert_eq!(
            arena.nodes[0].kind,
            NormalizedSchema::Const(CanonicalValue {
                json: "\"x\"".to_owned()
            })
        );
    }

    #[test]
    fn overlapping_one_of_is_explicitly_unsupported() {
        let schema = include_str!("../testdata/regressions/oneof_overlap.json");
        assert!(matches!(
            parse(schema),
            Err(CompileError::UnsupportedCombination { .. })
        ));
    }

    #[test]
    fn resolves_pointer_escapes() {
        let arena =
            parse(r##"{"$defs":{"a/b":{"const":1},"a~b":{"const":2}},"$ref":"#/$defs/a~1b"}"##)
                .unwrap();
        assert_eq!(arena.reference_edges.len(), 1);
        let arena = parse(r##"{"$defs":{"a~b":{"const":2}},"$ref":"#/$defs/a~0b"}"##).unwrap();
        assert_eq!(arena.reference_edges.len(), 1);
    }

    #[test]
    fn tiny_node_budget_returns_typed_error() {
        let mut options = CompileOptions::default();
        options.limits.max_schema_nodes = 1;
        let result = parse_and_normalize(
            br##"{"$defs":{"x":{"const":1}},"$ref":"#/$defs/x"}"##,
            &options,
        );
        assert!(matches!(
            result,
            Err(CompileError::ResourceLimitExceeded {
                stage: CompileStage::SchemaIndex,
                ..
            })
        ));
    }

    #[test]
    fn canonical_objects_sort_keys() {
        let value: Value = serde_json::from_str(r#"{"z":1,"a":[true,null]}"#).unwrap();
        assert_eq!(
            canonical_value(&value).unwrap().json,
            r#"{"a":[true,null],"z":1}"#
        );
    }
}
