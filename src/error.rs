//! The Errors that may occur within the crate.

use thiserror::Error;

use crate::schema::{DiagnosticLocation, SchemaPointer};

pub type Result<T, E = crate::Error> = std::result::Result<T, E>;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Compile(#[from] CompileError),
    // Index Errors
    #[error("Failed to build DFA {0}")]
    IndexDfaError(#[from] Box<regex_automata::dfa::dense::BuildError>),
    #[error("Index failed since anchored universal start state doesn't exist")]
    DfaHasNoStartState,
    // Vocabulary Errors
    #[error("EOS token should not be inserted into Vocabulary")]
    EOSTokenDisallowed,
    #[error("token ID {token_id} maps to byte strings that reach different DFA states")]
    AmbiguousTokenId { token_id: u32 },
    #[error(transparent)]
    TokenizersError(#[from] tokenizers::Error),
    #[error("Unsupported tokenizer for {model}: {reason}, please open an issue with the full error message: https://github.com/dhanavanthesh/oc-earley/issues")]
    UnsupportedTokenizer { model: String, reason: String },
    #[error("Unable to locate EOS token for {model}")]
    UnableToLocateEosTokenId { model: String },
    #[error("Tokenizer is not supported by token processor")]
    UnsupportedByTokenProcessor,
    #[error("Decoder unpacking failed for token processor")]
    DecoderUnpackingFailed,
    #[error("Token processing failed for byte level processor")]
    ByteProcessorFailed,
    #[error("Token processing failed for byte fallback level processor")]
    ByteFallbackProcessorFailed,
    // Json Schema errors
    #[error("serde json error")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("Unsupported JSON Schema structure {0} \nMake sure it is valid to the JSON Schema specification and supported by OC-Earley.\nIf it should be supported, please open an issue.")]
    UnsupportedJsonSchema(Box<serde_json::Value>),
    #[error("'properties' not found or not an object")]
    PropertiesNotFound,
    #[error("'allOf' must be an array")]
    AllOfMustBeAnArray,
    #[error("'anyOf' must be an array")]
    AnyOfMustBeAnArray,
    #[error("'oneOf' must be an array")]
    OneOfMustBeAnArray,
    #[error("'prefixItems' must be an array")]
    PrefixItemsMustBeAnArray,
    #[error("Unsupported data type in enum: {0}")]
    UnsupportedEnumDataType(Box<serde_json::Value>),
    #[error("'enum' must be an array")]
    EnumMustBeAnArray,
    #[error("Unsupported data type in const: {0}")]
    UnsupportedConstDataType(Box<serde_json::Value>),
    #[error("'const' key not found in object")]
    ConstKeyNotFound,
    #[error("'$ref' must be a string")]
    RefMustBeAString,
    #[error("External references are not supported: {0}")]
    ExternalReferencesNotSupported(Box<str>),
    #[error("Invalid reference format: {0}")]
    InvalidReferenceFormat(Box<str>),
    #[error("'type' must be a string or an array of string")]
    TypeMustBeAStringOrArray,
    #[error("Unsupported type: {0}")]
    UnsupportedType(Box<str>),
    #[error("maxLength must be greater than or equal to minLength")]
    MaxBoundError,
    #[error("Numeric bound '{0}' is not supported by OC-Earley")]
    UnsupportedNumericBound(Box<str>),
    #[error("Format {0} is not supported by OC-Earley")]
    StringTypeUnsupportedFormat(Box<str>),
    #[error("Invalid reference path: {0}")]
    InvalidRefecencePath(Box<str>),
    #[error("Ref recusion limit reached: {0}")]
    RefRecursionLimitReached(usize),
    #[error("The vocabulary provided is incompatible with the regex '{regex}'. Found no transitions from state {error_state}, missing tokens corresponding to at least one of the following characters: {missing_tokens:?}. This may be due to an encoding issue in your vocabulary.")]
    IncompatibleVocabulary {
        regex: String,
        error_state: u32,
        missing_tokens: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileStage {
    Parse,
    SchemaIndex,
    ReferenceResolution,
    Normalization,
    Lowering,
    GrammarReduction,
    SccAnalysis,
    RegularCertification,
    NfaConstruction,
    DfaDeterminization,
    VocabularyProjection,
    ResidualGrammar,
    TerminalCompilation,
    LrConstruction,
    LalrMerge,
    LalrTable,
    EarleyPreparation,
    LeoPreparation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeResource {
    InputBytes,
    ParseStack,
    ChartColumns,
    ItemsPerColumn,
    TotalItems,
    ActiveScans,
    LeoItems,
    CheckpointHistory,
}

impl std::fmt::Display for RuntimeResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum RuntimeError {
    #[error("byte 0x{byte:02x} is rejected at position {position}")]
    RejectedByte { position: u32, byte: u8 },
    #[error("runtime limit exceeded for {resource}: {observed} > {limit}")]
    ResourceLimitExceeded {
        resource: RuntimeResource,
        observed: usize,
        limit: usize,
    },
    #[error("checkpoint generation {found_generation} does not match {expected_generation}")]
    InvalidCheckpoint {
        expected_generation: u64,
        found_generation: u64,
    },
    #[error("runtime invariant failed: {message}")]
    InternalInvariant { message: &'static str },
}

impl std::fmt::Display for CompileStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("invalid JSON: {message}")]
    InvalidJson { message: String },
    #[error("duplicate schema key `{key}` at {location}")]
    DuplicateSchemaKey {
        location: DiagnosticLocation,
        key: String,
    },
    #[error("unsupported dialect `{found}`")]
    UnsupportedDialect { found: String },
    #[error("invalid value for `{keyword}` at {location}: {reason}")]
    InvalidKeywordValue {
        location: DiagnosticLocation,
        keyword: String,
        reason: String,
    },
    #[error("unsupported keyword `{keyword}` at {location}")]
    UnsupportedKeyword {
        location: DiagnosticLocation,
        keyword: String,
    },
    #[error("unsupported keyword combination at {location}: {reason}")]
    UnsupportedCombination {
        location: DiagnosticLocation,
        reason: String,
    },
    #[error("remote reference is unsupported at {location}: {reference}")]
    RemoteReferenceUnsupported {
        location: DiagnosticLocation,
        reference: String,
    },
    #[error("reference target not found at {location}: {reference}")]
    ReferenceNotFound {
        location: DiagnosticLocation,
        reference: String,
    },
    #[error("invalid JSON Pointer `{pointer}` at {location}: {reason}")]
    InvalidJsonPointer {
        location: DiagnosticLocation,
        pointer: String,
        reason: String,
    },
    #[error("resource limit exceeded during {stage}: {observed} > {limit}")]
    ResourceLimitExceeded {
        stage: CompileStage,
        observed: usize,
        limit: usize,
    },
    #[error("a structural backend is required at {location}")]
    StructuralBackendRequired { location: DiagnosticLocation },
    #[error("backend `{backend}` is unavailable at {location}: {reason}")]
    BackendUnavailable {
        backend: &'static str,
        location: DiagnosticLocation,
        reason: String,
    },
    #[error("compiled format version {found} is unsupported; expected {expected}")]
    IncompatibleCompiledFormat { found: u32, expected: u32 },
    #[error("regular automaton construction failed at {pointer}: {message}")]
    AutomatonBuild {
        pointer: SchemaPointer,
        message: String,
    },
    #[error("internal invariant failed: {message}")]
    InternalInvariant { message: &'static str },
}

impl Error {
    pub fn is_recursion_limit(&self) -> bool {
        matches!(self, Self::RefRecursionLimitReached(_))
    }
}

#[cfg(feature = "python-bindings")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        use pyo3::exceptions::PyValueError;
        use pyo3::PyErr;
        PyErr::new::<PyValueError, _>(e.to_string())
    }
}
