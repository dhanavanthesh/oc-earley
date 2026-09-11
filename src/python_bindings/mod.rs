//! Provides tools and interfaces to integrate the crate's functionality with Python.

use std::collections::VecDeque;
use std::sync::Arc;

use bincode::{config, Decode, Encode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::wrap_pyfunction;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
#[cfg(feature = "hugginface-hub")]
use tokenizers::FromPretrainedParameters;

use crate::index::Index;
use crate::json_schema;
use crate::prelude::*;
use crate::schema::COMPILED_FORMAT_VERSION;

const SERIAL_MAGIC: &[u8; 8] = b"OCEARLEY";
const SERIAL_HEADER_LEN: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum ObjectKind {
    Index = 1,
    Guide = 2,
    Vocabulary = 3,
    CompiledSchema = 4,
}

fn encode_object<T: Encode>(value: &T, kind: ObjectKind, label: &str) -> PyResult<Vec<u8>> {
    let payload = bincode::encode_to_vec(value, config::standard()).map_err(|error| {
        PyValueError::new_err(format!("Serialization of {label} failed: {error}"))
    })?;
    let capacity = SERIAL_HEADER_LEN
        .checked_add(payload.len())
        .ok_or_else(|| PyValueError::new_err(format!("Serialization of {label} is too large")))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| PyValueError::new_err(format!("Serialization of {label} is too large")))?;
    output.extend_from_slice(SERIAL_MAGIC);
    output.extend_from_slice(&COMPILED_FORMAT_VERSION.to_le_bytes());
    output.push(kind as u8);
    output.extend_from_slice(&payload);
    Ok(output)
}

fn decode_object<T: Decode<()>>(
    binary_data: &[u8],
    expected_kind: ObjectKind,
    label: &str,
) -> PyResult<T> {
    if binary_data.len() < SERIAL_HEADER_LEN {
        return Err(PyValueError::new_err(format!(
            "Deserialization of {label} failed: truncated header"
        )));
    }
    if &binary_data[..SERIAL_MAGIC.len()] != SERIAL_MAGIC {
        return Err(PyValueError::new_err(format!(
            "Deserialization of {label} failed: invalid magic bytes"
        )));
    }
    let mut version_bytes = [0; 4];
    version_bytes.copy_from_slice(&binary_data[8..12]);
    let version = u32::from_le_bytes(version_bytes);
    if version != COMPILED_FORMAT_VERSION {
        return Err(PyValueError::new_err(format!(
            "compiled format version {version} is unsupported; expected {COMPILED_FORMAT_VERSION}"
        )));
    }
    if binary_data[12] != expected_kind as u8 {
        return Err(PyValueError::new_err(format!(
            "Deserialization of {label} failed: wrong object kind"
        )));
    }
    let payload = &binary_data[SERIAL_HEADER_LEN..];
    let (value, consumed): (T, usize) = bincode::decode_from_slice(payload, config::standard())
        .map_err(|error| {
            PyValueError::new_err(format!("Deserialization of {label} failed: {error}"))
        })?;
    if consumed != payload.len() {
        return Err(PyValueError::new_err(format!(
            "Deserialization of {label} failed: trailing data"
        )));
    }
    Ok(value)
}

macro_rules! type_name {
    ($obj:expr) => {
        // Safety: obj is always initialized and tp_name is a C-string
        unsafe { std::ffi::CStr::from_ptr((&*(&*$obj.as_ptr()).ob_type).tp_name) }
    };
}

/// Guide object based on Index.
#[pyclass(name = "Guide", module = "oc_earley", from_py_object)]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyGuide {
    state: StateId,
    index: PyIndex,
    state_cache: VecDeque<StateId>,
}

#[pymethods]
impl PyGuide {
    /// Creates a Guide object based on Index.
    #[new]
    #[pyo3(signature = (index, max_rollback=32))]
    fn __new__(index: PyIndex, max_rollback: usize) -> Self {
        PyGuide {
            state: index.get_initial_state(),
            index,
            state_cache: VecDeque::with_capacity(max_rollback),
        }
    }

    /// Retrieves current state id of the Guide.
    fn get_state(&self) -> StateId {
        self.state
    }

    /// Gets the list of allowed tokens for the current state.
    fn get_tokens(&self) -> PyResult<Vec<TokenId>> {
        self.index
            .get_allowed_tokens(self.state)
            // Since Guide advances only through the states offered by the Index, it means
            // None here shouldn't happen and it's an issue at Index creation step
            .ok_or(PyErr::new::<PyValueError, _>(format!(
                "No allowed tokens available for the state {}",
                self.state
            )))
    }

    /// Get the number of rollback steps available.
    fn get_allowed_rollback(&self) -> usize {
        self.state_cache.len()
    }

    /// Guide moves to the next state provided by the token id and returns a list of allowed tokens, unless return_tokens is False.
    #[pyo3(signature = (token_id, return_tokens=None))]
    fn advance(
        &mut self,
        token_id: TokenId,
        return_tokens: Option<bool>,
    ) -> PyResult<Option<Vec<TokenId>>> {
        match self.index.get_next_state(self.state, token_id) {
            Some(new_state) => {
                // Free up space in state_cache if needed.
                if self.state_cache.len() == self.state_cache.capacity() {
                    self.state_cache.pop_front();
                }
                self.state_cache.push_back(self.state);
                self.state = new_state;
                if return_tokens.unwrap_or(true) {
                    self.get_tokens().map(Some)
                } else {
                    Ok(None)
                }
            }
            None => Err(PyErr::new::<PyValueError, _>(format!(
                "No next state found for the current state: {} with token ID: {token_id}",
                self.state
            ))),
        }
    }

    /// Rollback the Guide state `n` tokens (states).
    /// Fails if `n` is greater than stored prior states.
    fn rollback_state(&mut self, n: usize) -> PyResult<()> {
        if n == 0 {
            return Ok(());
        }
        if n > self.get_allowed_rollback() {
            return Err(PyValueError::new_err(format!(
                "Cannot roll back {n} step(s): only {available} states stored (max_rollback = {cap}). \
                 You must advance through at least {n} state(s) before rolling back {n} step(s).",
                 cap = self.state_cache.capacity(),
                 available = self.get_allowed_rollback(),
            )));
        }
        let mut new_state: u32 = self.state;
        for _ in 0..n {
            new_state = self
                .state_cache
                .pop_back()
                .ok_or_else(|| PyValueError::new_err("Rollback history changed during rollback"))?;
        }
        self.state = new_state;
        Ok(())
    }

    // Returns a boolean indicating if the sequence leads to a valid state in the DFA
    fn accepts_tokens(&self, sequence: Vec<u32>) -> bool {
        let mut state = self.state;
        for t in sequence {
            match self.index.get_next_state(state, t) {
                Some(s) => state = s,
                None => return false,
            }
        }
        true
    }

    /// Checks if the automaton is in a final state.
    fn is_finished(&self) -> bool {
        self.index.is_final_state(self.state)
    }

    /// Write the mask of allowed tokens into the memory specified by data_ptr.
    /// Size of the memory to be written to is indicated by `numel`, and `element_size`.
    /// `element_size` must be 4.
    ///
    /// `data_ptr` should be the data ptr to a `torch.tensor`, or `np.ndarray`, `mx.array` or other
    /// contiguous memory array.
    fn write_mask_into(&self, data_ptr: usize, numel: usize, element_size: usize) -> PyResult<()> {
        let expected_elements = self.index.0.vocab_size().div_ceil(32);
        if element_size != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid element size: got {} bytes per element, expected 4 bytes (32-bit integer).",
                    element_size
                ),
            ));
        } else if data_ptr == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid data pointer: received a null pointer.",
            ));
        } else if data_ptr % 4 != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid data pointer alignment: pointer address {} is not a multiple of 4.",
                data_ptr
            )));
        }
        let byte_len = numel
            .checked_mul(element_size)
            .ok_or_else(|| PyValueError::new_err("Invalid buffer size: byte length overflowed."))?;
        let expected_bytes = expected_elements.checked_mul(4).ok_or_else(|| {
            PyValueError::new_err("Invalid buffer size: expected byte length overflowed.")
        })?;
        if byte_len > isize::MAX as usize || data_ptr.checked_add(byte_len).is_none() {
            return Err(PyValueError::new_err(
                "Invalid buffer size: address range overflowed.",
            ));
        }
        if numel < expected_elements {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Invalid buffer size: got {} elements ({} bytes), expected {} elements ({} bytes). \
                    Ensure that the mask tensor has shape (1, (vocab_size + 31) // 32) and uses 32-bit integers.",
                    numel,
                    byte_len,
                    expected_elements,
                    expected_bytes
                )
            ));
        }
        // Safety: the caller provides a writable aligned buffer; range checks above prevent overflow.
        let slice = unsafe { std::slice::from_raw_parts_mut(data_ptr as *mut u32, numel) };
        slice.fill(0);
        if let Some(tokens) = self.index.0.allowed_tokens_iter(&self.state) {
            for &token in tokens {
                let token = usize::try_from(token)
                    .map_err(|_| PyValueError::new_err("Token ID does not fit usize."))?;
                let bucket = token / 32;
                if bucket < slice.len() {
                    slice[bucket] |= 1 << (token % 32);
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.state = self.index.get_initial_state();
        self.state_cache.clear();
    }

    /// Gets the debug string representation of the guide.
    fn __repr__(&self) -> String {
        format!(
            "Guide object with the state={:#?} and {:#?}",
            self.state, self.index
        )
    }

    /// Gets the string representation of the guide.
    fn __str__(&self) -> String {
        format!(
            "Guide object with the state={} and {}",
            self.state, self.index.0
        )
    }

    /// Compares whether two guides are the same.
    fn __eq__(&self, other: &PyGuide) -> bool {
        self == other
    }

    fn __reduce__(&self) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        Python::attach(|py| {
            let cls = PyModule::import(py, "oc_earley")?.getattr("Guide")?;
            let binary_data = encode_object(self, ObjectKind::Guide, "Guide")?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        decode_object(&binary_data, ObjectKind::Guide, "Guide")
    }
}

/// Index object based on regex and vocabulary.
#[pyclass(name = "Index", module = "oc_earley", frozen, from_py_object)]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyIndex(Arc<Index>);

#[pymethods]
impl PyIndex {
    /// Creates an index from a regex and vocabulary.
    #[new]
    fn __new__(py: Python<'_>, regex: &str, vocabulary: &PyVocabulary) -> PyResult<Self> {
        py.detach(|| {
            Index::new(regex, &vocabulary.0)
                .map(|x| PyIndex(Arc::new(x)))
                .map_err(Into::into)
        })
    }

    /// Returns allowed tokens in this state.
    fn get_allowed_tokens(&self, state: StateId) -> Option<Vec<TokenId>> {
        self.0.allowed_tokens(&state)
    }

    /// Updates the state.
    fn get_next_state(&self, state: StateId, token_id: TokenId) -> Option<StateId> {
        self.0.next_state(&state, &token_id)
    }

    /// Determines whether the current state is a final state.
    fn is_final_state(&self, state: StateId) -> bool {
        self.0.is_final_state(&state)
    }

    /// Get all final states.
    fn get_final_states(&self) -> HashSet<StateId> {
        self.0.final_states().clone()
    }

    /// Returns the Index as a Python Dict object.
    fn get_transitions(&self) -> HashMap<StateId, HashMap<TokenId, StateId>> {
        self.0.transitions().clone()
    }

    /// Returns the ID of the initial state of the index.
    fn get_initial_state(&self) -> StateId {
        self.0.initial_state()
    }

    /// Gets the debug string representation of the index.
    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    /// Gets the string representation of the index.
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Compares whether two indexes are the same.
    fn __eq__(&self, other: &PyIndex) -> bool {
        *self.0 == *other.0
    }

    /// Makes a deep copy of the Index.
    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyIndex(Arc::new((*self.0).clone()))
    }

    fn __reduce__(&self) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        Python::attach(|py| {
            let cls = PyModule::import(py, "oc_earley")?.getattr("Index")?;
            let binary_data = encode_object(&self.0, ObjectKind::Index, "Index")?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let index = decode_object(&binary_data, ObjectKind::Index, "Index")?;
        Ok(PyIndex(Arc::new(index)))
    }
}

/// LLM vocabulary.
#[pyclass(name = "Vocabulary", module = "oc_earley", from_py_object)]
#[derive(Clone, Debug, Encode, Decode)]
pub struct PyVocabulary(Vocabulary);

#[pymethods]
impl PyVocabulary {
    /// Creates a vocabulary from eos token id and a map of tokens to token ids.
    #[new]
    fn __new__(py: Python<'_>, eos_token_id: TokenId, map: Py<PyAny>) -> PyResult<PyVocabulary> {
        if let Ok(dict) = map.extract::<HashMap<String, Vec<TokenId>>>(py) {
            return Ok(PyVocabulary(Vocabulary::try_from((eos_token_id, dict))?));
        }
        if let Ok(dict) = map.extract::<HashMap<Vec<u8>, Vec<TokenId>>>(py) {
            return Ok(PyVocabulary(Vocabulary::try_from((eos_token_id, dict))?));
        }

        let message = "Expected a dict with keys of type str or bytes and values of type list[int]";
        let tname = type_name!(map).to_string_lossy();
        if tname == "dict" {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Dict keys or/and values of the wrong types. {message}"
            )))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "{message}, got {tname}"
            )))
        }
    }

    /// Creates the vocabulary of a pre-trained model.
    #[staticmethod]
    #[pyo3(signature = (model, revision=None, token=None))]
    #[cfg(feature = "hugginface-hub")]
    fn from_pretrained(
        model: String,
        revision: Option<String>,
        token: Option<String>,
    ) -> PyResult<PyVocabulary> {
        let mut params = FromPretrainedParameters::default();
        if let Some(r) = revision {
            params.revision = r
        }
        if token.is_some() {
            params.token = token
        }
        let v = Vocabulary::from_pretrained(model.as_str(), Some(params))?;
        Ok(PyVocabulary(v))
    }

    /// Inserts new token with token_id or extends list of token_ids if token already present.
    fn insert(&mut self, py: Python<'_>, token: Py<PyAny>, token_id: TokenId) -> PyResult<()> {
        if let Ok(t) = token.extract::<String>(py) {
            return Ok(self.0.try_insert(t, token_id)?);
        }
        if let Ok(t) = token.extract::<Token>(py) {
            return Ok(self.0.try_insert(t, token_id)?);
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    /// Removes a token from vocabulary.
    fn remove(&mut self, py: Python<'_>, token: Py<PyAny>) -> PyResult<()> {
        if let Ok(t) = token.extract::<String>(py) {
            self.0.remove(t);
            return Ok(());
        }
        if let Ok(t) = token.extract::<Token>(py) {
            self.0.remove(t);
            return Ok(());
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    /// Gets token ids of a given token.
    fn get(&self, py: Python<'_>, token: Py<PyAny>) -> PyResult<Option<Vec<TokenId>>> {
        if let Ok(t) = token.extract::<String>(py) {
            return Ok(self.0.token_ids(t.into_bytes()).cloned());
        }
        if let Ok(t) = token.extract::<Token>(py) {
            return Ok(self.0.token_ids(&t).cloned());
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected a token of type str or bytes, got {:?}",
            type_name!(token)
        )))
    }

    /// Gets the end of sentence token id.
    fn get_eos_token_id(&self) -> TokenId {
        self.0.eos_token_id()
    }

    /// Gets the debug string representation of the vocabulary.
    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    /// Gets the string representation of the vocabulary.
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Compares whether two vocabularies are the same.
    fn __eq__(&self, other: &PyVocabulary) -> bool {
        self.0 == other.0
    }

    /// Returns length of Vocabulary's tokens, excluding EOS token.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Makes a deep copy of the Vocabulary.
    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyVocabulary(self.0.clone())
    }

    fn __reduce__(&self) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        Python::attach(|py| {
            let cls = PyModule::import(py, "oc_earley")?.getattr("Vocabulary")?;
            let binary_data = encode_object(self, ObjectKind::Vocabulary, "Vocabulary")?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        decode_object(&binary_data, ObjectKind::Vocabulary, "Vocabulary")
    }
}

#[derive(Encode, Decode)]
struct CompiledSchemaPayload {
    schema: Vec<u8>,
    vocabulary: Vocabulary,
}

/// A schema analysis and its selected runtime backend.
#[pyclass(
    name = "CompiledSchema",
    module = "oc_earley",
    frozen,
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyCompiledSchema {
    compiled: CompiledSchema,
    schema: Vec<u8>,
    vocabulary: Vocabulary,
}

#[pymethods]
impl PyCompiledSchema {
    /// Compiles a JSON Schema under the strict K1 profile.
    #[staticmethod]
    fn from_json_schema(
        py: Python<'_>,
        schema: &Bound<'_, PyAny>,
        vocabulary: &PyVocabulary,
    ) -> PyResult<Self> {
        let schema = schema_bytes(schema)?;
        let inner_vocabulary = vocabulary.0.clone();
        let compiled = py.detach(|| {
            CompiledSchema::compile(&schema, &inner_vocabulary, &CompileOptions::default())
        })?;
        Ok(Self {
            compiled,
            schema,
            vocabulary: inner_vocabulary,
        })
    }

    /// Returns the deterministic tier report as Python values.
    fn tier_report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_pyobject::to_pyobject(py, &self.compiled.report)
            .map(Bound::unbind)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Recompiles the schema and returns per-stage timing in nanoseconds.
    fn profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let profile = py.detach(|| {
            CompiledSchema::compile_profiled(
                &self.schema,
                &self.vocabulary,
                &CompileOptions::default(),
            )
            .map(|(_, profile)| profile)
        })?;
        serde_pyobject::to_pyobject(py, &profile)
            .map(Bound::unbind)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Returns the selected backend name.
    #[getter]
    fn backend(&self) -> &'static str {
        match self.compiled.report.selected_backend {
            crate::engine::BackendKind::WholeDfa => "whole_dfa",
            crate::engine::BackendKind::Lalr => "lalr",
            crate::engine::BackendKind::Earley => "earley",
        }
    }

    /// Creates an incremental byte recognizer for the selected backend.
    fn recognizer(&self) -> PyResult<PyRecognizer> {
        let state = self
            .compiled
            .recognizer(crate::schema::RuntimeLimits::default())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyRecognizer {
            state,
            checkpoints: Vec::new(),
        })
    }

    /// Creates a guide when the selected backend is available.
    #[pyo3(signature = (max_rollback=32))]
    fn guide(&self, max_rollback: usize) -> PyResult<PyGuide> {
        let index = self
            .compiled
            .index()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .clone();
        Ok(PyGuide::__new__(PyIndex(Arc::new(index)), max_rollback))
    }

    fn __repr__(&self) -> String {
        format!(
            "CompiledSchema(backend='{}', schema_nodes={})",
            self.backend(),
            self.compiled.report.schema_nodes
        )
    }

    fn __eq__(&self, other: &PyCompiledSchema) -> bool {
        self == other
    }

    fn __reduce__(&self) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        Python::attach(|py| {
            let cls = PyModule::import(py, "oc_earley")?.getattr("CompiledSchema")?;
            let payload = CompiledSchemaPayload {
                schema: self.schema.clone(),
                vocabulary: self.vocabulary.clone(),
            };
            let binary_data =
                encode_object(&payload, ObjectKind::CompiledSchema, "CompiledSchema")?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(py: Python<'_>, binary_data: Vec<u8>) -> PyResult<Self> {
        let payload: CompiledSchemaPayload =
            decode_object(&binary_data, ObjectKind::CompiledSchema, "CompiledSchema")?;
        let compiled = py.detach(|| {
            CompiledSchema::compile(
                &payload.schema,
                &payload.vocabulary,
                &CompileOptions::default(),
            )
        })?;
        Ok(Self {
            compiled,
            schema: payload.schema,
            vocabulary: payload.vocabulary,
        })
    }
}

/// An incremental recognizer over canonical JSON bytes.
#[pyclass(name = "Recognizer", module = "oc_earley", skip_from_py_object)]
pub struct PyRecognizer {
    state: crate::engine::RecognizerState,
    checkpoints: Vec<crate::engine::RecognizerCheckpoint>,
}

#[pymethods]
impl PyRecognizer {
    /// Advances transactionally by a string or byte sequence.
    fn advance(&mut self, data: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        let bytes = runtime_bytes(data)?;
        self.state
            .try_advance_bytes(&bytes)
            .map(|advance| match advance {
                crate::engine::Advance::Rejected => "rejected",
                crate::engine::Advance::Live => "live",
                crate::engine::Advance::Accepting => "accepting",
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn accepting(&self) -> bool {
        self.state.is_accepting()
    }

    #[getter]
    fn live(&self) -> bool {
        self.state.is_live()
    }

    /// Stores a checkpoint and returns its opaque handle.
    fn checkpoint(&mut self) -> PyResult<usize> {
        let limit = crate::schema::RuntimeLimits::default().max_checkpoint_history;
        if self.checkpoints.len() >= limit {
            return Err(PyValueError::new_err(format!(
                "runtime limit exceeded for CheckpointHistory: {} > {limit}",
                self.checkpoints.len().saturating_add(1)
            )));
        }
        self.checkpoints.push(self.state.checkpoint());
        Ok(self.checkpoints.len() - 1)
    }

    /// Restores a checkpoint created by this recognizer.
    fn restore(&mut self, handle: usize) -> PyResult<()> {
        let checkpoint = self
            .checkpoints
            .get(handle)
            .ok_or_else(|| PyValueError::new_err(format!("invalid checkpoint handle {handle}")))?;
        self.state
            .restore(checkpoint)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Returns deterministic recognizer counters.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_pyobject::to_pyobject(py, &self.state.stats())
            .map(Bound::unbind)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

fn runtime_bytes(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = data.extract::<Vec<u8>>() {
        return Ok(bytes);
    }
    if let Ok(text) = data.extract::<String>() {
        return Ok(text.into_bytes());
    }
    Err(PyValueError::new_err("expected str or bytes"))
}

fn schema_bytes(schema: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(schema) = schema.extract::<String>() {
        return Ok(schema.into_bytes());
    }
    if let Ok(schema) = schema.extract::<Vec<u8>>() {
        return Ok(schema);
    }
    let value: serde_json::Value = serde_pyobject::from_pyobject(schema.clone())
        .map_err(|error| PyValueError::new_err(format!("Invalid schema value: {error}")))?;
    serde_json::to_vec(&value)
        .map_err(|error| PyValueError::new_err(format!("Invalid schema value: {error}")))
}

/// Creates regex string from JSON schema with optional whitespace pattern.
#[pyfunction(name = "build_regex_from_schema")]
#[pyo3(signature = (json_schema, whitespace_pattern=None, max_recursion_depth=3))]
pub fn build_regex_from_schema_py(
    json_schema: String,
    whitespace_pattern: Option<&str>,
    max_recursion_depth: usize,
) -> PyResult<String> {
    let value = serde_json::from_str(&json_schema).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("Expected a valid JSON string.")
    })?;
    json_schema::regex_from_value(&value, whitespace_pattern, Some(max_recursion_depth))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn register_child_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "json_schema")?;
    parent_module.add_submodule(&m)?;

    m.add("BOOLEAN", json_schema::BOOLEAN)?;
    m.add("DATE", json_schema::DATE)?;
    m.add("DATE_TIME", json_schema::DATE_TIME)?;
    m.add("INTEGER", json_schema::INTEGER)?;
    m.add("NULL", json_schema::NULL)?;
    m.add("NUMBER", json_schema::NUMBER)?;
    m.add("STRING", json_schema::STRING)?;
    m.add("STRING_INNER", json_schema::STRING_INNER)?;
    m.add("TIME", json_schema::TIME)?;
    m.add("UUID", json_schema::UUID)?;
    m.add("WHITESPACE", json_schema::WHITESPACE)?;
    m.add("EMAIL", json_schema::EMAIL)?;
    m.add("URI", json_schema::URI)?;
    m.add_function(wrap_pyfunction!(build_regex_from_schema_py, &m)?)?;

    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules_bind = (sys.as_ref() as &Bound<PyAny>).getattr("modules")?;
    let sys_modules = sys_modules_bind.cast::<PyDict>()?;
    sys_modules.set_item("oc_earley.json_schema", &m)?;

    Ok(())
}

/// This package provides core functionality for structured generation, providing a convenient way to:
///
/// - build regular expressions from JSON schemas
///
/// - construct an Index object by combining a Vocabulary and regular expression to efficiently map tokens from a given Vocabulary to state transitions in a finite-state automation
#[pymodule(gil_used = false)]
fn oc_earley(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let version = env!("CARGO_PKG_VERSION");
    m.add("__version__", version)?;

    m.add_class::<PyIndex>()?;
    m.add_class::<PyVocabulary>()?;
    m.add_class::<PyGuide>()?;
    m.add_class::<PyCompiledSchema>()?;
    m.add_class::<PyRecognizer>()?;
    register_child_module(m)?;

    Ok(())
}
