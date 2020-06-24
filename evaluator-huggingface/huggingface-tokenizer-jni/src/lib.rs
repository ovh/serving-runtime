use std::fmt;
use std::fmt::{Display, Formatter};

use jni::sys::{jobjectArray, jsize};
use jni::JNIEnv;

// Re-export
pub use encoding::*;
pub use tokenizer::*;

mod encoding;
mod tokenizer;

/// Wrap JNI and Tokenizers errors
enum Error {
    Jni(jni::errors::Error),
    Tokenizers(tokenizers::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::Jni(error) => f.write_fmt(format_args!("{}", error)),
            Error::Tokenizers(error) => f.write_fmt(format_args!("{}", error)),
        }
    }
}

impl From<jni::errors::Error> for Error {
    fn from(error: jni::errors::Error) -> Self {
        Error::Jni(error)
    }
}

impl From<tokenizers::Error> for Error {
    fn from(error: tokenizers::Error) -> Self {
        Error::Tokenizers(error)
    }
}

/// Throw an exception if an error occurred
fn unwrap_or_throw<T>(env: &JNIEnv, result: Result<T, Error>, default: T) -> T {
    // Check if an exception is already thrown
    if env.exception_check().unwrap() {
        return default;
    }

    match result {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            let exception_class = env
                .find_class("com/ovh/mls/serving/runtime/exceptions/EvaluatorException")
                .unwrap();
            env.throw_new(exception_class, format!("{}", error))
                .unwrap();
            default
        }
    }
}

/// Convert a Java string array to a vec of string
fn jstring_array_to_vec(env: &JNIEnv, array: jobjectArray) -> Result<Vec<String>, Error> {
    let array_len = env.get_array_length(array)?;
    let mut vec = Vec::new();
    for index in 0..array_len {
        let item = env.get_object_array_element(array, index as jsize)?;
        let item: String = env.get_string(item.into())?.into();
        vec.push(item);
    }
    Ok(vec)
}
