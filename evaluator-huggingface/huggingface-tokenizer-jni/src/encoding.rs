use std::sync::MutexGuard;

use jni::objects::{JObject, JString, JValue};
use jni::sys::{jboolean, jint, jintArray, jlong, jobjectArray, jsize};
use jni::JNIEnv;
use tokenizers::Encoding;

use crate::{unwrap_or_throw, Error};

/// Native method Encoding::isEmpty
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_isEmpty(
    env: JNIEnv,
    encoding: JObject,
) -> jboolean {
    let result = || -> Result<jboolean, Error> {
        // Check if encoding is empty
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        Ok(encoding.is_empty() as jboolean)
    }();

    unwrap_or_throw(&env, result, false as jboolean)
}

/// Native method Encoding::size
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_size(
    env: JNIEnv,
    encoding: JObject,
) -> jlong {
    let result = || -> Result<jlong, Error> {
        // Get encoding size
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        Ok(encoding.len() as jlong)
    }();

    unwrap_or_throw(&env, result, 0 as jlong)
}

/// Native getter Encoding::getTokens
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getTokens(
    env: JNIEnv,
    encoding: JObject,
) -> jobjectArray {
    let result = || -> Result<jobjectArray, Error> {
        // Get encoding tokens
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let tokens: Vec<JString> = encoding
            .get_tokens()
            .iter()
            .map(|item| env.new_string(item))
            .collect::<Result<Vec<JString>, _>>()?;
        let string_class = env.find_class("java/lang/String")?;
        let tokens_array =
            env.new_object_array(tokens.len() as jsize, string_class, JObject::null())?;
        for (index, token) in tokens.iter().enumerate() {
            env.set_object_array_element(tokens_array, index as jsize, *token)?;
        }

        Ok(tokens_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native getter Encoding::getWords
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getWords(
    env: JNIEnv,
    encoding: JObject,
) -> jobjectArray {
    let result = || -> Result<jobjectArray, Error> {
        // Get encoding words
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let optional_class = env.find_class("java/util/Optional")?;
        let words: Vec<_> = encoding
            .get_words()
            .iter()
            .map(|&item| {
                match item {
                    Some(item) => {
                        let integer_class = env.find_class("java/lang/Integer")?;
                        let item =
                            env.new_object(integer_class, "(I)V", &[JValue::Int(item as jint)])?;
                        env.call_static_method(
                            optional_class,
                            "of",
                            "(Ljava/lang/Object;)Ljava/util/Optional;",
                            &[JValue::Object(item)],
                        )
                    }
                    None => env.call_static_method(
                        optional_class,
                        "empty",
                        "()Ljava/util/Optional;",
                        &[],
                    ),
                }
                .and_then(|item| item.l())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let words_array =
            env.new_object_array(words.len() as jsize, optional_class, JObject::null())?;
        for (index, word) in words.iter().enumerate() {
            env.set_object_array_element(words_array, index as jsize, *word)?;
        }

        Ok(words_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native getter Encoding::getIds
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getIds(
    env: JNIEnv,
    encoding: JObject,
) -> jintArray {
    let result = || -> Result<jintArray, Error> {
        // Get encoding ids
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let ids: Vec<i32> = encoding
            .get_ids()
            .iter()
            .map(|&item| item as jint)
            .collect();
        let ids_array = env.new_int_array(ids.len() as jsize)?;
        env.set_int_array_region(ids_array, 0, &ids)?;

        Ok(ids_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native getter Encoding::getTypeIds
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getTypeIds(
    env: JNIEnv,
    encoding: JObject,
) -> jintArray {
    let result = || -> Result<jintArray, Error> {
        // Get encoding type ids
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let type_ids: Vec<i32> = encoding
            .get_type_ids()
            .iter()
            .map(|&item| item as jint)
            .collect();
        let type_ids_array = env.new_int_array(type_ids.len() as jsize)?;
        env.set_int_array_region(type_ids_array, 0, &type_ids)?;

        Ok(type_ids_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native getter Encoding::getOffsets
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getOffsets(
    env: JNIEnv,
    encoding: JObject,
) -> jobjectArray {
    let result = || -> Result<jobjectArray, Error> {
        // Get encoding offsets
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let offset_class =
            env.find_class("com/ovh/mls/serving/runtime/huggingface/tokenizer/Offset")?;
        let offsets: Vec<_> = encoding
            .get_offsets()
            .iter()
            .map(|&item| {
                env.new_object(
                    offset_class,
                    "(JJ)V",
                    &[JValue::Long(item.0 as jlong), JValue::Long(item.1 as jlong)],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets_array =
            env.new_object_array(offsets.len() as jsize, offset_class, JObject::null())?;
        for (index, offset) in offsets.iter().enumerate() {
            env.set_object_array_element(offsets_array, index as jsize, *offset)?;
        }

        Ok(offsets_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native getter Encoding::getSpecialTokensMask
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getSpecialTokensMask(
    env: JNIEnv,
    encoding: JObject,
) -> jintArray {
    let result = || -> Result<jintArray, Error> {
        // Get encoding special tokens mask
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let special_tokens_mask: Vec<i32> = encoding
            .get_special_tokens_mask()
            .iter()
            .map(|&item| item as jint)
            .collect();
        let special_tokens_mask_array = env.new_int_array(special_tokens_mask.len() as jsize)?;
        env.set_int_array_region(special_tokens_mask_array, 0, &special_tokens_mask)?;

        Ok(special_tokens_mask_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native getter Encoding::getAttentionMask
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_getAttentionMask(
    env: JNIEnv,
    encoding: JObject,
) -> jintArray {
    let result = || -> Result<jintArray, Error> {
        // Get encoding attention mask
        let encoding: MutexGuard<Encoding> = env.get_rust_field(encoding, "handle")?;
        let attention_mask: Vec<i32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&item| item as jint)
            .collect();
        let attention_mask_array = env.new_int_array(attention_mask.len() as jsize)?;
        env.set_int_array_region(attention_mask_array, 0, &attention_mask)?;

        Ok(attention_mask_array)
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native method Encoding::releaseHandle
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Encoding_releaseHandle(
    env: JNIEnv,
    encoding: JObject,
) {
    let result = || -> Result<(), Error> {
        env.take_rust_field::<_, _, Box<Encoding>>(encoding, "handle")?;
        Ok(())
    }();

    unwrap_or_throw(&env, result, ());
}
