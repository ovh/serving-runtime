use std::sync::MutexGuard;

use jni::objects::{JClass, JObject, JString};
use jni::sys::{jobject, jobjectArray};
use jni::JNIEnv;
use tokenizers::{EncodeInput, Tokenizer};

use crate::{jstring_array_to_vec, unwrap_or_throw, Error};

/// Native function Tokenizer::fromFile
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Tokenizer_fromFile(
    env: JNIEnv,
    tokenizer_class: JClass,
    file: JObject,
) -> jobject {
    let result = || -> Result<jobject, Error> {
        // Convert Java Path to Rust String
        let file: JString = env
            .call_method(file, "toString", "()Ljava/lang/String;", &[])?
            .l()?
            .into();
        let file: String = env.get_string(file)?.into();

        // Load tokenizer from file
        let tokenizer = Tokenizer::from_file(file)?;

        // Wrap tokenizer in a Java object
        let tokenizer_object = env.alloc_object(tokenizer_class)?;
        env.set_rust_field(tokenizer_object, "handle", tokenizer)?;

        Ok(tokenizer_object.into_inner())
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native method Tokenizer::encode(String)
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Tokenizer_encode__Ljava_lang_String_2(
    env: JNIEnv,
    tokenizer: JObject,
    input: JString,
) -> jobject {
    let result = || -> Result<jobject, Error> {
        // Encode input
        let input: String = env.get_string(input)?.into();
        let tokenizer: MutexGuard<Tokenizer> = env.get_rust_field(tokenizer, "handle")?;
        let encoding = tokenizer.encode(input, false)?;

        // Wrap encoding in a Java object
        let encoding_class =
            env.find_class("com/ovh/mls/serving/runtime/huggingface/tokenizer/Encoding")?;
        let encoding_object = env.alloc_object(encoding_class)?;
        env.set_rust_field(encoding_object, "handle", encoding)?;

        Ok(encoding_object.into_inner())
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native method Tokenizer::encode(String, String)
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Tokenizer_encode__Ljava_lang_String_2Ljava_lang_String_2(
    env: JNIEnv,
    tokenizer: JObject,
    input1: JString,
    input2: JString,
) -> jobject {
    let result = || -> Result<jobject, Error> {
        // Encode input
        let input1: String = env.get_string(input1)?.into();
        let input2: String = env.get_string(input2)?.into();
        let tokenizer: MutexGuard<Tokenizer> = env.get_rust_field(tokenizer, "handle")?;
        let encoding = tokenizer.encode(EncodeInput::Dual(input1.into(), input2.into()), false)?;

        // Wrap encoding in a Java object
        let encoding_class =
            env.find_class("com/ovh/mls/serving/runtime/huggingface/tokenizer/Encoding")?;
        let encoding_object = env.alloc_object(encoding_class)?;
        env.set_rust_field(encoding_object, "handle", encoding)?;

        Ok(encoding_object.into_inner())
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native method Tokenizer::encode(String[])
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Tokenizer_encode___3Ljava_lang_String_2(
    env: JNIEnv,
    tokenizer: JObject,
    input: jobjectArray,
) -> jobject {
    let result = || -> Result<jobject, Error> {
        // Encode input
        let input_vec = jstring_array_to_vec(&env, input)?;
        let tokenizer: MutexGuard<Tokenizer> = env.get_rust_field(tokenizer, "handle")?;
        let encoding = tokenizer.encode(input_vec, false)?;

        // Wrap encoding in a Java object
        let encoding_class =
            env.find_class("com/ovh/mls/serving/runtime/huggingface/tokenizer/Encoding")?;
        let encoding_object = env.alloc_object(encoding_class)?;
        env.set_rust_field(encoding_object, "handle", encoding)?;

        Ok(encoding_object.into_inner())
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

//Native method Tokenizer::encode(String[], String[])
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Tokenizer_encode___3Ljava_lang_String_2_3Ljava_lang_String_2(
    env: JNIEnv,
    tokenizer: JObject,
    input1: jobjectArray,
    input2: jobjectArray,
) -> jobject {
    let result = || -> Result<jobject, Error> {
        // Encode input
        let input1_vec = jstring_array_to_vec(&env, input1)?;
        let input2_vec = jstring_array_to_vec(&env, input2)?;
        let tokenizer: MutexGuard<Tokenizer> = env.get_rust_field(tokenizer, "handle")?;
        let encoding = tokenizer.encode(
            EncodeInput::Dual(input1_vec.into(), input2_vec.into()),
            false,
        )?;

        // Wrap encoding in a Java object
        let encoding_class =
            env.find_class("com/ovh/mls/serving/runtime/huggingface/tokenizer/Encoding")?;
        let encoding_object = env.alloc_object(encoding_class)?;
        env.set_rust_field(encoding_object, "handle", encoding)?;

        Ok(encoding_object.into_inner())
    }();

    unwrap_or_throw(&env, result, JObject::null().into_inner())
}

/// Native method Tokenizer::releaseHandle
#[no_mangle]
pub extern "system" fn Java_com_ovh_mls_serving_runtime_huggingface_tokenizer_Tokenizer_releaseHandle(
    env: JNIEnv,
    tokenizer: JObject,
) {
    let result = || -> Result<(), Error> {
        env.take_rust_field::<_, _, Box<Tokenizer>>(tokenizer, "handle")?;
        Ok(())
    }();

    unwrap_or_throw(&env, result, ());
}
