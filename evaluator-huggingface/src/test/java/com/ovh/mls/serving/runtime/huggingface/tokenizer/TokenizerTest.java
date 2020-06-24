package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Optional;

class TokenizerTest {

    @Test
    void fromFileEncodeString() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/bpe.json"));
        Encoding encoding = tokenizer.encode("This is a test");

        Assertions.assertFalse(encoding.isEmpty());
        Assertions.assertEquals(4, encoding.size());
        Assertions.assertArrayEquals(
            new String[]{"This", "is", "at", "est"},
            encoding.getTokens()
        );
        Assertions.assertArrayEquals(
            new Optional[]{Optional.of(0), Optional.of(0), Optional.of(0), Optional.of(0)},
            encoding.getWords()
        );
        Assertions.assertArrayEquals(
            new int[]{1212, 271, 265, 395},
            encoding.getIds()
        );
        Assertions.assertArrayEquals(
            new int[]{0, 0, 0, 0},
            encoding.getTypeIds()
        );
        Assertions.assertArrayEquals(
            new Offset[]{new Offset(0, 4), new Offset(4, 6), new Offset(6, 8), new Offset(8, 11)},
            encoding.getOffsets()
        );
        Assertions.assertArrayEquals(
            new int[]{0, 0, 0, 0},
            encoding.getSpecialTokensMask()
        );
        Assertions.assertArrayEquals(
            new int[]{1, 1, 1, 1},
            encoding.getAttentionMask()
        );
    }

    @Test
    void fromFileEncodeTokens() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/bpe.json"));
        Encoding encoding = tokenizer.encode("This is a test".split(" "));

        Assertions.assertFalse(encoding.isEmpty());
        Assertions.assertEquals(4, encoding.size());
        Assertions.assertArrayEquals(
            new String[]{"This", "is", "a", "test"},
            encoding.getTokens()
        );
        Assertions.assertArrayEquals(
            new Optional[]{Optional.of(0), Optional.of(1), Optional.of(2), Optional.of(3)},
            encoding.getWords()
        );
    }

    @Test
    void fromFileEncodeDualString() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/bpe.json"));
        Encoding encoding = tokenizer.encode("This is a test", "and a second sentence");

        Assertions.assertFalse(encoding.isEmpty());
        Assertions.assertEquals(9, encoding.size());
        Assertions.assertArrayEquals(
            new String[]{"This", "is", "at", "est", "and", "ase", "cond", "sent", "ence"},
            encoding.getTokens()
        );
    }

    @Test
    void fromFileEncodeDualTokens() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/bpe.json"));
        Encoding encoding = tokenizer.encode("This is a test".split(" "), "and a second sentence".split(" "));

        Assertions.assertFalse(encoding.isEmpty());
        Assertions.assertEquals(9, encoding.size());
        Assertions.assertArrayEquals(
            new String[]{"This", "is", "a", "test", "and", "a", "second", "sent", "ence"},
            encoding.getTokens()
        );
    }
}
