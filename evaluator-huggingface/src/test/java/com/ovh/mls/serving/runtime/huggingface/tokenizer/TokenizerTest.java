package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Optional;

class TokenizerTest {

    @Test
    void fromFileEncodeString() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/tokenizer.json"));
        Encoding encoding = tokenizer.encode("This is a test");

        Assertions.assertFalse(encoding.isEmpty());
        Assertions.assertEquals(11, encoding.size());
        Assertions.assertArrayEquals(
            new String[]{"Ġ", "T", "h", "is", "Ġ", "is", "Ġa", "Ġt", "e", "s", "t"},
            encoding.getTokens()
        );
        Assertions.assertArrayEquals(
            new Optional[]{
                Optional.of(0), Optional.of(0), Optional.of(0), Optional.of(0), Optional.of(1), Optional.of(1),
                Optional.of(2), Optional.of(3), Optional.of(3), Optional.of(3), Optional.of(3)
            },
            encoding.getWords()
        );
        Assertions.assertArrayEquals(
            new int[]{83, 44, 58, 96, 83, 96, 93, 92, 55, 69, 70},
            encoding.getIds()
        );
        Assertions.assertArrayEquals(
            new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            encoding.getTypeIds()
        );
        Assertions.assertArrayEquals(
            new Offset[]{
                new Offset(0, 0),
                new Offset(0, 1),
                new Offset(1, 2),
                new Offset(2, 4),
                new Offset(4, 5),
                new Offset(5, 7),
                new Offset(7, 9),
                new Offset(9, 11),
                new Offset(11, 12),
                new Offset(12, 13),
                new Offset(13, 14),
            },
            encoding.getOffsets()
        );
        Assertions.assertArrayEquals(
            new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            encoding.getSpecialTokensMask()
        );
        Assertions.assertArrayEquals(
            new int[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            encoding.getAttentionMask()
        );
    }

    @Test
    void fromFileEncodeTokens() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/tokenizer.json"));
        Encoding encoding = tokenizer.encode("This is a test".split(" "));

        Assertions.assertArrayEquals(
            new String[]{"Ġ", "T", "h", "is", "Ġ", "is", "Ġa", "Ġt", "e", "s", "t"},
            encoding.getTokens()
        );
    }

    @Test
    void fromFileEncodeDualString() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/tokenizer.json"));
        Encoding encoding = tokenizer.encode("This is a test", "and a second sentence");

        Assertions.assertArrayEquals(
            new String[]{
                "Ġ", "T", "h", "is", "Ġ", "is", "Ġa", "Ġt", "e", "s", "t", "Ġa", "n", "d", "Ġa", "Ġ", "s", "e", "c",
                "o", "n", "d", "Ġ", "s", "e", "n", "t", "e", "n", "c", "e"
            },
            encoding.getTokens()
        );
    }

    @Test
    void fromFileEncodeDualTokens() {
        Tokenizer tokenizer = Tokenizer.fromFile(Path.of("src/test/resources/tokenizer.json"));
        Encoding encoding = tokenizer.encode("This is a test".split(" "), "and a second sentence".split(" "));

        Assertions.assertArrayEquals(
            new String[]{
                "Ġ", "T", "h", "is", "Ġ", "is", "Ġa", "Ġt", "e", "s", "t", "Ġa", "n", "d", "Ġa", "Ġ", "s", "e", "c",
                "o", "n", "d", "Ġ", "s", "e", "n", "t", "e", "n", "c", "e"
            },
            encoding.getTokens()
        );
    }
}
