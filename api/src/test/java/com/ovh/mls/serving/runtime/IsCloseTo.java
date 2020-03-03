package com.ovh.mls.serving.runtime;

import org.hamcrest.Description;
import org.hamcrest.Factory;
import org.hamcrest.Matcher;
import org.hamcrest.TypeSafeMatcher;

public class IsCloseTo extends TypeSafeMatcher<Float> {

    private final float delta;
    private final float value;

    public IsCloseTo(float value, float error) {
        this.delta = error;
        this.value = value;
    }

    /**
     * Creates a matcher of {@link Double}s that matches when an examined double is equal
     * to the specified <code>operand</code>, within a range of +/- <code>error</code>.
     * <p/>
     * For example:
     * <pre>assertThat(1.03, is(closeTo(1.0, 0.03)))</pre>
     *
     * @param operand the expected value of matching doubles
     * @param error   the delta (+/-) within which matches will be allowed
     */
    @Factory
    public static Matcher<Float> closeTo(float operand, float error) {
        return new IsCloseTo(operand, error);
    }

    @Override
    public boolean matchesSafely(Float item) {
        return actualDelta(item) <= 0.0;
    }

    @Override
    public void describeMismatchSafely(Float item, Description mismatchDescription) {
        mismatchDescription.appendValue(item)
            .appendText(" differed by ")
            .appendValue(actualDelta(item));
    }

    @Override
    public void describeTo(Description description) {
        description.appendText("a numeric value within ")
            .appendValue(delta)
            .appendText(" of ")
            .appendValue(value);
    }

    private double actualDelta(Float item) {
        return (Math.abs((item - value)) - delta);
    }
}
