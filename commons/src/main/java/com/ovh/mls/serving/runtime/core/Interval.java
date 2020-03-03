package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class Interval {

    private final Double lowerBound;
    private final Double upperBound;

    @JsonCreator
    public Interval(@JsonProperty("lower_bound") Double lowerBound, @JsonProperty("upper_bound") Double upperBound) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    public Double getLowerBound() {
        return lowerBound;
    }

    public Double getUpperBound() {
        return upperBound;
    }
}
