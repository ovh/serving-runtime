package com.ovh.mls.serving.runtime.processors;

public class MeanStd {

    private Double mean;

    private Double std;

    public Double getMean() {
        return mean;
    }

    public MeanStd setMean(Double mean) {
        this.mean = mean;
        return this;
    }

    public Double getStd() {
        return std;
    }

    public MeanStd setStd(Double std) {
        this.std = std;
        return this;
    }
}
