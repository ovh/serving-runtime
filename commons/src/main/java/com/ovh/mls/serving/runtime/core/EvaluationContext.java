package com.ovh.mls.serving.runtime.core;


public class EvaluationContext {
    private static final String INPUT_DEBUG_KEY = "input";
    private static final String OUTPUT_DEBUG_KEY = "output";


    private long evaluation = 0L;

    /**
     * Current number of the evaluator beeing evaluated
     */
    private int currentEvaluator = 0;

    /**
     * Attribute indicating if user asked for debugging of an evaluator (i.e. stop and display tensors)
     * If null: no debugging asked
     * If not null: debugging of the evaluator of that number asked
     */
    private Integer debugStep;

    /**
     * Boolean indicating :
     * If true: should debug the input of the wanted debug step
     * If false: should debug the output of the wanted debug step
     * If null: doesn't debug anything
     */
    private Boolean debugInput;

    /**
     * Indicates if the output should be simplified at the end
     */
    private final boolean shouldSimplify;

    public EvaluationContext() {
        this(null);
    }

    public EvaluationContext(String debugStep) {
        this.evaluation = 0L;
        this.currentEvaluator = 0;
        this.debugStep = null;
        this.debugInput = null;

        if (debugStep != null) {
            shouldSimplify = false;
            String[] split = debugStep.split(":");
            if (split.length == 2) {
                String key = split[0];
                Integer value = Integer.valueOf(split[1]);
                if (INPUT_DEBUG_KEY.equals(key)) {
                    this.debugInput = true;
                } else if (OUTPUT_DEBUG_KEY.equals(key)) {
                    this.debugInput = false;
                }
                this.debugStep = value;
            }
        } else {
            shouldSimplify = true;
        }
    }

    public void incCurrentEvaluator() {
        this.currentEvaluator++;
    }

    public void incEvaluation() {
        this.evaluation++;
    }

    public void incEvaluationBy(int val) {
        this.evaluation += val;
    }

    public Long totalEvaluation() {
        return this.evaluation;
    }

    public boolean shouldStop(boolean isCurrentStepInput) {
        return
            this.debugInput != null && this.debugInput == isCurrentStepInput &&
            this.debugStep != null && this.debugStep == this.currentEvaluator;
    }

    public boolean shouldSimplify() {
        return shouldSimplify;
    }
}
