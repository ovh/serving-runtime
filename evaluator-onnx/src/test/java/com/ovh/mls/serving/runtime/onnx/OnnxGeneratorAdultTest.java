package com.ovh.mls.serving.runtime.onnx;

import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorUtil;
import com.ovh.mls.serving.runtime.core.builder.InputStreamJsonIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class OnnxGeneratorAdultTest {
    private static final ClassLoader LOADER = OnnxGeneratorAdultTest.class.getClassLoader();
    private static final Config CONFIG = ConfigFactory.load();
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(CONFIG);

    private Evaluator onnxEvaluator;

    @BeforeEach
    public void create() throws EvaluatorException, FileNotFoundException {
        onnxEvaluator = new OnnxGenerator().generate(
            new File(LOADER.getResource("onnx/adult/transformation.onnx").getFile()),
            CONFIG
        );
    }

    @Test
    public void evaluate() throws EvaluationException {
        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/adult/single_gen.json"));

        final EvaluationContext evaluationContext = new EvaluationContext();

        TensorIO output = onnxEvaluator.evaluate(input, evaluationContext);

        assertEquals(1, evaluationContext.totalEvaluation());
        assertEquals(
            Set.of(
                "scaled_imputed_hours-per-week",
                "scaled_imputed_capital-gain",
                "scaled_imputed_capital-loss",
                "scaled_imputed_education-num",
                "scaled_imputed_age",
                "scaled_imputed_fnlwgt",
                "ohe_marital-status_0",
                "ohe_marital-status_1",
                "ohe_marital-status_2",
                "ohe_marital-status_3",
                "ohe_marital-status_4",
                "ohe_marital-status_5",
                "ohe_sex",
                "ohe_relationship_0",
                "ohe_relationship_1",
                "ohe_relationship_2",
                "ohe_relationship_3",
                "ohe_relationship_4",
                "ohe_relationship_5",
                "level_value_education",
                "ohe_race_0",
                "ohe_race_1",
                "ohe_race_2",
                "ohe_race_3",
                "ohe_race_4",
                "level_value_native-country",
                "ohe_workclass_0",
                "ohe_workclass_1",
                "ohe_workclass_2",
                "ohe_workclass_3",
                "ohe_workclass_4",
                "ohe_workclass_5",
                "level_value_occupation"
            ),
            output.tensorsNames()
        );

        Tensor scaledImputedHours = output.getTensor("scaled_imputed_hours-per-week");
        assertArrayEquals(new int[]{1, 1}, scaledImputedHours.getShapeAsArray());
        assertEquals(3.898834228515625F, ((float[][]) scaledImputedHours.getData())[0][0], 0.000001);
        Tensor scaledImputedCapitalGain = output.getTensor("scaled_imputed_capital-gain");
        assertArrayEquals(new int[]{1, 1}, scaledImputedCapitalGain.getShapeAsArray());
        assertEquals(4281540.0F, ((float[][]) scaledImputedCapitalGain.getData())[0][0], 0.000001);
        Tensor scaledImputedCapitalLoss = output.getTensor("scaled_imputed_capital-loss");
        assertArrayEquals(new int[]{1, 1}, scaledImputedCapitalLoss.getShapeAsArray());
        assertEquals(-41858.82421875F, ((float[][]) scaledImputedCapitalLoss.getData())[0][0], 0.000001);
        Tensor scaledImputedEducation = output.getTensor("scaled_imputed_education-num");
        assertArrayEquals(new int[]{1, 1}, scaledImputedEducation.getShapeAsArray());
        assertEquals(7.624780654907227F, ((float[][]) scaledImputedEducation.getData())[0][0], 0.000001);
        Tensor scaledImputedAge = output.getTensor("scaled_imputed_age");
        assertArrayEquals(new int[]{1, 1}, scaledImputedAge.getShapeAsArray());
        assertEquals(15.17330551147461, ((float[][]) scaledImputedAge.getData())[0][0], 0.000001);
        Tensor scaledImputedFnlwgt = output.getTensor("scaled_imputed_fnlwgt");
        assertArrayEquals(new int[]{1, 1}, scaledImputedFnlwgt.getShapeAsArray());
        assertEquals(-8.13452288E9F, ((float[][]) scaledImputedFnlwgt.getData())[0][0], 0.000001);
        Tensor oheMaritalStatus0 = output.getTensor("ohe_marital-status_0");
        assertArrayEquals(new int[]{1, 1}, oheMaritalStatus0.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus0.getData());
        Tensor oheMaritalStatus1 = output.getTensor("ohe_marital-status_1");
        assertArrayEquals(new int[]{1, 1}, oheMaritalStatus1.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{1}}, (long[][]) oheMaritalStatus1.getData());
        Tensor oheMaritalStatus2 = output.getTensor("ohe_marital-status_2");
        assertArrayEquals(new int[]{1, 1}, oheMaritalStatus2.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus2.getData());
        Tensor oheMaritalStatus3 = output.getTensor("ohe_marital-status_3");
        assertArrayEquals(new int[]{1, 1}, oheMaritalStatus3.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus3.getData());
        Tensor oheMaritalStatus4 = output.getTensor("ohe_marital-status_4");
        assertArrayEquals(new int[]{1, 1}, oheMaritalStatus4.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus4.getData());
        Tensor oheMaritalStatus5 = output.getTensor("ohe_marital-status_5");
        assertArrayEquals(new int[]{1, 1}, oheMaritalStatus5.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus5.getData());
        Tensor oheSex = output.getTensor("ohe_sex");
        assertArrayEquals(new int[]{1, 1}, oheSex.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheSex.getData());
        Tensor oheRelationship0 = output.getTensor("ohe_relationship_0");
        assertArrayEquals(new int[]{1, 1}, oheRelationship0.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRelationship0.getData());
        Tensor oheRelationship1 = output.getTensor("ohe_relationship_1");
        assertArrayEquals(new int[]{1, 1}, oheRelationship1.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{1}}, (long[][]) oheRelationship1.getData());
        Tensor oheRelationship2 = output.getTensor("ohe_relationship_2");
        assertArrayEquals(new int[]{1, 1}, oheRelationship2.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRelationship2.getData());
        Tensor oheRelationship3 = output.getTensor("ohe_relationship_3");
        assertArrayEquals(new int[]{1, 1}, oheRelationship3.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRelationship3.getData());
        Tensor oheRelationship4 = output.getTensor("ohe_relationship_4");
        assertArrayEquals(new int[]{1, 1}, oheRelationship4.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRelationship4.getData());
        Tensor oheRelationship5 = output.getTensor("ohe_relationship_5");
        assertArrayEquals(new int[]{1, 1}, oheRelationship5.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRelationship5.getData());
        Tensor levelValueEducation = output.getTensor("level_value_education");
        assertArrayEquals(new int[]{1, 1}, levelValueEducation.getShapeAsArray());
        assertEquals(0.7728285193443298F, ((float[][]) levelValueEducation.getData())[0][0], 0.000001);
        Tensor oheRace0 = output.getTensor("ohe_race_0");
        assertArrayEquals(new int[]{1, 1}, oheRace0.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus0.getData());
        Tensor oheRace1 = output.getTensor("ohe_race_1");
        assertArrayEquals(new int[]{1, 1}, oheRace1.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheMaritalStatus0.getData());
        Tensor oheRace2 = output.getTensor("ohe_race_2");
        assertArrayEquals(new int[]{1, 1}, oheRace2.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRace2.getData());
        Tensor oheRace3 = output.getTensor("ohe_race_3");
        assertArrayEquals(new int[]{1, 1}, oheRace3.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheRace3.getData());
        Tensor oheRace4 = output.getTensor("ohe_race_4");
        assertArrayEquals(new int[]{1, 1}, oheRace4.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{1}}, (long[][]) oheRace4.getData());
        Tensor levelValueNativeCountry = output.getTensor("level_value_native-country");
        assertArrayEquals(new int[]{1, 1}, levelValueNativeCountry.getShapeAsArray());
        assertEquals(0.7722786068916321F,
            ((float[][]) levelValueNativeCountry.getData())[0][0], 0.000001);
        Tensor oheWorkclass0 = output.getTensor("ohe_workclass_0");
        assertArrayEquals(new int[]{1, 1}, oheWorkclass0.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheWorkclass0.getData());
        Tensor oheWorkclass1 = output.getTensor("ohe_workclass_1");
        assertArrayEquals(new int[]{1, 1}, oheWorkclass1.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheWorkclass1.getData());
        Tensor oheWorkclass2 = output.getTensor("ohe_workclass_2");
        assertArrayEquals(new int[]{1, 1}, oheWorkclass2.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheWorkclass2.getData());
        Tensor oheWorkclass3 = output.getTensor("ohe_workclass_3");
        assertArrayEquals(new int[]{1, 1}, oheWorkclass3.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheWorkclass3.getData());
        Tensor oheWorkclass4 = output.getTensor("ohe_workclass_4");
        assertArrayEquals(new int[]{1, 1}, oheWorkclass4.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{0}}, (long[][]) oheWorkclass4.getData());
        Tensor oheWorkclass5 = output.getTensor("ohe_workclass_5");
        assertArrayEquals(new int[]{1, 1}, oheWorkclass5.getShapeAsArray());
        assertArrayEquals(new long[][]{new long[]{1}}, (long[][]) oheWorkclass5.getData());
        Tensor levelValueOccupation = output.getTensor("level_value_occupation");
        assertArrayEquals(new int[]{1, 1}, levelValueOccupation.getShapeAsArray());
        assertEquals(0.8360435F, ((float[][]) levelValueOccupation.getData())[0][0], 0.000001);
    }
}
