#include "terrain_classification.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

TF_LITE_MICRO_TEST_BEGIN

TF_LITE_MICRO_TEST(loadModelandPerformInference) {
  // set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.

  const tflite::Model* model = ::tflite::GetModel(terrain_classification);
  if (model->version != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal"
      "to supported version %d. \n",
      model->version(), TFLITE_SCHEMA_VERSION);
  }

  //This pulls in all the operation implementations we need
  tflite::ops::micro::AllOpsResolver resolver;

  //create an area of memory to use for input, output and intermediate arrays.

  const int tensor areana = 8 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  //Building an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);

  // Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  //Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  //Make sure the input has the properties we expect
  // first ensuring that the input to the model is not null
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);

  TF_LITE_MICRO_EXPECT_EQ(48, input->dims->size);

  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  // Providing an input value to test
  arr = [-2.31439495e+00,  5.97616716e-02, -2.47774845e+00, -2.14046450e+00,
         -2.30420794e+00, -2.33426924e+00, -2.28394371e+00, -1.30186937e+00,
          2.72320737e-01, -2.31265589e+00, -7.39296468e-01, -1.28263204e+00,
         -1.47990409e+00, -1.11049571e+00,  2.08532245e+00,  8.73432105e-02,
          1.82784239e+00,  2.32967389e+00,  2.08847336e+00,  2.03926225e+00,
          2.12828761e+00,  2.83834014e+00,  2.29770183e+01,  4.60000000e+01,
          2.13393521e-04,  5.35964767e-05,  4.76778233e-05,  4.92456723e-07,
          2.13393521e-04,  4.12774497e-05,  3.65835590e+00,  1.27874422e+01,
          1.00000000e+00,  5.44404498e-03,  1.13616446e-03,  1.16661724e-03,
          5.82489601e-05,  5.44404498e-03,  6.26408306e-04,  2.84357040e+00,
          2.06768944e+01,  3.20000000e+01,  5.97301188e-04,  1.67362571e-04,
          1.50204272e-04,  8.01359716e-08,  5.97301188e-04,  1.27957316e-04];


  for (int i = 0; i < 48; i++) {
    input->data.f[i] = arr[i]
  }
  
  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Obtain a pointer to the output tensor and make sure it has the
  // properties we expect.

  TfLite Tensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // Obtain the output value from the tensor
  float value_1 = output->data.f[0];
  float value_2 = output->data.f[1];
  float value_3 = output->data.f[2];
  float value_4 = output->data.f[3];

  TF_LITE_MICRO_EXPECT_NEAR(0., value_1, 0.05);
  TF_LITE_MICRO_EXPECT_NEAR(0., value_2, 0.05);
  TF_LITE_MICRO_EXPECT_NEAR(0., value_3, 0.05);
  TF_LITE_MICRO_EXPECT_NEAR(1., value_4, 0.05);
}

TF_LITE_MICRO_TESTS_END