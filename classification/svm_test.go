package classification

import (
   "testing"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
)

type svmTestCase struct {
   Name string
   Reducer features.Reducer
   TestData []base.Tuple
   Input []base.Tuple
   Expected []base.Feature
}

func TestSvmBase(t *testing.T) {
   var testCases []svmTestCase = []svmTestCase{
      svmTestCase{
         "Base - 3",
         features.NoReducer{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, true),
            base.NewIntTuple([]interface{}{9, 9}, true),
            base.NewIntTuple([]interface{}{11, 11}, true),
            base.NewIntTuple([]interface{}{-10, -10}, false),
            base.NewIntTuple([]interface{}{-9, -9}, false),
            base.NewIntTuple([]interface{}{-11, -11}, false),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.Bool(true),
            base.Bool(false),
         },
      },
      svmTestCase{
         "Defaults - 3",
         nil,
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, true),
            base.NewIntTuple([]interface{}{9, 9}, true),
            base.NewIntTuple([]interface{}{11, 11}, true),
            base.NewIntTuple([]interface{}{-10, -10}, false),
            base.NewIntTuple([]interface{}{-9, -9}, false),
            base.NewIntTuple([]interface{}{-11, -11}, false),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.Bool(true),
            base.Bool(false),
         },
      },
      svmTestCase{
         "Reduced - 3",
         features.NewManualReducer([]int{1, 3}),
         []base.Tuple{
            base.NewIntTuple([]interface{}{1,  10, 0,  10, 6}, true),
            base.NewIntTuple([]interface{}{2,  9,  0,  9,  5}, true),
            base.NewIntTuple([]interface{}{3,  11, 0,  11, 4}, true),
            base.NewIntTuple([]interface{}{4, -10, 0, -10, 3}, false),
            base.NewIntTuple([]interface{}{5, -9,  0, -9,  2}, false),
            base.NewIntTuple([]interface{}{6, -11, 0, -11, 1}, false),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{1,  8, 0,  8, 2}, nil),
            base.NewIntTuple([]interface{}{2, -8, 0, -8, 1}, nil),
         },
         []base.Feature{
            base.Bool(true),
            base.Bool(false),
         },
      },
   };

   for _, testCase := range(testCases) {
      var svm Classifier = NewSvm(testCase.Reducer);
      svm.Train(testCase.TestData);
      var actual []base.Feature;
      actual, _ = svm.Classify(testCase.Input);

      if (len(actual) != len(testCase.Expected)) {
         t.Errorf("(%s) -- Length of expected (%d) and actual (%d) do not match", testCase.Name, len(testCase.Expected), len(actual));
         continue;
      }

      // Go over each value explicitly to make output more readable.
      for i, _ := range(actual) {
         if (actual[i] != testCase.Expected[i]) {
            t.Errorf("(%s)[%d] -- Bad classification. Expected: %v, Got: %v", testCase.Name, i, testCase.Expected[i], actual[i]);
         }
      }
   }
}
