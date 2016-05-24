package classification

import (
   "testing"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
)

type knnTestCase struct {
   Name string
   K int
   Reducer features.Reducer
   Distancer base.Distancer
   TestData []base.Tuple
   Input []base.Tuple
   Expected []base.Feature
}

func TestBase(t *testing.T) {
   var testCases []knnTestCase = []knnTestCase{
      knnTestCase{
         "Base - 3",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, "A"),
            base.NewIntTuple([]interface{}{9, 9}, "A"),
            base.NewIntTuple([]interface{}{11, 11}, "A"),
            base.NewIntTuple([]interface{}{-10, -10}, "B"),
            base.NewIntTuple([]interface{}{-9, -9}, "B"),
            base.NewIntTuple([]interface{}{-11, -11}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.String("A"),
            base.String("B"),
         },
      },
      knnTestCase{
         "Defaults - 3",
         3,
         nil,
         nil,
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, "A"),
            base.NewIntTuple([]interface{}{9, 9}, "A"),
            base.NewIntTuple([]interface{}{11, 11}, "A"),
            base.NewIntTuple([]interface{}{-10, -10}, "B"),
            base.NewIntTuple([]interface{}{-9, -9}, "B"),
            base.NewIntTuple([]interface{}{-11, -11}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.String("A"),
            base.String("B"),
         },
      },
      knnTestCase{
         "Reduced - 3",
         3,
         features.NewManualReducer([]int{1, 3}),
         base.Euclidean{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{1,  10, 0,  10, 6}, "A"),
            base.NewIntTuple([]interface{}{2,  9,  0,  9,  5}, "A"),
            base.NewIntTuple([]interface{}{3,  11, 0,  11, 4}, "A"),
            base.NewIntTuple([]interface{}{4, -10, 0, -10, 3}, "B"),
            base.NewIntTuple([]interface{}{5, -9,  0, -9,  2}, "B"),
            base.NewIntTuple([]interface{}{6, -11, 0, -11, 1}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{1,  8, 0,  8, 2}, nil),
            base.NewIntTuple([]interface{}{2, -8, 0, -8, 1}, nil),
         },
         []base.Feature{
            base.String("A"),
            base.String("B"),
         },
      },
   };

   for _, testCase := range(testCases) {
      var knn Classifier = NewKnn(testCase.K, testCase.Reducer, testCase.Distancer);
      knn.Train(testCase.TestData);
      var actual []base.Feature = knn.Classify(testCase.Input);

      if (len(actual) != len(testCase.Expected)) {
         t.Errorf("(%s) -- Length of expected (%d) and actual (%d) do not match", testCase.Name, len(testCase.Expected), len(actual));
         continue;
      }

      // Go over each value explicitly to make output more readable.
      for i, _ := range(actual) {
         if (actual[i] != testCase.Expected[i]) {
            t.Errorf("(%s)[%d] -- Bad classification. Expected: %v, Got: %v", testCase.Expected[i], actual[i]);
         }
      }
   }
}
