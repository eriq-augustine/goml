package classification

import (
   "math"
   "testing"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
   "github.com/eriq-augustine/goml/util"
)

type knnTestCase struct {
   Name string
   K int
   Reducer features.Reducer
   Distancer base.Distancer
   TestData []base.Tuple
   Input []base.Tuple
   ExpectedClasses []base.Feature
   ExpectedConfidences []float64
}

// Confidence used in many of the test cases.
// Where k = 3, all nearest neighbors match the class and the distances are
// 1, 2, and 3.
var baseConfidence float64 = (1.0 / (math.Sqrt(2) + math.Sqrt(8) + math.Sqrt(18) + 1.0)) + 6.0;

func TestKnnBase(t *testing.T) {
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
         []float64{
            baseConfidence,
            baseConfidence,
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
         []float64{
            baseConfidence,
            baseConfidence,
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
         []float64{
            baseConfidence,
            baseConfidence,
         },
      },
      knnTestCase{
         "Confidence - 1",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 0}, "A"),
            base.NewIntTuple([]interface{}{0, 1}, "A"),
            base.NewIntTuple([]interface{}{-1, 0}, "A"),
            base.NewIntTuple([]interface{}{-100, -100}, "B"),
            base.NewIntTuple([]interface{}{-900, -900}, "B"),
            base.NewIntTuple([]interface{}{-110, -110}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.String("A"),
         },
         []float64{
            6.25,
         },
      },
      knnTestCase{
         "Confidence - 2",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 0}, "A"),
            base.NewIntTuple([]interface{}{0, 10}, "A"),
            base.NewIntTuple([]interface{}{-10, 0}, "A"),
            base.NewIntTuple([]interface{}{-100, -100}, "B"),
            base.NewIntTuple([]interface{}{-900, -900}, "B"),
            base.NewIntTuple([]interface{}{-110, -110}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.String("A"),
         },
         []float64{
            6.0 + (1.0 / 31.0),
         },
      },
      knnTestCase{
         "Confidence - 3",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 0}, "B"),
            base.NewIntTuple([]interface{}{0, 1}, "A"),
            base.NewIntTuple([]interface{}{-1, 0}, "A"),
            base.NewIntTuple([]interface{}{-100, -100}, "B"),
            base.NewIntTuple([]interface{}{-900, -900}, "B"),
            base.NewIntTuple([]interface{}{-110, -110}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.String("A"),
         },
         []float64{
            4.5,
         },
      },
      knnTestCase{
         "Confidence - 4",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewNumericTuple([]interface{}{0.5, 0}, "B"),
            base.NewIntTuple([]interface{}{0, 1}, "A"),
            base.NewIntTuple([]interface{}{-1, 0}, "A"),
            base.NewIntTuple([]interface{}{-100, -100}, "B"),
            base.NewIntTuple([]interface{}{-900, -900}, "B"),
            base.NewIntTuple([]interface{}{-110, -110}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.String("A"),
         },
         []float64{
            4.4,
         },
      },
      knnTestCase{
         "Confidence - 5",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 0}, "B"),
            base.NewIntTuple([]interface{}{0, 10}, "A"),
            base.NewIntTuple([]interface{}{-10, 0}, "A"),
            base.NewIntTuple([]interface{}{-100, -100}, "B"),
            base.NewIntTuple([]interface{}{-900, -900}, "B"),
            base.NewIntTuple([]interface{}{-110, -110}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.String("A"),
         },
         []float64{
            4.0 + (1.0 / 11.0),
         },
      },
      knnTestCase{
         "Confidence - 6",
         3,
         features.NoReducer{},
         base.Euclidean{},
         []base.Tuple{
            base.NewNumericTuple([]interface{}{0.5, 0}, "B"),
            base.NewIntTuple([]interface{}{0, 10}, "A"),
            base.NewIntTuple([]interface{}{-10, 0}, "A"),
            base.NewIntTuple([]interface{}{-100, -100}, "B"),
            base.NewIntTuple([]interface{}{-900, -900}, "B"),
            base.NewIntTuple([]interface{}{-110, -110}, "B"),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.String("A"),
         },
         []float64{
            4.0 + (1.0 / 20.5),
         },
      },
   };

   for _, testCase := range(testCases) {
      var knn Classifier = NewKnn(testCase.K, testCase.Reducer, testCase.Distancer);
      knn.Train(testCase.TestData);
      var actualClasses []base.Feature;
      var actualConfidences []float64;

      actualClasses, actualConfidences = knn.Classify(testCase.Input);

      if (len(actualClasses) != len(testCase.ExpectedClasses)) {
         t.Errorf("(%s) -- Length of expected (%d) and actual classes (%d) do not match", testCase.Name, len(testCase.ExpectedClasses), len(actualClasses));
         continue;
      }

      if (len(actualConfidences) != len(testCase.ExpectedConfidences)) {
         t.Errorf("(%s) -- Length of expected (%d) and actual classes (%d) do not match", testCase.Name, len(testCase.ExpectedClasses), len(actualClasses));
         continue;
      }

      // Go over each value explicitly to make output more readable.
      for i, _ := range(actualClasses) {
         if (actualClasses[i] != testCase.ExpectedClasses[i]) {
            t.Errorf("(%s)[%d] -- Bad classification. Expected classes: %v, Got: %v", testCase.Name, i, testCase.ExpectedClasses[i], actualClasses[i]);
         }

         if (!util.FloatEquals(actualConfidences[i], testCase.ExpectedConfidences[i])) {
            t.Errorf("(%s)[%d] -- Bad classification. Expected confidence: %v, Got: %v", testCase.Name, i, testCase.ExpectedConfidences[i], actualConfidences[i]);
         }
      }
   }
}
