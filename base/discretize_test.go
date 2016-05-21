package base

import (
   "testing"
)

type discretizeFeaturesTestData struct {
   Name string
   NumBuckets int
   RawTuples []Tuple
   DiscreteTuples []IntTuple
}

func TestDiscretizeNumericFeatureBase(t *testing.T) {
   var testData []discretizeFeaturesTestData = []discretizeFeaturesTestData{
      // Bad Input
      discretizeFeaturesTestData{
         "Zero Buckets",
         0,
         []Tuple{
            NewNumericTuple([]interface{}{1}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0}, "A"),
         },
      },
      discretizeFeaturesTestData{
         "Negative Buckets",
         -1,
         []Tuple{
            NewNumericTuple([]interface{}{1}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0}, "A"),
         },
      },
      // Real
      discretizeFeaturesTestData{
         "One Bucket",
         1,
         []Tuple{
            NewNumericTuple([]interface{}{1, 4, 7, 1, 4}, "A"),
            NewNumericTuple([]interface{}{2, 5, 8, 2, 5}, "A"),
            NewNumericTuple([]interface{}{3, 6, 9, 3, 6}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0, 0, 0, 0, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0, 0, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0, 0, 0}, "A"),
         },
      },
      discretizeFeaturesTestData{
         "Two Buckets",
         2,
         []Tuple{
            NewNumericTuple([]interface{}{1, 10, -1.0, -1, -10}, "A"),
            NewNumericTuple([]interface{}{2, 20, -0.5, -2, -20}, "A"),
            NewNumericTuple([]interface{}{3, 30,  0.0, -3, -30}, "A"),
            NewNumericTuple([]interface{}{4, 40,  0.5, -4, -40}, "A"),
            NewNumericTuple([]interface{}{5, 50,  1.0, -5, -50}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0, 0, 0, 1, 1}, "A"),
            NewIntTuple([]interface{}{0, 0, 0, 1, 1}, "A"),
            NewIntTuple([]interface{}{1, 1, 1, 1, 1}, "A"),
            NewIntTuple([]interface{}{1, 1, 1, 0, 0}, "A"),
            NewIntTuple([]interface{}{1, 1, 1, 0, 0}, "A"),
         },
      },
      discretizeFeaturesTestData{
         "Uneven Distribution",
         4,
         []Tuple{
            NewNumericTuple([]interface{}{1, 100000, 5,  0.0001}, "A"),
            NewNumericTuple([]interface{}{20, 20000, 10, 0.001}, "A"),
            NewNumericTuple([]interface{}{300, 3000, 1,  0.01}, "A"),
            NewNumericTuple([]interface{}{13000,400, 2,  0.1}, "A"),
            NewNumericTuple([]interface{}{50000, 50, 3,  0.0}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0, 3, 1, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 3, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0, 0}, "A"),
            NewIntTuple([]interface{}{1, 0, 0, 3}, "A"),
            NewIntTuple([]interface{}{3, 0, 0, 0}, "A"),
         },
      },
      discretizeFeaturesTestData{
         "Dups",
         2,
         []Tuple{
            NewNumericTuple([]interface{}{1, 10, -1.0, -1, -10}, "A"),
            NewNumericTuple([]interface{}{1, 10, -1.0, -1, -10}, "A"),
            NewNumericTuple([]interface{}{3, 30,  0.0, -3, -30}, "A"),
            NewNumericTuple([]interface{}{5, 50,  1.0, -5, -50}, "A"),
            NewNumericTuple([]interface{}{5, 50,  1.0, -5, -50}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0, 0, 0, 1, 1}, "A"),
            NewIntTuple([]interface{}{0, 0, 0, 1, 1}, "A"),
            NewIntTuple([]interface{}{1, 1, 1, 1, 1}, "A"),
            NewIntTuple([]interface{}{1, 1, 1, 0, 0}, "A"),
            NewIntTuple([]interface{}{1, 1, 1, 0, 0}, "A"),
         },
      },
      discretizeFeaturesTestData{
         "Same",
         2,
         []Tuple{
            NewNumericTuple([]interface{}{1, 0, -1}, "A"),
            NewNumericTuple([]interface{}{1, 0, -1}, "A"),
            NewNumericTuple([]interface{}{1, 0, -1}, "A"),
            NewNumericTuple([]interface{}{1, 0, -1}, "A"),
            NewNumericTuple([]interface{}{1, 0, -1}, "A"),
         },
         []IntTuple{
            NewIntTuple([]interface{}{0, 0, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0}, "A"),
            NewIntTuple([]interface{}{0, 0, 0}, "A"),
         },
      },
   };

   for _, testCase := range(testData) {
      var actual []IntTuple = DiscretizeNumericFeatures(testCase.RawTuples, testCase.NumBuckets);
      if (!dataEquals(actual, testCase.DiscreteTuples)) {
         t.Errorf("Failed feature discretization (%s). Expected: %v, Got: %v", testCase.Name, testCase.DiscreteTuples, actual);
      }
   }
}

func dataEquals(a []IntTuple, b []IntTuple) bool {
   for i, _ := range(a) {
      if (!TupleEquals(a[i], b[i])) {
         return false;
      }
   }
   return true;
}
