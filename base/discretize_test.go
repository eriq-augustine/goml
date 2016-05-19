package base

import (
   "testing"
)

type discretizeTupleTestData struct {
   Name string
   Min float64
   Max float64
   NumBuckets int
   RawTuple Tuple
   DiscreteTuple Tuple
}

type discretizeFeaturesTestData struct {
   Name string
   NumBuckets int
   RawTuples []Tuple
   DiscreteTuples []Tuple
}

func TestDiscretizeNumericFeatureBase(t *testing.T) {
   var testData []discretizeFeaturesTestData = []discretizeFeaturesTestData{
      // Bad Input
      discretizeFeaturesTestData{
         "Zero Buckets",
         0,
         []Tuple{
            Tuple{[]interface{}{1}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{1}, "A"},
         },
      },
      discretizeFeaturesTestData{
         "Negative Buckets",
         -1,
         []Tuple{
            Tuple{[]interface{}{1}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{1}, "A"},
         },
      },
      // Real
      discretizeFeaturesTestData{
         "One Bucket",
         1,
         []Tuple{
            Tuple{[]interface{}{1, 4, 7, 1, 4}, "A"},
            Tuple{[]interface{}{2, 5, 8, 2, 5}, "A"},
            Tuple{[]interface{}{3, 6, 9, 3, 6}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{0, 0, 0, 0, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0, 0, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0, 0, 0}, "A"},
         },
      },
      discretizeFeaturesTestData{
         "Two Buckets",
         2,
         []Tuple{
            Tuple{[]interface{}{1, 10, -1.0, -1, -10}, "A"},
            Tuple{[]interface{}{2, 20, -0.5, -2, -20}, "A"},
            Tuple{[]interface{}{3, 30,  0.0, -3, -30}, "A"},
            Tuple{[]interface{}{4, 40,  0.5, -4, -40}, "A"},
            Tuple{[]interface{}{5, 50,  1.0, -5, -50}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{0, 0, 0, 1, 1}, "A"},
            Tuple{[]interface{}{0, 0, 0, 1, 1}, "A"},
            Tuple{[]interface{}{1, 1, 1, 1, 1}, "A"},
            Tuple{[]interface{}{1, 1, 1, 0, 0}, "A"},
            Tuple{[]interface{}{1, 1, 1, 0, 0}, "A"},
         },
      },
      discretizeFeaturesTestData{
         "Uneven Distribution",
         4,
         []Tuple{
            Tuple{[]interface{}{1, 100000, 5,  0.0001}, "A"},
            Tuple{[]interface{}{20, 20000, 10, 0.001}, "A"},
            Tuple{[]interface{}{300, 3000, 1,  0.01}, "A"},
            Tuple{[]interface{}{13000,400, 2,  0.1}, "A"},
            Tuple{[]interface{}{50000, 50, 3,  0.0}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{0, 3, 1, 0}, "A"},
            Tuple{[]interface{}{0, 0, 3, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0, 0}, "A"},
            Tuple{[]interface{}{1, 0, 0, 3}, "A"},
            Tuple{[]interface{}{3, 0, 0, 0}, "A"},
         },
      },
      discretizeFeaturesTestData{
         "Dups",
         2,
         []Tuple{
            Tuple{[]interface{}{1, 10, -1.0, -1, -10}, "A"},
            Tuple{[]interface{}{1, 10, -1.0, -1, -10}, "A"},
            Tuple{[]interface{}{3, 30,  0.0, -3, -30}, "A"},
            Tuple{[]interface{}{5, 50,  1.0, -5, -50}, "A"},
            Tuple{[]interface{}{5, 50,  1.0, -5, -50}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{0, 0, 0, 1, 1}, "A"},
            Tuple{[]interface{}{0, 0, 0, 1, 1}, "A"},
            Tuple{[]interface{}{1, 1, 1, 1, 1}, "A"},
            Tuple{[]interface{}{1, 1, 1, 0, 0}, "A"},
            Tuple{[]interface{}{1, 1, 1, 0, 0}, "A"},
         },
      },
      discretizeFeaturesTestData{
         "Same",
         2,
         []Tuple{
            Tuple{[]interface{}{1, 0, -1}, "A"},
            Tuple{[]interface{}{1, 0, -1}, "A"},
            Tuple{[]interface{}{1, 0, -1}, "A"},
            Tuple{[]interface{}{1, 0, -1}, "A"},
            Tuple{[]interface{}{1, 0, -1}, "A"},
         },
         []Tuple{
            Tuple{[]interface{}{0, 0, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0}, "A"},
            Tuple{[]interface{}{0, 0, 0}, "A"},
         },
      },
   };

   for _, testCase := range(testData) {
      var actual []Tuple = DiscretizeNumericFeatures(testCase.RawTuples, testCase.NumBuckets);
      if (!dataEquals(actual, testCase.DiscreteTuples)) {
         t.Errorf("Failed feature discretization (%s). Expected: %v, Got: %v", testCase.Name, testCase.DiscreteTuples, actual);
      }
   }
}

func dataEquals(a []Tuple, b []Tuple) bool {
   for i, _ := range(a) {
      if (!a[i].Equals(b[i])) {
         return false;
      }
   }
   return true;
}

func TestDiscretizeNumericTupleBase(t *testing.T) {
   var testData []discretizeTupleTestData = []discretizeTupleTestData{
      // Bad Input
      discretizeTupleTestData{
         "Zero Buckets",
         0,
         0,
         0,
         Tuple{[]interface{}{1}, "A"},
         Tuple{[]interface{}{1}, "A"},
      },
      discretizeTupleTestData{
         "Negative Buckets",
         0,
         0,
         -1,
         Tuple{[]interface{}{1}, "A"},
         Tuple{[]interface{}{1}, "A"},
      },
      discretizeTupleTestData{
         "Min > Max",
         1,
         0,
         1,
         Tuple{[]interface{}{1}, "A"},
         Tuple{[]interface{}{1}, "A"},
      },
      // Real
      discretizeTupleTestData{
         "One Bucket",
         1,
         5,
         1,
         Tuple{[]interface{}{1, 2, 3, 4, 5}, "A"},
         Tuple{[]interface{}{0, 0, 0, 0, 0}, "A"},
      },
      discretizeTupleTestData{
         "Two Buckets",
         1,
         5,
         2,
         Tuple{[]interface{}{1, 2, 3, 4, 5}, "A"},
         Tuple{[]interface{}{0, 0, 1, 1, 1}, "A"},
      },
      discretizeTupleTestData{
         "Data Less Than Min",
         1,
         5,
         2,
         Tuple{[]interface{}{0, 1, 2, 3, 4, 5}, "A"},
         Tuple{[]interface{}{0, 0, 0, 1, 1, 1}, "A"},
      },
      discretizeTupleTestData{
         "Data Greater Than Max",
         1,
         5,
         2,
         Tuple{[]interface{}{1, 2, 3, 4, 5, 6}, "A"},
         Tuple{[]interface{}{0, 0, 1, 1, 1, 1}, "A"},
      },
      discretizeTupleTestData{
         "Ten",
         1,
         10,
         10,
         Tuple{[]interface{}{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "A"},
         Tuple{[]interface{}{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "A"},
      },
      discretizeTupleTestData{
         "Negative Ten",
         -10,
         -1,
         10,
         Tuple{[]interface{}{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "A"},
         Tuple{[]interface{}{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, "A"},
      },
      discretizeTupleTestData{
         "+/- 1",
         -1,
         1,
         4,
         Tuple{[]interface{}{-1.00, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.00}, "A"},
         Tuple{[]interface{}{0, 0, 1, 1, 2, 2, 3, 3, 3}, "A"},
      },
   };

   for _, testCase := range(testData) {
      var actual Tuple = DiscretizeNumericTuple(testCase.RawTuple, testCase.Min, testCase.Max, testCase.NumBuckets);
      if (!actual.Equals(testCase.DiscreteTuple)) {
         t.Errorf("Failed tuple discretization (%s). Expected: %v, Got: %v", testCase.Name, testCase.DiscreteTuple, actual);
      }
   }
}
