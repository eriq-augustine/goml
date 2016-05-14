package util

import (
   "testing"

   "github.com/eriq-augustine/goml/base"
)

type discretizeTestData struct {
   Name string
   Min float64
   Max float64
   NumBuckets int
   RawTuple base.Tuple
   DiscreteTuple base.Tuple
}

func TestDiscretizeNumericBase(t *testing.T) {
   var testData []discretizeTestData = []discretizeTestData{
      // Bad Input
      discretizeTestData{
         "Zero Buckets",
         0,
         0,
         0,
         base.Tuple{[]interface{}{1}, "A"},
         base.Tuple{[]interface{}{1}, "A"},
      },
      discretizeTestData{
         "Negative Buckets",
         0,
         0,
         -1,
         base.Tuple{[]interface{}{1}, "A"},
         base.Tuple{[]interface{}{1}, "A"},
      },
      discretizeTestData{
         "Min > Max",
         1,
         0,
         1,
         base.Tuple{[]interface{}{1}, "A"},
         base.Tuple{[]interface{}{1}, "A"},
      },
      // Real
      discretizeTestData{
         "One Bucket",
         1,
         5,
         1,
         base.Tuple{[]interface{}{1, 2, 3, 4, 5}, "A"},
         base.Tuple{[]interface{}{0, 0, 0, 0, 0}, "A"},
      },
      discretizeTestData{
         "Two Buckets",
         1,
         5,
         2,
         base.Tuple{[]interface{}{1, 2, 3, 4, 5}, "A"},
         base.Tuple{[]interface{}{0, 0, 1, 1, 1}, "A"},
      },
      discretizeTestData{
         "Data Less Than Min",
         1,
         5,
         2,
         base.Tuple{[]interface{}{0, 1, 2, 3, 4, 5}, "A"},
         base.Tuple{[]interface{}{0, 0, 0, 1, 1, 1}, "A"},
      },
      discretizeTestData{
         "Data Greater Than Max",
         1,
         5,
         2,
         base.Tuple{[]interface{}{1, 2, 3, 4, 5, 6}, "A"},
         base.Tuple{[]interface{}{0, 0, 1, 1, 1, 1}, "A"},
      },
      discretizeTestData{
         "Ten",
         1,
         10,
         10,
         base.Tuple{[]interface{}{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "A"},
         base.Tuple{[]interface{}{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "A"},
      },
      discretizeTestData{
         "Negative Ten",
         -10,
         -1,
         10,
         base.Tuple{[]interface{}{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "A"},
         base.Tuple{[]interface{}{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, "A"},
      },
      discretizeTestData{
         "+/- 1",
         -1,
         1,
         4,
         base.Tuple{[]interface{}{-1.00, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.00}, "A"},
         base.Tuple{[]interface{}{0, 0, 1, 1, 2, 2, 3, 3, 3}, "A"},
      },
   };

   for _, testCase := range(testData) {
      var actual base.Tuple = DiscretizeNumeric(testCase.RawTuple, testCase.Min, testCase.Max, testCase.NumBuckets);
      if (!actual.Equals(testCase.DiscreteTuple)) {
         t.Errorf("Failed discretization (%s). Expected: %v, Got: %v", testCase.Name, testCase.DiscreteTuple, actual);
      }
   }
}
