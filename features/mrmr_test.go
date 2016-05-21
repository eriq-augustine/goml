package features

import (
   "testing"

   "github.com/eriq-augustine/goml/base"
)

type mrmrTestData struct {
   Name string
   Features Features
   Data []base.Tuple
   ReducedData []base.Tuple
}

func TestDiscretizeNumericFeatureBase(t *testing.T) {
   var testData []mrmrTestData = []mrmrTestData{
      mrmrTestData{
         "Base",
         nil,
         []base.Tuple{
            base.Tuple{[]interface{}{1}, "A"},
         },
         []base.Tuple{
            base.Tuple{[]interface{}{1}, "A"},
         },
      },
      /*
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
      */
   };

   for _, testCase := range(testData) {
      var reducer MRMRReducer;
      reducer.Init(testCase.Features, testCase.Data);

      var actual []base.Tuple = reducer.Reduce(testCase.Data);
      if (!dataEquals(actual, testCase.ReducedData)) {
         t.Errorf("Failed mRMR reduction (%s). Expected: %v, Got: %v", testCase.Name, testCase.ReducedData, actual);
      }
   }
}

func dataEquals(a []base.Tuple, b []base.Tuple) bool {
   for i, _ := range(a) {
      if (!a[i].Equals(b[i])) {
         return false;
      }
   }
   return true;
}
