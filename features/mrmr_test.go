package features

import (
   "testing"

   "github.com/eriq-augustine/goml/base"
)

type mrmrTestData struct {
   Name string
   NumFeatures int
   NumBuckets int
   Data []base.Tuple
   RawTuple base.Tuple
   ReducedTuple base.Tuple
}

func TestDiscretizeNumericFeatureBase(t *testing.T) {
   var testData []mrmrTestData = []mrmrTestData{
      mrmrTestData{
         "Base",
         2,
         3,
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 2, 3}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 3}, "B"),
         },
         base.NewTuple([]interface{}{0, 1, 2}, "A"),
         base.NewTuple([]interface{}{1, 0}, "A"),
      },
      mrmrTestData{
         "FanIn < FanOut",
         5,
         3,
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 2, 3}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 3}, "B"),
         },
         base.NewTuple([]interface{}{0, 1, 2}, "A"),
         base.NewTuple([]interface{}{0, 1, 2}, "A"),
      },
      mrmrTestData{
         "Max Relevance",
         2,
         5,
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 1, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 2}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 3}, "A"),
            base.NewIntTuple([]interface{}{1, 2, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 2, 2}, "A"),
            base.NewIntTuple([]interface{}{1, 2, 3}, "A"),
            base.NewIntTuple([]interface{}{1, 3, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 3, 2}, "A"),
            base.NewIntTuple([]interface{}{1, 3, 3}, "A"),
            base.NewIntTuple([]interface{}{2, 1, 1}, "B"),
            base.NewIntTuple([]interface{}{2, 1, 2}, "B"),
            base.NewIntTuple([]interface{}{2, 1, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 1}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 2}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 3, 1}, "B"),
            base.NewIntTuple([]interface{}{2, 3, 2}, "B"),
            base.NewIntTuple([]interface{}{2, 3, 3}, "B"),
            base.NewIntTuple([]interface{}{3, 5, 1}, "C"),
            base.NewIntTuple([]interface{}{4, 5, 2}, "C"),
            base.NewIntTuple([]interface{}{5, 5, 3}, "C"),
            base.NewIntTuple([]interface{}{3, 5, 1}, "C"),
            base.NewIntTuple([]interface{}{4, 5, 2}, "C"),
            base.NewIntTuple([]interface{}{5, 5, 3}, "C"),
            base.NewIntTuple([]interface{}{3, 5, 1}, "C"),
            base.NewIntTuple([]interface{}{4, 5, 2}, "C"),
            base.NewIntTuple([]interface{}{5, 5, 3}, "C"),
         },
         base.NewTuple([]interface{}{0, 1, 2}, "A"),
         base.NewTuple([]interface{}{0, 1}, "A"),
      },
      mrmrTestData{
         "Min Redundency",
         2,
         5,
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 1, 1, 1, 5}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 1, 1, 5}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 1, 1, 5}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 2, 2, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 2, 2, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 2, 2, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 3, 3, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 3, 3, 1}, "A"),
            base.NewIntTuple([]interface{}{1, 1, 3, 3, 1}, "A"),
            base.NewIntTuple([]interface{}{2, 2, 1, 1, 4}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 1, 1, 4}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 1, 1, 4}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 2, 2, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 2, 2, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 2, 2, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 3, 3, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 3, 3, 3}, "B"),
            base.NewIntTuple([]interface{}{2, 2, 3, 3, 3}, "B"),
            base.NewIntTuple([]interface{}{3, 3, 5, 5, 1}, "C"),
            base.NewIntTuple([]interface{}{4, 4, 5, 5, 2}, "C"),
            base.NewIntTuple([]interface{}{5, 5, 5, 5, 3}, "C"),
            base.NewIntTuple([]interface{}{3, 3, 5, 5, 1}, "C"),
            base.NewIntTuple([]interface{}{4, 4, 5, 5, 2}, "C"),
            base.NewIntTuple([]interface{}{5, 5, 5, 5, 3}, "C"),
            base.NewIntTuple([]interface{}{3, 3, 5, 5, 1}, "C"),
            base.NewIntTuple([]interface{}{4, 4, 5, 5, 2}, "C"),
            base.NewIntTuple([]interface{}{5, 5, 5, 5, 3}, "C"),
         },
         base.NewTuple([]interface{}{0, 1, 2, 3, 4, 5}, "A"),
         base.NewTuple([]interface{}{0, 2}, "A"),
      },
   };

   for _, testCase := range(testData) {
      var reducer *MRMRReducer = NewMRMRReducer(testCase.NumFeatures, testCase.NumBuckets);
      reducer.Init(testCase.Data);

      var actual base.Tuple = reducer.Reduce([]base.Tuple{testCase.RawTuple})[0];
      if (!base.TupleEquals(actual, testCase.ReducedTuple)) {
         t.Errorf("Failed mRMR reduction (%s). Expected: %v, Got: %v", testCase.Name, testCase.ReducedTuple, actual);
      }
   }
}
