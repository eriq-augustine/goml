package classification

import (
   "sort"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
)

type Knn struct {
   k int
   reducer features.Reducer
   distancer base.Distancer
   trainingData []base.NumericTuple
}

func NewKnn(k int, reducer features.Reducer, distancer base.Distancer) *Knn {
   if (k <= 0) {
      panic("k must be >= 1");
   }

   if (reducer == nil) {
      reducer = features.NoReducer{};
   }

   if (distancer == nil) {
      distancer = base.Euclidean{};
   }

   var knn Knn = Knn{
      k: k,
      reducer: reducer,
      distancer: distancer,
      trainingData: nil,
   };

   return &knn;
}

// TODO(eriq): Verify dimensions.
// The Knn now owns |data|.
func (this *Knn) Train(data []base.Tuple) {
   this.reducer.Init(data);
   data = this.reducer.Reduce(data);

   this.trainingData = make([]base.NumericTuple, len(data));
   for i, tuple := range(data) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("KNN only supports taining on NumericTuple");
      }

      this.trainingData[i] = numericTuple;
   }
}

// TODO(eriq): Verify dimensions.
// TODO(eriq): Parallelize
func (this Knn) Classify(tuples []base.Tuple) []base.Feature {
   tuples = this.reducer.Reduce(tuples);

   var results []base.Feature = make([]base.Feature, len(tuples));

   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("KNN only supports classifying NumericTuple");
      }

      results[i] = this.classifySingle(numericTuple);
   }

   return results;
}

func (this Knn) classifySingle(numericTuple base.NumericTuple) base.Feature {
   var distances []DistanceRecord = make([]DistanceRecord, len(this.trainingData));

   for i, trainingTuple := range this.trainingData {
      distances[i] = DistanceRecord{this.distancer.Distance(trainingTuple, numericTuple), i};
   }

   sort.Sort(ByDistance(distances));

   var classes map[base.Feature]int = make(map[base.Feature]int);
   for i := 0; i < this.k; i++ {
      var targetTuple base.Tuple = this.trainingData[distances[i].Index];
      count, ok := classes[targetTuple.GetClass()];
      if ok {
         classes[targetTuple.GetClass()] = count + 1;
      } else {
         classes[targetTuple.GetClass()] = 1;
      }
   }

   return bestClass(classes);
}

func bestClass(classes map[base.Feature]int) base.Feature {
   var bestCount int = -1;
   var bestValue base.Feature = nil;

   for value, count := range(classes) {
      if (bestCount == -1 || count > bestCount) {
         bestCount = count;
         bestValue = value;
      }
   }

   return bestValue;
}

type DistanceRecord struct {
   Distance float64
   Index    int
}

type ByDistance []DistanceRecord;

func (a ByDistance) Len() int {
   return len(a);
}

func (a ByDistance) Swap(i, j int) {
   a[i], a[j] = a[j], a[i];
}

func (a ByDistance) Less(i, j int) bool {
   return a[i].Distance < a[j].Distance;
}
