package classification

import (
   "sort"

   "github.com/eriq-augustine/goml/base"
)

type Knn struct {
   k            int
   distancer    base.Distancer
   trainingData []base.NumericTuple
}

func NewKnn(k int, distancer base.Distancer) *Knn {
   if distancer == nil {
      distancer = base.Euclidean{};
   }

   var knn Knn = Knn{
      k:            k,
      distancer:    distancer,
      trainingData: nil,
   };

   return &knn;
}

// TODO(eriq): Verify dimensions.
// The Knn now owns |data|.
func (classy *Knn) Train(data []base.Tuple) {
   classy.trainingData = make([]base.NumericTuple, len(data));
   for i, tuple := range(data) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("KNN only supports taining on NumericTuple");
      }

      classy.trainingData[i] = numericTuple;
   }
}

// TODO(eriq): Verify dimensions.
// TODO(eriq): Parallelize
func (classy Knn) Classify(tuple base.Tuple) interface{} {
   numericTuple, ok := tuple.(base.NumericTuple);
   if (!ok) {
      panic("KNN only supports classifying NumericTuple");
   }

   var distances []DistanceRecord = make([]DistanceRecord, len(classy.trainingData));

   for i, trainingTuple := range classy.trainingData {
      distances[i] = DistanceRecord{classy.distancer.Distance(trainingTuple, numericTuple), i};
   }

   sort.Sort(ByDistance(distances));

   var classes map[interface{}]int = make(map[interface{}]int);
   for i := 0; i < classy.k; i++ {
      var targetTuple base.Tuple = classy.trainingData[distances[i].Index];
      count, ok := classes[targetTuple.GetClass()];
      if ok {
         classes[targetTuple.GetClass()] = count + 1;
      } else {
         classes[targetTuple.GetClass()] = 1;
      }
   }

   return bestClass(classes);
}

// TODO(eriq): Len
func bestClass(classes map[interface{}]int) interface{} {
   var bestCount int = 0;
   var bestValue interface{} = nil;

   for value, count := range classes {
      if count > bestCount {
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
