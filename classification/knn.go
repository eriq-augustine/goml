package classification

import (
   "sort"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
   "github.com/eriq-augustine/goml/util"
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
func (this Knn) Classify(tuples []base.Tuple) ([]base.Feature, []float64) {
   tuples = this.reducer.Reduce(tuples);

   var results []base.Feature = make([]base.Feature, len(tuples));
   var confidences []float64 = make([]float64, len(tuples));

   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("KNN only supports classifying NumericTuple");
      }

      results[i], confidences[i] = this.classifySingle(numericTuple);
   }

   return results, confidences;
}

func (this Knn) classifySingle(numericTuple base.NumericTuple) (base.Feature, float64) {
   var distances []DistanceRecord = make([]DistanceRecord, len(this.trainingData));

   for i, trainingTuple := range this.trainingData {
      distances[i] = DistanceRecord{this.distancer.Distance(trainingTuple, numericTuple), i};
   }

   sort.Sort(ByDistance(distances));

   // {class -> [distance, ...], ...}
   var classes map[base.Feature][]float64 = make(map[base.Feature][]float64);
   // K Nearest Neighbors
   for i := 0; i < this.k; i++ {
      var targetTuple base.Tuple = this.trainingData[distances[i].Index];
      classDistances, _ := classes[targetTuple.GetClass()];
      // No need to check for existance, on nil a new slice will be created.
      classes[targetTuple.GetClass()] = append(classDistances, distances[i].Distance);
   }

   var bestClass base.Feature = findBestClass(classes);

   return bestClass, calculateScore(bestClass, classes);
}

// (1 / sum(distances) + sign(sum(distances))) + (2 * k`)
// distances with a class that does not match the target class are negated.
// k` = the number of matching classes (len(classes[bestClass])).
// sign is +/- 1.
func calculateScore(bestClass base.Feature, classes map[base.Feature][]float64) float64 {
   var sum float64 = 0;
   for class, distances := range(classes) {
      for _, distance := range(distances) {
         if (class == bestClass) {
            sum += distance;
         } else {
            sum -= distance;
         }
      }
   }

   return (1.0 / (sum + float64(util.Sign(sum)))) + (2.0 * float64(len(classes[bestClass])));
}

func findBestClass(classes map[base.Feature][]float64) base.Feature {
   var bestCount int = -1;
   var bestValue base.Feature = nil;

   for value, distances := range(classes) {
      if (bestCount == -1 || len(distances) > bestCount) {
         bestCount = len(distances);
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
